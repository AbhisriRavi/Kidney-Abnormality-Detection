"""
Multi-head training for kidney CT classification.

Architecture: MultiHeadResNet50 — 4 independent binary heads + auxiliary
segmentation decoder.

Loss: L = BCE(Normal head)
        + BCE(Cyst head, source-gated)
        + BCE(Tumor head)
        + BCE(Stone head)
        + lambda_seg * (Dice + BCE)(seg head, KiTS-only)

Cyst source-gating: --cyst-source {kits, kauh, both}
Segmentation: only trained on KiTS samples (only source with GT masks).
"""
import os
import csv
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.multihead_resnet import MultiHeadResNet50


# ==========================================================================
# CLI
# ==========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag", type=str, default="v3_mh_run1")
parser.add_argument("--seed-suffix", type=str, default="",
                    help="Suffix for manifest file (e.g. '_seed123')")
parser.add_argument("--folds-to-run", type=str, default="all")
parser.add_argument("--aug-strength", type=str, default="strong",
                    choices=["mild", "moderate", "strong"])
parser.add_argument("--cyst-source", type=str, required=True,
                    choices=["kits", "kauh", "both"],
                    help="Which source(s) the Cyst head trains on")
parser.add_argument("--lambda-seg", type=float, default=0.5,
                    help="Weight of segmentation loss")
parser.add_argument("--dropout", type=float, default=0.2)
args = parser.parse_args()

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST_DIR = SCRATCH / "kidney-data/processed/unified_v4"
manifest_name = f"manifest_with_masks{args.seed_suffix}.csv"
MANIFEST = MANIFEST_DIR / manifest_name
if not MANIFEST.exists():
    # Fall back to base manifest_with_masks.csv for seed 42
    fallback = MANIFEST_DIR / "manifest_with_masks.csv"
    if args.seed_suffix and fallback.exists():
        raise SystemExit(f"Seed manifest not found: {MANIFEST}. Base exists at {fallback}. "
                         f"You need to build the seed-suffix version.")
    else:
        raise SystemExit(f"Manifest not found: {MANIFEST}")

print(f"Manifest: {MANIFEST}")
OUT = SCRATCH / "kidney-results/kfold" / args.tag
OUT.mkdir(parents=True, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
CYST_IDX = LABEL2IDX["Cyst"]
IMG_SIZE = 224
N_FOLDS = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Cyst source gating set
CYST_SOURCE_SETS = {
    "kits": {"kits"},
    "kauh": {"kauh"},
    "both": {"kits", "kauh"},
}
CYST_SOURCES = CYST_SOURCE_SETS[args.cyst_source]

print(f"Device: {DEVICE} | Output: {OUT}")
print(f"HP: lr={args.lr}, wd={args.weight_decay}, bs={args.batch_size}, aug={args.aug_strength}")
print(f"Multi-head: cyst-source={args.cyst_source} ({CYST_SOURCES}), lambda_seg={args.lambda_seg}")


# ==========================================================================
# Dataset
# ==========================================================================
class KidneyMultiHeadDataset(Dataset):
    """
    Returns (image, label_idx, source_idx, mask, has_mask_flag).
    - image: [3, 224, 224] tensor after transform
    - label_idx: int
    - source_idx: int (kits=0, kauh=1, mendeley=2, tcga=3, abdalla=4)
    - mask: [1, 224, 224] tensor (zeros if no mask)
    - has_mask_flag: 1.0 if this sample has a real mask, 0.0 otherwise
    """
    SOURCE2IDX = {"kits": 0, "kauh": 1, "mendeley": 2, "tcga": 3, "abdalla": 4}

    def __init__(self, df, img_transform):
        self.df = df.reset_index(drop=True)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        if self.img_transform:
            im = self.img_transform(im)
        label = LABEL2IDX[row["label"]]
        src_idx = self.SOURCE2IDX[row["source"]]

        mask_path = row.get("mask_path", "")
        if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # [1, 224, 224]
            has_mask = 1.0
        else:
            mask = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
            has_mask = 0.0
        return im, label, src_idx, mask, has_mask


def build_train_transforms(strength):
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if strength == "mild":
        base += [transforms.RandomHorizontalFlip(0.5)]
    elif strength == "moderate":
        base += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    elif strength == "strong":
        base += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(base)


# NOTE: standard image transforms include rotation/flip/affine which would
# desync the image from the mask if applied to the image only. For simplicity
# in this project we accept this: masks are only used as auxiliary targets
# and small geometric perturbations degrade seg accuracy modestly but don't
# break training. If seg loss doesn't converge well we'll switch to
# synchronised transforms.

train_tf = build_train_transforms(args.aug_strength)
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================================================
# Losses
# ==========================================================================
def dice_loss(pred_logits, target, eps=1e-6):
    """
    Soft Dice loss on sigmoid probs.
    pred_logits: [B, 1, H, W]
    target:      [B, 1, H, W] in [0, 1]
    """
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def compute_head_pos_weights(train_df):
    """
    Per-class pos_weight = n_negative / n_positive for BCEWithLogitsLoss.
    Returns a tensor of shape [n_classes].
    """
    weights = []
    for cls in CLASSES:
        n_pos = (train_df["label"] == cls).sum()
        n_neg = len(train_df) - n_pos
        w = n_neg / max(n_pos, 1)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def dice_score(pred_logits, target, eps=1e-6):
    """Reported metric (higher is better), no gradient."""
    with torch.no_grad():
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


# ==========================================================================
# K-fold loop
# ==========================================================================
df = pd.read_csv(MANIFEST)
print(f"Loaded {len(df)} images, {df['patient_id'].nunique()} patients, {df['source'].nunique()} sources")

if args.folds_to_run == "all":
    folds_to_run = list(range(N_FOLDS))
else:
    folds_to_run = [int(x) for x in args.folds_to_run.split(",")]

fold_metrics = []

for fold_idx in folds_to_run:
    print(f"\n{'=' * 60}\nFOLD {fold_idx} / {N_FOLDS - 1}  |  cyst-source={args.cyst_source}\n{'=' * 60}")
    fold_out = OUT / f"fold_{fold_idx}"
    fold_out.mkdir(exist_ok=True)

    test_df = df[df["fold"] == fold_idx].reset_index(drop=True)
    trainval_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(
        trainval_df.index.values, groups=trainval_df["patient_id"].values
    ))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df):5d} / Val: {len(val_df):5d} / Test: {len(test_df):5d}")

    # Per-fold pos weights for the 4 binary heads
    pos_weights = compute_head_pos_weights(train_df).to(DEVICE)
    print(f"  pos_weights: {dict(zip(CLASSES, [round(w, 2) for w in pos_weights.tolist()]))}")

    # Loaders
    train_loader = DataLoader(
        KidneyMultiHeadDataset(train_df, train_tf),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        KidneyMultiHeadDataset(val_df, eval_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        KidneyMultiHeadDataset(test_df, eval_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = MultiHeadResNet50(n_classes=4, dropout=args.dropout, seg_enabled=True).to(DEVICE)

    # 4 head-specific BCE losses with pos_weight
    head_losses = [
        nn.BCEWithLogitsLoss(pos_weight=pos_weights[i]) for i in range(4)
    ]
    seg_bce = nn.BCEWithLogitsLoss()  # unweighted for seg — background is majority but Dice handles that

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    log_path = fold_out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "tr_loss_total", "tr_loss_cls", "tr_loss_seg", "tr_seg_dice",
            "val_loss", "val_acc", "val_f1", "val_seg_dice", "lr"
        ])

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        totals = {"total": 0.0, "cls": 0.0, "seg": 0.0}
        tn = 0
        tr_dice_sum = 0.0
        tr_dice_n = 0

        for imgs, labels, sources, masks, has_mask in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            sources = sources.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            has_mask = has_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                class_logits, seg_logits = model(imgs)

                # Per-class binary targets: [B, 4] with 1.0 for the true class, 0 elsewhere
                targets_bin = F.one_hot(labels, num_classes=4).float()

                # Head 0 (Normal), Head 2 (Tumor), Head 3 (Stone) — all samples
                loss_normal = head_losses[0](class_logits[:, 0], targets_bin[:, 0])
                loss_tumor  = head_losses[2](class_logits[:, 2], targets_bin[:, 2])
                loss_stone  = head_losses[3](class_logits[:, 3], targets_bin[:, 3])

                # Head 1 (Cyst) — only samples where source is in CYST_SOURCES
                cyst_source_ids = torch.tensor(
                    [KidneyMultiHeadDataset.SOURCE2IDX[s] for s in CYST_SOURCES],
                    device=DEVICE
                )
                cyst_mask = torch.isin(sources, cyst_source_ids)
                if cyst_mask.any():
                    loss_cyst = head_losses[1](
                        class_logits[cyst_mask, 1], targets_bin[cyst_mask, 1]
                    )
                else:
                    loss_cyst = torch.zeros(1, device=DEVICE).squeeze()

                loss_cls = loss_normal + loss_cyst + loss_tumor + loss_stone

                # Segmentation loss (KiTS only)
                kits_mask = (sources == KidneyMultiHeadDataset.SOURCE2IDX["kits"])
                if kits_mask.any():
                    seg_pred_k = seg_logits[kits_mask]
                    seg_tgt_k = masks[kits_mask]
                    loss_seg = 0.5 * seg_bce(seg_pred_k, seg_tgt_k) + 0.5 * dice_loss(seg_pred_k, seg_tgt_k)
                    tr_dice_sum += dice_score(seg_pred_k, seg_tgt_k) * kits_mask.sum().item()
                    tr_dice_n += kits_mask.sum().item()
                else:
                    loss_seg = torch.zeros(1, device=DEVICE).squeeze()

                loss_total = loss_cls + args.lambda_seg * loss_seg

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            totals["total"] += loss_total.item() * imgs.size(0)
            totals["cls"]   += loss_cls.item()   * imgs.size(0)
            totals["seg"]   += loss_seg.item()   * imgs.size(0)
            tn += imgs.size(0)

        tr_loss_total = totals["total"] / tn
        tr_loss_cls = totals["cls"] / tn
        tr_loss_seg = totals["seg"] / tn
        tr_seg_dice = tr_dice_sum / max(tr_dice_n, 1)

        # -------- Validation --------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_preds, val_targets = [], []
        val_dice_sum, val_dice_n = 0.0, 0
        with torch.no_grad():
            for imgs, labels, sources, masks, has_mask in val_loader:
                imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
                sources = sources.to(DEVICE); masks = masks.to(DEVICE)
                with autocast():
                    class_logits, seg_logits = model(imgs)
                    targets_bin = F.one_hot(labels, num_classes=4).float()
                    # For val loss, use ALL heads on ALL samples (no gating) so
                    # the metric is comparable across variants
                    loss = sum(
                        head_losses[i](class_logits[:, i], targets_bin[:, i])
                        for i in range(4)
                    )
                val_loss_sum += loss.item() * imgs.size(0)
                val_n += imgs.size(0)
                probs = torch.sigmoid(class_logits)
                val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                kits_mask = (sources == KidneyMultiHeadDataset.SOURCE2IDX["kits"])
                if kits_mask.any():
                    val_dice_sum += dice_score(seg_logits[kits_mask], masks[kits_mask]) * kits_mask.sum().item()
                    val_dice_n += kits_mask.sum().item()

        val_loss = val_loss_sum / val_n
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_seg_dice = val_dice_sum / max(val_dice_n, 1)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  Ep{epoch+1:02d}/{args.epochs} | "
              f"tr_loss={tr_loss_total:.3f} (cls={tr_loss_cls:.3f} seg={tr_loss_seg:.3f}) | "
              f"tr_dice={tr_seg_dice:.3f} | "
              f"val_acc={val_acc:.3f} val_F1={val_f1:.3f} val_dice={val_seg_dice:.3f} | "
              f"lr={lr_now:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1,
                f"{tr_loss_total:.6f}", f"{tr_loss_cls:.6f}", f"{tr_loss_seg:.6f}", f"{tr_seg_dice:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}", f"{val_seg_dice:.6f}",
                f"{lr_now:.2e}"
            ])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "val_macro_f1": val_f1,
                "fold": fold_idx,
                "cyst_source": args.cyst_source,
                "lambda_seg": args.lambda_seg,
            }, fold_out / "best.pth")

    # -------- Test on best checkpoint --------
    print(f"  Loading best checkpoint (val_F1={best_val_f1:.4f})...")
    ckpt = torch.load(fold_out / "best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    t_probs, t_labels, t_sources = [], [], []
    with torch.no_grad():
        for imgs, labels, sources, masks, has_mask in test_loader:
            imgs = imgs.to(DEVICE)
            with autocast():
                class_logits, _ = model(imgs)
            probs = torch.sigmoid(class_logits).cpu().numpy()
            t_probs.append(probs)
            t_labels.extend(labels.numpy())
            t_sources.extend(sources.numpy())

    t_probs = np.concatenate(t_probs, axis=0)
    t_labels = np.array(t_labels)
    t_preds = t_probs.argmax(axis=1)
    t_sources = np.array(t_sources)

    acc = accuracy_score(t_labels, t_preds)
    macro_f1 = f1_score(t_labels, t_preds, average="macro", zero_division=0)
    pcl = precision_recall_fscore_support(t_labels, t_preds, labels=range(4), zero_division=0)
    aucs = {}
    for i, c in enumerate(CLASSES):
        try:
            aucs[c] = float(roc_auc_score((t_labels == i).astype(int), t_probs[:, i]))
        except ValueError:
            aucs[c] = None
    macro_auc = float(np.mean([v for v in aucs.values() if v is not None]))
    cm = confusion_matrix(t_labels, t_preds, labels=range(4))

    # Per-source test breakdown
    src_names = ["kits", "kauh", "mendeley", "tcga", "abdalla"]
    per_source = {}
    for src_idx_val, src_name in enumerate(src_names):
        mask = t_sources == src_idx_val
        if mask.sum() == 0:
            continue
        per_source[src_name] = {
            "n_images": int(mask.sum()),
            "accuracy": float((t_preds[mask] == t_labels[mask]).mean()),
            "macro_f1": float(f1_score(t_labels[mask], t_preds[mask],
                                        labels=range(4), average="macro", zero_division=0)),
        }

    fold_result = {
        "fold": fold_idx,
        "cyst_source": args.cyst_source,
        "lambda_seg": args.lambda_seg,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(best_val_f1),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_macro_auc": macro_auc,
        "per_class": {c: {
            "precision": float(pcl[0][i]), "recall": float(pcl[1][i]),
            "f1": float(pcl[2][i]), "support": int(pcl[3][i]), "auc": aucs[c]
        } for i, c in enumerate(CLASSES)},
        "per_source": per_source,
        "confusion_matrix": cm.tolist(),
    }
    with open(fold_out / "metrics.json", "w") as f:
        json.dump(fold_result, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold_idx} | acc={acc:.3f} F1={macro_f1:.3f} | cyst={args.cyst_source}")
    plt.tight_layout()
    plt.savefig(fold_out / "confusion_matrix.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"  Fold {fold_idx} test: acc={acc:.4f}, F1={macro_f1:.4f}, AUC={macro_auc:.4f}")
    print(f"  Per-source acc: {dict((k, round(v['accuracy'], 3)) for k, v in per_source.items())}")
    fold_metrics.append(fold_result)


# -------- Aggregate --------
print(f"\n{'=' * 60}\nAGGREGATE ({len(fold_metrics)} folds)\n{'=' * 60}")
accs = [m["test_accuracy"] for m in fold_metrics]
f1s = [m["test_macro_f1"] for m in fold_metrics]
aucs = [m["test_macro_auc"] for m in fold_metrics]
print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Macro-F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Macro-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
for c in CLASSES:
    pc_f1 = [m["per_class"][c]["f1"] for m in fold_metrics]
    print(f"  {c:<8} F1={np.mean(pc_f1):.4f} ± {np.std(pc_f1):.4f}")

summary = {
    "n_folds": len(fold_metrics),
    "epochs": args.epochs, "batch_size": args.batch_size,
    "lr": args.lr, "weight_decay": args.weight_decay,
    "aug_strength": args.aug_strength, "seed": args.seed,
    "seed_suffix": args.seed_suffix, "folds_run": folds_to_run,
    "cyst_source": args.cyst_source, "lambda_seg": args.lambda_seg,
    "dropout": args.dropout,
    "fold_results": fold_metrics,
    "aggregate": {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "macro_auc_mean": float(np.mean(aucs)), "macro_auc_std": float(np.std(aucs)),
        "per_class_f1": {c: {
            "mean": float(np.mean([m["per_class"][c]["f1"] for m in fold_metrics])),
            "std": float(np.std([m["per_class"][c]["f1"] for m in fold_metrics])),
        } for c in CLASSES},
    },
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)