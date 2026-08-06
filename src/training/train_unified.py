"""
Unified k-fold trainer. Replaces the fifteen train_kfold_*.py scripts.

WHY
---
Measured on the existing repository:
    train_kfold_v3c_focal.py  vs  train_kfold_v4_focal.py   ->  3 differing lines
    train_kfold_v2.py         vs  train_kfold_v3c_focal.py  ->  3 differing lines
    the three *_multihead.py variants  -> 543 lines each, near-identical
    the three *_dann.py variants       -> 434 lines each, near-identical

That is roughly four distinct programs copy-pasted twelve times, with the
differences almost entirely manifest paths that were already selectable via
--manifest. This file collapses them into one entry point.

Improvements folded in while consolidating:
  * torch.amp instead of the deprecated torch.cuda.amp
  * worker_init_fn + explicit generators, so augmentation is seed-reproducible
  * SCRATCH resolved with a fallback and a clear error message
  * per-image test predictions written to predictions.csv for every fold, so
    patient-level evaluation, bootstrap CIs, and calibration need no re-inference
  * the full argument namespace saved to config.json for provenance

Examples
--------
# plain ResNet50, corrected v3 manifest, weighted CE
python -m src.training.train_unified --arch resnet50 --manifest v3c --tag v3c_rn50_full

# focal loss + balanced sampler
python -m src.training.train_unified --arch resnet50 --manifest v3c \
    --loss focal --focal-gamma 2.0 --balanced-sampler --tag v3c_rn50_focal

# multi-head with Cyst gating
python -m src.training.train_unified --arch multihead --manifest v3c \
    --cyst-source both --lambda-seg 0.3 --tag v3c_mh_both

# DANN
python -m src.training.train_unified --arch dann --manifest v3c \
    --dann-lambda 0.1 --dann-schedule linear_ramp --tag v3c_dann

# arbitrary manifest (e.g. a dilated-ROI build)
python -m src.training.train_unified --arch resnet50 \
    --manifest-path $SCRATCH/kidney-data/processed/roi_dilated/margin_m025/manifest_with_folds.csv \
    --tag roi_margin_025
"""
import os
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.factory import build_model, forward_class_logits, probs_from_logits

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MANIFEST_DIRS = {
    "v2": "kidney-data/processed/unified_v2",
    "v2_roi": "kidney-data/processed/unified_v2_roi",
    "v3": "kidney-data/processed/unified_v3",
    "v3_roi": "kidney-data/processed/unified_v3_roi",
    "v3c": "kidney-data/processed/unified_v3_corrected",
    "v3c_roi": "kidney-data/processed/unified_v3c_roi",
    "v4": "kidney-data/processed/unified_v4",
    "v4_roi": "kidney-data/processed/unified_v4_roi",
}


# ===========================================================================
# Data
# ===========================================================================
class KidneyDataset(Dataset):
    """Returns (image, label, source_idx, has_mask, mask, index)."""

    def __init__(self, df, transform, source2idx, mask_col=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.source2idx = source2idx
        self.mask_col = mask_col if (mask_col and mask_col in self.df.columns) else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        im = self.transform(im)

        mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        has_mask = 0
        if self.mask_col:
            mp = row[self.mask_col]
            if isinstance(mp, str) and mp and Path(mp).exists():
                m = Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE),
                                                       Image.NEAREST)
                mask = torch.from_numpy(
                    (np.array(m) > 127).astype(np.float32))[None]
                has_mask = 1

        return (im, LABEL2IDX[row["label"]],
                self.source2idx[row["source"]], has_mask, mask, idx)


def build_train_transforms(strength):
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if strength == "mild":
        base += [transforms.RandomHorizontalFlip(0.5)]
    elif strength == "moderate":
        base += [transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomRotation(10),
                 transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    elif strength == "strong":
        base += [transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomRotation(10),
                 transforms.ColorJitter(brightness=0.1, contrast=0.1),
                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                         scale=(0.9, 1.1)),
                 transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]
    base += [transforms.ToTensor(),
             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(base)


# ===========================================================================
# Losses
# ===========================================================================
class FocalLoss(nn.Module):
    """L = -alpha_c (1 - p_c)^gamma log p_c"""

    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        tlp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        tp = tlp.exp()
        return (-self.alpha[targets] * (1.0 - tp) ** self.gamma * tlp).mean()


def dice_bce_loss(seg_logits, mask, has_mask):
    """Segmentation loss applied only to samples that carry a ground-truth mask."""
    sel = has_mask.bool()
    if sel.sum() == 0:
        return seg_logits.sum() * 0.0
    logits = seg_logits[sel]
    target = mask[sel]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


def multihead_loss(class_logits, targets, source_idx, cyst_gate_ids, n_classes=4):
    """
    Independent BCE per head. The Cyst head is optionally trained only on
    samples from the gated sources, which is the point of using independent
    heads rather than a shared softmax.
    """
    onehot = F.one_hot(targets, num_classes=n_classes).float()
    total = 0.0
    for c in range(n_classes):
        logit_c = class_logits[:, c]
        target_c = onehot[:, c]
        if c == LABEL2IDX["Cyst"] and cyst_gate_ids is not None:
            gate = torch.isin(source_idx, cyst_gate_ids)
            if gate.sum() == 0:
                continue
            logit_c, target_c = logit_c[gate], target_c[gate]
        total = total + F.binary_cross_entropy_with_logits(logit_c, target_c)
    return total


# ===========================================================================
# Reproducibility
# ===========================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def resolve_scratch():
    s = os.environ.get("SCRATCH")
    if not s:
        raise SystemExit(
            "SCRATCH is not set. On Aire: export SCRATCH=/mnt/scratch/$USER\n"
            "Locally you can point it anywhere, e.g. export SCRATCH=$HOME/kidney-scratch"
        )
    return Path(s)


# ===========================================================================
# Main
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--manifest", choices=list(MANIFEST_DIRS), default=None)
    p.add_argument("--manifest-path", default=None,
                   help="Explicit path; overrides --manifest")
    p.add_argument("--seed-suffix", default="")
    p.add_argument("--mask-col", default="mask_path",
                   help="Manifest column holding kidney mask paths (multihead only)")
    # model
    p.add_argument("--arch", default="resnet50",
                   choices=["resnet50", "cbam", "vit", "multihead", "dann"])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-seg", action="store_true",
                   help="Disable the auxiliary segmentation decoder (multihead ablation)")
    # loss
    p.add_argument("--loss", default="ce", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--lambda-seg", type=float, default=0.3)
    p.add_argument("--cyst-source", default="both",
                   choices=["kits", "kauh", "both", "all"],
                   help="Sources on which the Cyst head is trained (multihead only)")
    # DANN
    p.add_argument("--dann-lambda", type=float, default=0.1)
    p.add_argument("--dann-schedule", default="fixed",
                   choices=["fixed", "linear_ramp"])
    # optimisation
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--aug-strength", default="strong",
                   choices=["mild", "moderate", "strong"])
    p.add_argument("--balanced-sampler", action="store_true")
    # runtime
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--folds-to-run", default="all")
    p.add_argument("--no-dump-predictions", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    scratch = resolve_scratch()

    if args.manifest_path:
        manifest = Path(args.manifest_path)
    elif args.manifest:
        manifest = scratch / MANIFEST_DIRS[args.manifest] / \
            f"manifest_with_folds{args.seed_suffix}.csv"
    else:
        raise SystemExit("Provide --manifest or --manifest-path")
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    out = scratch / "kidney-results/kfold" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump({**vars(args), "manifest_resolved": str(manifest)}, f, indent=2)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(manifest)
    sources = sorted(df["source"].unique())
    source2idx = {s: i for i, s in enumerate(sources)}
    print(f"Manifest: {manifest}")
    print(f"  {len(df)} images | {df['patient_id'].nunique()} patients | "
          f"sources {sources}")
    print(f"Arch={args.arch} loss={args.loss} sampler="
          f"{'balanced' if args.balanced_sampler else 'shuffle'} "
          f"lr={args.lr} wd={args.weight_decay} bs={args.batch_size} "
          f"aug={args.aug_strength} seed={args.seed}")

    cyst_gate_ids = None
    if args.arch == "multihead" and args.cyst_source != "all":
        wanted = {"kits": ["kits"], "kauh": ["kauh"],
                  "both": ["kits", "kauh"]}[args.cyst_source]
        ids = [source2idx[s] for s in wanted if s in source2idx]
        cyst_gate_ids = torch.tensor(ids, device=device)
        print(f"  Cyst head gated to sources {wanted} -> indices {ids}")

    train_tf = build_train_transforms(args.aug_strength)
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    folds = list(range(args.n_folds)) if args.folds_to_run == "all" else \
        [int(x) for x in args.folds_to_run.split(",")]

    fold_metrics, pred_frames = [], []
    for fold_idx in folds:
        print(f"\n{'=' * 60}\nFOLD {fold_idx}\n{'=' * 60}")
        fold_out = out / f"fold_{fold_idx}"
        fold_out.mkdir(exist_ok=True)

        test_df = df[df["fold"] == fold_idx].reset_index(drop=True)
        trainval = df[df["fold"] != fold_idx].reset_index(drop=True)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10,
                                     random_state=args.seed)
        tr_i, va_i = next(splitter.split(trainval.index.values,
                                         groups=trainval["patient_id"].values))
        train_df = trainval.iloc[tr_i].reset_index(drop=True)
        val_df = trainval.iloc[va_i].reset_index(drop=True)
        print(f"  train {len(train_df):6d} / val {len(val_df):5d} / "
              f"test {len(test_df):5d} images")
        # Explicit leakage assertion -- cheap, and worth citing in the methodology.
        assert not (set(train_df["patient_id"]) & set(test_df["patient_id"])), \
            "Patient leakage between train and test"
        assert not (set(val_df["patient_id"]) & set(test_df["patient_id"])), \
            "Patient leakage between val and test"

        counts = train_df["label"].value_counts()
        total = len(train_df)
        class_weights = torch.tensor(
            [total / (len(CLASSES) * counts.get(c, 1)) for c in CLASSES],
            dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(args.seed + fold_idx)

        mask_col = args.mask_col if args.arch == "multihead" else None
        train_ds = KidneyDataset(train_df, train_tf, source2idx, mask_col)
        common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=True, worker_init_fn=seed_worker, generator=g)
        if args.balanced_sampler:
            w = torch.tensor([1.0 / counts.get(l, 1) for l in train_df["label"]],
                             dtype=torch.float)
            train_loader = DataLoader(
                train_ds, sampler=WeightedRandomSampler(
                    w, num_samples=len(train_df), replacement=True, generator=g),
                drop_last=True, **common)
        else:
            train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)

        val_loader = DataLoader(KidneyDataset(val_df, eval_tf, source2idx, mask_col),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(KidneyDataset(test_df, eval_tf, source2idx, mask_col),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        model = build_model(args.arch, n_classes=len(CLASSES),
                            n_sources=len(sources), dropout=args.dropout,
                            seg_enabled=not args.no_seg).to(device)

        if args.loss == "focal":
            criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma).to(device)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        source_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        log_path = fold_out / "log.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                    "val_acc", "val_macro_f1", "lambda", "lr"])

        best_val_f1, n_steps = -1.0, args.epochs * max(len(train_loader), 1)
        step = 0
        for epoch in range(args.epochs):
            model.train()
            tl = tc = tn = 0
            cur_lambda = args.dann_lambda
            for img, y, src, has_mask, mask, _ in train_loader:
                img = img.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                src = src.to(device, non_blocking=True)
                has_mask = has_mask.to(device)
                mask = mask.to(device)

                if args.arch == "dann" and args.dann_schedule == "linear_ramp":
                    p = step / max(n_steps, 1)
                    cur_lambda = args.dann_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    if args.arch == "multihead":
                        logits, seg = model(img)
                        loss = multihead_loss(logits, y, src, cyst_gate_ids)
                        if seg is not None:
                            loss = loss + args.lambda_seg * dice_bce_loss(
                                seg, mask, has_mask)
                    elif args.arch == "dann":
                        logits, src_logits = model(img, lambda_=cur_lambda)
                        loss = criterion(logits, y) + source_criterion(src_logits, src)
                    else:
                        logits = model(img)
                        loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1

                tl += loss.item() * img.size(0)
                tc += (logits.argmax(1) == y).sum().item()
                tn += img.size(0)

            tl /= tn
            ta = tc / tn

            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for img, y, _, _, _, _ in val_loader:
                    img = img.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        logits = forward_class_logits(model, args.arch, img)
                    vp.extend(logits.argmax(1).cpu().numpy())
                    vt.extend(y.numpy())
            va = accuracy_score(vt, vp)
            vf1 = f1_score(vt, vp, average="macro", zero_division=0)
            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch+1:02d}/{args.epochs} tr_acc={ta:.3f} "
                  f"val_acc={va:.3f} val_F1={vf1:.3f} lr={lr_now:.2e}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, f"{tl:.6f}", f"{ta:.6f}",
                                        f"{va:.6f}", f"{vf1:.6f}",
                                        f"{cur_lambda:.4f}", f"{lr_now:.2e}"])
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                            "val_macro_f1": vf1, "fold": fold_idx,
                            "arch": args.arch}, fold_out / "best.pth")

        # ---- test ----
        ckpt = torch.load(fold_out / "best.pth", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        probs = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
        with torch.no_grad():
            for img, _, _, _, _, idxs in test_loader:
                img = img.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = forward_class_logits(model, args.arch, img)
                probs[idxs.numpy()] = probs_from_logits(
                    logits.float(), args.arch).cpu().numpy()

        y_true = test_df["label"].map(LABEL2IDX).values
        y_pred = probs.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pcl = precision_recall_fscore_support(y_true, y_pred,
                                              labels=range(len(CLASSES)),
                                              zero_division=0)
        aucs = {}
        for i, c in enumerate(CLASSES):
            b = (y_true == i).astype(int)
            aucs[c] = float(roc_auc_score(b, probs[:, i])) if 0 < b.sum() < len(b) else None
        valid = [v for v in aucs.values() if v is not None]
        macro_auc = float(np.mean(valid)) if valid else float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

        fold_result = {
            "fold": fold_idx, "best_epoch": int(ckpt["epoch"]),
            "best_val_macro_f1": float(best_val_f1),
            "test_accuracy": float(acc), "test_macro_f1": float(macro_f1),
            "test_macro_auc": macro_auc,
            "per_class": {c: {"precision": float(pcl[0][i]),
                              "recall": float(pcl[1][i]),
                              "f1": float(pcl[2][i]),
                              "support": int(pcl[3][i]),
                              "auc": aucs[c]} for i, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist(),
        }
        with open(fold_out / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=2)
        fold_metrics.append(fold_result)

        if not args.no_dump_predictions:
            pf = test_df[["path", "patient_id", "source", "label"]].copy()
            pf.insert(0, "fold", fold_idx)
            pf["y_true"] = y_true
            pf["y_pred"] = y_pred
            for i, c in enumerate(CLASSES):
                pf[f"p_{c}"] = probs[:, i]
            pf.to_csv(fold_out / "predictions.csv", index=False)
            pred_frames.append(pf)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Fold {fold_idx} | acc={acc:.3f} macro-F1={macro_f1:.3f}")
        plt.tight_layout()
        plt.savefig(fold_out / "confusion_matrix.png", dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  fold {fold_idx}: acc={acc:.4f} macroF1={macro_f1:.4f} "
              f"macroAUC={macro_auc:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- aggregate ----
    accs = [m["test_accuracy"] for m in fold_metrics]
    f1s = [m["test_macro_f1"] for m in fold_metrics]
    aucs_ = [m["test_macro_auc"] for m in fold_metrics]
    print(f"\n{'=' * 60}\nAGGREGATE over {len(fold_metrics)} folds\n{'=' * 60}")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Macro-AUC: {np.mean(aucs_):.4f} ± {np.std(aucs_):.4f}")
    for c in CLASSES:
        v = [m["per_class"][c]["f1"] for m in fold_metrics]
        print(f"  {c:<8} F1 = {np.mean(v):.4f} ± {np.std(v):.4f}")

    summary = {
        **vars(args), "manifest_resolved": str(manifest),
        "sources": sources, "fold_results": fold_metrics,
        "aggregate": {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "macro_auc_mean": float(np.mean(aucs_)), "macro_auc_std": float(np.std(aucs_)),
            "per_class_f1": {c: {
                "mean": float(np.mean([m["per_class"][c]["f1"] for m in fold_metrics])),
                "std": float(np.std([m["per_class"][c]["f1"] for m in fold_metrics])),
            } for c in CLASSES},
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if pred_frames:
        pred_dir = scratch / "kidney-results/predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        allp = pd.concat(pred_frames, ignore_index=True)
        allp.to_csv(pred_dir / f"{args.tag}.csv", index=False)
        print(f"\nPredictions: {pred_dir / (args.tag + '.csv')}")
        print("Feed straight into:")
        print(f"  python -m src.evaluation.patient_level_eval --predictions {args.tag}")
        print(f"  python -m src.evaluation.calibration --conditions {args.tag}")

    print(f"Summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
