"""
K-fold training with CBAM-ResNet50.
Identical to train_kfold_v2.py except for the model construction.
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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cbam import add_cbam_to_resnet

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--manifest", type=str, default="v2",
                    choices=["v2", "v2_roi"])
parser.add_argument("--seed-suffix", type=str, default="",
                    help="Suffix for manifest file (e.g. '_seed123'). Default uses seed 42 manifest.")
parser.add_argument("--smoke", action="store_true",
                    help="Run only fold 0 for 3 epochs (smoke test)")
args = parser.parse_args()

SCRATCH = Path(os.environ["SCRATCH"])
manifest_dir_map = {
    "v2": SCRATCH / "kidney-data/processed/unified_v2",
    "v2_roi": SCRATCH / "kidney-data/processed/unified_v2_roi",
}
manifest_name = f"manifest_with_folds{args.seed_suffix}.csv"
MANIFEST = manifest_dir_map[args.manifest] / manifest_name
if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")
print(f"Using manifest: {MANIFEST}")
OUT = SCRATCH / "kidney-results/kfold" / args.tag
OUT.mkdir(parents=True, exist_ok=True)
print(f"Manifest: {args.manifest} ({MANIFEST})")
print(f"Output:   {OUT}")
print(f"Mode:     {'SMOKE TEST' if args.smoke else 'FULL'}")

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
N_FOLDS = 1 if args.smoke else 5
EPOCHS = 3 if args.smoke else args.epochs

print(f"Device: {DEVICE} | Folds: {N_FOLDS} | Epochs: {EPOCHS}")
df = pd.read_csv(MANIFEST)
print(f"Loaded {len(df)} images, {df['patient_id'].nunique()} patients")


class KidneyDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, LABEL2IDX[row["label"]]


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

fold_metrics = []
for fold in range(N_FOLDS):
    print(f"\n{'=' * 60}\nFOLD {fold} / {N_FOLDS - 1}\n{'=' * 60}")
    fold_out = OUT / f"fold_{fold}"
    fold_out.mkdir(exist_ok=True)

    test_df = df[df["fold"] == fold].reset_index(drop=True)
    trainval_df = df[df["fold"] != fold].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(
        trainval_df.index.values, groups=trainval_df["patient_id"].values))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df)} / Val: {len(val_df)} / Test: {len(test_df)}")

    class_counts = train_df["label"].value_counts()
    total = len(train_df)
    class_weights = torch.tensor(
        [total / (len(CLASSES) * class_counts.get(c, 1)) for c in CLASSES],
        dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(KidneyDataset(train_df, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(KidneyDataset(val_df, eval_tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(KidneyDataset(test_df, eval_tf),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # The only difference from train_kfold_v2: inject CBAM
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = add_cbam_to_resnet(model, reduction=16)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: CBAM-ResNet50, {n_params:.1f}M params")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    log_path = fold_out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                 "val_loss", "val_acc", "val_macro_f1", "lr"])

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for img, y in train_loader:
            img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(img); loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tl += loss.item() * img.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tn += img.size(0)
        tl /= tn; ta = tc / tn

        model.eval()
        vl, vc, vn, vp, vt = 0.0, 0, 0, [], []
        with torch.no_grad():
            for img, y in val_loader:
                img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast():
                    out = model(img); loss = criterion(out, y)
                vl += loss.item() * img.size(0)
                p = out.argmax(1)
                vc += (p == y).sum().item(); vn += img.size(0)
                vp.extend(p.cpu().numpy()); vt.extend(y.cpu().numpy())
        vl /= vn; va = vc / vn
        val_f1 = f1_score(vt, vp, average="macro", zero_division=0)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | tr_acc={ta:.3f} | "
              f"val_acc={va:.3f} val_F1={val_f1:.3f} | lr={lr_now:.2e}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{tl:.6f}", f"{ta:.6f}",
                                     f"{vl:.6f}", f"{va:.6f}",
                                     f"{val_f1:.6f}", f"{lr_now:.2e}"])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "val_macro_f1": val_f1, "fold": fold},
                       fold_out / "best.pth")

    # Test
    ckpt = torch.load(fold_out / "best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    t_p, t_t, t_pr = [], [], []
    with torch.no_grad():
        for img, y in test_loader:
            img = img.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(img)
            t_pr.extend(torch.softmax(out, dim=1).cpu().numpy())
            t_p.extend(out.argmax(1).cpu().numpy())
            t_t.extend(y.numpy())
    t_p = np.array(t_p); t_t = np.array(t_t); t_pr = np.array(t_pr)

    acc = accuracy_score(t_t, t_p)
    macro_f1 = f1_score(t_t, t_p, average="macro", zero_division=0)
    pcl = precision_recall_fscore_support(t_t, t_p, labels=range(len(CLASSES)), zero_division=0)
    aucs = {c: float(roc_auc_score((t_t == i).astype(int), t_pr[:, i]))
            if (t_t == i).sum() > 0 and (t_t == i).sum() < len(t_t) else None
            for i, c in enumerate(CLASSES)}
    macro_auc = float(np.mean([v for v in aucs.values() if v is not None]))
    cm = confusion_matrix(t_t, t_p, labels=range(len(CLASSES)))

    fold_result = {
        "fold": fold, "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(best_val_f1),
        "test_accuracy": float(acc), "test_macro_f1": float(macro_f1),
        "test_macro_auc": macro_auc,
        "per_class": {c: {"precision": float(pcl[0][i]), "recall": float(pcl[1][i]),
                          "f1": float(pcl[2][i]), "support": int(pcl[3][i]),
                          "auc": aucs[c]} for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
    }
    with open(fold_out / "metrics.json", "w") as f:
        json.dump(fold_result, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold} | acc={acc:.3f} macro-F1={macro_f1:.3f}")
    plt.tight_layout()
    plt.savefig(fold_out / "confusion_matrix.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"  Fold {fold} test: acc={acc:.4f}, macro-F1={macro_f1:.4f}, macro-AUC={macro_auc:.4f}")
    fold_metrics.append(fold_result)

# Aggregate
if not args.smoke:
    print(f"\n{'=' * 60}\nAGGREGATE\n{'=' * 60}")
    accs = [m["test_accuracy"] for m in fold_metrics]
    f1s = [m["test_macro_f1"] for m in fold_metrics]
    aucs = [m["test_macro_auc"] for m in fold_metrics]
    print(f"Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro-F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Macro-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("\nPer-class F1:")
    for c in CLASSES:
        vals = [m["per_class"][c]["f1"] for m in fold_metrics]
        avals = [m["per_class"][c]["auc"] for m in fold_metrics if m["per_class"][c]["auc"] is not None]
        print(f"  {c:<8} F1={np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"AUC={np.mean(avals):.4f} ± {np.std(avals):.4f}")

    summary = {
        "n_folds": N_FOLDS, "epochs": args.epochs, "tag": args.tag, "manifest": args.manifest,
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
    print(f"\nSaved aggregate summary to {OUT / 'summary.json'}")