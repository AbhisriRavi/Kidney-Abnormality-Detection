"""
KiTS-only re-tune with stronger regularization to address overfitting.
Adds dropout, higher weight decay, lower LR, early stopping.
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


parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, required=True,
                    choices=["resnet50", "efficientnet_b0", "densenet121"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--folds-to-run", type=str, default="all")
parser.add_argument("--aug-strength", type=str, default="strong")
args = parser.parse_args()

if args.tag is None:
    args.tag = f"pdreg_kits_{args.arch}_seed{args.seed}"

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST = SCRATCH / "kidney-data/processed/per_dataset/kits" / f"manifest_seed{args.seed}.csv"
if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")
print(f"Manifest: {MANIFEST}")

OUT = SCRATCH / "kidney-results/per_dataset" / args.tag
OUT.mkdir(parents=True, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Cyst", "Tumor"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
IMG_SIZE = 224
N_FOLDS = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"Device: {DEVICE} | Output: {OUT}")
print(f"KiTS regularized: arch={args.arch} lr={args.lr} wd={args.weight_decay} "
      f"dropout={args.dropout} epochs={args.epochs} patience={args.patience}")


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


def build_train_transforms(strength):
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if strength == "strong":
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


train_tf = build_train_transforms(args.aug_strength)
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(arch, n_classes, dropout):
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feat = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, n_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, n_classes))
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_feat = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, n_classes))
    return m


df = pd.read_csv(MANIFEST)
print(f"Loaded {len(df)} images, {df['patient_id'].nunique()} patients")

if args.folds_to_run == "all":
    folds_to_run = list(range(N_FOLDS))
else:
    folds_to_run = [int(x) for x in args.folds_to_run.split(",")]

fold_metrics = []

for fold_idx in folds_to_run:
    print(f"\n{'=' * 60}\nFOLD {fold_idx}\n{'=' * 60}")
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
    print(f"  Train: {len(train_df)} / Val: {len(val_df)} / Test: {len(test_df)}")

    class_counts = train_df["label"].value_counts()
    total = len(train_df)
    class_weights = torch.tensor(
        [total / (N_CLASSES * class_counts.get(c, 1)) for c in CLASSES],
        dtype=torch.float32
    ).to(DEVICE)

    train_loader = DataLoader(KidneyDataset(train_df, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(KidneyDataset(val_df, eval_tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(KidneyDataset(test_df, eval_tf),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.arch, N_CLASSES, args.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    log_path = fold_out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                 "val_loss", "val_acc", "val_macro_f1", "lr"])

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
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
        print(f"  Ep{epoch+1:02d}/{args.epochs} | tr_acc={ta:.3f} | "
              f"val_acc={va:.3f} val_F1={val_f1:.3f} | lr={lr_now:.2e}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{tl:.6f}", f"{ta:.6f}",
                                     f"{vl:.6f}", f"{va:.6f}",
                                     f"{val_f1:.6f}", f"{lr_now:.2e}"])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "val_macro_f1": val_f1, "fold": fold_idx,
                        "arch": args.arch}, fold_out / "best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"  Loading best checkpoint (val_F1={best_val_f1:.4f})...")
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
    pcl = precision_recall_fscore_support(t_t, t_p, labels=range(N_CLASSES), zero_division=0)
    aucs = {}
    for i, c in enumerate(CLASSES):
        try:
            aucs[c] = float(roc_auc_score((t_t == i).astype(int), t_pr[:, i]))
        except ValueError:
            aucs[c] = None
    valid_aucs = [v for v in aucs.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None
    cm = confusion_matrix(t_t, t_p, labels=range(N_CLASSES))

    fold_result = {
        "fold": fold_idx, "arch": args.arch,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(best_val_f1),
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_macro_auc": macro_auc,
        "per_class": {c: {
            "precision": float(pcl[0][i]), "recall": float(pcl[1][i]),
            "f1": float(pcl[2][i]), "support": int(pcl[3][i]), "auc": aucs[c]
        } for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
        "class_labels": CLASSES,
    }
    with open(fold_out / "metrics.json", "w") as f:
        json.dump(fold_result, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"KiTS reg / {args.arch} fold {fold_idx} | acc={acc:.3f} F1={macro_f1:.3f}")
    plt.tight_layout()
    plt.savefig(fold_out / "confusion_matrix.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"  Fold {fold_idx} test: acc={acc:.4f}, macro-F1={macro_f1:.4f}")
    fold_metrics.append(fold_result)

print(f"\n{'=' * 60}\nAGGREGATE ({len(fold_metrics)} folds)\n{'=' * 60}")
accs = [m["test_accuracy"] for m in fold_metrics]
f1s = [m["test_macro_f1"] for m in fold_metrics]
aucs = [m["test_macro_auc"] for m in fold_metrics if m["test_macro_auc"] is not None]
print(f"Accuracy:  {np.mean(accs):.4f} +- {np.std(accs):.4f}")
print(f"Macro-F1:  {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
if aucs:
    print(f"Macro-AUC: {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
for c in CLASSES:
    pc_f1 = [m["per_class"][c]["f1"] for m in fold_metrics]
    print(f"  {c:<8} F1={np.mean(pc_f1):.4f} +- {np.std(pc_f1):.4f}")

summary = {
    "n_folds": len(fold_metrics),
    "dataset": "kits", "arch": args.arch, "seed": args.seed,
    "epochs": args.epochs, "batch_size": args.batch_size,
    "lr": args.lr, "weight_decay": args.weight_decay,
    "dropout": args.dropout, "patience": args.patience,
    "aug_strength": args.aug_strength,
    "class_labels": CLASSES,
    "fold_results": fold_metrics,
    "aggregate": {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "macro_auc_mean": float(np.mean(aucs)) if aucs else None,
        "macro_auc_std": float(np.std(aucs)) if aucs else None,
        "per_class_f1": {c: {
            "mean": float(np.mean([m["per_class"][c]["f1"] for m in fold_metrics])),
            "std": float(np.std([m["per_class"][c]["f1"] for m in fold_metrics])),
        } for c in CLASSES},
    },
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
