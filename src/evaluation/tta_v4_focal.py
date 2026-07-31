"""
Test-time augmentation on corrected v3 focal + balanced checkpoints.

For each of the 15 fold-model checkpoints (3 seeds × 5 folds):
  1. Load model from v4_focal_seed{42,123,456}/fold_{0..4}/best.pth
  2. For each test image, run 5 TTA versions and average softmax
  3. Compute test metrics with TTA applied

Aggregates across 15 folds and reports comparison to non-TTA focal baseline.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"
V3C_DIR = SCRATCH / "kidney-data/processed/unified_v4"
OUT = SCRATCH / "kidney-results/v4_tta"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEEDS = [42, 123, 456]
RUN_TAGS = {
    42:  "v4_focal_seed42",
    123: "v4_focal_seed123",
    456: "v4_focal_seed456",
}
MANIFESTS = {
    42:  V3C_DIR / "manifest_with_masks.csv",
    123: V3C_DIR / "manifest_with_masks_seed123.csv",
    456: V3C_DIR / "manifest_with_masks_seed456.csv",
}


def make_tta_transforms():
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return [
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(), normalize,
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), normalize,
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(), normalize,
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(), normalize,
        ]),
        transforms.Compose([
            transforms.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1))),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(), normalize,
        ]),
    ]


TTA_TFS = make_tta_transforms()
N_TTA = len(TTA_TFS)


class KidneyDatasetTTA(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        return im, LABEL2IDX[row["label"]], idx


def tta_predict(model, test_df, batch_size=32):
    n = len(test_df)
    accum_probs = np.zeros((n, len(CLASSES)), dtype=np.float64)
    labels = np.zeros(n, dtype=int)

    dataset = KidneyDatasetTTA(test_df)

    for tta_i, tf in enumerate(TTA_TFS):
        def collate(batch):
            imgs = torch.stack([tf(im) for im, _, _ in batch])
            ys = torch.tensor([y for _, y, _ in batch], dtype=torch.long)
            idxs = torch.tensor([ix for _, _, ix in batch], dtype=torch.long)
            return imgs, ys, idxs

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate)

        with torch.no_grad():
            for imgs, ys, idxs in loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                with autocast():
                    out = model(imgs)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                for i, ix in enumerate(idxs.numpy()):
                    accum_probs[ix] += probs[i]
                    labels[ix] = ys[i].item()

    accum_probs /= N_TTA
    return accum_probs, labels


all_fold_metrics = []

for seed in SEEDS:
    tag = RUN_TAGS[seed]
    manifest = MANIFESTS[seed]
    if not manifest.exists():
        print(f"Missing manifest: {manifest}")
        continue

    df = pd.read_csv(manifest)
    seed_dir = OUT / tag
    seed_dir.mkdir(exist_ok=True)

    for fold_idx in range(5):
        ckpt_path = KFOLD / tag / f"fold_{fold_idx}" / "best.pth"
        if not ckpt_path.exists():
            print(f"Missing checkpoint: {ckpt_path}")
            continue

        print(f"\n=== Seed {seed}, Fold {fold_idx} ===")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(DEVICE).eval()

        test_df = df[df["fold"] == fold_idx].reset_index(drop=True)
        probs, y_true = tta_predict(model, test_df)
        y_pred = probs.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        pcl = precision_recall_fscore_support(y_true, y_pred,
                                               labels=range(len(CLASSES)),
                                               zero_division=0)
        aucs = {}
        for i, c in enumerate(CLASSES):
            try:
                aucs[c] = float(roc_auc_score((y_true == i).astype(int), probs[:, i]))
            except ValueError:
                aucs[c] = None
        macro_auc = float(np.mean([v for v in aucs.values() if v is not None]))
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

        fold_result = {
            "seed": seed,
            "fold": fold_idx,
            "n_tta": N_TTA,
            "test_accuracy": float(acc),
            "test_macro_f1": float(macro_f1),
            "test_macro_auc": macro_auc,
            "per_class": {c: {
                "precision": float(pcl[0][i]),
                "recall": float(pcl[1][i]),
                "f1": float(pcl[2][i]),
                "support": int(pcl[3][i]),
                "auc": aucs[c]
            } for i, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist(),
        }

        fold_out = seed_dir / f"fold_{fold_idx}"
        fold_out.mkdir(exist_ok=True)
        with open(fold_out / "tta_metrics.json", "w") as f:
            json.dump(fold_result, f, indent=2)

        print(f"  acc={acc:.4f}, F1={macro_f1:.4f}, AUC={macro_auc:.4f}")
        all_fold_metrics.append(fold_result)

print(f"\n{'=' * 70}\nTTA AGGREGATE (n={len(all_fold_metrics)} fold-level results)\n{'=' * 70}")

accs = [m["test_accuracy"] for m in all_fold_metrics]
f1s = [m["test_macro_f1"] for m in all_fold_metrics]
aucs = [m["test_macro_auc"] for m in all_fold_metrics]
print(f"Accuracy:  {np.mean(accs)*100:.2f}% +- {np.std(accs, ddof=1)*100:.2f}%")
print(f"Macro-F1:  {np.mean(f1s):.4f} +- {np.std(f1s, ddof=1):.4f}")
print(f"Macro-AUC: {np.mean(aucs):.4f} +- {np.std(aucs, ddof=1):.4f}")
print("Per-class F1:")
for c in CLASSES:
    pc_f1 = [m["per_class"][c]["f1"] for m in all_fold_metrics]
    print(f"  {c:<8} F1={np.mean(pc_f1):.4f} +- {np.std(pc_f1, ddof=1):.4f}")

summary = {
    "n_fold_results": len(all_fold_metrics),
    "n_tta_transforms": N_TTA,
    "fold_results": all_fold_metrics,
    "aggregate": {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)),
        "macro_auc_mean": float(np.mean(aucs)),
        "macro_auc_std": float(np.std(aucs, ddof=1)),
        "per_class_f1": {c: {
            "mean": float(np.mean([m["per_class"][c]["f1"] for m in all_fold_metrics])),
            "std": float(np.std([m["per_class"][c]["f1"] for m in all_fold_metrics], ddof=1)),
        } for c in CLASSES},
    },
}
with open(OUT / "tta_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: {OUT / 'tta_summary.json'}")