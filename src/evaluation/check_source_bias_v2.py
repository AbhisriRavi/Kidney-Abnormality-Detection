"""
Source-bias diagnostic for the v2 model.

Same as v1's diagnostic but reads the v2 manifest and v2 checkpoints.
Key question: with multi-source coverage of Cyst and Tumor, does the
model now make cross-source predictions, or has it still learned source identity?
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
KFOLD_ROOT = SCRATCH / "kidney-results/kfold/v2_run1"
OUT = SCRATCH / "kidney-results/source_bias_check_v2"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
N_FOLDS = 5
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        return im, idx


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

df = pd.read_csv(MANIFEST)
print(f"Loaded manifest with {len(df)} images across {df['source'].nunique()} sources")

all_records = []
for fold in range(N_FOLDS):
    ckpt_path = KFOLD_ROOT / f"fold_{fold}" / "best.pth"
    print(f"\nFold {fold}: loading {ckpt_path}")
    test_df = df[df["fold"] == fold].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} images")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(KidneyDataset(test_df, eval_tf),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)
    fold_probs, fold_idxs = [], []
    with torch.no_grad():
        for img, idx in loader:
            img = img.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(img)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            fold_probs.append(probs)
            fold_idxs.extend(idx.numpy().tolist())

    fold_probs = np.concatenate(fold_probs)
    fold_preds = fold_probs.argmax(axis=1)

    for local_i, prob_row, pred_idx in zip(fold_idxs, fold_probs, fold_preds):
        row = test_df.iloc[local_i]
        all_records.append({
            "fold": fold, "path": row["path"], "patient_id": row["patient_id"],
            "source": row["source"], "true_label": row["label"],
            "predicted_label": IDX2LABEL[pred_idx],
            "prob_Normal": float(prob_row[0]), "prob_Cyst": float(prob_row[1]),
            "prob_Tumor": float(prob_row[2]), "prob_Stone": float(prob_row[3]),
        })

preds_df = pd.DataFrame(all_records)
preds_df.to_csv(OUT / "global_predictions.csv", index=False)
print(f"\nSaved per-image predictions: {len(preds_df)} rows")

# Check 1: Source × true × predicted
print(f"\n{'=' * 70}")
print("CHECK 1: Prediction breakdown by source")
print('=' * 70)
for src in sorted(preds_df["source"].unique()):
    sub = preds_df[preds_df["source"] == src]
    print(f"\n--- Source: {src.upper()} (n={len(sub)}) ---")
    print("True distribution:")
    print("  " + str(dict(sub["true_label"].value_counts())))
    print("Predicted distribution:")
    print("  " + str(dict(sub["predicted_label"].value_counts())))
    cm = pd.crosstab(sub["true_label"], sub["predicted_label"]).reindex(CLASSES, axis=0, fill_value=0).reindex(CLASSES, axis=1, fill_value=0)
    print("Confusion matrix:")
    print(cm.to_string())
    print("Per-class recall:")
    for c in CLASSES:
        n_true = (sub["true_label"] == c).sum()
        if n_true:
            n_correct = ((sub["true_label"] == c) & (sub["predicted_label"] == c)).sum()
            print(f"  {c}: {n_correct}/{n_true} = {n_correct/n_true:.3f}")

# Check 2: Per-source predicted distributions
print(f"\n{'=' * 70}")
print("CHECK 2: For each source, what % of images get predicted as each class?")
print("(For the source-shortcut fix to work, predictions should be reasonably spread)")
print('=' * 70)
cross = pd.crosstab(preds_df["source"], preds_df["predicted_label"], normalize="index") * 100
cross = cross.reindex(columns=CLASSES, fill_value=0).round(1)
print(cross.to_string())

# Check 3: Critical comparison vs v1
print(f"\n{'=' * 70}")
print("CHECK 3: Has the model learned to make cross-source predictions?")
print('=' * 70)
for src in sorted(preds_df["source"].unique()):
    sub = preds_df[preds_df["source"] == src]
    pred_classes = sub["predicted_label"].value_counts()
    n_classes_predicted = len(pred_classes)
    print(f"  {src}: predicts {n_classes_predicted}/4 classes — {sorted(pred_classes.index.tolist())}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, src in zip(axes.flatten(), sorted(preds_df["source"].unique())):
    sub = preds_df[preds_df["source"] == src]
    cm = pd.crosstab(sub["true_label"], sub["predicted_label"]).reindex(CLASSES, axis=0, fill_value=0).reindex(CLASSES, axis=1, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Source = {src.upper()} (n={len(sub)})")
plt.tight_layout()
plt.savefig(OUT / "confusion_by_source.png", dpi=120, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9, 5))
cross.plot(kind="bar", ax=ax, edgecolor="black")
ax.set_ylabel("% of images predicted as class")
ax.set_xlabel("Source")
ax.set_title("v2 — Predicted-class distribution by source\n(spread across all 4 classes = source-shortcut broken)")
ax.legend(title="Predicted")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "predicted_distribution_by_source.png", dpi=120, bbox_inches="tight")

print(f"\nSaved plots to {OUT}")