"""
Diagnostic: is the multi-source kidney classifier learning pathology, or learning
to identify the source dataset (KiTS vs Mendeley)?

For each fold, load the best checkpoint and re-run inference on the test set,
recording per-image probabilities. Then analyse:
  1. Source × true_label × predicted_label cross-tab
  2. Per-source accuracy on the Normal class (the only one with both sources)
  3. Per-source predicted-class distributions
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST = SCRATCH / "kidney-data/processed/unified_kits_mendeley/manifest_with_folds.csv"
KFOLD_ROOT = SCRATCH / "kidney-results/kfold/run1"
OUT = SCRATCH / "kidney-results/source_bias_check"
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
        return im, idx  # return idx so we can map back to source/label


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Re-run inference per fold, save per-image predictions
# ---------------------------------------------------------------------------
df = pd.read_csv(MANIFEST)
print(f"Loaded manifest with {len(df)} images")

all_records = []
for fold in range(N_FOLDS):
    ckpt_path = KFOLD_ROOT / f"fold_{fold}" / "best.pth"
    print(f"\nFold {fold}: loading {ckpt_path}")
    test_df = df[df["fold"] == fold].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} images")

    # Build model and load weights
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
            "fold": fold,
            "path": row["path"],
            "patient_id": row["patient_id"],
            "source": row["source"],
            "true_label": row["label"],
            "predicted_label": IDX2LABEL[pred_idx],
            "prob_Normal": float(prob_row[0]),
            "prob_Cyst": float(prob_row[1]),
            "prob_Tumor": float(prob_row[2]),
            "prob_Stone": float(prob_row[3]),
        })

preds_df = pd.DataFrame(all_records)
preds_df.to_csv(OUT / "global_predictions.csv", index=False)
print(f"\nSaved per-image predictions: {OUT / 'global_predictions.csv'}")
print(f"Total predictions across all 5 folds: {len(preds_df)}")

# ---------------------------------------------------------------------------
# Check 1: Source × true_label × predicted_label cross-tabs
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("CHECK 1: Prediction breakdown by source")
print('=' * 70)

for src in sorted(preds_df["source"].unique()):
    sub = preds_df[preds_df["source"] == src]
    print(f"\n--- Source: {src.upper()} (n={len(sub)}) ---")
    print(f"True class distribution:")
    print(sub["true_label"].value_counts().to_string())
    print(f"\nPredicted class distribution:")
    print(sub["predicted_label"].value_counts().to_string())
    print(f"\nConfusion matrix for {src} (rows=true, cols=predicted):")
    cm = pd.crosstab(sub["true_label"], sub["predicted_label"],
                     margins=False).reindex(CLASSES, axis=0, fill_value=0)
    cm = cm.reindex(CLASSES, axis=1, fill_value=0)
    print(cm.to_string())
    # Per-class accuracy on this source
    print(f"\nPer-class recall on {src}:")
    for c in CLASSES:
        true_c = sub[sub["true_label"] == c]
        if len(true_c) == 0:
            continue
        correct = (true_c["predicted_label"] == c).sum()
        print(f"  {c}: {correct}/{len(true_c)} = {correct/len(true_c):.3f}")

# ---------------------------------------------------------------------------
# Check 2: Normal-class accuracy by source (the critical comparison)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("CHECK 2: Normal-class performance by source")
print("(Normal is the only class with BOTH sources — fair comparison)")
print('=' * 70)
normal_only = preds_df[preds_df["true_label"] == "Normal"]
print(f"\nTotal Normal images: {len(normal_only)}")
for src in sorted(normal_only["source"].unique()):
    src_sub = normal_only[normal_only["source"] == src]
    correct = (src_sub["predicted_label"] == "Normal").sum()
    print(f"  {src}: {correct}/{len(src_sub)} = {correct/len(src_sub):.3f} accuracy")
    print(f"    Predicted distribution: "
          f"{dict(src_sub['predicted_label'].value_counts())}")

# ---------------------------------------------------------------------------
# Check 3: Per-source predicted-class distributions
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("CHECK 3: For each source, how often is each class predicted?")
print("(If Mendeley → almost never predicts Cyst/Tumor → source-bias)")
print('=' * 70)
cross = pd.crosstab(preds_df["source"], preds_df["predicted_label"], normalize="index") * 100
cross = cross.reindex(columns=CLASSES, fill_value=0).round(1)
print("\nPredicted class proportions per source (%):")
print(cross.to_string())

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, src in zip(axes, sorted(preds_df["source"].unique())):
    sub = preds_df[preds_df["source"] == src]
    cm = pd.crosstab(sub["true_label"], sub["predicted_label"])
    cm = cm.reindex(CLASSES, axis=0, fill_value=0).reindex(CLASSES, axis=1, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Source = {src.upper()} (n={len(sub)})")
plt.tight_layout()
plt.savefig(OUT / "confusion_by_source.png", dpi=120, bbox_inches="tight")

# Predicted-class distribution per source
fig, ax = plt.subplots(figsize=(8, 5))
cross.plot(kind="bar", ax=ax, edgecolor="black")
ax.set_ylabel("% of images predicted as class")
ax.set_xlabel("Source")
ax.set_title("Predicted-class distribution by source\n"
             "(If one source = one class strongly, source-bias is present)")
ax.legend(title="Predicted class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "predicted_distribution_by_source.png", dpi=120, bbox_inches="tight")

print(f"\n\nSaved diagnostic plots and CSV to: {OUT}")