"""
Per-source matched-subset analysis: does the ROI vs Full trade-off
hold uniformly across the 4 source datasets, or is it source-driven?

Reuses v2-full predictions (from v2_run1) and v2-ROI predictions (from v2_roi_run1),
restricted to the 6,185 ROI-eligible images, broken down per (class, source).
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
from sklearn.metrics import f1_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRATCH = Path(os.environ["SCRATCH"])
V2_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
ROI_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2_roi/manifest_with_folds.csv"

V2_KFOLD = SCRATCH / "kidney-results/kfold/v2_run1"
ROI_KFOLD = SCRATCH / "kidney-results/kfold/v2_roi_run1"

OUT = SCRATCH / "kidney-results/per_source_matched"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
N_FOLDS = 5
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


def predict_all_folds(kfold_root, manifest_df):
    """Run inference fold-by-fold using each fold's checkpoint on its test slice.
    Returns dataframe with: source, true_label, pred_label, fold."""
    rows = []
    for fold in range(N_FOLDS):
        ckpt_path = kfold_root / f"fold_{fold}" / "best.pth"
        test_df = manifest_df[manifest_df["fold"] == fold].reset_index(drop=True)

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model = model.to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        loader = DataLoader(KidneyDataset(test_df, eval_tf), batch_size=64,
                            shuffle=False, num_workers=4, pin_memory=True)
        preds, idxs = [], []
        with torch.no_grad():
            for img, idx in loader:
                img = img.to(DEVICE, non_blocking=True)
                with autocast():
                    out = model(img)
                preds.extend(out.argmax(1).cpu().numpy().tolist())
                idxs.extend(idx.numpy().tolist())
        for li, p in zip(idxs, preds):
            row = test_df.iloc[li]
            rows.append({"source": row["source"], "true_label": row["label"],
                         "pred_label": IDX2LABEL[p], "fold": fold})
    return pd.DataFrame(rows)


print("Loading manifests...")
v2_df = pd.read_csv(V2_MANIFEST)
roi_df = pd.read_csv(ROI_MANIFEST)

# Restrict v2-full to ROI-eligible subset for fair matched comparison
roi_stems = set(roi_df["stem"].tolist())
v2_matched = v2_df[v2_df["stem"].isin(roi_stems)].reset_index(drop=True)
print(f"Matched subset: {len(v2_matched)} v2-full images, {len(roi_df)} v2-ROI images")
assert len(v2_matched) == len(roi_df), "Mismatch between matched subsets!"

print("\nRunning v2-FULL inference on matched subset...")
full_preds = predict_all_folds(V2_KFOLD, v2_matched)
full_preds["model"] = "v2_full"

print("Running v2-ROI inference...")
roi_preds = predict_all_folds(ROI_KFOLD, roi_df)
roi_preds["model"] = "v2_roi"

print("\n=== Per-source per-class F1 (matched subset) ===")
rows = []
for src in sorted(v2_matched["source"].unique()):
    for cls in CLASSES:
        for model_name, preds in [("v2_full", full_preds), ("v2_roi", roi_preds)]:
            sub = preds[preds["source"] == src]
            if len(sub) == 0 or (sub["true_label"] == cls).sum() == 0:
                f1 = None
            else:
                y_true = (sub["true_label"] == cls).astype(int)
                y_pred = (sub["pred_label"] == cls).astype(int)
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
            rows.append({"source": src, "class": cls, "model": model_name,
                         "f1": f1, "n_samples": len(sub)})

per_source = pd.DataFrame(rows)

# Pivot for readability
pivot = per_source.pivot_table(index=["source", "class"], columns="model", values="f1")
pivot["delta_roi_vs_full"] = pivot["v2_roi"] - pivot["v2_full"]
pivot = pivot.reset_index()
pivot.to_csv(OUT / "per_source_per_class_f1.csv", index=False)

print(pivot.to_string(index=False))

# Visualisation: grouped bar of full vs roi F1 per (source, class)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, cls in zip(axes.flatten(), CLASSES):
    sub = pivot[pivot["class"] == cls].copy()
    sub = sub.dropna(subset=["v2_full", "v2_roi"])
    if len(sub) == 0:
        ax.set_title(f"{cls}: no data")
        continue
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x - width/2, sub["v2_full"], width, label="Full-image", color="#2E86AB", edgecolor="black")
    ax.bar(x + width/2, sub["v2_roi"], width, label="ROI", color="#A23B72", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["source"], rotation=0)
    ax.set_ylabel("F1 score")
    ax.set_title(f"{cls}: Full vs ROI by source")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    # annotate delta
    for i, (_, row) in enumerate(sub.iterrows()):
        delta = row["delta_roi_vs_full"]
        ax.annotate(f"Δ {delta:+.2f}", xy=(i, max(row["v2_full"], row["v2_roi"]) + 0.04),
                    ha='center', fontsize=9, fontweight='bold',
                    color='green' if delta > 0 else 'red')
plt.suptitle("Per-source F1: Full vs ROI (matched 6,185 images)", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "per_source_per_class_f1.png", dpi=130, bbox_inches="tight")

print(f"\nSaved per-source breakdown to: {OUT}")
print(f"  - per_source_per_class_f1.csv")
print(f"  - per_source_per_class_f1.png")