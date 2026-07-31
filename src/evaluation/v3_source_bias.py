"""
Source-bias diagnostic for v3 RN50-full.

Loads the best-fold model from seed 42 (matches v2 protocol) and produces a
5-panel confusion matrix — one panel per source (kits, mendeley, kauh, tcga,
abdalla). The key question: with two Stone sources now (Mendeley + Abdalla),
does the source→class shortcut break?

Output:
  $SCRATCH/kidney-results/v3_analysis/source_bias/
    confusion_by_source.png     5-panel figure
    predictions.csv             raw per-image predictions
    per_source_summary.csv      per-source class distribution + accuracy
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
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"
V3_MANIFEST = SCRATCH / "kidney-data/processed/unified_v3/manifest_with_folds.csv"
RUN_TAG = "v3_rn50_full_run1_seed42"
OUT = SCRATCH / "kidney-results/v3_analysis/source_bias"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# Find best fold by test macro-F1 in the summary
# ----------------------------------------------------------------------
summary_path = KFOLD / RUN_TAG / "summary.json"
if not summary_path.exists():
    raise SystemExit(f"Missing run summary: {summary_path}")

with open(summary_path) as f:
    s = json.load(f)

best_fold = int(np.argmax([fr["test_macro_f1"] for fr in s["fold_results"]]))
best_ckpt = KFOLD / RUN_TAG / f"fold_{best_fold}" / "best.pth"
print(f"Using best-fold model: {best_ckpt}")
print(f"  Fold {best_fold} test F1 = {s['fold_results'][best_fold]['test_macro_f1']:.4f}")

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
print("Loading RN50...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE).eval()

# ----------------------------------------------------------------------
# Load test-fold data
# ----------------------------------------------------------------------
df = pd.read_csv(V3_MANIFEST)
test_df = df[df["fold"] == best_fold].reset_index(drop=True)
print(f"Test-fold images: {len(test_df)}")
print(f"Per source in test fold: {dict(test_df['source'].value_counts())}")

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class KidneyDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        return self.transform(im), LABEL2IDX[row["label"]], idx


loader = DataLoader(KidneyDataset(test_df, eval_tf),
                    batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------
print("Predicting on test fold...")
preds = np.zeros(len(test_df), dtype=int)
with torch.no_grad():
    for imgs, ys, idxs in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast():
            out = model(imgs)
        p = out.argmax(1).cpu().numpy()
        for i, ix in zip(p, idxs.numpy()):
            preds[ix] = i

test_df["prediction"] = [IDX2LABEL[p] for p in preds]
test_df["correct"] = test_df["prediction"] == test_df["label"]
test_df.to_csv(OUT / "predictions.csv", index=False)

# ----------------------------------------------------------------------
# 5-panel confusion matrix
# ----------------------------------------------------------------------
SOURCES = sorted(test_df["source"].unique())
print(f"\nSources in test fold: {SOURCES}")

n_src = len(SOURCES)
ncols = min(3, n_src)
nrows = (n_src + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
if n_src == 1:
    axes = np.array([axes])
axes_flat = axes.flatten() if hasattr(axes, "flatten") else np.array([axes])

per_src_rows = []
for idx, src in enumerate(SOURCES):
    ax = axes_flat[idx]
    sub = test_df[test_df["source"] == src]
    y_true = [LABEL2IDX[l] for l in sub["label"]]
    y_pred = [LABEL2IDX[l] for l in sub["prediction"]]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Source = {src.upper()} (n={len(sub)})")
    per_src_rows.append({
        "source": src,
        "n_images": len(sub),
        "accuracy": float(sub["correct"].mean()),
        **{f"true_{c}": int((sub["label"] == c).sum()) for c in CLASSES},
        **{f"pred_{c}": int((sub["prediction"] == c).sum()) for c in CLASSES},
    })

for j in range(n_src, len(axes_flat)):
    axes_flat[j].axis("off")

plt.suptitle(f"v3 RN50-full: confusion matrices by source "
             f"(fold {best_fold}, seed 42)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "confusion_by_source.png", dpi=130, bbox_inches="tight")
plt.close()

per_src_df = pd.DataFrame(per_src_rows)
per_src_df.to_csv(OUT / "per_source_summary.csv", index=False)

print()
print("Per-source summary:")
print(per_src_df.to_string(index=False))

# ----------------------------------------------------------------------
# Key question: does Stone still map from a single source?
# ----------------------------------------------------------------------
print()
print("=" * 70)
print("SOURCE-SHORTCUT DIAGNOSTIC — STONE CLASS")
print("=" * 70)
stone_by_src = test_df[test_df["label"] == "Stone"].groupby("source")["correct"].agg(["count", "mean"])
print("\nStone recall by true source (in v3, both mendeley and abdalla have Stone):")
print(stone_by_src.to_string())

stone_pred_by_src = test_df[test_df["prediction"] == "Stone"].groupby("source").size()
print("\nWhere are Stone predictions coming from (by source of image)?")
print(stone_pred_by_src.to_string())

print()
print(f"Files saved to: {OUT}")
