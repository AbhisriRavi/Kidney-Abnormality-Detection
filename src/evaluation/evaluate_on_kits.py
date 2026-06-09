"""
Evaluate both trained classifiers (full-image and ROI) on the KiTS external test set.

This is the cross-dataset evaluation — the model has never seen any of these
patients during training. Results here are the leakage-free generalisation estimate.

Outputs (one set per mode):
  $SCRATCH/kidney-results/cross_eval_kits/<mode>/
    metrics.json         - accuracy, macro-F1, per-class P/R/F1/AUC, confusion matrix
    confusion_matrix.png
    predictions.csv      - per-image probabilities for downstream analysis
"""
import os
import json
import argparse
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
    confusion_matrix, f1_score, roc_auc_score, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["full", "roi"], required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
args = parser.parse_args()

SCRATCH = Path(os.environ["SCRATCH"])
CKPT = SCRATCH / "kidney-results/classification" / args.mode / "best.pth"
MANIFEST = SCRATCH / "kidney-data/processed/kits_external_test/manifest.csv"
OUT = SCRATCH / "kidney-results/cross_eval_kits" / args.mode
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMPORTANT: KiTS doesn't have Stone, so we evaluate only 3 classes.
# Classifier was trained on 4, so we softmax over all 4 but only count
# Normal/Cyst/Tumor in the confusion matrix.
TRAINED_CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
EVAL_CLASSES = ["Normal", "Cyst", "Tumor"]
LABEL2IDX = {c: i for i, c in enumerate(TRAINED_CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
print(f"Mode={args.mode}  Checkpoint={CKPT}")

df = pd.read_csv(MANIFEST)
print(f"External test images: {len(df)}")
print(df["label"].value_counts().to_string())

PATH_COL = "full_image_path" if args.mode == "full" else "roi_path"

class KitsTestDataset(Dataset):
    def __init__(self, df, path_col, transform):
        self.df = df
        self.path_col = path_col
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row[self.path_col]).convert("L").convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, LABEL2IDX[row["label"]]

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = KitsTestDataset(df, PATH_COL, eval_tf)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

# Load model + checkpoint
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(TRAINED_CLASSES))
model = model.to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded {args.mode} classifier (best val_macro_f1={ckpt['val_macro_f1']:.4f})")

# Inference
all_preds, all_targets, all_probs = [], [], []
with torch.no_grad():
    for img, y in loader:
        img = img.to(DEVICE, non_blocking=True)
        with autocast():
            out = model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_targets.extend(y.numpy())
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

# Stone predictions are valid outputs, but absent from the ground truth.
# Report the full confusion matrix including Stone column to show where the
# model places its confusion.
acc = accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, labels=[LABEL2IDX[c] for c in EVAL_CLASSES],
                    average="macro", zero_division=0)
per_class = precision_recall_fscore_support(
    all_targets, all_preds, labels=[LABEL2IDX[c] for c in EVAL_CLASSES], zero_division=0
)
auc_per_class = {}
for c in EVAL_CLASSES:
    i = LABEL2IDX[c]
    try:
        auc_per_class[c] = float(roc_auc_score((all_targets == i).astype(int), all_probs[:, i]))
    except ValueError:
        auc_per_class[c] = None
macro_auc = float(np.mean([v for v in auc_per_class.values() if v is not None]))

# Full 4x4 confusion matrix (rows = true Normal/Cyst/Tumor; cols = predicted 4 classes)
cm = confusion_matrix(all_targets, all_preds, labels=range(len(TRAINED_CLASSES)))

metrics = {
    "mode": args.mode,
    "n_test": int(len(df)),
    "test_accuracy_3class": float(acc),
    "test_macro_f1_3class": float(macro_f1),
    "test_macro_auc_3class": macro_auc,
    "per_class": {c: {"precision": float(per_class[0][i]),
                      "recall": float(per_class[1][i]),
                      "f1": float(per_class[2][i]),
                      "support": int(per_class[3][i]),
                      "auc": auc_per_class[c]} for i, c in enumerate(EVAL_CLASSES)},
    "confusion_matrix_4x4": cm.tolist(),
    "confusion_matrix_columns": TRAINED_CLASSES,
    "confusion_matrix_rows": TRAINED_CLASSES,
    "stone_predictions_count": int((all_preds == LABEL2IDX["Stone"]).sum()),
}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

pred_df = df.copy()
pred_df["predicted_label"] = [IDX2LABEL[p] for p in all_preds]
for i, c in enumerate(TRAINED_CLASSES):
    pred_df[f"prob_{c}"] = all_probs[:, i]
pred_df.to_csv(OUT / "predictions.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=TRAINED_CLASSES, yticklabels=TRAINED_CLASSES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"KiTS external test, mode={args.mode}\n3-class acc={acc:.3f}, macro-F1={macro_f1:.3f}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=110, bbox_inches="tight")

print(f"\n=== Cross-dataset results (mode={args.mode}, KiTS) ===")
print(f"3-class accuracy: {acc:.4f}")
print(f"3-class macro-F1: {macro_f1:.4f}")
print(f"3-class macro-AUC: {macro_auc:.4f}")
print(f"Stone predictions (false alarms — there are no real Stones in KiTS): {metrics['stone_predictions_count']}")
print("\nPer-class:")
for c in EVAL_CLASSES:
    pc = metrics["per_class"][c]
    auc_str = f"{pc['auc']:.4f}" if pc["auc"] is not None else "N/A"
    print(f"  {c:<8} P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f} AUC={auc_str} (n={pc['support']})")
print(f"\nSaved to {OUT}")