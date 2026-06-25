"""
Re-evaluate v2-full models on the ROI-eligible subset only (6,185 images).
This makes the comparison v2-full vs v2-ROI apples-to-apples on the same images.
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
V2_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
ROI_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2_roi/manifest_with_folds.csv"
KFOLD_ROOT = SCRATCH / "kidney-results/kfold/v2_run1"
OUT = SCRATCH / "kidney-results/matched_subset_eval"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
N_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Find which v2-full images survived ROI extraction
roi_df = pd.read_csv(ROI_MANIFEST)
v2_df = pd.read_csv(V2_MANIFEST)

# The ROI manifest's "stem" column matches what was originally in v2's stem column
# So we use that to identify the surviving images
roi_stems = set(roi_df["stem"].tolist())
v2_subset = v2_df[v2_df["stem"].isin(roi_stems)].reset_index(drop=True)
print(f"ROI-eligible subset of v2-full: {len(v2_subset)} images (of {len(v2_df)})")


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


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_preds, all_targets, all_probs = [], [], []
fold_metrics = []

for fold in range(N_FOLDS):
    ckpt_path = KFOLD_ROOT / f"fold_{fold}" / "best.pth"
    print(f"\nFold {fold}: loading {ckpt_path}")
    test_df = v2_subset[v2_subset["fold"] == fold].reset_index(drop=True)
    print(f"  Test set (matched subset): {len(test_df)} images")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(KidneyDataset(test_df, eval_tf), batch_size=64,
                        shuffle=False, num_workers=4, pin_memory=True)

    p_fold, t_fold, pr_fold = [], [], []
    with torch.no_grad():
        for img, y in loader:
            img = img.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(img)
            pr_fold.extend(torch.softmax(out, dim=1).cpu().numpy())
            p_fold.extend(out.argmax(1).cpu().numpy())
            t_fold.extend(y.numpy())

    p_fold = np.array(p_fold); t_fold = np.array(t_fold); pr_fold = np.array(pr_fold)
    acc = accuracy_score(t_fold, p_fold)
    f1 = f1_score(t_fold, p_fold, average="macro", zero_division=0)
    aucs = {}
    for i, c in enumerate(CLASSES):
        try:
            aucs[c] = float(roc_auc_score((t_fold == i).astype(int), pr_fold[:, i]))
        except ValueError:
            aucs[c] = None
    macro_auc = float(np.mean([v for v in aucs.values() if v is not None]))
    pcl = precision_recall_fscore_support(t_fold, p_fold, labels=range(len(CLASSES)), zero_division=0)

    fold_metrics.append({
        "fold": fold,
        "n_test": len(test_df),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_auc": macro_auc,
        "per_class_f1": {c: float(pcl[2][i]) for i, c in enumerate(CLASSES)},
        "per_class_auc": aucs,
    })
    print(f"  Fold {fold}: acc={acc:.4f}, F1={f1:.4f}, AUC={macro_auc:.4f}")
    all_preds.extend(p_fold); all_targets.extend(t_fold); all_probs.extend(pr_fold)

# Aggregate
accs = [m["accuracy"] for m in fold_metrics]
f1s = [m["macro_f1"] for m in fold_metrics]
aucs = [m["macro_auc"] for m in fold_metrics]

print(f"\n{'=' * 60}")
print(f"v2-FULL on MATCHED ROI-eligible subset (n={len(v2_subset)})")
print('=' * 60)
print(f"Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Macro-F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Macro-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print()
print("Per-class F1 (mean ± std):")
for c in CLASSES:
    vals = [m["per_class_f1"][c] for m in fold_metrics]
    print(f"  {c:<8} F1={np.mean(vals):.4f} ± {np.std(vals):.4f}")

with open(OUT / "v2_full_on_roi_subset.json", "w") as f:
    json.dump({"n_test": int(len(v2_subset)), "fold_metrics": fold_metrics,
                "aggregate": {"accuracy_mean": float(np.mean(accs)),
                              "accuracy_std": float(np.std(accs)),
                              "macro_f1_mean": float(np.mean(f1s)),
                              "macro_f1_std": float(np.std(f1s)),
                              "macro_auc_mean": float(np.mean(aucs)),
                              "macro_auc_std": float(np.std(aucs))}}, f, indent=2)
print(f"\nSaved: {OUT / 'v2_full_on_roi_subset.json'}")