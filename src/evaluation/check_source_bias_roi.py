"""Source-bias diagnostic on the v2_roi_run1 model."""
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
MANIFEST = SCRATCH / "kidney-data/processed/unified_v2_roi/manifest_with_folds.csv"
KFOLD_ROOT = SCRATCH / "kidney-results/kfold/v2_roi_run1"
OUT = SCRATCH / "kidney-results/source_bias_check_roi"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

df = pd.read_csv(MANIFEST)
all_records = []
for fold in range(5):
    ckpt = torch.load(KFOLD_ROOT / f"fold_{fold}" / "best.pth", map_location=DEVICE, weights_only=False)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_df = df[df["fold"] == fold].reset_index(drop=True)
    loader = DataLoader(KidneyDataset(test_df, eval_tf), batch_size=64,
                        shuffle=False, num_workers=4, pin_memory=True)
    probs, idxs = [], []
    with torch.no_grad():
        for img, idx in loader:
            img = img.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(img)
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
            idxs.extend(idx.numpy().tolist())
    probs = np.concatenate(probs)
    preds = probs.argmax(axis=1)
    for li, pr, pi in zip(idxs, probs, preds):
        row = test_df.iloc[li]
        all_records.append({"source": row["source"], "true_label": row["label"],
                             "predicted_label": IDX2LABEL[pi],
                             "patient_id": row["patient_id"], "fold": fold})

preds_df = pd.DataFrame(all_records)
preds_df.to_csv(OUT / "global_predictions.csv", index=False)

print("=== ROI MODEL: Predicted class counts per source ===")
print(pd.crosstab(preds_df["source"], preds_df["predicted_label"]).to_string())
print("\n=== ROI MODEL: Predicted class proportions per source (%) ===")
pct = pd.crosstab(preds_df["source"], preds_df["predicted_label"], normalize="index") * 100
print(pct.reindex(columns=CLASSES, fill_value=0).round(2).to_string())

print("\n=== Cross-source predictions ===")
sources = sorted(preds_df["source"].unique())
for src in sources:
    sub = preds_df[preds_df["source"] == src]
    distinct = sub["predicted_label"].nunique()
    print(f"  {src}: predicts {distinct}/4 classes — {sorted(sub['predicted_label'].value_counts().index.tolist())}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, src in zip(axes.flatten(), sources):
    sub = preds_df[preds_df["source"] == src]
    cm = pd.crosstab(sub["true_label"], sub["predicted_label"]).reindex(CLASSES, axis=0, fill_value=0).reindex(CLASSES, axis=1, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"ROI model | Source = {src.upper()} (n={len(sub)})")
plt.tight_layout()
plt.savefig(OUT / "confusion_by_source.png", dpi=120, bbox_inches="tight")
print(f"\nSaved plots to {OUT}")