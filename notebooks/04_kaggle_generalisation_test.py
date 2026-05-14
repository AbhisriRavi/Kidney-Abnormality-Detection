"""
Apply the trained KiTS U-Net to Kaggle CT images and visualise the predicted kidney masks.
Critical test: does cross-dataset transfer actually work?
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.networks.nets import UNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRATCH = Path(os.environ["SCRATCH"])
CKPT = SCRATCH / "kidney-results/segmenter/best.pth"
MANIFEST = SCRATCH / "kidney-data/processed/manifest.csv"
OUT = SCRATCH / "kidney-results/kaggle_generalisation"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
    dropout=0.1,
).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint (Dice={ckpt['val_dice']:.4f}, epoch={ckpt['epoch']})")

# ---------------------------------------------------------------------------
# Sample Kaggle images: 3 per class per orientation = 24 total
# ---------------------------------------------------------------------------
df = pd.read_csv(MANIFEST)
samples = []
for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    for view in ["axial", "coronal"]:
        subset = df[(df.label == cls) & (df.likely_view == view)]
        if len(subset) >= 3:
            samples.append(subset.sample(3, random_state=42))
samples = pd.concat(samples).reset_index(drop=True)
print(f"Selected {len(samples)} test images")

# ---------------------------------------------------------------------------
# Preprocess + predict
# ---------------------------------------------------------------------------
def preprocess(path):
    """Match the preprocessing used at training time as closely as possible."""
    with Image.open(path).convert("L") as im:
        arr = np.array(im, dtype=np.float32)
    # Min-max normalise to [0,1] — best we can do without HU values for Kaggle JPGs
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # Resize to 256x256
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return t.to(DEVICE), arr

predictions = []
with torch.no_grad():
    for _, row in samples.iterrows():
        img_t, img_np = preprocess(row["path"])
        logits = model(img_t)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
        predictions.append((row, img_np, pred))

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(8, 6, figsize=(18, 24))
for i, (row, img_np, pred) in enumerate(predictions):
    r = i // 3
    c = (i % 3) * 2
    # Original
    axes[r, c].imshow(img_np, cmap="gray")
    axes[r, c].set_title(f"{row['label']} / {row['likely_view']}\n{row['width']}x{row['height']}",
                         fontsize=9)
    axes[r, c].axis("off")
    # Overlay prediction
    axes[r, c+1].imshow(img_np, cmap="gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))
    axes[r, c+1].imshow(pred, cmap="Greens", alpha=0.5, extent=(0, IMG_SIZE, IMG_SIZE, 0))
    axes[r, c+1].set_title(f"Predicted kidney mask\n({int(pred.sum())} pixels)", fontsize=9)
    axes[r, c+1].axis("off")

plt.tight_layout()
out = OUT / "kaggle_predictions.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved generalisation visualisation to: {out}")

# Quick numeric summary
summary = []
for row, img_np, pred in predictions:
    summary.append({
        "label": row["label"],
        "view": row["likely_view"],
        "predicted_kidney_pixels": int(pred.sum()),
    })
summary_df = pd.DataFrame(summary)
print("\nMean predicted kidney pixels by class & view:")
print(summary_df.groupby(["label", "view"])["predicted_kidney_pixels"].mean().round(0).to_string())