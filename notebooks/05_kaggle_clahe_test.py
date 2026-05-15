"""
Re-run the Kaggle generalisation test with CLAHE preprocessing applied before
inference. CLAHE = Contrast Limited Adaptive Histogram Equalization, a standard
domain-adaptation trick for cross-dataset CT.

If predicted kidney pixels increase meaningfully (especially Cyst class going from
0 to non-zero), CLAHE is a viable adaptation strategy. If not, we move to Path 2
(fine-tune on a small annotated Kaggle subset).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
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
OUT = SCRATCH / "kidney-results/kaggle_clahe"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# CLAHE config — typical defaults for medical imaging
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
# Sample SAME 24 images as yesterday for fair comparison
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
# Preprocess + predict — WITH CLAHE
# ---------------------------------------------------------------------------
def preprocess_with_clahe(path):
    with Image.open(path).convert("L") as im:
        arr = np.array(im, dtype=np.uint8)  # CLAHE needs uint8

    # Apply CLAHE (operates on uint8)
    arr_eq = clahe.apply(arr)

    # Normalise to [0, 1] float
    arr_float = arr_eq.astype(np.float32) / 255.0

    # Resize to 256x256
    t = torch.from_numpy(arr_float).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return t.to(DEVICE), arr_float, arr  # return both pre- and post-CLAHE for viz

predictions = []
with torch.no_grad():
    for _, row in samples.iterrows():
        img_t, img_eq, img_raw = preprocess_with_clahe(row["path"])
        logits = model(img_t)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
        predictions.append((row, img_raw, img_eq, pred))

# ---------------------------------------------------------------------------
# Visualise: 4 columns per image (raw | CLAHE'd | overlay)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(8, 9, figsize=(27, 24))
for i, (row, img_raw, img_eq, pred) in enumerate(predictions):
    r = i // 3
    c = (i % 3) * 3
    # Original
    axes[r, c].imshow(img_raw, cmap="gray")
    axes[r, c].set_title(f"{row['label']} / {row['likely_view']}\noriginal", fontsize=8)
    axes[r, c].axis("off")
    # After CLAHE
    axes[r, c+1].imshow(img_eq, cmap="gray")
    axes[r, c+1].set_title("after CLAHE", fontsize=8)
    axes[r, c+1].axis("off")
    # Prediction overlay
    axes[r, c+2].imshow(img_eq, cmap="gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))
    axes[r, c+2].imshow(pred, cmap="Greens", alpha=0.5, extent=(0, IMG_SIZE, IMG_SIZE, 0))
    axes[r, c+2].set_title(f"prediction\n({int(pred.sum())} px)", fontsize=8)
    axes[r, c+2].axis("off")

plt.tight_layout()
out = OUT / "kaggle_predictions_clahe.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"Saved CLAHE visualisation to: {out}")

# ---------------------------------------------------------------------------
# Numeric summary + comparison
# ---------------------------------------------------------------------------
summary = []
for row, _, _, pred in predictions:
    summary.append({
        "label": row["label"],
        "view": row["likely_view"],
        "predicted_kidney_pixels": int(pred.sum()),
    })
summary_df = pd.DataFrame(summary)

print("\n=== CLAHE preprocessing results ===")
print("Mean predicted kidney pixels by class & view:")
print(summary_df.groupby(["label", "view"])["predicted_kidney_pixels"].mean().round(0).to_string())

print("\nNon-zero prediction count (out of 3 samples each):")
print(summary_df.assign(nonzero=summary_df.predicted_kidney_pixels > 50)
      .groupby(["label", "view"])["nonzero"].sum().to_string())

print("\nFor reference, yesterday's baseline (no CLAHE):")
print("""
label   view   
Cyst    axial         0.0
        coronal       0.0
Normal  axial      1156.0
        coronal     410.0
Stone   axial       379.0
        coronal       0.0
Tumor   axial       476.0
        coronal      29.0
""")