"""
Re-run the Kaggle generalisation test with CLAHE + histogram matching to a
reference KiTS slice. If histogram matching closes the Cyst class gap, we have
a no-annotation domain adaptation that works. If not, intensity-only methods
are exhausted and we move to fine-tuning.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from skimage.exposure import match_histograms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.networks.nets import UNet

SCRATCH = Path(os.environ["SCRATCH"])
CKPT = SCRATCH / "kidney-results/segmenter/best.pth"
MANIFEST = SCRATCH / "kidney-data/processed/manifest.csv"
KITS_SLICES = SCRATCH / "kidney-data/processed/kits_axial_slices"
OUT = SCRATCH / "kidney-results/kaggle_histmatch"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ---------------------------------------------------------------------------
# Build a reference histogram from several KiTS slices (more stable than one)
# ---------------------------------------------------------------------------
print("Building KiTS reference distribution...")
ref_files = list(KITS_SLICES.glob("case_00000_*_img.npy"))[:20]
ref_imgs = []
for f in ref_files:
    img = np.load(f)
    # Bring back to 0-255 uint8 range to match the Kaggle 8-bit space
    img_uint8 = (img * 255).astype(np.uint8)
    ref_imgs.append(img_uint8)
reference = np.stack(ref_imgs).mean(axis=0).astype(np.uint8)
print(f"Reference shape: {reference.shape}, range: [{reference.min()}, {reference.max()}]")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = UNet(spatial_dims=2, in_channels=1, out_channels=2,
             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
             num_res_units=2, norm="batch", dropout=0.1).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint (Dice={ckpt['val_dice']:.4f})")

# ---------------------------------------------------------------------------
# Same 24 samples for direct comparison
# ---------------------------------------------------------------------------
df = pd.read_csv(MANIFEST)
samples = []
for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    for view in ["axial", "coronal"]:
        subset = df[(df.label == cls) & (df.likely_view == view)]
        if len(subset) >= 3:
            samples.append(subset.sample(3, random_state=42))
samples = pd.concat(samples).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Preprocess with CLAHE + histogram matching
# ---------------------------------------------------------------------------
def preprocess(path):
    with Image.open(path).convert("L") as im:
        arr = np.array(im, dtype=np.uint8)
    # Resize raw to match reference size before histogram matching
    arr_resized = cv2.resize(arr, reference.shape[::-1], interpolation=cv2.INTER_AREA)
    # CLAHE first (local contrast)
    arr_clahe = clahe.apply(arr_resized)
    # Then histogram match (global intensity distribution)
    arr_matched = match_histograms(arr_clahe, reference).astype(np.uint8)
    # Normalise to [0,1]
    arr_float = arr_matched.astype(np.float32) / 255.0
    t = torch.from_numpy(arr_float).unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=IMG_SIZE, mode="bilinear", align_corners=False)
    return t.to(DEVICE), arr_resized, arr_matched

predictions = []
with torch.no_grad():
    for _, row in samples.iterrows():
        img_t, img_raw, img_matched = preprocess(row["path"])
        logits = model(img_t)
        pred = logits.argmax(dim=1).cpu().numpy()[0]
        predictions.append((row, img_raw, img_matched, pred))

# ---------------------------------------------------------------------------
# Visualise + summarise
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(8, 9, figsize=(27, 24))
for i, (row, img_raw, img_matched, pred) in enumerate(predictions):
    r = i // 3
    c = (i % 3) * 3
    axes[r, c].imshow(img_raw, cmap="gray")
    axes[r, c].set_title(f"{row['label']} / {row['likely_view']}\noriginal", fontsize=8)
    axes[r, c].axis("off")
    axes[r, c+1].imshow(img_matched, cmap="gray")
    axes[r, c+1].set_title("CLAHE + hist match", fontsize=8)
    axes[r, c+1].axis("off")
    axes[r, c+2].imshow(img_matched, cmap="gray", extent=(0, IMG_SIZE, IMG_SIZE, 0))
    axes[r, c+2].imshow(pred, cmap="Greens", alpha=0.5, extent=(0, IMG_SIZE, IMG_SIZE, 0))
    axes[r, c+2].set_title(f"prediction\n({int(pred.sum())} px)", fontsize=8)
    axes[r, c+2].axis("off")
plt.tight_layout()
plt.savefig(OUT / "kaggle_predictions_histmatch.png", dpi=100, bbox_inches="tight")
print(f"Saved to {OUT / 'kaggle_predictions_histmatch.png'}")

summary = pd.DataFrame([{"label": r["label"], "view": r["likely_view"],
                          "predicted_kidney_pixels": int(p.sum())}
                         for r, _, _, p in predictions])
print("\n=== CLAHE + histogram matching results ===")
print("Mean predicted kidney pixels:")
print(summary.groupby(["label", "view"])["predicted_kidney_pixels"].mean().round(0).to_string())
print("\nNon-zero count (out of 3):")
print(summary.assign(nonzero=summary.predicted_kidney_pixels > 50)
      .groupby(["label", "view"])["nonzero"].sum().to_string())

print("""
=== Comparison ===
              Baseline   CLAHE    Hist+CLAHE
Cyst axial:      0.0       8.0     <see above>
Normal axial: 1156.0    1626.0     <see above>
Tumor axial:   476.0    1056.0     <see above>
Stone axial:   379.0     320.0     <see above>
""")