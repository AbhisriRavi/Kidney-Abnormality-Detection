"""
Build the ROI version of the v2 manifest.

For every image in unified_v2/manifest_with_folds.csv:
  1. Load image (already 224x224 PNG)
  2. If source is mendeley or kauh, apply CLAHE + histmatch (already done at preprocessing
     time for our manifest, but applying again here is a no-op for the U-Net input prep)
  3. Predict kidney mask via U-Net
  4. Keep largest connected component, add 10% margin to bounding box
  5. Crop original 224x224 image to that bounding box, resize to 224x224
  6. Save as ROI PNG with same filename structure

Drops images where the U-Net finds < MIN_KIDNEY_PIXELS pixels of kidney.

Output:
  $SCRATCH/kidney-data/processed/unified_v2_roi/
    images/<source>/<class>/<patient>_<slice>.png
    manifest_with_folds.csv       (ROI manifest, fold column copied from v2)
    extraction_log.csv            (per-image status)
"""
import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from monai.networks.nets import UNet
import cv2
from PIL import Image
from skimage import measure
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])

# Inputs
MANIFEST_IN = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
SEGMENTER_CKPT = SCRATCH / "kidney-results/segmenter/best.pth"

# Outputs
OUT_ROOT = SCRATCH / "kidney-data/processed/unified_v2_roi"
IMG_OUT = OUT_ROOT / "images"
MANIFEST_OUT = OUT_ROOT / "manifest_with_folds.csv"
LOG_OUT = OUT_ROOT / "extraction_log.csv"
IMG_OUT.mkdir(parents=True, exist_ok=True)

# Config
IMG_SIZE = 224
BBOX_MARGIN = 0.10            # 10% padding around the bounding box
MIN_KIDNEY_PIXELS = 200       # below this, we drop the image (no detection)
SEG_INPUT_SIZE = 256          # what the U-Net was trained on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load segmenter
# ---------------------------------------------------------------------------
print("Loading U-Net segmenter...")
seg_model = UNet(
    spatial_dims=2, in_channels=1, out_channels=2,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch", dropout=0.1,
).to(DEVICE)
ckpt = torch.load(SEGMENTER_CKPT, map_location=DEVICE, weights_only=False)
seg_model.load_state_dict(ckpt["model_state"])
seg_model.eval()
print(f"  Segmenter loaded (best val Dice = {ckpt.get('val_dice', '?')})")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def predict_mask(img_gray_224):
    """img_gray_224: uint8 224x224. Returns binary mask 224x224 uint8."""
    # Resize to U-Net's training size
    inp = cv2.resize(img_gray_224, (SEG_INPUT_SIZE, SEG_INPUT_SIZE),
                     interpolation=cv2.INTER_AREA)
    # Normalize to [0,1] and add batch+channel dims
    t = torch.from_numpy(inp.astype(np.float32) / 255.0)[None, None].to(DEVICE)
    with torch.no_grad(), autocast():
        logits = seg_model(t)
    pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    # Resize mask back to 224x224
    mask = cv2.resize(pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return mask

def clean_mask(mask):
    """Keep largest connected component. Returns (cleaned mask, n_pixels)."""
    if mask.sum() == 0:
        return mask, 0
    labels = measure.label(mask, connectivity=2)
    if labels.max() == 0:
        return mask, 0
    sizes = np.bincount(labels.flat)
    sizes[0] = 0  # ignore background
    best_label = sizes.argmax()
    cleaned = (labels == best_label).astype(np.uint8)
    return cleaned, int(cleaned.sum())

def crop_to_mask(img, mask, margin_pct=BBOX_MARGIN):
    """Crop image to bounding box of mask with margin. Returns 224x224 crop or None."""
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    my = int(round(h * margin_pct))
    mx = int(round(w * margin_pct))
    y0 = max(0, y0 - my); y1 = min(img.shape[0] - 1, y1 + my)
    x0 = max(0, x0 - mx); x1 = min(img.shape[1] - 1, x1 + mx)
    crop = img[y0:y1+1, x0:x1+1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------
# Process every image in the manifest
# ---------------------------------------------------------------------------
df = pd.read_csv(MANIFEST_IN)
print(f"\nProcessing {len(df)} images...")

new_rows = []
log_rows = []
n_ok, n_no_kidney, n_crop_fail = 0, 0, 0

# Group by source for organised output
for idx, row in tqdm(df.iterrows(), total=len(df)):
    src_path = row["path"]
    label = row["label"]
    source = row["source"]
    patient_id = row["patient_id"]
    fold = int(row["fold"])

    try:
        img = np.array(Image.open(src_path).convert("L"), dtype=np.uint8)
    except Exception as e:
        log_rows.append({"path": src_path, "status": "load_failed", "n_kidney_px": 0})
        n_crop_fail += 1
        continue

    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    mask = predict_mask(img)
    mask, n_kid = clean_mask(mask)

    if n_kid < MIN_KIDNEY_PIXELS:
        log_rows.append({"path": src_path, "status": "no_kidney", "n_kidney_px": int(n_kid)})
        n_no_kidney += 1
        continue

    cropped = crop_to_mask(img, mask)
    if cropped is None:
        log_rows.append({"path": src_path, "status": "crop_failed", "n_kidney_px": int(n_kid)})
        n_crop_fail += 1
        continue

    # Save in a parallel directory structure
    out_dir = IMG_OUT / source / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = Path(src_path).name
    out_path = out_dir / out_name
    Image.fromarray(cropped).save(out_path)

    new_rows.append({
        "path": str(out_path),
        "label": label,
        "patient_id": patient_id,
        "source": source,
        "stem": row.get("stem", out_name),
        "fold": fold,
    })
    log_rows.append({"path": src_path, "status": "ok", "n_kidney_px": int(n_kid)})
    n_ok += 1

# ---------------------------------------------------------------------------
# Save manifest and log
# ---------------------------------------------------------------------------
with open(MANIFEST_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label", "patient_id", "source", "stem", "fold"])
    w.writeheader()
    w.writerows(new_rows)

with open(LOG_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "status", "n_kidney_px"])
    w.writeheader()
    w.writerows(log_rows)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
roi_df = pd.DataFrame(new_rows)
print(f"\n{'=' * 60}")
print("ROI extraction complete")
print('=' * 60)
print(f"  Total input images:     {len(df)}")
print(f"  ROI saved OK:           {n_ok}")
print(f"  No kidney detected:     {n_no_kidney}")
print(f"  Crop failed:            {n_crop_fail}")
print(f"  Retention rate:         {n_ok / len(df) * 100:.1f}%")
print()
print("ROI images per class:")
print(roi_df["label"].value_counts().to_string())
print()
print("Per-source retention (% of original kept):")
src_total = df.groupby("source").size()
src_kept = roi_df.groupby("source").size()
for src in sorted(src_total.index):
    kept = src_kept.get(src, 0)
    total = src_total[src]
    print(f"  {src:<10s}: {kept:5d} / {total:5d} = {kept/total*100:.1f}%")
print()
print("Per-class retention:")
cls_total = df.groupby("label").size()
cls_kept = roi_df.groupby("label").size()
for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    kept = cls_kept.get(cls, 0)
    total = cls_total[cls]
    print(f"  {cls:<8s}: {kept:5d} / {total:5d} = {kept/total*100:.1f}%")
print()
print(f"Manifest: {MANIFEST_OUT}")
print(f"Log:      {LOG_OUT}")