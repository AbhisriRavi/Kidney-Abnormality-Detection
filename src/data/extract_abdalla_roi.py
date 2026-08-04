"""
Run the KiTS-trained U-Net on Abdalla Original images and extract kidney ROI crops.

For each Abdalla original image:
  - Predict kidney mask with U-Net
  - Score prediction quality (bilateral / area / symmetry)
  - If score >= threshold: extract tight bounding box, save cropped image
  - Otherwise: log as "skipped" (no valid ROI)

Output:
  $SCRATCH/kidney-data/processed/abdalla_roi/
    images/<stem>.png            (ROI-cropped images)
    extraction_log.csv           (per-image: kept/skipped, score, reason)
    summary.txt                  (aggregate statistics)
"""
import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from monai.networks.nets import UNet
import cv2
from PIL import Image
from skimage import measure
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])
V3_MANIFEST = SCRATCH / "kidney-data/processed/unified_v3/manifest.csv"
SEGMENTER_CKPT = SCRATCH / "kidney-results/segmenter/best.pth"
OUT_ROOT = SCRATCH / "kidney-data/processed/abdalla_roi"
IMG_OUT = OUT_ROOT / "images"
LOG_OUT = OUT_ROOT / "extraction_log.csv"
SUMMARY_OUT = OUT_ROOT / "summary.txt"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
SEG_INPUT_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Score threshold — same heuristic used for Mendeley scoring
MIN_SCORE = 60
MIN_PIXELS = 500

# Load segmenter
print("Loading U-Net segmenter...")
seg_model = UNet(
    spatial_dims=2, in_channels=1, out_channels=2,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch", dropout=0.1,
).to(DEVICE)
ckpt = torch.load(SEGMENTER_CKPT, map_location=DEVICE, weights_only=False)
seg_model.load_state_dict(ckpt["model_state"])
seg_model.eval()


def predict_mask(img_path):
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    orig_shape = img.shape
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    inp = cv2.resize(img, (SEG_INPUT_SIZE, SEG_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(inp.astype(np.float32) / 255.0)[None, None].to(DEVICE)
    with torch.no_grad(), autocast():
        logits = seg_model(t)
    pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST), img


def score_prediction(mask):
    """Returns (score [0-100], details dict)."""
    total_px = int(mask.sum())
    if total_px < MIN_PIXELS:
        return 0, {"n_components": 0, "total_px": total_px}
    labels = measure.label(mask, connectivity=2)
    components = [c for c in measure.regionprops(labels) if c.area > 50]
    n_comp = len(components)
    if n_comp == 2:
        bilat = 50
    elif n_comp == 1:
        bilat = 25
    else:
        bilat = 10
    if 500 <= total_px <= 3000:
        size = 30
    elif 300 <= total_px < 500 or 3000 < total_px <= 4000:
        size = 20
    else:
        size = 10
    if n_comp == 2:
        xs = sorted([c.centroid[1] for c in components])
        sym = 20 if (xs[0] < IMG_SIZE/2 and xs[1] > IMG_SIZE/2) else 5
    else:
        sym = 5
    return bilat + size + sym, {"n_components": n_comp, "total_px": total_px}


def bbox_from_mask(mask, padding=10):
    """Returns (x0, y0, x1, y1) padded bounding box around mask, or None if empty."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    x0 = max(0, xs.min() - padding)
    y0 = max(0, ys.min() - padding)
    x1 = min(IMG_SIZE, xs.max() + padding)
    y1 = min(IMG_SIZE, ys.max() + padding)
    return x0, y0, x1, y1


# Load v3 manifest, filter Abdalla rows
df = pd.read_csv(V3_MANIFEST)
abdalla_df = df[df["source"] == "abdalla"].reset_index(drop=True)
print(f"Processing {len(abdalla_df)} Abdalla images...")

kept_rows, skipped_rows = [], []
for _, row in tqdm(abdalla_df.iterrows(), total=len(abdalla_df)):
    img_path = row["path"]
    mask, orig = predict_mask(img_path)
    score, details = score_prediction(mask)

    if score >= MIN_SCORE:
        bbox = bbox_from_mask(mask, padding=10)
        if bbox is None:
            skipped_rows.append({**row.to_dict(), "score": score,
                                  "reason": "no_bbox", **details})
            continue
        x0, y0, x1, y1 = bbox
        crop = orig[y0:y1, x0:x1]
        # Resize crop to 224x224 for consistency
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        out_path = IMG_OUT / f"{row['stem']}.png"
        Image.fromarray(crop).save(out_path)
        kept_row = {
            "path": str(out_path.resolve()),
            "label": row["label"],
            "source": row["source"],
            "patient_id": row["patient_id"],
            "stem": row["stem"],
            "score": score,
            **details,
        }
        kept_rows.append(kept_row)
    else:
        skipped_rows.append({**row.to_dict(), "score": score,
                              "reason": "low_score", **details})

# Write log
all_rows = kept_rows + skipped_rows
log_df = pd.DataFrame(all_rows)
log_df.to_csv(LOG_OUT, index=False)

# Summary
n_total = len(abdalla_df)
n_kept = len(kept_rows)
n_skipped = len(skipped_rows)
n_zero = sum(1 for r in skipped_rows if r.get("n_components", 0) == 0)

summary = f"""Abdalla ROI extraction summary
====================================
Total Abdalla images:      {n_total}
ROI crops extracted:       {n_kept} ({100*n_kept/n_total:.1f}%)
Skipped (low score):       {n_skipped} ({100*n_skipped/n_total:.1f}%)
  of which zero-detection: {n_zero} ({100*n_zero/n_total:.1f}%)

Score distribution (all images):
{log_df['score'].describe().to_string()}

Per-class retention:
"""
for c in ["Normal", "Stone"]:
    total_c = (abdalla_df["label"] == c).sum()
    kept_c = sum(1 for r in kept_rows if r["label"] == c)
    summary += f"  {c}: {kept_c}/{total_c} ({100*kept_c/max(total_c,1):.1f}%)\n"

summary += "\nPer-site retention:\n"
for site in sorted(abdalla_df["abdalla_site"].unique()):
    total_s = (abdalla_df["abdalla_site"] == site).sum()
    kept_s = sum(1 for r in kept_rows
                 if str(r.get("stem", "")).split("_")[1] == site)
    summary += f"  {site}: {kept_s}/{total_s} ({100*kept_s/max(total_s,1):.1f}%)\n"

with open(SUMMARY_OUT, "w") as f:
    f.write(summary)
print("\n" + summary)
print(f"Saved:")
print(f"  log:     {LOG_OUT}")
print(f"  summary: {SUMMARY_OUT}")
print(f"  images:  {IMG_OUT} (kept {n_kept})")