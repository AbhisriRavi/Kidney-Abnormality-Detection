"""
Apply the trained U-Net (with CLAHE + histogram matching preprocessing) to all
axial Kaggle images. Crop each image to the bounding box of the predicted kidney
mask, saving as the ROI dataset for downstream classification.

Output: $SCRATCH/kidney-data/processed/kaggle_axial_roi/<class>/<filename>.png
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from monai.networks.nets import UNet

SCRATCH = Path(os.environ["SCRATCH"])
CKPT = SCRATCH / "kidney-results/segmenter/best.pth"
MANIFEST = SCRATCH / "kidney-data/processed/manifest.csv"
KITS_SLICES = SCRATCH / "kidney-data/processed/kits_axial_slices"
OUT_DIR = SCRATCH / "kidney-data/processed/kaggle_axial_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
ROI_OUTPUT_SIZE = 224  # target size for downstream classifier (ResNet/EfficientNet)
MARGIN_FRAC = 0.10     # 10% margin around predicted bounding box
MIN_MASK_PIXELS = 200  # skip images where prediction is essentially empty
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def match_histograms(source, reference):
    src_values, src_idx, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(reference.ravel(), return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / reference.size
    interp = np.interp(src_cdf, ref_cdf, ref_values)
    return interp[src_idx].reshape(source.shape).astype(source.dtype)

# Build KiTS reference for histogram matching
print("Building KiTS reference...")
ref_files = list(KITS_SLICES.glob("case_00000_*_img.npy"))[:20]
ref_imgs = [(np.load(f) * 255).astype(np.uint8) for f in ref_files]
reference = np.stack(ref_imgs).mean(axis=0).astype(np.uint8)

# Load segmenter
print("Loading segmenter...")
model = UNet(spatial_dims=2, in_channels=1, out_channels=2,
             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
             num_res_units=2, norm="batch", dropout=0.1).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Iterate over all axial Kaggle images
df = pd.read_csv(MANIFEST)
axial_df = df[df.likely_view == "axial"].reset_index(drop=True)
print(f"Processing {len(axial_df)} axial images...")

stats = {"saved": 0, "skipped_empty_mask": 0, "skipped_error": 0}

for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    (OUT_DIR / cls).mkdir(exist_ok=True)

with torch.no_grad():
    for _, row in tqdm(axial_df.iterrows(), total=len(axial_df)):
        try:
            # Load + preprocess
            with Image.open(row["path"]).convert("L") as im:
                raw = np.array(im, dtype=np.uint8)
            resized = cv2.resize(raw, reference.shape[::-1], interpolation=cv2.INTER_AREA)
            eq = clahe.apply(resized)
            matched = match_histograms(eq, reference)
            tensor = torch.from_numpy(matched.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(tensor, size=IMG_SIZE, mode="bilinear",
                                                       align_corners=False).to(DEVICE)

            # Predict mask
            logits = model(tensor)
            pred = logits.argmax(dim=1).cpu().numpy()[0]

            # Skip if essentially no kidney detected
            if pred.sum() < MIN_MASK_PIXELS:
                stats["skipped_empty_mask"] += 1
                continue

            # Find bounding box of mask in IMG_SIZE coords, then map back to raw image
            ys, xs = np.where(pred > 0)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            # Add margin
            h, w = y_max - y_min, x_max - x_min
            y_min = max(0, int(y_min - MARGIN_FRAC * h))
            y_max = min(IMG_SIZE, int(y_max + MARGIN_FRAC * h))
            x_min = max(0, int(x_min - MARGIN_FRAC * w))
            x_max = min(IMG_SIZE, int(x_max + MARGIN_FRAC * w))

            # Map bbox from 256-space back to raw-image-space
            scale_y = raw.shape[0] / IMG_SIZE
            scale_x = raw.shape[1] / IMG_SIZE
            y_min_raw = int(y_min * scale_y)
            y_max_raw = int(y_max * scale_y)
            x_min_raw = int(x_min * scale_x)
            x_max_raw = int(x_max * scale_x)

            # Crop raw image and save as fixed-size PNG
            crop = raw[y_min_raw:y_max_raw, x_min_raw:x_max_raw]
            crop_resized = cv2.resize(crop, (ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE),
                                       interpolation=cv2.INTER_AREA)
            out_name = Path(row["path"]).stem + ".png"
            out_path = OUT_DIR / row["label"] / out_name
            Image.fromarray(crop_resized).save(out_path)
            stats["saved"] += 1

        except Exception as e:
            stats["skipped_error"] += 1
            if stats["skipped_error"] < 5:
                print(f"Error on {row['path']}: {e}")

print(f"\nDone. Saved: {stats['saved']}, "
      f"skipped (empty mask): {stats['skipped_empty_mask']}, "
      f"skipped (error): {stats['skipped_error']}")
print(f"ROI dataset: {OUT_DIR}")