"""
Build a multi-class test set from KiTS23 for cross-dataset evaluation.

For each KiTS case, extract axial slices and label them as:
  - Normal: slice has kidney (label 1) but NO tumor (2) and NO cyst (3)
  - Tumor:  slice has tumor (label 2)
  - Cyst:   slice has cyst (label 3) AND no tumor

If a slice has both tumor and cyst, we prefer tumor (more clinically relevant
and the more conservative call for a classification system).

Output:
  $SCRATCH/kidney-data/processed/kits_external_test/
    full/<class>/case_XXXXX_slice_YYYY.png    (full-image version)
    roi/<class>/case_XXXXX_slice_YYYY.png     (ROI-cropped version)
    manifest.csv                              (paths + labels)
"""
import os
import csv
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])
KITS_DIR = SCRATCH / "kidney-data/raw/kits23/kits23/dataset"
OUT = SCRATCH / "kidney-data/processed/kits_external_test"
(OUT / "full").mkdir(parents=True, exist_ok=True)
(OUT / "roi").mkdir(parents=True, exist_ok=True)
for cls in ["Normal", "Cyst", "Tumor"]:
    (OUT / "full" / cls).mkdir(exist_ok=True)
    (OUT / "roi" / cls).mkdir(exist_ok=True)

# HU windowing identical to training-time preprocessing
HU_MIN, HU_MAX = -200, 300
IMG_OUT_SIZE = 224

# Limits to keep the external set balanced and manageable
MAX_PER_CLASS = 800           # cap each class to avoid huge imbalance
MIN_KIDNEY_PIXELS = 200       # require some kidney visible for Normal
MIN_LESION_PIXELS = 50        # require lesion to be substantial for Cyst/Tumor


def window_and_norm(img):
    """Match the training-time preprocessing for KiTS."""
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    return (img * 255).astype(np.uint8)


def crop_to_kidney_bbox(img_2d, kidney_mask_2d, margin=0.10):
    """Crop the image to the bounding box of the kidney mask + margin."""
    ys, xs = np.where(kidney_mask_2d > 0)
    if len(ys) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h, w = y_max - y_min, x_max - x_min
    y_min = max(0, int(y_min - margin * h))
    y_max = min(img_2d.shape[0], int(y_max + margin * h))
    x_min = max(0, int(x_min - margin * w))
    x_max = min(img_2d.shape[1], int(x_max + margin * w))
    return img_2d[y_min:y_max, x_min:x_max]


def save_pair(img_2d, kidney_mask_2d, full_path, roi_path):
    """Save full-image and ROI-cropped versions, both 224x224."""
    img_uint8 = window_and_norm(img_2d.astype(np.float32))
    # Full image, resized
    full_resized = cv2.resize(img_uint8, (IMG_OUT_SIZE, IMG_OUT_SIZE), interpolation=cv2.INTER_AREA)
    Image.fromarray(full_resized).save(full_path)
    # ROI crop, resized
    roi_crop = crop_to_kidney_bbox(img_uint8, kidney_mask_2d)
    if roi_crop is None or roi_crop.size == 0:
        return False
    roi_resized = cv2.resize(roi_crop, (IMG_OUT_SIZE, IMG_OUT_SIZE), interpolation=cv2.INTER_AREA)
    Image.fromarray(roi_resized).save(roi_path)
    return True


# ---------------------------------------------------------------------------
# Walk through KiTS cases and harvest slices
# ---------------------------------------------------------------------------
manifest_rows = []
counters = {"Normal": 0, "Cyst": 0, "Tumor": 0}
cases = sorted(KITS_DIR.glob("case_*"))
print(f"Processing {len(cases)} KiTS cases...")

for case in tqdm(cases):
    if all(counters[c] >= MAX_PER_CLASS for c in counters):
        break
    try:
        img_vol = nib.load(case / "imaging.nii.gz").get_fdata()
        seg_vol = nib.load(case / "segmentation.nii.gz").get_fdata()
    except Exception as e:
        print(f"Skipping {case.name}: {e}")
        continue

    # iterate through axial slices (first axis based on earlier verification)
    for i in range(img_vol.shape[0]):
        seg_slice = seg_vol[i]
        kidney_mask = (seg_slice > 0).astype(np.uint8)  # any kidney structure
        tumor_pixels = int((seg_slice == 2).sum())
        cyst_pixels = int((seg_slice == 3).sum())
        kidney_pixels = int(kidney_mask.sum())

        # Classify this slice
        if tumor_pixels >= MIN_LESION_PIXELS:
            cls = "Tumor"
        elif cyst_pixels >= MIN_LESION_PIXELS:
            cls = "Cyst"
        elif kidney_pixels >= MIN_KIDNEY_PIXELS and tumor_pixels == 0 and cyst_pixels == 0:
            cls = "Normal"
        else:
            continue

        if counters[cls] >= MAX_PER_CLASS:
            continue

        stem = f"{case.name}_slice{i:04d}"
        full_path = OUT / "full" / cls / f"{stem}.png"
        roi_path = OUT / "roi" / cls / f"{stem}.png"
        ok = save_pair(img_vol[i], kidney_mask, full_path, roi_path)
        if ok:
            manifest_rows.append({
                "stem": stem, "label": cls,
                "full_image_path": str(full_path),
                "roi_path": str(roi_path),
            })
            counters[cls] += 1

# ---------------------------------------------------------------------------
# Save manifest + summary
# ---------------------------------------------------------------------------
with open(OUT / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stem", "label", "full_image_path", "roi_path"])
    w.writeheader()
    w.writerows(manifest_rows)

print("\n=== KiTS external test set built ===")
print(f"Total slices: {len(manifest_rows)}")
for cls in counters:
    print(f"  {cls:<8}: {counters[cls]}")
print(f"Saved to: {OUT}")