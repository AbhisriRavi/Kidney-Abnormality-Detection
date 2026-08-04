"""
Build the unified manifest combining KiTS23 and Mendeley sources for the
multi-source 4-class kidney abnormality classification project.

Classes (final 4):
  Normal  - Mendeley Non-Stone (full images) + KiTS slices with kidney only
  Cyst    - KiTS slices containing cyst (label 3)
  Tumor   - KiTS slices containing tumor (label 2)
  Stone   - Mendeley Stone

For KiTS we save 2D slices as PNG so the rest of the pipeline doesn't need
to load NIfTI volumes repeatedly.

Output:
  $SCRATCH/kidney-data/processed/unified_kits_mendeley/
    images/<class>/<patient_id>_<slice>.png   (224x224 PNGs)
    manifest.csv                              (path, label, patient_id, source)
"""
import os
import csv
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])

# Sources
KITS_DIR = SCRATCH / "kidney-data/raw/kits23/kits23/dataset"
MENDELEY_ROOT = (SCRATCH / "kidney-data/raw/mendeley_stone" /
                 "Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research" /
                 "Kindey Stone Dataset/Original")
MENDELEY_STONE = MENDELEY_ROOT / "Stone"
MENDELEY_NONSTONE = MENDELEY_ROOT / "Non-Stone"

# Output
OUT_ROOT = SCRATCH / "kidney-data/processed/unified_kits_mendeley"
IMG_OUT = OUT_ROOT / "images"
MANIFEST_PATH = OUT_ROOT / "manifest.csv"
for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    (IMG_OUT / cls).mkdir(parents=True, exist_ok=True)

# KiTS preprocessing constants (match the segmenter training)
HU_MIN, HU_MAX = -200, 300
IMG_SIZE = 224

# Caps to keep classes balanced
MAX_KITS_PER_CLASS = 1000   # cap KiTS-derived slices per class
MIN_KIDNEY_PIXELS = 200
MIN_LESION_PIXELS = 50

# Patient-ID parsers
MENDELEY_PATIENT_RE = re.compile(r"^(P\d+)_")


def hu_window_to_uint8(img2d):
    img = np.clip(img2d.astype(np.float32), HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    return (img * 255).astype(np.uint8)


def parse_mendeley_patient(filename):
    m = MENDELEY_PATIENT_RE.match(filename)
    if not m:
        return None
    return f"mendeley_{m.group(1)}"


def save_image_224(arr_uint8, out_path):
    """Resize to 224x224 and save as PNG."""
    resized = cv2.resize(arr_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized).save(out_path)


# Process Mendeley
print("Processing Mendeley Non-Stone (Normal)...")
rows = []
for f in tqdm(sorted(MENDELEY_NONSTONE.glob("*.jpg"))):
    pid = parse_mendeley_patient(f.name)
    if pid is None:
        continue
    img = np.array(Image.open(f).convert("L"), dtype=np.uint8)
    out_name = f"{pid}_{f.stem}.png"
    out_path = IMG_OUT / "Normal" / out_name
    save_image_224(img, out_path)
    rows.append({"path": str(out_path), "label": "Normal",
                  "patient_id": pid, "source": "mendeley", "stem": out_name})

print("Processing Mendeley Stone...")
for f in tqdm(sorted(MENDELEY_STONE.glob("*.jpg"))):
    pid = parse_mendeley_patient(f.name)
    if pid is None:
        continue
    img = np.array(Image.open(f).convert("L"), dtype=np.uint8)
    out_name = f"{pid}_{f.stem}.png"
    out_path = IMG_OUT / "Stone" / out_name
    save_image_224(img, out_path)
    rows.append({"path": str(out_path), "label": "Stone",
                  "patient_id": pid, "source": "mendeley", "stem": out_name})

# KiTS — Normal, Cyst, Tumor slices
print("Processing KiTS volumes...")
counters = {"Normal": 0, "Cyst": 0, "Tumor": 0}
cases = sorted(KITS_DIR.glob("case_*"))

for case in tqdm(cases):
    if all(counters[c] >= MAX_KITS_PER_CLASS for c in counters):
        break
    try:
        img_vol = nib.load(case / "imaging.nii.gz").get_fdata()
        seg_vol = nib.load(case / "segmentation.nii.gz").get_fdata()
    except Exception as e:
        print(f"Skip {case.name}: {e}")
        continue

    pid = f"kits_{case.name}"
    for i in range(img_vol.shape[0]):
        seg = seg_vol[i]
        tumor_px = int((seg == 2).sum())
        cyst_px = int((seg == 3).sum())
        kidney_px = int((seg > 0).sum())

        if tumor_px >= MIN_LESION_PIXELS:
            cls = "Tumor"
        elif cyst_px >= MIN_LESION_PIXELS:
            cls = "Cyst"
        elif kidney_px >= MIN_KIDNEY_PIXELS and tumor_px == 0 and cyst_px == 0:
            cls = "Normal"
        else:
            continue

        if counters[cls] >= MAX_KITS_PER_CLASS:
            continue

        arr = hu_window_to_uint8(img_vol[i])
        out_name = f"{pid}_slice{i:04d}.png"
        out_path = IMG_OUT / cls / out_name
        save_image_224(arr, out_path)
        rows.append({"path": str(out_path), "label": cls,
                      "patient_id": pid, "source": "kits", "stem": out_name})
        counters[cls] += 1

# manifest
with open(MANIFEST_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label", "patient_id", "source", "stem"])
    w.writeheader()
    w.writerows(rows)

# Summary
import pandas as pd
df = pd.DataFrame(rows)

print("\n=== Unified manifest built ===")
print(f"Total images: {len(df)}")
print(f"Manifest:     {MANIFEST_PATH}")
print()
print("Class × source distribution:")
print(df.groupby(["label", "source"]).size().unstack(fill_value=0).to_string())
print()
print("Unique patients per class:")
print(df.groupby("label")["patient_id"].nunique().to_string())
print()
print(f"Total unique patients: {df['patient_id'].nunique()}")