"""
Extract the KAUH 'Normal case with cyst' patients as additional Cyst examples.

Pipeline:
  1. Read Excel, filter to Situation == 'Normal case with cyst' (28 patients)
  2. For each, locate Dataset/NN.zip and extract
  3. Subsample 10 slices per patient from middle 60% of the series
  4. Apply CLAHE + histogram matching to a KiTS reference image
  5. Resize to 224x224, save as PNG with namespaced filename
  6. Write manifest.csv

Output:
  $SCRATCH/kidney-data/processed/kauh_cyst/
    images/kauh_P###/slice_NN.png
    manifest.csv
    work/  (temp extracted JPGs, can be deleted after)
"""
import os
import csv
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.exposure import match_histograms

SCRATCH = Path(os.environ["SCRATCH"])

# Inputs
KAUH_ROOT = SCRATCH / "kidney-data/raw/kauh_jordan/github_extras/KidneyTumor/Dataset"
EXCEL_PATH = KAUH_ROOT / "00Kidney_Patients.xlsx"

# Outputs
OUT_ROOT = SCRATCH / "kidney-data/processed/kauh_cyst"
IMG_OUT = OUT_ROOT / "images"
WORK_DIR = OUT_ROOT / "work"
MANIFEST_OUT = OUT_ROOT / "manifest.csv"
IMG_OUT.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Reference image for histogram matching — pick one KiTS-derived Normal slice
# that already exists in our processed pool
KITS_REFERENCE = SCRATCH / "kidney-data/processed/unified_kits_mendeley/images/Normal"

# Config
SLICES_PER_PATIENT = 10
MID_FRACTION = 0.6
IMG_SIZE = 224


def find_kits_reference():
    """Find any KiTS-sourced Normal image to use as the histogram-matching reference."""
    candidates = sorted(KITS_REFERENCE.glob("kits_case_*_slice*.png"))
    if not candidates:
        raise SystemExit(f"No KiTS reference found in {KITS_REFERENCE}")
    # Pick one from roughly the middle of the list (a healthy mid-abdomen slice)
    ref_path = candidates[len(candidates) // 2]
    ref = np.array(Image.open(ref_path).convert("L"), dtype=np.uint8)
    return ref, ref_path


def adapt_image(img_path, reference):
    """Apply CLAHE + histogram matching + resize to make a KAUH JPG look like a KiTS slice."""
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # Histogram match to KiTS reference
    try:
        img = match_histograms(img, reference).astype(np.uint8)
    except Exception:
        pass  # if matching fails, keep CLAHE-only
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def process_patient(patient_num, reference):
    """Extract one patient's zip, subsample slices, adapt, save."""
    zip_name = f"{int(patient_num):02d}.zip"
    zip_path = KAUH_ROOT / zip_name
    if not zip_path.exists():
        # Try other numbering formats (e.g. "1.zip" without zero-pad)
        for alt in [f"{int(patient_num)}.zip", f"{patient_num:03d}.zip"]:
            if (KAUH_ROOT / alt).exists():
                zip_path = KAUH_ROOT / alt
                break
        else:
            return None, "zip not found"

    pid = f"kauh_P{int(patient_num):03d}"
    extract_dir = WORK_DIR / pid
    extract_dir.mkdir(exist_ok=True)

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        return None, f"extract failed: {e}"

    # Find all JPGs in the extracted folder (likely under a single subfolder named e.g. '1/')
    jpgs = sorted(extract_dir.rglob("*.jpg")) + sorted(extract_dir.rglob("*.JPG"))
    if not jpgs:
        return None, "no JPGs"

    # Subsample 10 from middle 60%
    n = len(jpgs)
    margin = (1 - MID_FRACTION) / 2
    lo = int(n * margin)
    hi = int(n * (1 - margin))
    if hi - lo < SLICES_PER_PATIENT:
        chosen = list(range(lo, hi))
    else:
        chosen = np.linspace(lo, hi - 1, SLICES_PER_PATIENT, dtype=int).tolist()

    # Adapt + save
    pat_out = IMG_OUT / pid
    pat_out.mkdir(exist_ok=True)
    records = []
    for out_i, src_i in enumerate(chosen):
        img = adapt_image(jpgs[src_i], reference)
        out_name = f"slice_{out_i:02d}.png"
        out_path = pat_out / out_name
        Image.fromarray(img).save(out_path)
        records.append({
            "path": str(out_path),
            "label": "Cyst",
            "patient_id": pid,
            "source": "kauh",
            "stem": f"{pid}_{out_name}",
        })

    # Clean up the extracted JPGs to save space
    shutil.rmtree(extract_dir, ignore_errors=True)
    return records, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading KAUH metadata...")
df = pd.read_excel(EXCEL_PATH)
cyst_patients = df[df["Situation"] == "Normal case with cyst"].copy()
print(f"Cyst patients in metadata: {len(cyst_patients)}\n")

print("Loading KiTS reference image for histogram matching...")
ref, ref_path = find_kits_reference()
print(f"Reference: {ref_path.name}\n")

all_records = []
n_ok, n_failed = 0, 0
for idx, row in cyst_patients.iterrows():
    pid_num = row["Patient_Num"]
    print(f"  Patient {pid_num}...", end=" ", flush=True)
    records, status = process_patient(pid_num, ref)
    if records is None:
        print(f"FAILED ({status})")
        n_failed += 1
        continue
    all_records.extend(records)
    n_ok += 1
    print(f"wrote {len(records)} slices")

# Manifest
with open(MANIFEST_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label", "patient_id", "source", "stem"])
    w.writeheader()
    w.writerows(all_records)

print(f"\n{'=' * 60}")
print("KAUH cyst extraction complete")
print('=' * 60)
print(f"  Patients OK:        {n_ok}")
print(f"  Patients failed:    {n_failed}")
print(f"  Total slices saved: {len(all_records)}")
print(f"  Manifest:           {MANIFEST_OUT}")