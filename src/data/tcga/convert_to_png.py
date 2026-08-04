"""
Convert TCGA-KIRC downloaded DICOMs to 2D PNGs.

For each patient:
  - Read all DICOMs in their chosen series
  - Sort by InstanceNumber (CT slice order)
  - Apply HU windowing [-200, 300], rescale to [0,255]
  - Subsample SLICES_PER_PATIENT (default 10) evenly from the middle MID_FRACTION (default 60%)
  - Resize to 224x224
  - Save as PNG: tcga_<PatientID>_slice<N>.png

Output:
  $SCRATCH/kidney-data/processed/tcga_kirc/
    images/<PatientID>/slice_<NN>.png    (all chosen slices, namespaced)
    manifest.csv                          (path, patient_id, source, label)

All TCGA-KIRC patients are kidney tumor cases, so every image is labelled "Tumor".
"""
import os
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import cv2
from PIL import Image

SCRATCH = Path(os.environ["SCRATCH"])
RAW = SCRATCH / "kidney-data/raw/tcga_kirc/full"
DICOM_ROOT = RAW / "dicom"
CHOSEN_CSV = RAW / "chosen_series.csv"

OUT_ROOT = SCRATCH / "kidney-data/processed/tcga_kirc"
IMG_OUT = OUT_ROOT / "images"
MANIFEST_OUT = OUT_ROOT / "manifest.csv"
IMG_OUT.mkdir(parents=True, exist_ok=True)

# Config
HU_MIN, HU_MAX = -200, 300
IMG_SIZE = 224
SLICES_PER_PATIENT = 10
MID_FRACTION = 0.6  # take from middle 60% of the series


def hu_window(pixel_array, slope, intercept):
    hu = pixel_array.astype(np.float32) * slope + intercept
    hu = np.clip(hu, HU_MIN, HU_MAX)
    hu = (hu - HU_MIN) / (HU_MAX - HU_MIN)
    return (hu * 255).astype(np.uint8)


def process_patient(pid, series_dir):
    """Read, sort, subsample, convert. Returns list of dicts (one per saved slice)."""
    dcm_paths = list(series_dir.rglob("*.dcm"))
    if not dcm_paths:
        return []

    # Read all DICOMs and pair with their InstanceNumber for proper ordering
    pairs = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            instance_no = int(getattr(ds, "InstanceNumber", 0))
            pairs.append((instance_no, p, ds))
        except Exception as e:
            print(f"  skip unreadable {p.name}: {e}")
    if not pairs:
        return []

    pairs.sort(key=lambda x: x[0])
    n = len(pairs)

    # Take SLICES_PER_PATIENT evenly-spaced indices from the middle MID_FRACTION of the series
    margin = (1 - MID_FRACTION) / 2
    lo = int(n * margin)
    hi = int(n * (1 - margin))
    if hi - lo < SLICES_PER_PATIENT:
        # series too short for our subsampling — take everything in the mid range
        chosen_idxs = list(range(lo, hi))
    else:
        chosen_idxs = np.linspace(lo, hi - 1, SLICES_PER_PATIENT, dtype=int).tolist()

    pat_dir = IMG_OUT / pid
    pat_dir.mkdir(exist_ok=True)
    records = []

    for out_i, src_i in enumerate(chosen_idxs):
        _, src_path, ds = pairs[src_i]
        arr = ds.pixel_array
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = hu_window(arr, slope, intercept)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        out_name = f"slice_{out_i:02d}.png"
        out_path = pat_dir / out_name
        Image.fromarray(img).save(out_path)
        records.append({
            "path": str(out_path),
            "label": "Tumor",
            "patient_id": f"tcga_{pid}",
            "source": "tcga",
            "stem": f"tcga_{pid}_{out_name}",
        })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not CHOSEN_CSV.exists():
    print(f"ERROR: chosen_series.csv not found. Did the download finish?")
    sys.exit(1)

chosen = pd.read_csv(CHOSEN_CSV)
print(f"Processing {len(chosen)} patients from TCGA-KIRC...")
print(f"Strategy: {SLICES_PER_PATIENT} slices per patient from middle {int(MID_FRACTION*100)}% of series\n")

all_records = []
n_ok, n_missing, n_empty = 0, 0, 0

for idx, row in chosen.iterrows():
    pid = row["PatientID"]
    suid = row["SeriesInstanceUID"]
    series_dir = DICOM_ROOT / pid / suid

    if not series_dir.exists():
        n_missing += 1
        if n_missing <= 5:
            print(f"[{idx+1}/{len(chosen)}] {pid}: series folder missing — skipping")
        continue

    records = process_patient(pid, series_dir)
    if not records:
        n_empty += 1
        print(f"[{idx+1}/{len(chosen)}] {pid}: no usable DICOMs")
        continue

    all_records.extend(records)
    n_ok += 1
    if (idx + 1) % 25 == 0 or idx + 1 == len(chosen):
        print(f"[{idx+1}/{len(chosen)}] {pid}: wrote {len(records)} slices "
              f"(running totals: {n_ok} ok, {n_missing} missing, {n_empty} empty)")

# Write the manifest
with open(MANIFEST_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label", "patient_id", "source", "stem"])
    w.writeheader()
    w.writerows(all_records)

# Summary
print("\n" + "=" * 60)
print("TCGA-KIRC conversion complete")
print("=" * 60)
print(f"  Patients processed OK:    {n_ok}")
print(f"  Patients with missing dl: {n_missing}")
print(f"  Patients with empty data: {n_empty}")
print(f"  Total slices written:     {len(all_records)}")
print(f"  Output manifest:          {MANIFEST_OUT}")
print(f"  Image directory:          {IMG_OUT}")