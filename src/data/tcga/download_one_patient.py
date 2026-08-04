"""
TCGA-KIRC smoke test, part 2: download one patient end-to-end.

1. Pick the first patient with a clean post-contrast series
2. Download just that series via TCIA API
3. Convert the DICOMs to 2D PNG slices using HU windowing
4. Show summary

If this works smoothly, the full download pipeline will too.
"""
import os
from pathlib import Path
from tcia_utils import nbia
import pandas as pd
import numpy as np
import pydicom
import cv2
from PIL import Image

SCRATCH = Path(os.environ["SCRATCH"])
ROOT = SCRATCH / "kidney-data/raw/tcga_kirc/smoke_test"
DICOM_OUT = ROOT / "dicom"
PNG_OUT = ROOT / "png"
DICOM_OUT.mkdir(parents=True, exist_ok=True)
PNG_OUT.mkdir(parents=True, exist_ok=True)

# HU windowing constants (same as KiTS)
HU_MIN, HU_MAX = -200, 300
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# Reapply the same filtering logic to find ONE good series
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / "all_ct_series.csv")
df["desc_upper"] = df["SeriesDescription"].fillna("").str.upper()
df["bpe_upper"] = df["BodyPartExamined"].fillna("").str.upper()
good_bp = df["bpe_upper"].isin(["KIDNEY", "ABDOMEN", "ABDOMENPELVIS", "ABDOMENPELVI", "ABD"])
good_size = df["ImageCount"].between(30, 400)
not_scout = ~df["desc_upper"].str.contains("SCOUT|LOC|TOPOGRAM|SURVIEW", na=False)
candidates = df[good_bp & good_size & not_scout].copy()

# Categorise phase
def cat_phase(d):
    d = d.upper()
    if any(s in d for s in ["PRE-CONTRAST", "PRE CONTRAST", "PRE LIVER", "NON CONTRAST", "NONCONTRAST"]):
        return "pre"
    if any(s in d for s in ["DELAY", "EXCRETORY"]):
        return "delayed"
    if any(s in d for s in ["POST CONTRAST", "POST-CONTRAST", "ARTERIAL", "PORTAL",
                             "NEPHROGRAPHIC", "CORTICOMEDULLARY", "RENAL", "C/A/P", "A/P", "CAP "]):
        return "post"
    return "unknown"

candidates["phase"] = candidates["desc_upper"].apply(cat_phase)

# Pick the FIRST post-contrast series we can find
post = candidates[candidates["phase"] == "post"].sort_values("ImageCount", ascending=False)
target = post.iloc[0]
print(f"Selected series:")
print(f"  PatientID:         {target['PatientID']}")
print(f"  SeriesDescription: {target['SeriesDescription']}")
print(f"  ImageCount:        {target['ImageCount']}")
print(f"  FileSize:          {target['FileSize'] / 1e6:.1f} MB")
print(f"  Phase:             {target['phase']}")
print(f"  SeriesInstanceUID: {target['SeriesInstanceUID']}\n")

# ---------------------------------------------------------------------------
# Download just this series
# ---------------------------------------------------------------------------
print("Downloading series via TCIA API...")
nbia.downloadSeries(
    series_data=[{"SeriesInstanceUID": target["SeriesInstanceUID"]}],
    path=str(DICOM_OUT),
    csv_filename="downloaded.csv",
)
print("Download complete.\n")

# ---------------------------------------------------------------------------
# Find the DICOMs we just got
# ---------------------------------------------------------------------------
dcm_files = sorted(DICOM_OUT.rglob("*.dcm"))
print(f"Found {len(dcm_files)} DICOM files on disk")
if not dcm_files:
    print("ERROR: No DICOM files found. Check download output above.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Convert to PNG with HU windowing
# ---------------------------------------------------------------------------
def hu_window(pixel_array, slope, intercept):
    hu = pixel_array.astype(np.float32) * slope + intercept
    hu = np.clip(hu, HU_MIN, HU_MAX)
    hu = (hu - HU_MIN) / (HU_MAX - HU_MIN)
    return (hu * 255).astype(np.uint8)

pid = target["PatientID"]
print(f"\nConverting DICOMs to PNG for {pid}...")
for i, dcm_path in enumerate(dcm_files):
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = hu_window(arr, slope, intercept)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    out_name = f"{pid}_slice{i:04d}.png"
    Image.fromarray(img).save(PNG_OUT / out_name)

png_files = sorted(PNG_OUT.glob(f"{pid}_*.png"))
print(f"Wrote {len(png_files)} PNGs")
print(f"\nSamples:")
for f in png_files[:3]:
    print(f"  {f}")
print("  ...")
for f in png_files[-3:]:
    print(f"  {f}")

print(f"\n=== Smoke test complete ===")
print(f"DICOMs: {DICOM_OUT}")
print(f"PNGs:   {PNG_OUT}")
print(f"Inspect a sample PNG in VSCode to confirm it looks like a kidney CT slice.")