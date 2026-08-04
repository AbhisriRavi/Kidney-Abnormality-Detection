"""
TCGA-KIRC smoke test.

Goal: confirm we can:
  1. List patients and series in TCGA-KIRC via TCIA API
  2. Download a single patient's CT series
  3. Convert the DICOM files to 2D PNG slices

This script does ONLY that — no full download yet.
"""
import os
from pathlib import Path
from tcia_utils import nbia
import pandas as pd

SCRATCH = Path(os.environ["SCRATCH"])
OUT = SCRATCH / "kidney-data/raw/tcga_kirc/smoke_test"
OUT.mkdir(parents=True, exist_ok=True)

COLLECTION = "TCGA-KIRC"

# ---------------------------------------------------------------------------
# 1. List patients
# ---------------------------------------------------------------------------
print(f"Querying patients in {COLLECTION}...")
patients = nbia.getPatient(collection=COLLECTION)
patients_df = pd.DataFrame(patients)
print(f"Total patients: {len(patients_df)}")
print(f"Columns: {list(patients_df.columns)}")
print(f"\nFirst 5 patients:")
print(patients_df.head().to_string())

# ---------------------------------------------------------------------------
# 2. List all CT series
# ---------------------------------------------------------------------------
print(f"\n\nQuerying CT series in {COLLECTION}...")
series = nbia.getSeries(collection=COLLECTION, modality="CT")
series_df = pd.DataFrame(series)
print(f"Total CT series: {len(series_df)}")
print(f"Columns: {list(series_df.columns)}")
print(f"\nFirst 3 series:")
print(series_df.head(3).to_string())
print(f"\nImage count distribution per series:")
if "ImageCount" in series_df.columns:
    print(series_df["ImageCount"].describe().to_string())

# Save for next step
series_df.to_csv(OUT / "all_ct_series.csv", index=False)
print(f"\nSaved series list to: {OUT / 'all_ct_series.csv'}")