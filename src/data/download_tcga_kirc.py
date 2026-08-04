"""
Download TCGA-KIRC CT scans from TCIA using their public REST API.

This script:
  1. Queries TCIA for all CT-modality series in TCGA-KIRC
  2. Downloads each series as DICOM to scratch
  3. Saves a manifest of what was downloaded
"""
import os
from pathlib import Path
from tcia_utils import nbia
import pandas as pd

SCRATCH = Path(os.environ["SCRATCH"])
OUT_DIR = SCRATCH / "kidney-data/raw/tcga_kirc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION = "TCGA-KIRC"

print(f"Querying TCIA for {COLLECTION} series...")
# Get all series in this collection (returns a list of dicts)
series_list = nbia.getSeries(collection=COLLECTION)
print(f"Total series in collection: {len(series_list)}")

# Filter to CT only (collection also has some other modalities)
ct_series = [s for s in series_list if s.get("Modality") == "CT"]
print(f"CT series: {len(ct_series)}")

# Save the full series list as a manifest before downloading
df = pd.DataFrame(ct_series)
manifest_path = OUT_DIR / "series_manifest.csv"
df.to_csv(manifest_path, index=False)
print(f"Saved series manifest: {manifest_path}")

# Show what we're about to download
print("\nFirst 5 series:")
print(df[["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality",
         "ImageCount"]].head().to_string() if "PatientID" in df.columns else df.head())

print(f"\nUnique patients: {df['PatientID'].nunique() if 'PatientID' in df.columns else 'unknown'}")
print(f"Total images across all series: "
      f"{df['ImageCount'].sum() if 'ImageCount' in df.columns else 'unknown'}")

# Download each series
print(f"\nDownloading to {OUT_DIR}...")
nbia.downloadSeries(
    series_data=ct_series,
    path=str(OUT_DIR),
    csv_filename="downloaded_series.csv",
)
print("Done.")