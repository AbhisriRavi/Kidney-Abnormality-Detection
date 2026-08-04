"""
Build unified_v3_roi manifest = v2_roi + Abdalla ROI crops.

Reads:
  $SCRATCH/kidney-data/processed/unified_v2_roi/manifest.csv   (existing v2 ROI)
  $SCRATCH/kidney-data/processed/abdalla_roi/extraction_log.csv (kept only)

Writes:
  $SCRATCH/kidney-data/processed/unified_v3_roi/manifest.csv
  $SCRATCH/kidney-data/processed/unified_v3_roi/manifest_with_folds.csv
  $SCRATCH/kidney-data/processed/unified_v3_roi/manifest_with_folds_seed123.csv
  $SCRATCH/kidney-data/processed/unified_v3_roi/manifest_with_folds_seed456.csv
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
V2_ROI_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2_roi/manifest.csv"
ABDALLA_LOG = SCRATCH / "kidney-data/processed/abdalla_roi/extraction_log.csv"
V3_ROI_DIR = SCRATCH / "kidney-data/processed/unified_v3_roi"
V3_ROI_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
N_FOLDS = 5
SEEDS = [42, 123, 456]

# Load v2_roi
v2_roi = pd.read_csv(V2_ROI_MANIFEST)
print(f"v2_roi: {len(v2_roi)} images, {v2_roi['patient_id'].nunique()} patients")

# Load Abdalla ROI extraction results and keep only "kept" rows
abd_log = pd.read_csv(ABDALLA_LOG)
abd_kept = abd_log[abd_log["reason"].isna() | (abd_log["reason"] == "")].copy()
# Rows in log for kept images have no "reason" column set — they were added directly
# For a robust filter, filter by whether the extracted file path column exists and points to abdalla_roi
if "reason" in abd_log.columns:
    abd_kept = abd_log[~abd_log["path"].str.contains("/abdalla2025/", na=False)]
else:
    abd_kept = abd_log

# Ensure only the columns we need
required_cols = ["path", "label", "source", "patient_id", "stem"]
abd_kept = abd_kept[required_cols].copy()
print(f"Abdalla ROI kept: {len(abd_kept)} images, {abd_kept['patient_id'].nunique()} patients")

# Pad schema
v2_roi["abdalla_site"]     = ""
v2_roi["abdalla_sex"]      = ""
v2_roi["abdalla_aug_type"] = ""
abd_kept["abdalla_site"]     = ""  # backfill later if needed
abd_kept["abdalla_sex"]      = ""
abd_kept["abdalla_aug_type"] = ""

v3_roi = pd.concat([v2_roi, abd_kept], ignore_index=True, sort=False)
v3_roi.to_csv(V3_ROI_DIR / "manifest.csv", index=False)

print(f"\nv3_roi: {len(v3_roi)} images, {v3_roi['patient_id'].nunique()} patients")
print(f"  per-class: {dict(v3_roi['label'].value_counts())}")
print(f"  per-source: {dict(v3_roi['source'].value_counts())}")

# Generate folds for each seed
for seed in SEEDS:
    df = v3_roi.copy()
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(splitter.split(
        X=df.index.values, y=df["label"].values, groups=df["patient_id"].values
    )):
        df.loc[df.index[test_idx], "fold"] = fold_idx
    assert (df["fold"] >= 0).all()
    assert (df.groupby("patient_id")["fold"].nunique() == 1).all()

    suffix = "" if seed == 42 else f"_seed{seed}"
    out = V3_ROI_DIR / f"manifest_with_folds{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"  seed {seed}: {out} ({dict(df.groupby('fold').size())})")

print("\nDone.")