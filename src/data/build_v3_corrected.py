"""
Build corrected v3 manifests by dropping the duplicated 'mendeley' source.

Corrected v3:
  - 8,924 images, 561 patients across 4 sources
  - KiTS (104 pts), Abdalla (201 pts), KAUH (19 pts), TCGA (237 pts)

Generates 5 seed variants: base (seed 42) + _seed123 / _seed456 / _seed789 / _seed1011.
Each has a 'fold' column with patient-grouped stratified 5-fold splits.

Also generates 'manifest_with_masks*.csv' with the mask_path column preserved
from the KiTS segmentation extraction (already run in Phase 1).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
V3_DIR = SCRATCH / "kidney-data/processed/unified_v3"
V3C_DIR = SCRATCH / "kidney-data/processed/unified_v3_corrected"
V3C_DIR.mkdir(parents=True, exist_ok=True)

KEEP_SOURCES = ["kits", "abdalla", "kauh", "tcga"]
SEEDS = [42, 123, 456, 789, 1011]

# Load base v3 manifest with masks
base = pd.read_csv(V3_DIR / "manifest_with_masks.csv")
print(f"Original v3 manifest_with_masks.csv: {len(base)} rows")

# Drop duplicated mendeley
corrected = base[base["source"].isin(KEEP_SOURCES)].reset_index(drop=True)
print(f"Corrected v3 (dropped 'mendeley'): {len(corrected)} rows, "
      f"{corrected['patient_id'].nunique()} patients")

# Assemble class labels for stratification (use patient-level majority class)
patient_class = corrected.groupby("patient_id")["label"].agg(
    lambda s: s.value_counts().index[0]
).to_dict()

# Build per-seed manifests with folds
for seed in SEEDS:
    df = corrected.copy()
    df["_pat_class"] = df["patient_id"].map(patient_class)

    # StratifiedGroupKFold: stratifies by class while keeping patients together
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(df, df["_pat_class"], groups=df["patient_id"])):
        df.loc[test_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "Some rows unassigned to a fold"
    df = df.drop(columns=["_pat_class"])

    # Verify no patient overlap across folds
    for i in range(5):
        for j in range(i+1, 5):
            overlap = set(df[df["fold"] == i]["patient_id"]) & set(df[df["fold"] == j]["patient_id"])
            assert len(overlap) == 0, f"Patient overlap between folds {i} and {j}!"

    # Write manifest
    if seed == 42:
        out_name = "manifest_with_masks.csv"
    else:
        out_name = f"manifest_with_masks_seed{seed}.csv"
    df.to_csv(V3C_DIR / out_name, index=False)

    # Fold distribution summary
    print(f"\n=== Seed {seed} ({out_name}) ===")
    fold_summary = df.groupby("fold").agg(
        n_imgs=("path", "count"),
        n_patients=("patient_id", "nunique"),
    )
    print(fold_summary)
    # Per-fold class distribution
    ct = df.groupby(["fold", "label"]).size().unstack(fill_value=0)
    print(f"Per-fold class distribution:")
    print(ct)

print(f"\nDone. Corrected v3 manifests written to {V3C_DIR}")
print(f"Files:")
for f in sorted(V3C_DIR.glob("manifest_with_masks*.csv")):
    print(f"  {f.name}")