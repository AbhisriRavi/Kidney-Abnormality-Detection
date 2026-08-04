"""
Add a 'fold' column to the unified manifest using StratifiedGroupKFold:
  - 5 folds
  - Stratified by class (each fold has roughly proportional class counts)
  - Grouped by patient_id (no patient appears in more than one fold)

These splits are saved once and reused by every downstream experiment
(full-image, ROI, etc.) so the comparison is fair.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST_IN = SCRATCH / "kidney-data/processed/unified_kits_mendeley/manifest.csv"
MANIFEST_OUT = SCRATCH / "kidney-data/processed/unified_kits_mendeley/manifest_with_folds.csv"

N_FOLDS = 5
SEED = 42

df = pd.read_csv(MANIFEST_IN)
print(f"Loaded manifest with {len(df)} images, {df['patient_id'].nunique()} unique patients\n")

# StratifiedGroupKFold needs labels, groups, and X. X is unused for splitting but required by the API.
splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Assign each row to a fold
df["fold"] = -1
for fold_idx, (_, test_idx) in enumerate(splitter.split(
    X=df.index.values, y=df["label"].values, groups=df["patient_id"].values
)):
    df.loc[df.index[test_idx], "fold"] = fold_idx

assert (df["fold"] >= 0).all(), "Some rows didn't get assigned a fold"

# Sanity-check: no patient should appear in multiple folds
patient_fold_count = df.groupby("patient_id")["fold"].nunique()
n_leaky = (patient_fold_count > 1).sum()
print(f"Patients appearing in multiple folds: {n_leaky}")
assert n_leaky == 0, "Patient leakage detected — splitter failed"

# Save
df.to_csv(MANIFEST_OUT, index=False)
print(f"Saved: {MANIFEST_OUT}\n")

# Reports
print("=== Fold composition ===\n")
print("Images per fold per class:")
print(df.groupby(["fold", "label"]).size().unstack(fill_value=0).to_string())
print()
print("Patients per fold per class:")
print(df.groupby(["fold", "label"])["patient_id"].nunique().unstack(fill_value=0).to_string())
print()
print("Source distribution per fold:")
print(df.groupby(["fold", "source"]).size().unstack(fill_value=0).to_string())
print()
print("Class proportion per fold (%):")
pct = df.groupby(["fold", "label"]).size().unstack(fill_value=0)
pct = pct.div(pct.sum(axis=1), axis=0) * 100
print(pct.round(1).to_string())