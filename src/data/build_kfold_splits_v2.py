"""
Add a 'fold' column to the v2 unified manifest using StratifiedGroupKFold:
  - 5 folds
  - Stratified by class
  - Grouped by patient_id (no patient appears in more than one fold)
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST_IN = SCRATCH / "kidney-data/processed/unified_v2/manifest.csv"
MANIFEST_OUT = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"

N_FOLDS = 5
SEED = 42

df = pd.read_csv(MANIFEST_IN)
print(f"Loaded manifest with {len(df)} images, {df['patient_id'].nunique()} unique patients\n")

splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

df["fold"] = -1
for fold_idx, (_, test_idx) in enumerate(splitter.split(
    X=df.index.values, y=df["label"].values, groups=df["patient_id"].values
)):
    df.loc[df.index[test_idx], "fold"] = fold_idx

assert (df["fold"] >= 0).all(), "Some rows didn't get assigned a fold"

# Sanity-check
patient_fold_count = df.groupby("patient_id")["fold"].nunique()
n_leaky = (patient_fold_count > 1).sum()
print(f"Patients appearing in multiple folds: {n_leaky}")
assert n_leaky == 0, "Patient leakage detected"

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

# CRITICAL CHECK: each class should have at least 2 sources represented per fold
# (for Normal/Cyst/Tumor — Stone is single-source so we exclude it)
print("\n=== Source coverage per (fold × class) — the methodological check ===")
print("Each cell shows the number of sources contributing to that fold/class")
cov = df.groupby(["fold", "label"])["source"].nunique().unstack(fill_value=0)
print(cov.to_string())
print("\nFor multi-source classes (Cyst, Tumor, Normal), this should be ≥ 2 in every fold")
print("for the source-shortcut fix to apply in every fold.")