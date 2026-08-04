"""
Generate independent patient-grouped 5-fold splits at multiple seeds.

For each seed in [123, 456]:
  - Re-run StratifiedGroupKFold on the unified v2 manifest
  - Save as manifest_with_folds_seed<N>.csv
  - Same for v2_roi manifest

This gives us 3 fully independent CV replications across seeds [42, 123, 456].
Seed 42 manifest already exists from build_kfold_splits_v2.py.
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])

MANIFESTS = {
    "v2": SCRATCH / "kidney-data/processed/unified_v2/manifest.csv",
    "v2_roi": SCRATCH / "kidney-data/processed/unified_v2_roi/manifest.csv",
}
OUT_DIRS = {
    "v2": SCRATCH / "kidney-data/processed/unified_v2",
    "v2_roi": SCRATCH / "kidney-data/processed/unified_v2_roi",
}

N_FOLDS = 5
SEEDS = [123, 456]

# v2_roi manifest may have been saved as "manifest_with_folds.csv" only
# (it was generated from the v2 fold assignments). Use whichever exists.
def find_base_manifest(key):
    candidates = [
        OUT_DIRS[key] / "manifest.csv",
        OUT_DIRS[key] / "manifest_with_folds.csv",  # use this and drop the fold column
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"No base manifest found for {key}")

for key in ["v2", "v2_roi"]:
    base_path = find_base_manifest(key)
    print(f"\n=== {key.upper()} (base: {base_path}) ===")
    df_base = pd.read_csv(base_path)
    # Drop existing fold column if present so we start clean
    if "fold" in df_base.columns:
        df_base = df_base.drop(columns=["fold"])

    for seed in SEEDS:
        df = df_base.copy()
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        df["fold"] = -1
        for fold_idx, (_, test_idx) in enumerate(splitter.split(
            X=df.index.values, y=df["label"].values, groups=df["patient_id"].values
        )):
            df.loc[df.index[test_idx], "fold"] = fold_idx
        assert (df["fold"] >= 0).all()

        # Verify zero patient leakage
        per_patient_folds = df.groupby("patient_id")["fold"].nunique()
        assert (per_patient_folds == 1).all(), "Patient leakage in fold split!"

        out_path = OUT_DIRS[key] / f"manifest_with_folds_seed{seed}.csv"
        df.to_csv(out_path, index=False)
        print(f"  seed {seed}: saved {out_path}")
        # Quick summary
        print(f"    images per class: {dict(df['label'].value_counts())}")
        print(f"    patients per fold: {dict(df.groupby('fold')['patient_id'].nunique())}")