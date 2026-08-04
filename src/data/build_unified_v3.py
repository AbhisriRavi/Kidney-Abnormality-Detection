"""
Build unified_v3 by adding Abdalla Original (3,364 images, 201 patients) to unified_v2.

Output:
  $SCRATCH/kidney-data/processed/unified_v3/
    manifest.csv
    manifest_with_folds.csv
    manifest_with_folds_seed123.csv
    manifest_with_folds_seed456.csv
    abdalla_augmented_heldout.csv   (35,457 augmented images, separate)

Filename parsing for Abdalla:
  P037_FA_M_S_I01.jpg     (original)
  P037_FA_M_S_I01_CR.jpg  (augmented; suffix = augmentation type)

Patient grouping:
  patient_id = "abdalla_P037"  (prefixed to avoid collisions across sources)
  source     = "abdalla"
  abdalla_site = FA/ME/RA       (extra column for site-level analyses)
"""
import os
import re
import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])

V2_DIR = SCRATCH / "kidney-data/processed/unified_v2"
V3_DIR = SCRATCH / "kidney-data/processed/unified_v3"
V3_DIR.mkdir(parents=True, exist_ok=True)

ABDALLA_ROOT = (SCRATCH / "kidney-data/raw/abdalla2025" /
                "Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection "
                "A Resource for Deep Learning Research" / "Kindey Stone Dataset")

ABDALLA_ORIG_STONE   = ABDALLA_ROOT / "Original"  / "Stone"
ABDALLA_ORIG_NONSTONE= ABDALLA_ROOT / "Original"  / "Non-Stone"
ABDALLA_AUG_STONE    = ABDALLA_ROOT / "Augmented" / "Stone"
ABDALLA_AUG_NONSTONE = ABDALLA_ROOT / "Augmented" / "Non-Stone"

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 1011]

# ----------------------------------------------------------------------
# Parse Abdalla filenames
# ----------------------------------------------------------------------
FNAME_RE_ORIG = re.compile(r"^(P\d+)_([A-Z]+)_([MF])_(NS|S)_(I\d+)\.jpg$")
FNAME_RE_AUG  = re.compile(r"^(P\d+)_([A-Z]+)_([MF])_(NS|S)_(I\d+)_([A-Z0-9]+)\.jpg$")

CLASS_FROM_CODE = {"S": "Stone", "NS": "Normal"}


def parse_filename(fname, augmented=False):
    """Return dict of parsed fields, or None if it doesn't match."""
    rex = FNAME_RE_AUG if augmented else FNAME_RE_ORIG
    m = rex.match(fname)
    if not m:
        return None
    pid, site, sex, cls_code, slice_idx = m.groups()[:5]
    aug_type = m.group(6) if augmented else None
    return {
        "patient_short": pid,
        "site": site,
        "sex": sex,
        "class_code": cls_code,
        "slice_idx": slice_idx,
        "aug_type": aug_type,
    }


# ----------------------------------------------------------------------
# Build Abdalla rows
# ----------------------------------------------------------------------
def build_rows(folder, augmented):
    rows = []
    skipped = 0
    for f in sorted(folder.iterdir()):
        if not f.name.endswith(".jpg"):
            continue
        parsed = parse_filename(f.name, augmented=augmented)
        if not parsed:
            skipped += 1
            continue
        row = {
            "path": str(f.resolve()),
            "label": CLASS_FROM_CODE[parsed["class_code"]],
            "source": "abdalla",
            "patient_id": f"abdalla_{parsed['patient_short']}",
            "stem": f.stem,
            "abdalla_site": parsed["site"],
            "abdalla_sex": parsed["sex"],
            "abdalla_aug_type": parsed["aug_type"] if augmented else "",
        }
        rows.append(row)
    return rows, skipped


print("Scanning Abdalla Original/Stone ...")
orig_s, sk1 = build_rows(ABDALLA_ORIG_STONE, augmented=False)
print(f"  {len(orig_s)} stone originals  ({sk1} skipped)")

print("Scanning Abdalla Original/Non-Stone ...")
orig_ns, sk2 = build_rows(ABDALLA_ORIG_NONSTONE, augmented=False)
print(f"  {len(orig_ns)} normal originals ({sk2} skipped)")

abdalla_orig_df = pd.DataFrame(orig_s + orig_ns)

print("Scanning Abdalla Augmented/Stone ...")
aug_s, sk3 = build_rows(ABDALLA_AUG_STONE, augmented=True)
print(f"  {len(aug_s)} stone augmented  ({sk3} skipped)")

print("Scanning Abdalla Augmented/Non-Stone ...")
aug_ns, sk4 = build_rows(ABDALLA_AUG_NONSTONE, augmented=True)
print(f"  {len(aug_ns)} normal augmented ({sk4} skipped)")

abdalla_aug_df = pd.DataFrame(aug_s + aug_ns)

print(f"\nAbdalla totals: {len(abdalla_orig_df)} originals, "
      f"{len(abdalla_aug_df)} augmented")
print(f"Abdalla unique patients (originals): "
      f"{abdalla_orig_df['patient_id'].nunique()}")
print(f"Abdalla per-class (originals): "
      f"{dict(abdalla_orig_df['label'].value_counts())}")
print(f"Abdalla per-site (originals): "
      f"{dict(abdalla_orig_df['abdalla_site'].value_counts())}")


# ----------------------------------------------------------------------
# Merge with v2 manifest
# ----------------------------------------------------------------------
v2_path = V2_DIR / "manifest.csv"
print(f"\nLoading existing v2 manifest: {v2_path}")
v2_df = pd.read_csv(v2_path)
print(f"  {len(v2_df)} images, {v2_df['patient_id'].nunique()} patients, "
      f"sources={sorted(v2_df['source'].unique())}")

# Ensure schema compatibility — pad extra Abdalla-specific columns into v2 as empty
v2_df["abdalla_site"]     = ""
v2_df["abdalla_sex"]      = ""
v2_df["abdalla_aug_type"] = ""

v3_df = pd.concat([v2_df, abdalla_orig_df], ignore_index=True, sort=False)
v3_path = V3_DIR / "manifest.csv"
v3_df.to_csv(v3_path, index=False)

print(f"\nv3 manifest written: {v3_path}")
print(f"  {len(v3_df)} images, "
      f"{v3_df['patient_id'].nunique()} patients, "
      f"sources={sorted(v3_df['source'].unique())}")
print(f"  per-class: {dict(v3_df['label'].value_counts())}")
print(f"  per-source: {dict(v3_df['source'].value_counts())}")

# Quick sanity: classes per source
print("\nClasses present per source:")
for src in sorted(v3_df["source"].unique()):
    classes_in_src = sorted(v3_df.loc[v3_df["source"] == src, "label"].unique())
    n = (v3_df["source"] == src).sum()
    print(f"  {src:10s} ({n:5d} imgs): {classes_in_src}")


# ----------------------------------------------------------------------
# Generate 5-fold splits for each seed
# ----------------------------------------------------------------------
print(f"\nGenerating {N_FOLDS}-fold splits for seeds {SEEDS} ...")
for seed in SEEDS:
    df = v3_df.copy()
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                     random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(splitter.split(
        X=df.index.values, y=df["label"].values,
        groups=df["patient_id"].values
    )):
        df.loc[df.index[test_idx], "fold"] = fold_idx
    assert (df["fold"] >= 0).all()

    # Verify zero patient leakage
    per_patient_folds = df.groupby("patient_id")["fold"].nunique()
    assert (per_patient_folds == 1).all(), "Patient leakage!"

    suffix = "" if seed == 42 else f"_seed{seed}"
    out = V3_DIR / f"manifest_with_folds{suffix}.csv"
    df.to_csv(out, index=False)

    print(f"\n  seed {seed}: saved {out}")
    print(f"    patients per fold: "
          f"{dict(df.groupby('fold')['patient_id'].nunique())}")
    print(f"    images per fold:   "
          f"{dict(df.groupby('fold').size())}")

# ----------------------------------------------------------------------
# Save Abdalla augmented as a separate hold-out test manifest
# ----------------------------------------------------------------------
aug_out = V3_DIR / "abdalla_augmented_heldout.csv"
abdalla_aug_df.to_csv(aug_out, index=False)
print(f"\nHeld-out augmented manifest: {aug_out}")
print(f"  {len(abdalla_aug_df)} augmented images")
print(f"  per-class: {dict(abdalla_aug_df['label'].value_counts())}")
print(f"  per-augtype: "
      f"{dict(abdalla_aug_df['abdalla_aug_type'].value_counts().head(15))}")

print("\nDone.")
