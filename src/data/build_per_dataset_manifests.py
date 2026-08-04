"""
Build per-dataset manifests for Chapter 5 (per-dataset chapter).
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
OUT_ROOT = SCRATCH / "kidney-data/processed/per_dataset"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
N_FOLDS = 5


def build_folds(df, seeds):
    patient_class = df.groupby("patient_id")["label"].agg(
        lambda s: s.value_counts().index[0]
    ).to_dict()
    df = df.copy()
    df["_pat_class"] = df["patient_id"].map(patient_class)

    out = {}
    for seed in seeds:
        this = df.copy()
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        this["fold"] = -1
        for fold_idx, (_, test_idx) in enumerate(
            sgkf.split(this, this["_pat_class"], groups=this["patient_id"])
        ):
            this.loc[test_idx, "fold"] = fold_idx
        assert (this["fold"] >= 0).all()

        for i in range(N_FOLDS):
            for j in range(i + 1, N_FOLDS):
                overlap = set(this[this["fold"] == i]["patient_id"]) & set(
                    this[this["fold"] == j]["patient_id"]
                )
                assert len(overlap) == 0, f"Overlap: folds {i}/{j}, seed {seed}"

        this = this.drop(columns=["_pat_class"])
        out[seed] = this
    return out


def print_summary(name, df_by_seed):
    print(f"\n=== {name.upper()} ===")
    for seed, df in df_by_seed.items():
        print(f"\nSeed {seed}: {len(df)} images, {df['patient_id'].nunique()} patients")
        fs = df.groupby("fold").agg(n_imgs=("path", "count"), n_patients=("patient_id", "nunique"))
        print(fs.to_string())
        ct = df.groupby(["fold", "label"]).size().unstack(fill_value=0)
        print("Per-fold class distribution:")
        print(ct.to_string())


# =========================================================================
# 1. MENDELEY
# =========================================================================
print("=" * 60)
print("Building MENDELEY manifest (4-class, 217 patients)")
print("=" * 60)

MEND_ROOT = SCRATCH / "kidney-data/raw/kaggle_ct_kidney_patient/Grouped images"
rows = []
for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    cls_dir = MEND_ROOT / cls
    if not cls_dir.exists():
        print(f"Missing: {cls_dir}")
        continue
    for group_dir in sorted(cls_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        m = re.match(r"Group\s+(\d+)", group_dir.name)
        if not m:
            print(f"Unexpected: {group_dir.name}")
            continue
        group_num = int(m.group(1))
        patient_id = f"mendeley_{cls}_G{group_num:03d}"
        for img_path in sorted(group_dir.glob("*.jpg")):
            rows.append({
                "path": str(img_path),
                "label": cls,
                "patient_id": patient_id,
                "source": "mendeley",
            })

mend_df = pd.DataFrame(rows)
print(f"\nTotal Mendeley rows: {len(mend_df)}")
print(f"Patients: {mend_df['patient_id'].nunique()}")
print(f"Class counts: {mend_df['label'].value_counts().to_dict()}")

mend_by_seed = build_folds(mend_df, SEEDS)
print_summary("mendeley", mend_by_seed)

mend_out = OUT_ROOT / "mendeley"
mend_out.mkdir(parents=True, exist_ok=True)
for seed, df in mend_by_seed.items():
    df.to_csv(mend_out / f"manifest_seed{seed}.csv", index=False)
    print(f"Wrote {mend_out / f'manifest_seed{seed}.csv'}")


# =========================================================================
# 2. KITS
# =========================================================================
print("\n" + "=" * 60)
print("Building KITS manifest (3-class, 104 patients)")
print("=" * 60)

V3C = SCRATCH / "kidney-data/processed/unified_v3_corrected"
v3c_df = pd.read_csv(V3C / "manifest_with_masks.csv")
kits_df = v3c_df[v3c_df["source"] == "kits"][["path", "label", "patient_id", "source"]].reset_index(drop=True)
print(f"\nTotal KiTS rows: {len(kits_df)}")
print(f"Patients: {kits_df['patient_id'].nunique()}")
print(f"Class counts: {kits_df['label'].value_counts().to_dict()}")

kits_by_seed = build_folds(kits_df, SEEDS)
print_summary("kits", kits_by_seed)

kits_out = OUT_ROOT / "kits"
kits_out.mkdir(parents=True, exist_ok=True)
for seed, df in kits_by_seed.items():
    df.to_csv(kits_out / f"manifest_seed{seed}.csv", index=False)
    print(f"Wrote {kits_out / f'manifest_seed{seed}.csv'}")


# =========================================================================
# 3. ABDALLA
# =========================================================================
print("\n" + "=" * 60)
print("Building ABDALLA manifest (2-class, 201 patients)")
print("=" * 60)

abd_df = v3c_df[v3c_df["source"] == "abdalla"][["path", "label", "patient_id", "source"]].reset_index(drop=True)
print(f"\nTotal Abdalla rows: {len(abd_df)}")
print(f"Patients: {abd_df['patient_id'].nunique()}")
print(f"Class counts: {abd_df['label'].value_counts().to_dict()}")

abd_by_seed = build_folds(abd_df, SEEDS)
print_summary("abdalla", abd_by_seed)

abd_out = OUT_ROOT / "abdalla"
abd_out.mkdir(parents=True, exist_ok=True)
for seed, df in abd_by_seed.items():
    df.to_csv(abd_out / f"manifest_seed{seed}.csv", index=False)
    print(f"Wrote {abd_out / f'manifest_seed{seed}.csv'}")


print("\n" + "=" * 60)
print("All per-dataset manifests built.")
print(f"Root: {OUT_ROOT}")
print("=" * 60)
