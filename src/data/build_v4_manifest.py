"""
Build v4 unified manifest with 5 sources:
  - KiTS 3-class (from v3_corrected)
  - Abdalla 2-class (from v3_corrected)
  - KAUH Cyst-only (from v3_corrected)
  - TCGA Tumor-only (from v3_corrected)
  - Real Mendeley 4-class, DOWNSAMPLED to ~3,000 imgs (patient-preserving)

Total v4 target: ~11,900 images, ~665 patients across 5 sources.
Generates 5 seed variants with patient-grouped StratifiedGroupKFold splits.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
V3C_DIR = SCRATCH / "kidney-data/processed/unified_v3_corrected"
MENDELEY_DIR = SCRATCH / "kidney-data/processed/per_dataset/mendeley"
OUT_DIR = SCRATCH / "kidney-data/processed/unified_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
TARGET_MENDELEY_SIZE = 3000
N_FOLDS = 5


def downsample_mendeley_by_patient(df, target_size, seed):
    rng = np.random.default_rng(seed)
    pat_class = df.groupby("patient_id")["label"].first().to_dict()
    pat_size = df.groupby("patient_id").size().to_dict()

    per_class_patients = {}
    for cls in df["label"].unique():
        pats = [p for p, c in pat_class.items() if c == cls]
        rng.shuffle(pats)
        per_class_patients[cls] = pats

    class_props = df["label"].value_counts(normalize=True).to_dict()
    class_targets = {c: int(target_size * p) for c, p in class_props.items()}

    selected = set()
    for cls, target in class_targets.items():
        running = 0
        for p in per_class_patients[cls]:
            if running >= target:
                break
            selected.add(p)
            running += pat_size[p]

    sampled = df[df["patient_id"].isin(selected)].reset_index(drop=True)
    print(f"  Downsampled Mendeley: target={target_size}, actual={len(sampled)}")
    print(f"    Patients: {sampled['patient_id'].nunique()}")
    print(f"    Class distribution: {sampled['label'].value_counts().to_dict()}")
    return sampled


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


# 1. Load v3-corrected non-mendeley sources
print("=" * 60)
print("Loading v3_corrected sources...")
print("=" * 60)
v3c = pd.read_csv(V3C_DIR / "manifest_with_masks.csv")
non_mend = v3c[v3c["source"].isin(["kits", "abdalla", "kauh", "tcga"])].copy()
print(f"v3c non-mendeley rows: {len(non_mend)}, patients: {non_mend['patient_id'].nunique()}")
for src in ["kits", "abdalla", "kauh", "tcga"]:
    sub = non_mend[non_mend["source"] == src]
    print(f"  {src}: {len(sub)} imgs, {sub['patient_id'].nunique()} pts, {sub['label'].value_counts().to_dict()}")


# 2. Load real Mendeley 4-class
print("\n" + "=" * 60)
print("Loading real Mendeley...")
print("=" * 60)
mend_full = pd.read_csv(MENDELEY_DIR / "manifest_seed42.csv")
mend_full = mend_full[["path", "label", "patient_id", "source"]].copy()
print(f"Full Mendeley: {len(mend_full)} imgs, {mend_full['patient_id'].nunique()} patients")
print(f"  Class distribution: {mend_full['label'].value_counts().to_dict()}")


# 3. Build v4 per seed (Mendeley downsampled per seed)
for seed in SEEDS:
    print(f"\n{'=' * 60}\nBuilding v4 for seed {seed}\n{'=' * 60}")
    mend_sampled = downsample_mendeley_by_patient(mend_full, TARGET_MENDELEY_SIZE, seed)
    combined = pd.concat([non_mend, mend_sampled], ignore_index=True)
    print(f"\nCombined v4 (seed {seed}): {len(combined)} imgs, {combined['patient_id'].nunique()} patients")
    for src in ["kits", "abdalla", "kauh", "tcga", "mendeley"]:
        sub = combined[combined["source"] == src]
        print(f"  {src}: {len(sub)} imgs, {sub['patient_id'].nunique()} pts")

    if "mask_path" not in combined.columns:
        combined["mask_path"] = ""
    combined["mask_path"] = combined["mask_path"].fillna("")

    df_by_seed = build_folds(combined, [seed])
    df_folded = df_by_seed[seed]

    fs = df_folded.groupby("fold").agg(n_imgs=("path", "count"), n_patients=("patient_id", "nunique"))
    print(f"Per-fold sizes:")
    print(fs.to_string())

    ct = df_folded.groupby(["fold", "label"]).size().unstack(fill_value=0)
    print(f"Per-fold class distribution:")
    print(ct.to_string())

    if seed == 42:
        out_name = "manifest_with_masks.csv"
    else:
        out_name = f"manifest_with_masks_seed{seed}.csv"
    df_folded.to_csv(OUT_DIR / out_name, index=False)
    print(f"\nWrote: {OUT_DIR / out_name}")


# 4. Symlinks for scripts that expect manifest_with_folds naming
print("\n" + "=" * 60)
print("Creating manifest_with_folds symlinks...")
print("=" * 60)
os.chdir(OUT_DIR)
for seed in SEEDS:
    if seed == 42:
        src = "manifest_with_masks.csv"
        dst = "manifest_with_folds.csv"
    else:
        src = f"manifest_with_masks_seed{seed}.csv"
        dst = f"manifest_with_folds_seed{seed}.csv"
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print(f"  {dst} -> {src}")

print(f"\nDone. v4 manifests in {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
