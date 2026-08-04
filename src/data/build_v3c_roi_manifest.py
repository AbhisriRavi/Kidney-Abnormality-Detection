"""
Build v3-corrected ROI manifest from existing ROI crops on disk.

Sources:
  - KiTS from unified_v2_roi/ (2,774 crops)
  - KAUH from unified_v2_roi/ (132 crops)
  - TCGA from unified_v2_roi/ (1,380 crops)
  - Abdalla from abdalla_roi/ (1,483 crops)

Excludes: 'mendeley' source in v2_roi (which was preprocessed Abdalla).
Total: ~5,769 images across 4 sources. Patient counts smaller due to
ROI retention rates (Abdalla 44%, KAUH 70%, TCGA 58%, KiTS 92%).
"""
import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRATCH = Path(os.environ["SCRATCH"])
V2_ROI_DIR = SCRATCH / "kidney-data/processed/unified_v2_roi"
ABDALLA_ROI_DIR = SCRATCH / "kidney-data/processed/abdalla_roi"
OUT_DIR = SCRATCH / "kidney-data/processed/unified_v3c_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
N_FOLDS = 5


def parse_abdalla_filename(filename):
    """
    P037_FA_M_S_I01.png -> patient_id=abdalla_P037, label=Stone (S) / Normal (NS)
    """
    m = re.match(r"P(\d+)_([A-Z]{2})_[MF]_(N?S)_I\d+\.png", filename)
    if not m:
        return None, None
    pat_num = int(m.group(1))
    label = "Stone" if m.group(3) == "S" else "Normal"
    return f"abdalla_P{pat_num:03d}", label


# ---------------------------
# 1. Load v2_roi and drop the leaky "mendeley" source
# ---------------------------
print("=" * 60)
print("Loading v2_roi and dropping leaky 'mendeley' source...")
print("=" * 60)
v2_roi = pd.read_csv(V2_ROI_DIR / "manifest_with_folds.csv")
print(f"v2_roi: {len(v2_roi)} rows, by source: {v2_roi['source'].value_counts().to_dict()}")

clean = v2_roi[v2_roi["source"].isin(["kits", "kauh", "tcga"])][["path", "label", "patient_id", "source"]].copy()
print(f"After dropping mendeley: {len(clean)} rows, by source: {clean['source'].value_counts().to_dict()}")


# ---------------------------
# 2. Build abdalla ROI manifest from filenames
# ---------------------------
print("\n" + "=" * 60)
print("Building abdalla ROI manifest from filenames...")
print("=" * 60)
abdalla_imgs = ABDALLA_ROI_DIR / "images"
if not abdalla_imgs.exists():
    raise SystemExit(f"Missing: {abdalla_imgs}")

abd_rows = []
for img_path in sorted(abdalla_imgs.rglob("*.png")):
    filename = img_path.name
    pat_id, label = parse_abdalla_filename(filename)
    if pat_id is None:
        print(f"  Skipping unparsed: {filename}")
        continue
    abd_rows.append({
        "path": str(img_path),
        "label": label,
        "patient_id": pat_id,
        "source": "abdalla",
    })
abd_df = pd.DataFrame(abd_rows)
print(f"Abdalla ROI: {len(abd_df)} imgs, {abd_df['patient_id'].nunique()} patients")
print(f"  Labels: {abd_df['label'].value_counts().to_dict()}")


# ---------------------------
# 3. Combine
# ---------------------------
combined = pd.concat([clean, abd_df], ignore_index=True)
print(f"\nTotal v3c_roi: {len(combined)} imgs, {combined['patient_id'].nunique()} patients")
print(f"By source:")
for src in ["kits", "abdalla", "kauh", "tcga"]:
    sub = combined[combined["source"] == src]
    print(f"  {src}: {len(sub)} imgs, {sub['patient_id'].nunique()} pts, {sub['label'].value_counts().to_dict()}")


# ---------------------------
# 4. Build folds per seed
# ---------------------------
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


by_seed = build_folds(combined, SEEDS)
for seed, df in by_seed.items():
    print(f"\n=== Seed {seed} ===")
    print(df.groupby("fold").agg(n_imgs=("path", "count"), n_patients=("patient_id", "nunique")).to_string())
    print(df.groupby(["fold", "label"]).size().unstack(fill_value=0).to_string())
    out_name = "manifest_with_folds.csv" if seed == 42 else f"manifest_with_folds_seed{seed}.csv"
    df.to_csv(OUT_DIR / out_name, index=False)
    print(f"Wrote: {OUT_DIR / out_name}")

# Also create manifest_with_masks aliases (needed by scripts that check that name)
os.chdir(OUT_DIR)
for seed in SEEDS:
    if seed == 42:
        src = "manifest_with_folds.csv"; dst = "manifest_with_masks.csv"
    else:
        src = f"manifest_with_folds_seed{seed}.csv"; dst = f"manifest_with_masks_seed{seed}.csv"
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)

print(f"\nDone. v3c_roi manifests in {OUT_DIR}")
