"""
Build train/val/test manifest covering the 5,509 ROI-eligible axial images.
Both Approach A (full image) and B (ROI) will use this exact same split.
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SCRATCH = Path(os.environ["SCRATCH"])
ROI_DIR = SCRATCH / "kidney-data/processed/kaggle_axial_roi"
ORIGINAL_MANIFEST = SCRATCH / "kidney-data/processed/manifest.csv"
OUT = SCRATCH / "kidney-data/processed/classification_manifest.csv"

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
SEED = 42

# Original axial images — has full paths and labels
orig = pd.read_csv(ORIGINAL_MANIFEST)
axial = orig[orig.likely_view == "axial"].copy()
axial["stem"] = axial["path"].apply(lambda p: Path(p).stem)

# Build set of ROI-eligible stems
roi_stems = {f.stem for cls in CLASSES for f in (ROI_DIR / cls).glob("*.png")}
print(f"ROI-eligible image stems: {len(roi_stems)}")

# Keep only rows with ROI versions
eligible = axial[axial["stem"].isin(roi_stems)].reset_index(drop=True)
eligible["full_image_path"] = eligible["path"]
eligible["roi_path"] = eligible.apply(
    lambda r: str(ROI_DIR / r["label"] / f"{r['stem']}.png"), axis=1
)
eligible = eligible[["stem", "label", "full_image_path", "roi_path"]]
print(f"Manifest rows: {len(eligible)}")
print("Class distribution:")
print(eligible["label"].value_counts().to_string())

# Stratified 70/10/20 split
train_df, test_df = train_test_split(
    eligible, test_size=0.20, stratify=eligible["label"], random_state=SEED
)
train_df, val_df = train_test_split(
    train_df, test_size=0.125, stratify=train_df["label"], random_state=SEED
)
# Mark the split
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"
full = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print("\nClass distribution per split:")
print(full.groupby(["split", "label"]).size().unstack(fill_value=0))

full.to_csv(OUT, index=False)
print(f"\nSaved manifest to {OUT}")