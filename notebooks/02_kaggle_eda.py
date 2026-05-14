"""
EDA for the Kaggle CT Kidney Dataset (Normal/Cyst/Tumor/Stone).

Outputs:
  - data/processed/manifest.csv          : path, label, dimensions, aspect ratio, likely view
  - results-scratch/eda_samples.png      : 4x8 grid of sample images (8 per class)
  - results-scratch/aspect_histogram.png : aspect ratio distribution per class
  - Console summary: class counts, dimensions, orientation breakdown
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless backend, safe on login/compute nodes
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRATCH = Path(os.environ["SCRATCH"])
DATA_RAW = SCRATCH / "kidney-data/raw/kaggle_ct_kidney"
DATA_PROCESSED = SCRATCH / "kidney-data/processed"
RESULTS = SCRATCH / "kidney-results"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Locate the class-folder parent (handles potential nesting)
candidates = list(DATA_RAW.rglob("Normal"))
assert candidates, f"Couldn't find a Normal/ folder under {DATA_RAW}"
DATASET_ROOT = candidates[0].parent
print(f"Dataset root: {DATASET_ROOT}\n")

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

# ---------------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------------
print("Scanning files...")
records = []
for cls in CLASSES:
    cls_dir = DATASET_ROOT / cls
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img_path in cls_dir.glob(ext):
            records.append({"path": str(img_path), "label": cls})

df = pd.DataFrame(records)
assert len(df) > 0, "No images found — check the dataset path."

# Extract dimensions and aspect ratio for every image (fast: only metadata)
print(f"Reading dimensions for {len(df)} images...")
widths, heights = [], []
for p in df["path"]:
    with Image.open(p) as im:
        widths.append(im.size[0])
        heights.append(im.size[1])
df["width"] = widths
df["height"] = heights
df["aspect"] = df["width"] / df["height"]

# Classify view by aspect ratio (rough heuristic).
# Axial CT slices are typically near-square because the body cross-section is roughly circular.
# Coronal/sagittal slices tend to be taller than wide (whole torso height).
def classify_view(row):
    if row["width"] == 512 and row["height"] == 512:
        return "axial"
    else:
        return "coronal"

df["likely_view"] = df.apply(classify_view, axis=1)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"Total images: {len(df)}")
print("=" * 60)

print("\nClass distribution:")
print(df["label"].value_counts().to_string())

print("\nImage dimensions (across all images):")
print(f"  Width  : min={df['width'].min()}, max={df['width'].max()}, "
      f"median={int(df['width'].median())}")
print(f"  Height : min={df['height'].min()}, max={df['height'].max()}, "
      f"median={int(df['height'].median())}")

print("\nMost common (width, height) pairs:")
pair_counts = df.groupby(["width", "height"]).size().sort_values(ascending=False).head(10)
print(pair_counts.to_string())

print("\nLikely view orientation by class (heuristic from aspect ratio):")
view_breakdown = df.groupby(["label", "likely_view"]).size().unstack(fill_value=0)
# Ensure consistent column order
for col in ["axial", "coronal"]:
    if col not in view_breakdown.columns:
        view_breakdown[col] = 0
view_breakdown = view_breakdown[["axial", "coronal"]]
print(view_breakdown.to_string())

# Percentages
print("\nLikely view orientation by class (percentages):")
view_pct = view_breakdown.div(view_breakdown.sum(axis=1), axis=0) * 100
print(view_pct.round(1).to_string())

# ---------------------------------------------------------------------------
# Save manifest
# ---------------------------------------------------------------------------
manifest_path = DATA_PROCESSED / "manifest.csv"
df.to_csv(manifest_path, index=False)
print(f"\nSaved manifest with {len(df)} rows to: {manifest_path}")

# ---------------------------------------------------------------------------
# Visualisation 1: sample grid (8 per class)
# ---------------------------------------------------------------------------
SAMPLES_PER_CLASS = 8
print(f"\nBuilding sample grid ({SAMPLES_PER_CLASS} images per class)...")

fig, axes = plt.subplots(len(CLASSES), SAMPLES_PER_CLASS,
                          figsize=(SAMPLES_PER_CLASS * 3, len(CLASSES) * 3))
for i, cls in enumerate(CLASSES):
    samples = df[df.label == cls].sample(SAMPLES_PER_CLASS, random_state=0)
    for j, (_, row) in enumerate(samples.iterrows()):
        ax = axes[i, j]
        with Image.open(row["path"]) as im:
            ax.imshow(im, cmap="gray")
        ax.set_title(f"{cls}\n{row['width']}x{row['height']} ({row['likely_view']})",
                     fontsize=9)
        ax.axis("off")

plt.tight_layout()
grid_path = RESULTS / "eda_samples.png"
plt.savefig(grid_path, dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved sample grid to: {grid_path}")

# ---------------------------------------------------------------------------
# Visualisation 2: aspect ratio histogram
# ---------------------------------------------------------------------------
print("Building aspect ratio histogram...")
fig, axes = plt.subplots(1, len(CLASSES), figsize=(len(CLASSES) * 4, 4), sharey=True)
for ax, cls in zip(axes, CLASSES):
    subset = df[df.label == cls]["aspect"]
    ax.hist(subset, bins=40, edgecolor="black", alpha=0.75)
    ax.axvspan(0.85, 1.15, color="green", alpha=0.15, label="axial range")
    ax.set_title(f"{cls} (n={len(subset)})")
    ax.set_xlabel("Aspect ratio (W/H)")
    ax.set_xlim(0, 2.0)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Number of images")
plt.tight_layout()
hist_path = RESULTS / "aspect_histogram.png"
plt.savefig(hist_path, dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved aspect histogram to: {hist_path}")

print("\nEDA complete.")