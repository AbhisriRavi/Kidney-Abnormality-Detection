import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(os.environ["SCRATCH"])
ROI_DIR = SCRATCH / "kidney-data/processed/kaggle_axial_roi"
MANIFEST = SCRATCH / "kidney-data/processed/manifest.csv"
OUT = SCRATCH / "kidney-results"

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

# Count saved ROI files per class
print("ROI dataset class distribution:")
roi_counts = {}
for cls in CLASSES:
    n = len(list((ROI_DIR / cls).glob("*.png")))
    roi_counts[cls] = n
    print(f"  {cls}: {n}")

# Compare to original axial counts
df = pd.read_csv(MANIFEST)
axial = df[df.likely_view == "axial"]
print("\nOriginal axial class distribution:")
orig_counts = axial["label"].value_counts().to_dict()
for cls in CLASSES:
    print(f"  {cls}: {orig_counts.get(cls, 0)}")

print("\nRetention rate by class:")
print(f"{'Class':<10} {'Orig':>8} {'ROI':>8} {'Retained':>10}")
for cls in CLASSES:
    orig = orig_counts.get(cls, 0)
    roi = roi_counts[cls]
    pct = (roi / orig * 100) if orig else 0
    print(f"{cls:<10} {orig:>8} {roi:>8} {pct:>9.1f}%")

# Visual spot-check: 4 random ROI crops per class
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, cls in enumerate(CLASSES):
    files = sorted((ROI_DIR / cls).glob("*.png"))
    rng = np.random.RandomState(42)
    samples = rng.choice(files, size=min(4, len(files)), replace=False)
    for j, f in enumerate(samples):
        with Image.open(f) as im:
            axes[i, j].imshow(im, cmap="gray")
        axes[i, j].set_title(f"{cls}\n{f.stem[:20]}", fontsize=8)
        axes[i, j].axis("off")
plt.tight_layout()
out_path = OUT / "roi_dataset_samples.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print(f"\nSaved spot-check grid to: {out_path}")