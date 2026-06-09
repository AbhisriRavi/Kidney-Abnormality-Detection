import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SLICES = Path(os.environ["SCRATCH"]) / "kidney-data/processed/kits_axial_slices"
RESULTS = Path(os.environ["SCRATCH"]) / "kidney-results"

# Counts
img_files = sorted(SLICES.glob("*_img.npy"))
mask_files = sorted(SLICES.glob("*_mask.npy"))
print(f"Image files: {len(img_files)}")
print(f"Mask files:  {len(mask_files)}")
assert len(img_files) == len(mask_files), "Image/mask count mismatch"

# Disk usage
total_mb = sum(f.stat().st_size for f in img_files + mask_files) / 1e6
print(f"Total disk: {total_mb:.0f} MB")

# Unique cases
cases = set(f.name.split("_slice")[0] for f in img_files)
print(f"Cases represented: {len(cases)}")

# Visual spot-check: 8 random pairs
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
sample_indices = np.random.RandomState(0).choice(len(img_files), 8, replace=False)
for i, idx in enumerate(sample_indices):
    img = np.load(img_files[idx])
    mask = np.load(mask_files[idx])
    r = i // 2
    c = (i % 2) * 2
    axes[r, c].imshow(img, cmap="gray")
    axes[r, c].set_title(f"Image\n{img_files[idx].name[:25]}", fontsize=8)
    axes[r, c].axis("off")
    axes[r, c+1].imshow(img, cmap="gray")
    axes[r, c+1].imshow(mask, cmap="Reds", alpha=0.4)
    axes[r, c+1].set_title(f"Mask overlay\nkidney pixels: {mask.sum()}", fontsize=8)
    axes[r, c+1].axis("off")
plt.tight_layout()
out = RESULTS / "kits_preprocessing_check.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved overlay check to: {out}")