"""
Sanity-check the KiTS external test set:
- 4 random samples per class for both full and ROI versions
- Confirm labels look anatomically plausible (kidney visible, lesions present where expected)
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(os.environ["SCRATCH"])
ROOT = SCRATCH / "kidney-data/processed/kits_external_test"
RESULTS = SCRATCH / "kidney-results"
CLASSES = ["Normal", "Cyst", "Tumor"]

fig, axes = plt.subplots(6, 4, figsize=(14, 21))
rng = np.random.RandomState(42)

for ci, cls in enumerate(CLASSES):
    full_files = sorted((ROOT / "full" / cls).glob("*.png"))
    samples = rng.choice(full_files, size=4, replace=False)
    for j, f in enumerate(samples):
        roi_file = ROOT / "roi" / cls / f.name
        with Image.open(f) as im:
            axes[ci*2, j].imshow(im, cmap="gray")
        axes[ci*2, j].set_title(f"{cls} (full)\n{f.stem[:30]}", fontsize=8)
        axes[ci*2, j].axis("off")
        with Image.open(roi_file) as im:
            axes[ci*2+1, j].imshow(im, cmap="gray")
        axes[ci*2+1, j].set_title(f"{cls} (ROI)", fontsize=8)
        axes[ci*2+1, j].axis("off")

plt.tight_layout()
out = RESULTS / "kits_test_set_samples.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved sample grid to: {out}")