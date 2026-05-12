# notebooks/01_kaggle_eda.py
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

DATA_ROOT = Path("/mnt/scratch") / os.environ["USER"] / "kidney-data" / "raw" / "kaggle_ct_kidney"
# The unzipped folder may be nested — adjust:
candidates = list(DATA_ROOT.rglob("Normal"))
assert candidates, f"Couldn't find Normal/ under {DATA_ROOT}"
DATA_ROOT = candidates[0].parent
print("Found dataset at:", DATA_ROOT)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

records = []
for cls in CLASSES:
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in (DATA_ROOT / cls).glob(ext):
            records.append({"path": str(img_path), "label": cls})
df = pd.DataFrame(records)

print(f"\nTotal images: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())

# Sample dimensions
sample_dims = []
for p in df["path"].sample(min(200, len(df)), random_state=0):
    with Image.open(p) as im:
        sample_dims.append(im.size)
unique_dims = pd.Series(sample_dims).value_counts().head(10)
print("\nMost common (W,H) dimensions in 200-image sample:")
print(unique_dims)

# Visualise
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, cls in enumerate(CLASSES):
    samples = df[df.label == cls].sample(4, random_state=0)
    for j, p in enumerate(samples["path"]):
        with Image.open(p) as im:
            axes[i, j].imshow(im, cmap="gray")
        axes[i, j].set_title(cls)
        axes[i, j].axis("off")
plt.tight_layout()
out = Path("/mnt/scratch") / os.environ["USER"] / "kidney-results" / "eda_samples.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"\nSaved sample grid to {out}")

# Save manifest
manifest_path = Path("/mnt/scratch") / os.environ["USER"] / "kidney-data" / "processed" / "manifest.csv"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(manifest_path, index=False)
print(f"Saved manifest to {manifest_path}")