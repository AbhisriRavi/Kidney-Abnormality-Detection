"""
Extract 2D kidney segmentation masks aligned with KiTS training images.

For every KiTS row in the v3 manifest:
  1. Parse patient_id → case_XXXXX
  2. Parse filename slice N
  3. Load case_XXXXX/segmentation.nii.gz
  4. Extract slice N along axis 0
  5. Threshold (mask > 0) → binary kidney mask (kidney + tumor + cyst all count as "kidney region")
  6. Resize to 224×224 with nearest-neighbour interpolation
  7. Save as PNG at parallel path

Outputs:
  - Aligned masks under $SCRATCH/kidney-data/processed/kits_seg_masks/
  - Extended manifest with a mask_path column (or None for non-KiTS rows)
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])
KITS_RAW = SCRATCH / "kidney-data/raw/kits23/kits23/dataset"
MANIFEST_IN = SCRATCH / "kidney-data/processed/unified_v3/manifest.csv"
MASKS_OUT = SCRATCH / "kidney-data/processed/kits_seg_masks"
MANIFEST_OUT = SCRATCH / "kidney-data/processed/unified_v3/manifest_with_masks.csv"

MASKS_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST_IN)
print(f"Loaded manifest: {len(df)} rows, {df['source'].nunique()} sources")

# Add mask_path column, default None
df["mask_path"] = ""

kits = df[df["source"] == "kits"]
print(f"KiTS rows: {len(kits)} ({kits['patient_id'].nunique()} patients)")

# Cache NIfTIs per case to avoid re-loading (each patient contributes ~30 slices)
nii_cache = {}
skipped = []
successes = 0

for idx, row in tqdm(kits.iterrows(), total=len(kits), desc="Extracting masks"):
    patient_id = row["patient_id"]  # e.g. "kits_case_00003"
    case_num = patient_id.replace("kits_", "")  # "case_00003"

    # Parse slice number from filename
    fname = os.path.basename(row["path"])
    m = re.search(r"slice(\d+)", fname)
    if not m:
        skipped.append((patient_id, fname, "no slice number parseable"))
        continue
    slice_idx = int(m.group(1))

    # Load segmentation if not cached
    if case_num not in nii_cache:
        seg_path = KITS_RAW / case_num / "segmentation.nii.gz"
        if not seg_path.exists():
            skipped.append((patient_id, fname, f"missing NIfTI {seg_path}"))
            continue
        nii_cache[case_num] = nib.load(seg_path).get_fdata()
        # Keep at most 20 cases cached to bound memory
        if len(nii_cache) > 20:
            oldest = next(iter(nii_cache))
            del nii_cache[oldest]

    seg_vol = nii_cache[case_num]

    # Check slice bounds
    if slice_idx >= seg_vol.shape[0]:
        skipped.append((patient_id, fname, f"slice {slice_idx} out of bounds {seg_vol.shape}"))
        continue

    # Extract slice, threshold, resize
    mask_2d = seg_vol[slice_idx, :, :]
    kidney_mask = (mask_2d > 0).astype(np.uint8) * 255  # binary 0/255

    # Resize 512x512 -> 224x224 with nearest-neighbour (preserves binary)
    mask_pil = Image.fromarray(kidney_mask, mode="L").resize((224, 224), Image.NEAREST)

    # Save
    out_path = MASKS_OUT / f"{patient_id}_slice{slice_idx:04d}.png"
    mask_pil.save(out_path)
    df.at[idx, "mask_path"] = str(out_path)
    successes += 1

# Save extended manifest
df.to_csv(MANIFEST_OUT, index=False)

print(f"\n{'=' * 60}")
print(f"Success: {successes} masks extracted")
print(f"Skipped: {len(skipped)}")
if skipped:
    print("First 5 skipped:")
    for s in skipped[:5]:
        print(f"  {s}")
print(f"Extended manifest: {MANIFEST_OUT}")
print(f"Masks directory: {MASKS_OUT}")

# Sanity check: how many KiTS rows got masks, and what proportion have non-empty masks?
kits_with_masks = df[(df["source"] == "kits") & (df["mask_path"] != "")]
print(f"\nKiTS rows with masks: {len(kits_with_masks)} / {len(kits)}")

# Sample 5 random masks and report pixel coverage
sample = kits_with_masks.sample(min(10, len(kits_with_masks)), random_state=42)
print(f"\nSample mask statistics (kidney pixel coverage):")
for _, r in sample.iterrows():
    mask = np.array(Image.open(r["mask_path"]))
    coverage = (mask > 0).mean() * 100
    print(f"  {os.path.basename(r['mask_path'])}: {coverage:.1f}% kidney pixels")