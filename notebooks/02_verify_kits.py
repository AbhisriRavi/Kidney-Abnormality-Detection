import os
import nibabel as nib
import numpy as np
from pathlib import Path

DATASET = Path(os.environ["SCRATCH"]) / "kidney-data/raw/kits23/kits23/dataset"
cases = sorted(DATASET.glob("case_*"))
print(f"Found {len(cases)} case folders\n")

# Spot-check the first, middle, and last cases
indices = [0, len(cases) // 2, len(cases) - 1]
for i in indices:
    case = cases[i]
    print(f"=== {case.name} ===")
    try:
        img = nib.load(case / "imaging.nii.gz")
        seg = nib.load(case / "segmentation.nii.gz")
        img_data = img.get_fdata()
        seg_data = seg.get_fdata()

        print(f"  Imaging: shape={img.shape}, voxel size={img.header.get_zooms()}")
        print(f"  HU range: {img_data.min():.0f} to {img_data.max():.0f}")
        print(f"  Segmentation: shape={seg.shape}, unique labels={np.unique(seg_data).astype(int).tolist()}")
        print(f"  Voxels per label: " + 
              ", ".join(f"{int(l)}={int((seg_data == l).sum())}" for l in np.unique(seg_data)))
    except Exception as e:
        print(f"  ERROR: {e}")
    print()