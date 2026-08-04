"""
Build the v2 unified manifest combining all 4 sources:
  - KiTS (Normal, Cyst, Tumor) + Mendeley (Normal, Stone) from v1 manifest
  - TCGA-KIRC (Tumor)
  - KAUH (Cyst)

Output: $SCRATCH/kidney-data/processed/unified_v2/manifest.csv

Schema: path, label, patient_id, source, stem
All patient_ids are already namespaced (kits_*, mendeley_*, tcga_*, kauh_*).
"""
import os
from pathlib import Path
import pandas as pd

SCRATCH = Path(os.environ["SCRATCH"])

# Inputs
V1 = SCRATCH / "kidney-data/processed/unified_kits_mendeley/manifest.csv"
TCGA = SCRATCH / "kidney-data/processed/tcga_kirc/manifest.csv"
KAUH = SCRATCH / "kidney-data/processed/kauh_cyst/manifest.csv"

# Output
OUT_ROOT = SCRATCH / "kidney-data/processed/unified_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_ROOT / "manifest.csv"

print("Loading source manifests...")
dfs = []
for name, path in [("v1 (KiTS+Mendeley)", V1), ("TCGA-KIRC", TCGA), ("KAUH cyst", KAUH)]:
    if not path.exists():
        raise SystemExit(f"Missing manifest: {path}")
    d = pd.read_csv(path)
    print(f"  {name:25s}: {len(d):5d} images, {d['patient_id'].nunique():4d} patients")
    dfs.append(d)

# Concatenate
merged = pd.concat(dfs, ignore_index=True)

# Sanity checks
print(f"\nMerged total: {len(merged)} images, {merged['patient_id'].nunique()} patients")

# Confirm no path collisions (since each source uses its own folder, this should be true)
assert merged["path"].is_unique, "Duplicate paths detected in merged manifest!"

# Confirm no patient_id collisions (they should be namespaced)
patient_sources = merged.groupby("patient_id")["source"].nunique()
multi_source_patients = patient_sources[patient_sources > 1]
if len(multi_source_patients) > 0:
    print(f"\nWARNING: {len(multi_source_patients)} patient IDs appear in multiple sources!")
    print(multi_source_patients.head())
else:
    print("✓ All patient_ids are uniquely tied to one source (no namespace collisions)")

# Save
merged.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# Summary tables
print("\n" + "=" * 60)
print("Unified v2 manifest composition")
print("=" * 60)

print("\nImages per (class × source):")
print(merged.groupby(["label", "source"]).size().unstack(fill_value=0).to_string())

print("\nPatients per (class × source):")
print(merged.groupby(["label", "source"])["patient_id"].nunique().unstack(fill_value=0).to_string())

print("\nTotal images per class:")
print(merged["label"].value_counts().to_string())

print("\nTotal patients per class:")
print(merged.groupby("label")["patient_id"].nunique().to_string())

print(f"\nGrand total: {len(merged)} images, {merged['patient_id'].nunique()} unique patients")

# Critical check: do Cyst and Tumor each span 2+ sources?
print("\n" + "=" * 60)
print("Source-coverage diagnostic (the key methodological check)")
print("=" * 60)
for cls in ["Normal", "Cyst", "Tumor", "Stone"]:
    sources = merged[merged["label"] == cls]["source"].unique()
    print(f"  {cls:8s}: {len(sources)} source(s) — {', '.join(sorted(sources))}")