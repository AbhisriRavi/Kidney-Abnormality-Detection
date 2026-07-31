#!/bin/bash
# Create v4 versions of the v3c training scripts.
set -e
cd ~/projects/Kidney-Abnormality-Detection/src/training

# --- 1. Baseline (from v3c baseline)
cp train_kfold_v3_baseline.py train_kfold_v4_baseline.py
sed -i 's|unified_v3_corrected|unified_v4|g' train_kfold_v4_baseline.py
echo "Patched train_kfold_v4_baseline.py"

# --- 2. Focal (from v3c focal)
cp train_kfold_v3c_focal.py train_kfold_v4_focal.py
# The v3c focal has a manifest choice mechanism - add v4 to it
python3 << 'PY'
with open("train_kfold_v4_focal.py") as f:
    src = f.read()
# Add v4 to manifest choices
src = src.replace('"v3", "v3_roi", "v3c"', '"v3", "v3_roi", "v3c", "v4"')
# Add v4 mapping
src = src.replace(
    '"v3c": SCRATCH / "kidney-data/processed/unified_v3_corrected",',
    '"v3c": SCRATCH / "kidney-data/processed/unified_v3_corrected",\n    "v4": SCRATCH / "kidney-data/processed/unified_v4",'
)
with open("train_kfold_v4_focal.py", "w") as f:
    f.write(src)
print("Patched train_kfold_v4_focal.py")
PY

# --- 3. DANN (from v3c DANN)
cp train_kfold_v3c_dann.py train_kfold_v4_dann.py
sed -i 's|unified_v3_corrected|unified_v4|g' train_kfold_v4_dann.py
echo "Patched train_kfold_v4_dann.py"

# --- 4. Multihead (from v3c multihead)
cp train_kfold_v3c_multihead.py train_kfold_v4_multihead.py
sed -i 's|unified_v3_corrected|unified_v4|g' train_kfold_v4_multihead.py
echo "Patched train_kfold_v4_multihead.py"

# Verify
echo ""
echo "=== Verifying patches ==="
for f in train_kfold_v4_*.py; do
    echo ""
    echo "--- $f ---"
    grep -n "unified_v" "$f" | head -3
done
