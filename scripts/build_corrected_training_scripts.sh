#!/bin/bash
# Create v3c versions of existing training scripts by patching the manifest path.
set -e
cd ~/projects/Kidney-Abnormality-Detection/src/training

# 1. Focal + balanced — the existing train_kfold_v2.py handles all v2/v3 variants
#    via --manifest v3 arg. We need a v3c variant.
if [ -f train_kfold_v2.py ]; then
    cp train_kfold_v2.py train_kfold_v3c_focal.py
    # Add v3c to the manifest choices
    sed -i 's|"v2", "v2_roi", "v3", "v3_roi"|"v2", "v2_roi", "v3", "v3_roi", "v3c"|' train_kfold_v3c_focal.py
    # Add v3c mapping
    python3 << 'PY'
import re
with open("train_kfold_v3c_focal.py") as f:
    src = f.read()
# Add v3c to manifest_dir_map
new_map = '''    "v3": SCRATCH / "kidney-data/processed/unified_v3",
    "v3_roi": SCRATCH / "kidney-data/processed/unified_v3_roi",
    "v3c": SCRATCH / "kidney-data/processed/unified_v3_corrected",
}'''
old_map = '''    "v3": SCRATCH / "kidney-data/processed/unified_v3",
    "v3_roi": SCRATCH / "kidney-data/processed/unified_v3_roi",
}'''
src = src.replace(old_map, new_map)
with open("train_kfold_v3c_focal.py", "w") as f:
    f.write(src)
print("Patched train_kfold_v3c_focal.py")
PY
fi

# 2. DANN — copy train_kfold_v3_dann.py, change manifest dir
if [ -f train_kfold_v3_dann.py ]; then
    cp train_kfold_v3_dann.py train_kfold_v3c_dann.py
    sed -i 's|V3_DIR = SCRATCH / "kidney-data/processed/unified_v3"|V3_DIR = SCRATCH / "kidney-data/processed/unified_v3_corrected"|' train_kfold_v3c_dann.py
    echo "Patched train_kfold_v3c_dann.py"
fi

# 3. Multi-head — copy train_kfold_v3_multihead.py, change manifest dir
if [ -f train_kfold_v3_multihead.py ]; then
    cp train_kfold_v3_multihead.py train_kfold_v3c_multihead.py
    sed -i 's|MANIFEST_DIR = SCRATCH / "kidney-data/processed/unified_v3"|MANIFEST_DIR = SCRATCH / "kidney-data/processed/unified_v3_corrected"|' train_kfold_v3c_multihead.py
    echo "Patched train_kfold_v3c_multihead.py"
fi

# Verify the patches worked
echo ""
echo "=== Verification ==="
for f in train_kfold_v3c_focal.py train_kfold_v3c_dann.py train_kfold_v3c_multihead.py; do
    if [ -f "$f" ]; then
        echo ""
        echo "--- $f ---"
        grep -n "unified_v3\(_corrected\)\?" "$f" | head -5
    fi
done
