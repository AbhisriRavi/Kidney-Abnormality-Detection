#!/bin/bash
# ---------------------------------------------------------------------------
# Consolidate the 15 near-duplicate training scripts behind train_unified.py.
#
# LOW-RISK PROCEDURE: nothing is deleted. The old scripts move to
# src/training/legacy/ and stay runnable, so any result you have already
# reported remains reproducible from the exact code that produced it.
#
# Run the verification step below BEFORE committing: it retrains one fold with
# the new entry point and diffs the metrics against the old run.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p src/training/legacy
cat > src/training/legacy/README.md <<'MD'
# Legacy training scripts

Superseded by `src/training/train_unified.py`. Retained verbatim so that every
result reported in the dissertation remains reproducible from the exact code
that produced it.

Mapping to the unified entry point:

| Legacy script | Equivalent invocation |
|---|---|
| `train_kfold_v2.py` | `--arch resnet50 --manifest v2 --loss ce` |
| `train_kfold_v3_baseline.py` | `--arch resnet50 --manifest v3 --loss ce` |
| `train_kfold_v3c_focal.py` | `--arch resnet50 --manifest v3c --loss focal --balanced-sampler` |
| `train_kfold_v4_focal.py` | `--arch resnet50 --manifest v4 --loss focal --balanced-sampler` |
| `train_kfold_cbam.py` | `--arch cbam --manifest v2` |
| `train_kfold_vit.py` | `--arch vit --manifest v2 --batch-size 32 --lr 5e-5` |
| `train_kfold_v3c_multihead.py` | `--arch multihead --manifest v3c --cyst-source {kits,kauh,both}` |
| `train_kfold_v3c_dann.py` | `--arch dann --manifest v3c --dann-lambda 0.1` |

Measured duplication before consolidation: `train_kfold_v3c_focal.py` and
`train_kfold_v4_focal.py` differed by 3 lines out of 387; the three
`*_multihead.py` variants were 543 lines each and near-identical.
MD

for f in train_kfold.py train_kfold_v2.py train_kfold_cbam.py train_kfold_vit.py \
         train_kfold_v3_baseline.py train_kfold_v3_dann.py train_kfold_v3_multihead.py \
         train_kfold_v3c_dann.py train_kfold_v3c_focal.py train_kfold_v3c_multihead.py \
         train_kfold_v3c_roi_baseline.py train_kfold_v4_baseline.py \
         train_kfold_v4_dann.py train_kfold_v4_focal.py train_kfold_v4_multihead.py ; do
    [ -e "src/training/$f" ] || continue
    git mv "src/training/$f" "src/training/legacy/$f" 2>/dev/null || mv "src/training/$f" "src/training/legacy/$f"
done

git add src/training/legacy/README.md 2>/dev/null || true
echo
echo "Moved. Lines now in legacy/: $(cat src/training/legacy/*.py | wc -l)"
echo
echo "VERIFY before committing -- retrain fold 0 and compare:"
echo "  python -m src.training.train_unified --arch resnet50 --manifest v3c \\"
echo "      --loss focal --balanced-sampler --folds-to-run 0 --tag verify_unified"
echo "  # then diff test_macro_f1 against your existing v3c focal run's fold 0"
