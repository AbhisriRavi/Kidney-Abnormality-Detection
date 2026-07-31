#!/bin/bash
# Launch all v4 experiments: 6 configs x 3 seeds = 18 jobs, each runs 5 folds.
set -e
cd ~/projects/Kidney-Abnormality-Detection

echo "==== Submitting v4 experiments ===="

# Baseline
for SEED in 42 123 456; do
    SUFFIX=""; if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
    TAG="v4_rn50_baseline_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" scripts/v4_baseline_train.sh
done

# Focal
for SEED in 42 123 456; do
    SUFFIX=""; if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
    TAG="v4_focal_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" scripts/v4_focal_train.sh
done

# DANN
for SEED in 42 123 456; do
    SUFFIX=""; if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
    TAG="v4_dann_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" scripts/v4_dann_train.sh
done

# Multihead (3 Cyst variants)
for CYST_SRC in kauh kits both; do
    for SEED in 42 123 456; do
        SUFFIX=""; if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
        TAG="v4_mh_${CYST_SRC}_seed${SEED}"
        sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX",CYST_SRC="$CYST_SRC" scripts/v4_multihead_train.sh
    done
done

echo ""
echo "==== Submitted 18 v4 jobs (3 baseline + 3 focal + 3 DANN + 9 multi-head) ===="
echo "TTA runs after focal completes - will submit separately."
squeue -u $USER | head -30
