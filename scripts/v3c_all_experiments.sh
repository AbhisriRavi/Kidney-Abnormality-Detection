#!/bin/bash
set -e
cd ~/projects/Kidney-Abnormality-Detection

echo "==== Submitting corrected v3 experiments ===="

for SEED in 42 123 456; do
    SUFFIX=""
    if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
    TAG="v3c_focal_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" scripts/v3c_focal_train.sh
done

for SEED in 42 123 456; do
    SUFFIX=""
    if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
    TAG="v3c_dann_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" scripts/v3c_dann_train.sh
done

for CYST_SRC in kauh kits both; do
    for SEED in 42 123 456; do
        SUFFIX=""
        if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
        TAG="v3c_mh_${CYST_SRC}_seed${SEED}"
        sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX",CYST_SRC="$CYST_SRC" scripts/v3c_multihead_train.sh
    done
done

echo ""
echo "==== Submitted 15 jobs (3 focal + 3 DANN + 9 multi-head) ===="
squeue -u $USER
