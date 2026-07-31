#!/bin/bash
# Final RN50 v3-full runs: 3 seeds × 5 folds using HP-tuned config

cd ~/projects/Kidney-Abnormality-Detection

LR=2e-4
WD=1e-5
BS=64
AUG=strong

for SEED in 42 123 456; do
    SUFFIX=""
    if [ "$SEED" != "42" ]; then
        SUFFIX="_seed${SEED}"
    fi
    TAG="v3_rn50_full_run1_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX",LR="$LR",WD="$WD",BS="$BS",AUG="$AUG" \
           scripts/final_v3_rn50_train.sh
done

echo "Submitted 3 final v3 RN50 jobs."
squeue -u $USER