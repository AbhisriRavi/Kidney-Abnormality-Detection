#!/bin/bash
cd ~/projects/Kidney-Abnormality-Detection

for SEED in 42 123 456; do
    SUFFIX=""
    if [ "$SEED" != "42" ]; then
        SUFFIX="_seed${SEED}"
    fi
    TAG="v3c_rn50_baseline_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/v3c_baseline_train.sh
done

echo "Submitted 3 corrected v3 baseline jobs (each runs 5 folds)."
squeue -u $USER