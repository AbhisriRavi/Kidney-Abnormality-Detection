#!/bin/bash
# Launch v3c_roi baseline: 3 seeds x 5 folds internal = 3 jobs, 15 fold results.
set -e
cd ~/projects/Kidney-Abnormality-Detection

for SEED in 42 123 456; do
    SUFFIX=""; if [ "$SEED" != "42" ]; then SUFFIX="_seed${SEED}"; fi
    TAG="v3c_roi_baseline_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX" scripts/v3c_roi_train.sh
done

echo "Submitted 3 v3c_roi jobs"
squeue -u $USER | grep v3c_roi
