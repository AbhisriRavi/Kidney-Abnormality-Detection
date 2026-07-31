#!/bin/bash
# Final v3 DANN runs: 3 seeds × 5 folds at lambda=1.0

cd ~/projects/Kidney-Abnormality-Detection

LAMBDA=1.0
for SEED in 42 123 456; do
    SUFFIX=""
    if [ "$SEED" != "42" ]; then
        SUFFIX="_seed${SEED}"
    fi
    TAG="v3_dann_final_lam${LAMBDA}_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX",LAMBDA="$LAMBDA" \
           scripts/final_v3_dann_train.sh
done

echo "Submitted 3 final v3 DANN jobs (each runs 5 folds)."
squeue -u $USER