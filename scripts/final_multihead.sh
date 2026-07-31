#!/bin/bash
# Full 45-run matrix: 3 Cyst variants × 3 seeds × 5 folds
cd ~/projects/Kidney-Abnormality-Detection

for CYST_SRC in kauh kits both; do
    for SEED in 42 123 456; do
        SUFFIX=""
        if [ "$SEED" != "42" ]; then
            SUFFIX="_seed${SEED}"
        fi
        TAG="v3_mh_${CYST_SRC}_seed${SEED}"
        sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX",CYST_SRC="$CYST_SRC" \
               scripts/final_multihead_train.sh
    done
done

echo "Submitted 9 jobs (3 Cyst variants x 3 seeds), each runs 5 folds."
squeue -u $USER
