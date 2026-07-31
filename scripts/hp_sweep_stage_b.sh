#!/bin/bash
# HP sweep Stage B: batch × augmentation, using Stage A winner (lr=2e-4, wd=1e-5)

cd ~/projects/Kidney-Abnormality-Detection

for BS in 32 64 128; do
    for AUG in mild moderate strong; do
        TAG="hp_sweep/stage_b/bs${BS}_aug${AUG}"
        sbatch --export=ALL,TAG="$TAG",BS="$BS",AUG="$AUG" \
               scripts/hp_sweep_train_b.sh
    done
done

echo "Submitted 9 Stage B jobs."
squeue -u $USER