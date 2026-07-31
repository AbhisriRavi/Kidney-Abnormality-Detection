#!/bin/bash
# Additional seeds 789 and 1011 — only submit if compute is available

cd ~/projects/Kidney-Abnormality-Detection

LR=2e-4
WD=1e-5
BS=64
AUG=strong

for SEED in 789 1011; do
    SUFFIX="_seed${SEED}"
    TAG="v3_rn50_full_run1_seed${SEED}"
    sbatch --export=ALL,TAG="$TAG",SEED="$SEED",SUFFIX="$SUFFIX",LR="$LR",WD="$WD",BS="$BS",AUG="$AUG" \
           scripts/final_v3_rn50_train.sh
done

echo "Submitted 2 additional seeds (789, 1011)."
squeue -u $USER