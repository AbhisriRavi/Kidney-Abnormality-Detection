#!/bin/bash
# Pilot: try 3 lambdas on fold 0 seed 42 to pick the best

cd ~/projects/Kidney-Abnormality-Detection

for LAMBDA in 0.01 0.1 1.0; do
    TAG="v3_dann_pilot_lam${LAMBDA}"
    sbatch --export=ALL,TAG="$TAG",LAMBDA="$LAMBDA" \
           scripts/dann_train.sh
done

echo "Submitted 3 pilot jobs."
squeue -u $USER  