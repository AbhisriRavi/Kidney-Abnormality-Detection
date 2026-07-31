#!/bin/bash
set -e
cd ~/projects/Kidney-Abnormality-Detection

for ARCH in resnet50 efficientnet_b0 densenet121; do
    for SEED in 42 123 456; do
        sbatch --export=ALL,ARCH="$ARCH",SEED="$SEED" scripts/pdreg_kits_train.sh
    done
done

echo ""
echo "Submitted 9 KiTS regularized jobs"
squeue -u $USER
