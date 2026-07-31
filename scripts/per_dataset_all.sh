#!/bin/bash
# Launch all 27 per-dataset jobs: 3 datasets x 3 architectures x 3 seeds
set -e
cd ~/projects/Kidney-Abnormality-Detection

for DATASET in mendeley kits abdalla; do
    for ARCH in resnet50 efficientnet_b0 densenet121; do
        for SEED in 42 123 456; do
            sbatch --export=ALL,DATASET="$DATASET",ARCH="$ARCH",SEED="$SEED" \
                   scripts/per_dataset_train.sh
        done
    done
done

echo ""
echo "==== Submitted 27 per-dataset jobs ===="
squeue -u $USER | head -30
