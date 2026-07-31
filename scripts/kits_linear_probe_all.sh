#!/bin/bash
set -e
cd ~/projects/Kidney-Abnormality-Detection

for SEED in 42 123 456; do
    sbatch --export=ALL,SEED="$SEED" scripts/kits_linear_probe_train.sh
done

echo ""
echo "Submitted 3 KiTS linear probe jobs (RN50, all folds)"
squeue -u $USER
