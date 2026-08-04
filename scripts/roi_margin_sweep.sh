#!/bin/bash
#SBATCH --job-name=roi_sweep
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Train one classifier per ROI dilation margin, then plot Stone F1 and Cyst F1
# against margin. A crossing point confirms the anatomical explanation for the
# Stone/ROI trade-off.

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection

ROOT=$SCRATCH/kidney-data/processed/roi_dilated
for M in 000 015 025 040; do
    echo "=== ROI margin $M ==="
    python -u -m src.training.train_unified \
        --arch resnet50 \
        --manifest-path $ROOT/margin_m$M/manifest_with_folds.csv \
        --tag roi_margin_$M \
        --loss ce --balanced-sampler \
        --epochs 25 --lr 2e-4 --weight-decay 1e-5 \
        --batch-size 64 --aug-strength strong
done

python -u -m src.evaluation.roi_margin_curve \
    --tags roi_margin_000 roi_margin_015 roi_margin_025 roi_margin_040 \
    --margins 0.00 0.15 0.25 0.40
