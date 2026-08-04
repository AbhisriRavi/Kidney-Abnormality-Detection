#!/bin/bash
#SBATCH --job-name=loso
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Leave-one-source-out external validation.
# One training run per source: ~5 runs on the 5-source v3-corrected manifest.
# This is the single most valuable remaining experiment -- it converts the
# source-shortcut claim from indirect evidence into a direct measurement.

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection

python -u -m src.training.train_loso \
    --manifest v3c \
    --arch resnet50 \
    --tag loso_rn50_v3c \
    --hold-out all \
    --epochs 25 \
    --lr 2e-4 --weight-decay 1e-5 --batch-size 64 \
    --aug-strength strong --use-balanced-sampler

# Optional: repeat on the ROI manifest to test whether region focusing
# improves cross-source transfer (a plausible and reportable secondary result).
# python -u -m src.training.train_loso --manifest v3c_roi --arch resnet50 \
#     --tag loso_rn50_v3c_roi --hold-out all
