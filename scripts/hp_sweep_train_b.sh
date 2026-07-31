#!/bin/bash
#SBATCH --job-name=hp_B
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection

python -u src/training/train_kfold_v2.py \
    --tag "$TAG" \
    --manifest v3 \
    --seed 42 \
    --folds-to-run 0 \
    --epochs 15 \
    --lr 2e-4 \
    --weight-decay 1e-5 \
    --batch-size "$BS" \
    --aug-strength "$AUG"