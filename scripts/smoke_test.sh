#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/training/train_kfold_v2.py \
    --tag smoke_test --manifest v2 --seed 42 \
    --folds-to-run 0 --epochs 5 \
    --lr 1e-4 --weight-decay 1e-4 --batch-size 64 --aug-strength mild
