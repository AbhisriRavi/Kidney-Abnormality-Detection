#!/bin/bash
#SBATCH --job-name=v3c_roi
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH

python -u src/training/train_kfold_v3c_roi_baseline.py \
    --tag "$TAG" --seed "$SEED" --seed-suffix "$SUFFIX" \
    --folds-to-run all \
    --epochs 25 --lr 2e-4 --weight-decay 1e-5 --batch-size 64 --aug-strength strong
