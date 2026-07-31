#!/bin/bash
#SBATCH --job-name=v4_focal
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH

python -u src/training/train_kfold_v4_focal.py \
    --tag "$TAG" --manifest v4 --seed "$SEED" --seed-suffix "$SUFFIX" \
    --folds-to-run all \
    --epochs 25 --lr 2e-4 --weight-decay 1e-5 --batch-size 64 --aug-strength strong \
    --use-balanced-sampler --use-focal-loss --focal-gamma 2.0
