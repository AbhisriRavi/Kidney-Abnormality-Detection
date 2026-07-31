#!/bin/bash
#SBATCH --job-name=v3_foc
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection

python -u src/training/train_kfold_v2.py \
    --tag "$TAG" \
    --manifest v3 \
    --seed "$SEED" \
    --seed-suffix "$SUFFIX" \
    --folds-to-run all \
    --epochs 25 \
    --lr "$LR" \
    --weight-decay "$WD" \
    --batch-size "$BS" \
    --aug-strength "$AUG" \
    --use-balanced-sampler \
    --use-focal-loss \
    --focal-gamma 2.0