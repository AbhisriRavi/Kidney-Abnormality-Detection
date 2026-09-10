#!/bin/bash
#SBATCH --job-name=v3c_arch
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH

python -u "$TRAINER" \
    --tag "$TAG" \
    --manifest "$MANIFEST" \
    --seed 42 \
    --epochs 25 \
    --lr "$LR" --weight-decay "$WD" --batch-size "$BS"
