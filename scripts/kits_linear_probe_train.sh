#!/bin/bash
#SBATCH --job-name=kits_lp
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH

python -u src/training/train_kits_linear_probe.py \
    --seed "$SEED" --folds-to-run all \
    --epochs 25 --lr 1e-3 --weight-decay 1e-4 \
    --batch-size 64 --aug-strength strong
