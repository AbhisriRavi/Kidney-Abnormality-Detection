#!/bin/bash
#SBATCH --job-name=smoke_mh
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH
python -u src/training/train_kfold_v3_multihead.py \
    --tag smoke_mh_kauh --seed 42 --folds-to-run 0 --epochs 25 \
    --lr 2e-4 --weight-decay 1e-5 --batch-size 64 --aug-strength strong \
    --cyst-source kauh --lambda-seg 0.5
