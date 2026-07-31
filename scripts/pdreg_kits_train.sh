#!/bin/bash
#SBATCH --job-name=pdreg_kits
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH

python -u src/training/train_per_dataset_kits_reg.py \
    --arch "$ARCH" --seed "$SEED" \
    --folds-to-run all \
    --epochs 15 --lr 1e-4 --weight-decay 1e-4 \
    --dropout 0.3 --patience 5 \
    --batch-size 64 --aug-strength strong
