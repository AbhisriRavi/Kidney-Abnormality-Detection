#!/bin/bash
#SBATCH --job-name=train_unet
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney

cd ~/projects/Kidney-Abnormality-Detection
python -u src/training/train_segmenter.py