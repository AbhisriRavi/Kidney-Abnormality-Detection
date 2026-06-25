#!/bin/bash
#SBATCH --job-name=kfold_v2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

TAG=${1:-v2_run1}
MANIFEST=${2:-v2}

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/training/train_kfold_v2.py --tag $TAG --manifest $MANIFEST