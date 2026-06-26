#!/bin/bash
#SBATCH --job-name=bias_vit
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

TAG=$1
MANIFEST=$2

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/evaluation/check_source_bias_vit.py --tag $TAG --manifest $MANIFEST