#!/bin/bash
#SBATCH --job-name=extract_masks
#SBATCH --time=00:45:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/data/extract_kits_seg_masks.py