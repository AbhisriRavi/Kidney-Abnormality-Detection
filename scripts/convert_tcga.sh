#!/bin/bash
#SBATCH --job-name=tcga_conv
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/data/tcga/convert_to_png.py