#!/bin/bash
#SBATCH --job-name=kauh_cyst
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/data/build_kauh_cyst_manifest.py