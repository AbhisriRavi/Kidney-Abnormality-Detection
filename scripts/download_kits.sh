#!/bin/bash
#SBATCH --job-name=kits_download
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

module load miniforge/24.7.1
conda activate kidney
cd /mnt/scratch/users/$USER/kidney-data/raw/kits23/kits23
kits23_download_data