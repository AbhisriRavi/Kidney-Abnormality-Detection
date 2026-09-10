#!/bin/bash
#SBATCH --job-name=extract_abdalla
#SBATCH --time=00:30:00
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney

ROOT="${SCRATCH:-/mnt/scratch/$USER}/kidney-data/raw/abdalla2025"
INNER="$ROOT/Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research"

cd "$INNER"
echo "Working dir: $(pwd)"
echo "Starting unrar at $(date)"
unrar x -y "Kindey Stone Dataset.rar"
echo "Finished at $(date)"
ls -lh
