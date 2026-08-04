#!/bin/bash
#SBATCH --job-name=roi_dilate
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection

python -u -m src.data.build_roi_dilated \
    --in-manifest $SCRATCH/kidney-data/processed/unified_v3_corrected/manifest_with_folds.csv \
    --out-root    $SCRATCH/kidney-data/processed/roi_dilated \
    --segmenter   $SCRATCH/kidney-results/segmenter/best.pth \
    --margin 0.00 0.15 0.25 0.40
