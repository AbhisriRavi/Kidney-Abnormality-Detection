#!/bin/bash
#SBATCH --job-name=eval_kits
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

MODE=$1
if [ -z "$MODE" ]; then
    echo "Usage: sbatch $0 {full|roi}"
    exit 1
fi

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
python -u src/evaluation/evaluate_on_kits.py --mode $MODE