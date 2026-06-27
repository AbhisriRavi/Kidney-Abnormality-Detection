#!/bin/bash
#SBATCH --job-name=multi_seed
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Arguments via env: TAG, MANIFEST, SEED, SUFFIX
# Positional: $1 = architecture (rn50/cbam/vit)

ARCH=$1

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection

case $ARCH in
    rn50)
        python -u src/training/train_kfold_v2.py \
            --tag $TAG --manifest $MANIFEST --seed $SEED --seed-suffix $SUFFIX
        ;;
    cbam)
        python -u src/training/train_kfold_cbam.py \
            --tag $TAG --manifest $MANIFEST --seed $SEED --seed-suffix $SUFFIX
        ;;
    vit)
        python -u src/training/train_kfold_vit.py \
            --tag $TAG --manifest $MANIFEST --seed $SEED --seed-suffix $SUFFIX
        ;;
    *)
        echo "Unknown architecture: $ARCH"
        exit 1
        ;;
esac