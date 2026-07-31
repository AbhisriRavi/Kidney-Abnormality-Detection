cat > ~/projects/Kidney-Abnormality-Detection/scripts/final_multihead_train.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=v3_mh
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load miniforge/24.7.1
conda activate kidney
cd ~/projects/Kidney-Abnormality-Detection
export PYTHONPATH=$PWD:$PYTHONPATH

python -u src/training/train_kfold_v3_multihead.py \
    --tag "$TAG" \
    --seed "$SEED" \
    --seed-suffix "$SUFFIX" \
    --folds-to-run all \
    --epochs 25 \
    --lr 2e-4 --weight-decay 1e-5 --batch-size 64 --aug-strength strong \
    --cyst-source "$CYST_SRC" \
    --lambda-seg 0.5
EOF