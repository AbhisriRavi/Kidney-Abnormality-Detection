#!/bin/bash
# Submit all 12 multi-seed runs.
#
# Combines: 3 architectures x 2 manifests x 2 seeds (excluding the original seed 42 runs)
#
# Each job uses the same training script as the original runs, with extra args
# for seed and manifest suffix.

cd ~/projects/Kidney-Abnormality-Detection

for SEED in 123 456; do
    SUFFIX="_seed${SEED}"

    # Plain ResNet50 — full
    sbatch --export=ALL,TAG="v2_run1_seed${SEED}",MANIFEST="v2",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/multi_seed_train.sh rn50

    # Plain ResNet50 — ROI
    sbatch --export=ALL,TAG="v2_roi_run1_seed${SEED}",MANIFEST="v2_roi",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/multi_seed_train.sh rn50

    # CBAM-ResNet50 — full
    sbatch --export=ALL,TAG="cbam_full_run1_seed${SEED}",MANIFEST="v2",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/multi_seed_train.sh cbam

    # CBAM-ResNet50 — ROI
    sbatch --export=ALL,TAG="cbam_roi_run1_seed${SEED}",MANIFEST="v2_roi",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/multi_seed_train.sh cbam

    # ViT-Base — full
    sbatch --export=ALL,TAG="vit_full_run1_seed${SEED}",MANIFEST="v2",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/multi_seed_train.sh vit

    # ViT-Base — ROI
    sbatch --export=ALL,TAG="vit_roi_run1_seed${SEED}",MANIFEST="v2_roi",SEED="$SEED",SUFFIX="$SUFFIX" \
           scripts/multi_seed_train.sh vit
done

echo
echo "Submitted 12 multi-seed jobs. Check queue with: squeue -u \$USER"