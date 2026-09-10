#!/bin/bash
cd ~/projects/Kidney-Abnormality-Detection

submit () {
  sbatch --export=ALL,TRAINER="$1",MANIFEST="$2",TAG="$3",LR="$4",WD="$5",BS="$6" \
         scripts/v3c_arch_train.sh
}

submit src/training/train_kfold_cbam.py v3c     v3c_cbam_full_seed42 1e-4 1e-4 64
submit src/training/train_kfold_cbam.py v3c_roi v3c_cbam_roi_seed42  1e-4 1e-4 64
submit src/training/train_kfold_vit.py  v3c     v3c_vit_full_seed42  5e-5 1e-4 32
submit src/training/train_kfold_vit.py  v3c_roi v3c_vit_roi_seed42   5e-5 1e-4 32

echo "Submitted 4 jobs (5 folds each)."
squeue -u $USER
