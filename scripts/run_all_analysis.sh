#!/bin/bash
# ---------------------------------------------------------------------------
# End-to-end analysis over EXISTING checkpoints. No training, no GPU hours
# beyond inference. Produces every new figure and table in one pass.
# ---------------------------------------------------------------------------
set -euo pipefail
cd ~/projects/Kidney-Abnormality-Detection

# Edit this list to match your run tags and architectures.
# Format: "<tag>:<arch>:<manifest>"
CONDITIONS=(
  "v2_run1:resnet50:v2"
  "v2_roi_run1:resnet50:v2_roi"
  "cbam_full_run1:cbam:v2"
  "cbam_roi_run1:cbam:v2_roi"
  "vit_full_run1:vit:v2"
  "vit_roi_run1:vit:v2_roi"
)

echo "### Step 1/4 — dump per-image predictions from saved checkpoints"
for c in "${CONDITIONS[@]}"; do
    IFS=':' read -r tag arch man <<< "$c"
    if [ -f "$SCRATCH/kidney-results/predictions/$tag.csv" ]; then
        echo "  $tag already dumped, skipping"
        continue
    fi
    python -u -m src.evaluation.dump_predictions \
        --tag "$tag" --arch "$arch" --manifest "$man"
done

TAGS=$(printf "%s " "${CONDITIONS[@]}" | tr ' ' '\n' | cut -d: -f1 | tr '\n' ' ')

echo "### Step 2/4 — slice-level vs patient-level evaluation"
python -u -m src.evaluation.patient_level_eval --predictions $TAGS --agg mean
python -u -m src.evaluation.patient_level_eval --predictions $TAGS --agg max

echo "### Step 3/4 — patient-clustered bootstrap CIs and McNemar tests"
python -u -m src.evaluation.robust_stats --conditions $TAGS \
    --level patient --all-pairs --n-boot 2000

echo "### Step 4/4 — calibration and selective prediction"
python -u -m src.evaluation.calibration --conditions $TAGS \
    --level patient --temperature-scale

echo
echo "All outputs under $SCRATCH/kidney-results/comparison/"
