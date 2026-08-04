# Setup and reproduction

## 1. Environment

```bash
conda env create -f environment.yml
conda activate kidney
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On the Aire HPC cluster:

```bash
module load miniforge/24.7.1
conda activate kidney
export SCRATCH=/mnt/scratch/$USER
```

`SCRATCH` must be set. Every script resolves data and results paths relative to
it, and `train_unified.py` fails with an explicit message if it is missing.

Expected layout under `$SCRATCH`:

```
$SCRATCH/
  kidney-data/
    raw/                       Downloaded archives, untouched
    processed/
      unified_v2/              manifest_with_folds.csv + images
      unified_v3/
      unified_v3_corrected/    Primary manifest (patient-leakage corrected)
      unified_v3_corrected_roi/
      roi_dilated/margin_m000/ ... margin_m040/
  kidney-results/
    kfold/<tag>/               config.json, summary.json, fold_*/
    predictions/<tag>.csv      Per-image predictions, input to all analysis
    loso/<tag>/
    segmenter/
    comparison/                Patient-level, robust stats, calibration outputs
```

## 2. Data acquisition

Sources must be obtained under their own licence terms. None is redistributed
in this repository.

```bash
sbatch scripts/download_kits.sh          # KiTS23
sbatch scripts/download_tcga_kirc.sh     # TCGA-KIRC via TCIA
sbatch scripts/convert_tcga.sh           # DICOM -> PNG
sbatch scripts/extract_abdalla.sh        # Abdalla et al. axial slices
# Mendeley and KAUH: download manually, then place under $SCRATCH/kidney-data/raw/
```

## 3. Preprocessing and manifest construction

```bash
sbatch scripts/build_unified_manifest.sh   # -> unified_v3_corrected
sbatch scripts/extract_kits_masks.sh       # KiTS ground-truth kidney masks
sbatch scripts/train_segmenter.sh          # U-Net kidney segmenter (for sources without masks)
sbatch scripts/build_roi.sh                # Tight ROI crops
sbatch scripts/build_roi_dilated.sh        # Margin sweep: 0.00 / 0.15 / 0.25 / 0.40
```

Every manifest is a CSV with columns:
`path, label, patient_id, source, fold` (plus `mask_path` where masks exist,
and `bbox_*`, `roi_margin`, `mask_fallback` for ROI manifests).

**Patient grouping is enforced at manifest construction.** All slices from one
patient carry the same `patient_id` and are assigned to the same fold. This is
the fix for the leakage discovered in the Kaggle CT Kidney Dataset, where
distinct slices from the same scan were previously distributed across the
train/test boundary.

## 4. Training

Single entry point:

```bash
python -m src.training.train_unified --arch ARCH --manifest MANIFEST --tag TAG [options]
```

| Option | Values | Notes |
|---|---|---|
| `--arch` | `resnet50` `cbam` `vit` `multihead` `dann` | |
| `--manifest` | `v2` `v2_roi` `v3` `v3_roi` `v3c` `v3c_roi` `v4` `v4_roi` | |
| `--manifest-path` | any CSV | Overrides `--manifest` |
| `--loss` | `ce` `focal` | `--focal-gamma` defaults to 2.0 |
| `--balanced-sampler` | flag | 25% per class per batch in expectation |
| `--cyst-source` | `kits` `kauh` `both` `all` | Multi-head Cyst gating |
| `--lambda-seg` | float | Auxiliary segmentation weight (multi-head) |
| `--dann-lambda`, `--dann-schedule` | float, `fixed`/`linear_ramp` | DANN only |
| `--folds-to-run` | `all` or `0,2,4` | For smoke tests |

Headline configuration:

```bash
python -m src.training.train_unified \
    --arch resnet50 --manifest v3c --loss focal --balanced-sampler \
    --lr 2e-4 --weight-decay 1e-5 --batch-size 64 --aug-strength strong \
    --epochs 25 --tag v3c_rn50_full
```

Hyperparameters were selected by a two-stage sweep on fold 0, **ranked on
validation macro-F1** (`src/evaluation/hp_sweep_aggregate.py`). Test folds are
never used for selection.

## 5. Analysis

Everything downstream runs from `predictions/<tag>.csv`. If a run predates the
unified trainer, regenerate predictions from its checkpoints — no retraining:

```bash
python -m src.evaluation.dump_predictions --tag v2_run1 --arch resnet50 --manifest v2
```

Then:

```bash
bash scripts/run_all_analysis.sh
```

which runs, in order:

1. `patient_level_eval.py` — slice-level vs patient-level metrics (`mean`,
   `max`, `vote` aggregation)
2. `robust_stats.py` — patient-clustered bootstrap CIs, paired bootstrap on
   Δmetric, McNemar tests
3. `calibration.py` — ECE, MCE, Brier, reliability diagrams,
   selective-prediction curves

External validation and the ROI mechanism test:

```bash
sbatch scripts/loso_train.sh          # leave-one-source-out
sbatch scripts/roi_margin_sweep.sh    # ROI dilation sweep + crossing-point figure
```

## 6. Reproducibility notes

- Seeds are set for `random`, `numpy`, `torch`, and CUDA. DataLoaders use
  explicit generators and `worker_init_fn`, so augmentation is reproducible
  across runs at fixed `--num-workers`.
- Exact bitwise reproducibility on GPU additionally requires
  `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. This is not enabled by default because it
  materially slows training; enable it if you need bitwise-identical runs.
- Multi-seed results use `--seed {42,123,456,789,1011}` with matching
  `--seed-suffix` manifests, giving 25 fold-level observations per condition.
- Every run's full argument namespace is written to `config.json`.

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `SCRATCH is not set` | `export SCRATCH=...` |
| `Manifest not found` | Manifest build step not run, or wrong `--seed-suffix` |
| `AssertionError: Patient leakage` | Manifest fold assignment is not patient-grouped — rebuild it |
| Missing keys on `load_state_dict` | Expected when loading a multi-head or DANN checkpoint for classification-only inference; auxiliary heads are unused |
| CUDA OOM with `--arch vit` | Use `--batch-size 32 --lr 5e-5` |
