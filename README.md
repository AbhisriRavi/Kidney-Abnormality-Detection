# Kidney Abnormality Detection Using CT Imaging and Deep Neural Networks

Four-class classification of kidney CT (Normal, Cyst, Tumor, Stone) with
patient-level evaluation across four independent acquisition sources.

MSc Advanced Computer Science (Artificial Intelligence) dissertation,
School of Computer Science, University of Leeds, 2025/2026.
Author: Abhisri Ravi. Supervisor: Dr Samson Fabiyi. Assessor: Dr Sharib Ali.

## What this project found

Published classifiers for this task routinely report accuracies above 99%, which is
hard to reconcile with the inter-observer agreement expert radiologists achieve on the
same distinctions. This project tested two mechanisms that can produce such numbers
without diagnostic ability, and found both.

**Patient-level leakage.** Partitioning slices at image level puts slices from one
patient in both training and test sets. A classifier scoring above 99% under
image-level partitioning fell to roughly 38% on independently acquired patients,
against 33.3% for assigning a single class to everything.

**Source-label collinearity.** Pooling public datasets in which acquisition source
predicts class label lets a model recognise the source instead of the pathology. In the
corrected dataset, two of the four sources contribute a single class each, so
recognising the source is sufficient to predict the label for those patients. Under
leave-one-source-out evaluation, accuracy on held-out sources ranged from 0.00 to 0.33,
against 0.76 to 0.97 in distribution. On one fold the model was correct on 6 of 3,364
images. Six mitigation strategies produced no improvement, and a control experiment
removing preprocessing differences produced no recovery of transfer.

The binding constraint is dataset composition, not model capacity. Where source and
class are collinear, no loss function, architecture or invariance penalty recovers
transferable performance, because the training data does not contain the evidence
needed to separate pathology from provenance.

## Results

Corrected dataset (v3c): 8,924 images from 561 patients across four sources, after a
duplicated fifth source was identified and excluded.

Patient-level accuracy and macro-F1, from Table 4.3. The two columns cover different
patient populations, because ROI extraction discards 17.5% of patients to segmentation
failure.

| Architecture | Full image (n=561) | | ROI (n=463) | |
|---|---|---|---|---|
| | Accuracy | Macro-F1 | Accuracy | Macro-F1 |
| ResNet50 | 0.8503 | 0.8211 | 0.8747 | 0.8320 |
| CBAM | 0.7986 | 0.7317 | 0.8747 | 0.8474 |
| ViT-B/16 | 0.8431 | 0.7888 | 0.8790 | 0.8470 |
| DANN | 0.6381 | 0.6405 | | |

**Do not read the ROI column as an improvement.** Comparing 0.8503 against 0.8747
confounds the effect of cropping with the effect of having discarded the hardest 17.5%
of patients from one arm. On the 463 patients present in both conditions, the effect of
ROI extraction is -0.024 ([-0.061, +0.011]), a null result. Every per-class interval
nonetheless excludes zero: Cyst and Tumor favour cropped input, Normal and Stone favour
full images. Cropping also improves calibration.

No architecture outperformed the baseline.

Kidney segmentation for ROI extraction uses a MONAI U-Net, validation Dice 0.938.

## Data

No imaging data is redistributed here. The four constituent sources are obtained from
their providers under their own licences; see `DATA.md` for terms and acquisition
steps. A fifth source was acquired and excluded after it proved to be a preprocessed
duplicate of another, overlapping on 201 of 201 patients.

`metadata/tcga/` holds TCIA download manifests only. Per-series metadata is regenerable
via `src/data/tcga/download_full.py`.

## Repository structure
```text
configs/ experiment configuration files
docs/ setup and reproduction notes
metadata/ regenerable source metadata
notebooks/ exploratory and verification analyses
scripts/ SLURM launch and pipeline scripts
src/data/ acquisition, preprocessing and manifests
src/models/ model definitions
src/training/ training entry points
src/evaluation/ metrics, statistics and diagnostics
src/utils/ shared utilities
tests/ software checks
dissertation/ result summaries and writing artefacts
```


### Notes on naming and provenance

Dataset versions (v2, v3, v3c, v4) are a property of the manifest, not the code. The
reported baseline was produced by `train_kfold_v3_baseline.py` running against the v3c
manifest, so a `v3_` filename does not indicate superseded results. Files keep their
original names to preserve the link to the SLURM job logs.

`v4` manifests and trainers are exploratory work that was not carried into the
dissertation. Nothing in the report is based on them.

Results under `dissertation/results/v3c/` are the corrected, reported set. Files under
`dissertation/figures/` predate the duplicate-source correction and are retained only
for history.

Two corrections to Appendix C of the dissertation, found while preparing this
repository:

- The ROI manifest directory is `unified_v3c_roi`, not `unified_v3_corrected_roi`.
- The headline `train_unified` command in Appendix C specifies `--loss focal
  --balanced-sampler`. The reported Table 4.1 baseline was launched through
  `scripts/v3c_baseline.sh`, which uses weighted cross-entropy and no balanced sampler.
  Both are valid configurations; the launch script is the record of what produced the
  reported numbers.

Per-run provenance is split: `summary.json` records optimisation settings (learning
rate, batch size, epochs, augmentation strength, seed, folds), while architecture, loss
and manifest are fixed in the launch scripts, which are version-controlled.

## Reproducing

Requires a CUDA GPU and a writable scratch filesystem.

```bash
conda env create -f environment.yml
conda activate kidney
export SCRATCH=/path/to/writable/scratch
```

Acquire the four sources per `DATA.md`, then build manifests, train and analyse. Full
command sequence in `docs/setup.md` and Appendix C of the dissertation.

The reported experiment matrix was launched through the retained SLURM scripts:

```bash
bash   scripts/v3c_baseline.sh
bash   scripts/v3c_roi_all.sh
sbatch scripts/cbam_full.sh cbam_v3c_full <v3c-full-manifest>
sbatch scripts/vit_full.sh  vit_v3c_full  <v3c-full-manifest>
bash   scripts/v3c_all_experiments.sh
sbatch scripts/loso_train.sh
bash   scripts/run_all_analysis.sh
```

A unified entry point covers the same architectures and manifests for new runs:

```bash
python -m src.training.train_unified \
  --arch resnet50 --manifest v3c --tag my_run \
  --lr 2e-4 --weight-decay 1e-5 --batch-size 64 \
  --aug-strength strong --epochs 25
```

Evaluation is driven from per-image prediction CSVs, so patient-level aggregation,
bootstrap intervals, McNemar tests and calibration can be regenerated without
retraining.

```bash
pytest tests/ -v
```

The tests verify patient-disjoint folds, the manifest schema, the reported dataset
sizes, and the per-source class composition on which the collinearity analysis rests.
They skip cleanly when the data is not present.

## Traceability

| Reported item | Module | Input |
|---|---|---|
| Table 4.1 | `src/evaluation/ch4_v3c.py`, `patient_level_eval.py` | per-image prediction CSVs |
| Figure 4.2, Table 4.2 | `src/evaluation/robust_stats.py` | matched patient predictions |
| Table 4.4 | `src/evaluation/check_source_bias.py`, `per_source_matched_eval.py` | predictions with source field |
| Table 4.5 | `src/training/train_loso.py` | `kidney-results/loso/` |
| Table 4.6 | `src/evaluation/ch4_intervention.py` | baseline and intervention predictions |
| Table 4.7 | `src/evaluation/calibration.py` | patient prediction probabilities |

## Compute

207 SLURM jobs, 48.1 GPU hours, NVIDIA L40S, University of Leeds Aire HPC service.

## Licence

Code released under the MIT Licence, see `LICENSE`. Each data source retains its own
licence and none is redistributed here; see `DATA.md`.

## Disclaimer

A research artefact. Not clinically validated, and must not inform patient care.