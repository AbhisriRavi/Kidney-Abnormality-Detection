# Kidney Abnormality Detection from CT Imaging

MSc Advanced Computer Science (AI) dissertation project, University of Leeds.
Investigates whether kidney region-of-interest (ROI) extraction improves
deep-learning classification of kidney abnormalities (Normal / Cyst / Tumor /
Stone) compared with conventional full-slice input.

**Author:** Abhisri Ravi · **Supervisor:** Dr Samson Fabiyi · **Module:** COMP5200M

---

## Research questions

**RQ1.** Does kidney ROI extraction improve classification performance relative
to full-image input, and is any effect uniform across abnormality classes?

**RQ2.** Does architectural sophistication (channel/spatial attention, vision
transformers, multi-head outputs with an auxiliary segmentation objective)
improve on a well-tuned plain ResNet50 baseline on this task?

**RQ3.** To what extent do models exploit the structural covariance between
data source and class label rather than pathology-relevant image features, and
does that covariance limit cross-source generalisation?

## Headline findings

1. **Patient-level leakage in the Kaggle CT Kidney Dataset** inflates apparent
   accuracy from an honest ~38% to ~99%. Slices from the same patient appear in
   both training and test splits under the naive random split used by much of
   the published literature on this dataset.
2. **ROI extraction is a class-dependent trade-off, not a uniform gain.**
   Replicated across three architectures with different inductive biases:
   Cyst F1 +0.12 to +0.15, Tumor F1 +0.01 to +0.04, Stone F1 −0.06 to −0.19.
   The Stone degradation has an anatomical explanation — calculi frequently sit
   in the calyces, renal pelvis, and proximal ureter, outside a tight
   parenchyma-bounded crop.
3. **Architectural complexity does not help.** Plain ResNet50 outperforms
   CBAM-ResNet50 and ViT-Base on accuracy, macro-F1, and macro-AUC, and has the
   tightest cross-fold variance.
4. **Source-shortcut learning is architecture-independent.** The dominant
   source→class mapping is near-identical across all three architectures,
   indicating a property of the data structure rather than the model.
5. **KiTS23 performance sits near chance for every architecture.** KiTS is the
   only source carrying three of the four classes, and therefore the only
   source on which the task cannot be solved by recognising the scanner. This
   is the strongest single piece of evidence for finding 4.

---

## Repository layout

```
src/
  data/          Dataset acquisition, preprocessing, manifest construction, ROI extraction
  models/        Architecture definitions (factory.py is the single entry point)
  training/      train_unified.py (main entry point), train_loso.py, train_segmenter.py
                 legacy/  — superseded per-experiment scripts, retained for reproducibility
  evaluation/    Metrics, patient-level aggregation, robust statistics, calibration,
                 source-bias diagnostics, Grad-CAM/attention visualisation
scripts/         SLURM submission scripts for the Aire HPC cluster
notebooks/       Exploratory analysis and verification scripts
configs/         Experiment configuration files
dissertation/    Figures, tables, and writing notes
metadata/tcga/   TCGA-KIRC per-series metadata (regenerable)
docs/            Setup and reproduction instructions
```

## Datasets

| Source | Classes contributed | Licence | Access |
|---|---|---|---|
| KiTS23 | Cyst, Normal, Tumor | CC BY-NC-SA 4.0 | github.com/neheller/kits23 |
| Mendeley (Islam et al.) | Normal, Stone | CC BY 4.0 | Mendeley Data |
| KAUH (Alzu'bi et al.) | Cyst | See publication | Journal of Healthcare Engineering |
| TCGA-KIRC | Tumor | TCIA Data Usage Policy | The Cancer Imaging Archive |
| Abdalla et al. (2025) | Stone, Normal | See publication | — |

All data is publicly available and anonymised. See
`dissertation/ethics_and_licensing.md` for the full statement.

**Source × class structure is deliberately reported**, because it is not
balanced: Cyst appears in 2 sources, Tumor in 2, Stone in 2, Normal in 3. This
partial confounding between source and label is a central object of study in
this project rather than an incidental limitation. Run
`python -m src.training.train_loso --manifest v3c --hold-out all` to regenerate
the cross-tabulation and the quantitative cross-source transfer results.

## Quick start

```bash
conda env create -f environment.yml
conda activate kidney
export SCRATCH=/mnt/scratch/$USER        # or any writable directory

# Train
python -m src.training.train_unified --arch resnet50 --manifest v3c \
    --loss focal --balanced-sampler --tag v3c_rn50_full

# Evaluate (slice- and patient-level, CIs, calibration)
bash scripts/run_all_analysis.sh
```

Full instructions, including data acquisition and the Aire HPC workflow, are in
[`docs/setup.md`](docs/setup.md).

## Evaluation protocol

- **Patient-grouped 5-fold cross-validation.** No patient appears in more than
  one fold; the trainer asserts this at run time.
- **Two units of analysis.** Slice-level metrics are reported for continuity
  with the literature; **patient-level metrics are the primary results**, since
  patients are the independent unit and slice counts vary widely per patient.
- **Model selection on validation macro-F1 only.** Hyperparameters and
  checkpoints are chosen on a patient-disjoint validation split carved from the
  training folds; test folds are never consulted during selection.
- **Patient-clustered bootstrap CIs and McNemar tests**, rather than paired
  t-tests over 5 fold means, which are badly underpowered after correction for
  multiple comparisons.
- **Leave-one-source-out external validation** to measure cross-source transfer.
- **Calibration** (ECE, MCE, Brier, reliability diagrams, selective-prediction
  curves), because accuracy alone does not establish clinical usability.

## Reproducing the reported results

Every experiment writes a `config.json` and `summary.json` under
`$SCRATCH/kidney-results/kfold/<tag>/`, containing the resolved manifest path,
the complete argument namespace, and per-fold metrics. Per-image predictions
are written to `$SCRATCH/kidney-results/predictions/<tag>.csv`, which is the
input to all downstream analysis.

## Citation

If you use this code, please cite the dissertation:

> Ravi, A. (2026). *Kidney Abnormality Detection Using CT Imaging and Deep
> Neural Networks.* MSc dissertation, School of Computer Science, University of
> Leeds.
