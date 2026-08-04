# Ethics, data governance, and fairness

*Draft section for the dissertation. Expands Appendix A of the scoping document
from a general statement into a per-dataset, auditable one.*

## Data provenance and licensing

All imaging data used in this project is publicly available, de-identified, and
used in accordance with its licence. No data was collected from patients by the
author, no identifiable information was accessed, and no attempt was made to
re-identify any individual. No dataset is redistributed in the project
repository; acquisition scripts fetch each source from its original provider.

| Source | Contributed classes | Licence / terms | Conditions observed |
|---|---|---|---|
| KiTS23 | Cyst, Normal, Tumor | CC BY-NC-SA 4.0 | Non-commercial academic use; attribution to Heller et al.; derivative manifests shared under the same terms |
| Mendeley CT Kidney (Islam et al., 2021) | Normal, Stone | CC BY 4.0 | Attribution |
| KAUH (Alzu'bi et al., 2022) | Cyst | Terms in source publication | Research use; attribution |
| TCGA-KIRC | Tumor | TCIA Data Usage Policy / CC BY 3.0 | Attribution to TCIA and TCGA Research Network; no re-identification |
| Abdalla et al. (2025) | Stone, Normal | Terms in source publication | Research use; attribution |

Ethical approval was not required, as the project involves only secondary
analysis of already-published anonymised datasets. This is consistent with
University of Leeds School of Computer Science guidance for projects using
public data.

## Data integrity: the patient-leakage finding

An important data-governance issue was identified during this project. The
widely-used Kaggle CT Kidney Dataset distributes multiple axial slices per
patient without patient identifiers exposed in the directory structure. Under
the naive random train/test split used in much of the literature built on this
dataset, slices from the same scan appear on both sides of the split. Because
adjacent slices of one kidney are near-duplicates, a model can achieve very
high apparent accuracy by memorising patient-specific appearance rather than
pathology.

Quantified here: apparent accuracy of approximately 99% under a random split
falls to approximately 38% under a patient-grouped split with cross-dataset
verification.

This is reported not as a criticism of the dataset authors but as a caution
about how the dataset is used. It also carries an ethical dimension: published
performance figures that are inflated by leakage overstate the readiness of
such systems for clinical use, and could contribute to misplaced confidence in
automated diagnostic tools. All experiments in this dissertation use
patient-grouped splits, and the trainer asserts patient disjointness at run
time.

## Fairness and representational bias

The unified dataset exhibits a structural covariance between imaging source and
class label: no source contains all four classes, so the label distribution is
partially predictable from the acquisition site alone. This has consequences
that go beyond measurement validity.

A model that has learned "TCGA-like appearance implies tumour" has effectively
learned a scanner-and-institution prior rather than a pathological one. Because
the constituent datasets originate from different countries, institutions, and
scanner generations, and because their underlying patient populations differ in
ways not documented in the released metadata, such a model would be expected to
fail unpredictably on populations or equipment underrepresented in training.
The failure mode is particularly concerning because it is silent: the model
remains confident while being wrong.

Three concrete limitations follow, and are stated plainly rather than hedged:

1. **No demographic metadata.** Age, sex, and ethnicity are unavailable for
   most sources, so subgroup performance cannot be audited. Any claim of
   equitable performance would be unsupported.
2. **Geographic concentration.** The Abdalla data derives from three Iraqi
   hospitals, TCGA-KIRC predominantly from North American centres, KAUH from
   Jordan. Coverage is neither globally representative nor deliberately
   stratified.
3. **Single-slice, 2D formulation.** Radiologists read volumes with clinical
   context; this system classifies isolated axial slices. Performance figures
   here are not comparable to reported radiologist accuracy and should not be
   presented as such.

Leave-one-source-out evaluation is included specifically to measure this: it
reports how much performance is lost when the model encounters an institution
it has not seen, which is the closest available proxy for deployment at a new
site.

## Clinical positioning and intended use

This system is a research artefact. It is not validated for clinical use, has
not been assessed by any regulator, and must not be used to inform patient
care. Three points bound the claim:

- **Assistive, not autonomous.** The plausible role for a model at this
  performance level is triage or second-reader support, never independent
  diagnosis. The selective-prediction analysis quantifies this directly: it
  reports accuracy as a function of the fraction of cases auto-reported, so a
  coverage threshold can be chosen at which the remaining cases are routed to a
  radiologist.
- **Calibration matters as much as accuracy.** A confidently wrong model is
  more dangerous in a triage workflow than a diffident one, which is why
  expected calibration error and reliability diagrams are reported alongside
  accuracy.
- **Error asymmetry is not modelled.** A missed tumour and a false-positive
  cyst carry very different clinical costs. All metrics here treat errors
  symmetrically. Cost-sensitive evaluation with clinician-elicited weights is
  necessary future work.

## Reproducibility and research integrity

- All code, configurations, and per-fold results are version-controlled.
- Hyperparameters are selected on validation splits only; test folds are never
  consulted during model or hyperparameter selection.
- Negative and null results are reported in full, including the six
  interventions that failed to improve on the baseline and the DANN result that
  significantly degraded performance. Suppressing these would misrepresent the
  research process and inflate the apparent effectiveness of the approaches
  tested.
- Statistical tests, correction for multiple comparisons, and the number of
  comparisons run are reported explicitly, so the reader can assess the
  evidential weight of each claim.
