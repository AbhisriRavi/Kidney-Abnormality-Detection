# Framing and structure

*Working notes. Replaces `corrected_story.md`, which framed the project around
a null result.*

## The problem with the current framing

`corrected_story.md` currently opens on six failed interventions and concludes
that "the bottleneck is representational." That is defensible but it reads as a
project that did not work. It also buries the actual answer to the question the
project specification poses, which is about ROI extraction, not about
architectural interventions.

Two changes fix this without any new experiments.

### Change 1: RQ1 is the spine, and its answer is positive

The submitted specification and scoping document both frame this project around
one question: does kidney ROI extraction improve classification relative to
full-image input? The answer obtained is more interesting than a plain yes:

> ROI extraction is not a uniform improvement but a class-dependent trade-off
> with an anatomical cause, replicated across three architectures with
> substantially different inductive biases.

Cyst F1 improves by 0.12–0.15, Tumor F1 by 0.01–0.04, and Stone F1 degrades by
0.06–0.19. The degradation is explained by the anatomy of renal calculi, which
frequently sit in the calyces, renal pelvis, and proximal ureter — structures
outside a parenchyma-bounded crop. Replication across a CNN, an
attention-augmented CNN, and a transformer rules out an architecture-specific
artefact.

"It depends, and here is precisely why, and here is the design implication" is
a stronger dissertation result than a flat improvement would have been. The ROI
dilation sweep (`scripts/roi_margin_sweep.sh`) converts the explanation from
plausible to tested, and yields a transferable recommendation: how tight should
a kidney ROI be?

### Change 2: KiTS near-chance and the failed interventions are the same finding

Currently these are reported separately: KiTS performs near chance, and six
interventions produced null results. They are one thing.

Look at the source × class structure:

| Source | Classes present |
|---|---|
| KAUH | Cyst only |
| TCGA | Tumor only |
| Mendeley | Normal, Stone |
| Abdalla | Stone, Normal |
| KiTS | Cyst, Normal, Tumor |

KAUH Cyst F1 is 0.996 and TCGA Tumor F1 is 1.000. Those are not measurements of
pathology detection; on a single-class source, predicting the class correctly is
achievable by recognising the scanner. **KiTS is the only source on which the
task cannot be solved that way**, because it carries three of the four classes.
Its near-chance performance is therefore not an anomaly to be explained away —
it is the cleanest estimate in the study of how well the model actually
discriminates kidney pathology once the shortcut is unavailable.

This reframing also explains the DANN result. DANN degraded accuracy by 8.5pp
because, in a dataset where source identity is partially *diagnostic* of class,
adversarially removing source information necessarily removes label-relevant
signal. That is a mechanistic explanation for a negative result, which is a
genuine contribution rather than a failure.

## Revised summary paragraph

> Across a patient-grouped, five-source kidney CT dataset (8,924 images, 561
> patients), region-of-interest extraction was found to produce a
> class-dependent trade-off rather than a uniform improvement: Cyst F1 improved
> by 0.12–0.15 and Tumor F1 by 0.01–0.04, while Stone F1 degraded by 0.06–0.19.
> The pattern replicated across a CNN, an attention-augmented CNN, and a vision
> transformer, and admits an anatomical explanation, since renal calculi
> frequently lie outside a parenchyma-bounded crop. Diagnostic analysis revealed
> that the dataset carries a structural covariance between imaging source and
> class label: no constituent source contains all four classes, and per-source
> scores on single-class sources approach ceiling (KAUH Cyst F1 = 0.996,
> TCGA Tumor F1 = 1.000). Performance on KiTS23 — the only source carrying three
> classes, and therefore the only source on which the task cannot be solved by
> recognising the acquisition site — falls to near chance across every
> architecture tested. Leave-one-source-out evaluation quantifies the resulting
> generalisation gap directly. Six architectural and training-time interventions
> intended to mitigate the shortcut, including focal loss with class-balanced
> sampling, adversarial source removal, test-time augmentation, and multi-head
> architectures with an auxiliary segmentation objective, produced no improvement
> over a well-tuned plain ResNet50 baseline under Bonferroni correction across 36
> tests. Adversarial source removal significantly degraded performance
> (−8.5pp accuracy, p = 0.001), consistent with the interpretation that in this
> dataset the source-related features it removes are themselves partially
> label-carrying. The bottleneck is therefore identified as a property of dataset
> composition rather than of model capacity, and the practical implication is
> that source-balanced multi-pathology data collection, not architectural
> sophistication, is the binding constraint on progress in this task.

## Chapter structure

Use the conventional structure. Five finding-chapters would fragment the ROI
narrative and make it harder for an assessor to map the dissertation back onto
the submitted objectives.

1. **Introduction** — clinical motivation, RQ1–RQ3, contributions, structure
2. **Background and Related Work** — CNNs in medical imaging, ROI/attention
   approaches, kidney CT specifically, shortcut learning and dataset bias,
   the gap this project addresses
3. **Data and Methodology**
   - 3.1 Datasets and the source × class structure (put the cross-tab here)
   - 3.2 **The patient-leakage finding** — named subsection, not a bullet
   - 3.3 Preprocessing and manifest construction
   - 3.4 Kidney segmentation and ROI extraction, including the dilation margin
   - 3.5 Architectures
   - 3.6 Evaluation protocol — patient-grouped CV, patient-level metrics,
     bootstrap CIs, McNemar, calibration, leave-one-source-out
   - 3.7 Ethical considerations *(see ethics_and_licensing.md)*
4. **Results**
   - 4.1 Baseline performance, slice-level and patient-level
   - 4.2 **RQ1: ROI vs full image** — the class-dependent trade-off, plus the
     dilation sweep
   - 4.3 **RQ2: architecture comparison** — simpler wins
   - 4.4 **RQ3: source-shortcut diagnostics** — per-source breakdown, KiTS,
     leave-one-source-out, Grad-CAM/attention evidence
   - 4.5 Intervention study — the six null results and the DANN degradation
   - 4.6 Calibration and selective prediction
5. **Discussion** — what the trade-off means for system design, why complexity
   did not help, dataset composition as the binding constraint, clinical
   positioning, limitations
6. **Conclusion and Future Work**

## On the supervisor questions

**"Simpler model wins" as headline or secondary?** Secondary. It supports the
representational-bottleneck argument rather than standing alone, and RQ1 is what
the specification committed to. Give it a full results section, not the abstract.

**Stone ROI degradation: limitation or test an alternative?** Test it. The
dilation sweep is roughly two days and converts an observed limitation into a
validated causal mechanism plus a design recommendation.

**Source-shortcut: document or run DANN?** Already run, and the negative result
is more informative than a positive one would have been. Add leave-one-source-out
as the quantitative measurement, which is the piece currently missing.

## Scope-evolution paragraph

Include something like this early in Chapter 3, so the growth beyond the
original specification reads as methodological necessity rather than drift:

> The scope of this project evolved during execution in response to a data
> integrity finding. The original specification proposed a direct comparison
> between full-image and ROI-based classification on a single primary dataset.
> Early experiments produced accuracies above 99%, which cross-dataset
> evaluation revealed to be an artefact of patient-level leakage rather than
> genuine performance. Correcting this required rebuilding the dataset with
> patient-grouped splits, which in turn required incorporating additional
> sources to obtain sufficient per-class patient counts. The resulting
> multi-source dataset exposed a structural covariance between source and label
> that became a substantive object of study in its own right. The ROI question
> posed in the original specification remains the primary research question and
> is answered in Section 4.2; the additional investigations reported in Sections
> 4.4 and 4.5 were necessary to establish that the answer is not itself an
> artefact of dataset composition.
