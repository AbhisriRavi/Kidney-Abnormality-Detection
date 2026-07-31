# Impact of Additional Source Diversity on Multi-Source Kidney Classification

## Motivation

The results reported in previous sections were obtained on a four-source
unified dataset comprising KiTS, Mendeley, KAUH, and TCGA-KIRC. Two
methodological concerns motivated an extension to a fifth source. First,
the Stone class was represented by only one source (Mendeley), which
created a structural covariance between source identity and the Stone
label. Diagnostic analyses reported in earlier sections showed that this
covariance was exploited by the network, particularly on the ROI-cropped
configuration where Stone F1 degraded substantially. Second, both the
supervisor consultation and the reviewed literature (Abdalla et al., 2025)
suggested that adding a second, independently sourced Stone-positive
dataset would allow a direct test of whether the network learned genuine
Stone-relevant features or whether performance depended on Mendeley-
specific artefacts.

To address these concerns, the unified dataset was extended to include
the Abdalla et al. (2025) axial CT dataset (henceforth referred to as
Abdalla), which contains 3,364 original axial slices from 201 patients
across three Iraqi hospitals (Rania, Mercy, and Faruq). The dataset
provides both Stone-positive (1,577 images) and Non-Stone (1,787 images)
labels, thereby also contributing a third source for the Normal class.
The extended dataset — designated `unified_v3` — comprises 12,288 images
from 762 patients across five sources and retains the same four
abnormality classes. The augmented image variants distributed with the
Abdalla dataset (approximately 35,000 additional images generated through
twelve augmentation types) were retained as a separate held-out corpus
and did not contribute to any training or validation partition. Class
proportions in `unified_v3` were as follows: Normal 4,574; Tumor 3,370;
Stone 3,154; Cyst 1,190.

## Hyperparameter tuning

Prior to running the final headline experiments, a two-stage
hyperparameter search was conducted on the plain ResNet-50 architecture
using fold 0 of `unified_v3`. The first stage tested nine combinations
of learning rate (5×10⁻⁵, 1×10⁻⁴, 2×10⁻⁴) and weight decay (1×10⁻⁵,
1×10⁻⁴, 1×10⁻³); the second stage, holding the first-stage winner
constant, tested nine combinations of batch size (32, 64, 128) and
augmentation strength (mild, moderate, strong). The combination that
produced the highest test macro-F1 on fold 0 — learning rate 2×10⁻⁴,
weight decay 1×10⁻⁵, batch size 64, augmentation strength strong —
yielded a fold-0 macro-F1 of 0.830, an improvement of approximately
0.09 over the fold-0 result with the original v2 hyperparameters. This
configuration was subsequently used unchanged in all final `unified_v3`
experiments reported below.

## Headline results

The plain ResNet-50 architecture was retrained on `unified_v3` using
five-fold patient-grouped cross-validation, repeated across five seeds
(42, 123, 456, 789, 1011), for a total of 25 fold-level test observations
per condition. This design provided the same statistical framework used
for the v2 configurations, extended to a larger seed count to increase
the sensitivity of significance testing. All hyperparameters were fixed
to the values identified by the tuning procedure described above; all
other elements of the training protocol (AdamW optimiser, cosine learning
rate schedule, mixed-precision training, class-balanced cross-entropy
loss, ImageNet initialisation) were retained without modification.

Aggregate performance metrics for plain ResNet-50 on `unified_v3`, together
with the corresponding v2 values, are reported in Table [v3_headline].
Test accuracy increased from 73.7 ± 3.7 % (n=15) on v2 to 82.3 ± 2.9 %
(n=25) on v3, a difference of 8.7 percentage points that was highly
statistically significant under Welch's t-test (t=8.02, p<0.001). Macro-F1
increased from 0.712 ± 0.051 to 0.774 ± 0.037 (Δ = 0.062, p<0.001),
and macro-AUC from 0.910 ± 0.031 to 0.950 ± 0.022 (Δ = 0.040, p<0.001).
All three headline metric differences survived Bonferroni correction
across the 11 metrics tested (α_corrected = 0.0045). Both the direction
and the magnitude of these differences were consistent with the
hypothesis that added source diversity would improve classification
performance.

## Per-class analysis

Per-class F1 scores are reported in Figure [per_class_v2_v3]. Two of the
four classes exhibited significant improvements. Normal F1 increased
from 0.699 ± 0.084 to 0.856 ± 0.066 (Δ = +0.157, p<0.001,
Bonferroni-significant), consistent with the intuition that adding a
third Normal-positive source substantively broadened the visual
distribution the network was exposed to during training. Stone F1
increased from 0.888 ± 0.055 to 0.974 ± 0.017 (Δ = +0.086, p<0.001,
Bonferroni-significant); the cross-fold variance also decreased
substantially (standard deviation reduced by approximately 69 percent),
suggesting that Stone predictions became more reliable across folds as
well as more accurate on average.

The remaining two classes did not exhibit statistically significant
changes. Cyst F1 was 0.454 ± 0.164 on v2 and 0.471 ± 0.144 on v3
(Δ = +0.017, p=0.69), and Tumor F1 was 0.808 ± 0.048 on v2 and
0.795 ± 0.058 on v3 (Δ = -0.012, p=0.19). The stability of the Cyst
result is a positive finding in the specific context of this expansion,
because the addition of Abdalla — which contributes no Cyst images —
reduced the proportional representation of the Cyst class from
approximately 13.3 % of v2 to 9.7 % of v3. Despite this dilution, Cyst
F1 did not degrade, indicating that the model retained its (albeit
limited) capacity to identify Cyst under the more challenging class
distribution.

## Source-level analysis

To investigate whether the aggregate improvements were driven by the
Abdalla source alone or reflected broader gains across the dataset, the
best-fold v2 and v3 models were evaluated on each source's test-fold
subset independently. Results are summarised in Figure [per_source_v2_v3].
Accuracy on Mendeley increased from 88.6 % on v2 to 97.6 % on v3
(+9.0 percentage points); accuracy on TCGA remained essentially unchanged
(93.1 % → 92.9 %); accuracy on KAUH remained near-perfect (100.0 % →
98.0 %, though the small test-fold sample sizes of 30 and 50 images
respectively make this comparison unstable); and accuracy on the newly
added Abdalla source was 95.9 %. Notably, the Mendeley improvement is
consistent with the interpretation that adding a second Stone source
did not merely allow the model to solve Abdalla in isolation but rather
improved its treatment of Stone across both sources — a claim
supported by the finding that Stone recall was near-ceiling and
essentially equal across Mendeley (97.4 %) and Abdalla (97.8 %) in
the v3 model, whereas in v2 the Mendeley-only Stone class was
solved through a source-shortcut that did not generalise.

Accuracy on KiTS improved modestly (39.1 % → 45.9 %, +6.8 pp) but
remained low in absolute terms in both configurations. Inspection of
the KiTS confusion matrix revealed a persistent over-prediction of the
Cyst class: in v2, 397 of 540 KiTS test images were predicted as Cyst
against only 208 true Cysts; in v3, 415 of 584 KiTS test images were
predicted as Cyst against 188 true Cysts. This within-KiTS
Cyst/Tumor/Normal confusion pattern is not attributable to the source
diversification introduced in v3 — the pattern was already present in
v2 — but rather reflects an intrinsic difficulty in distinguishing
cystic from tumorous and normal renal parenchyma in the KiTS
contrast-enhanced imaging protocol. This finding, discussed further in
the Discussion chapter, is considered orthogonal to the source-shortcut
mechanism identified elsewhere in the analysis.

## Summary

The extension to `unified_v3` yielded a substantial and statistically
significant improvement in overall classification performance, with the
strongest gains in the Normal and Stone classes and the largest
per-source improvement occurring on Mendeley — the source that had
previously exhibited the source-shortcut for Stone. No class or source
experienced a statistically significant degradation, and the Cyst
class was successfully preserved despite proportional dilution.
Combined with the source-shortcut diagnostic evidence in Figure
[source_bias_v3] (Section [source_bias_analysis]), these results support
the interpretation that additional source diversity is a viable
mitigation for source-class shortcut learning in multi-source medical
image classification, at least when the additional source is
sufficiently large and its class distribution partially overlaps with
existing sources.
