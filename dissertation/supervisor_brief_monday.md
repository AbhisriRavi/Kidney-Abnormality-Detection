Kidney Abnormality Detection — Status Update

Abhisri Ravi · MSc Advanced Computer Science (AI) · Supervisor: Dr Samson Fabiyi

Prepared for meeting on Monday

Summary

Complete experimental matrix is now done. Three architectures × two input regions = six trained models, all on patient-grouped 5-fold CV across a four-source dataset. Source-bias diagnostics performed on every model. Ready to begin writing.

What's been done since the last meeting

    Multi-source dataset built: KiTS23 + Mendeley + KAUH + TCGA-KIRC. 8,924 images across 561 patients. All classes multi-source except Stone (no public alternative).

    Patient-grouped 5-fold CV pipeline: zero patient leakage between folds verified.

    Six trained models: Plain ResNet50, CBAM-ResNet50, ViT-Base — each on full-image and ROI inputs.

    Source-bias diagnostics: run on all six conditions.

    Matched-subset comparison: removes training-set-size confound from ROI vs full comparison.

Headline numbers (5-fold CV, mean ± std)

Architecture	Region	Accuracy	Macro-F1	Macro-AUC

Plain ResNet50	Full	73.6% ± 1.4%	0.715 ± 0.018	0.918 ± 0.020

Plain ResNet50	ROI	71.8% ± 2.6%	0.706 ± 0.025	0.899 ± 0.036

CBAM-ResNet50	Full	68.9% ± 5.1%	0.645 ± 0.071	0.890 ± 0.030

CBAM-ResNet50	ROI	66.4% ± 3.8%	0.652 ± 0.030	0.885 ± 0.023

ViT-Base	Full	71.7% ± 4.6%	0.696 ± 0.053	0.900 ± 0.044

ViT-Base	ROI	68.6% ± 3.5%	0.662 ± 0.037	0.902 ± 0.015

Per-class F1 — the architecture-independent ROI trade-off

Class	RN50-Full	RN50-ROI	CBAM-Full	CBAM-ROI	ViT-Full	ViT-ROI

Normal	0.698	0.635	0.588	0.515	0.677	0.576

Cyst	0.479	0.629	0.434	0.585	0.481	0.603

Tumor	0.810	0.823	0.801	0.807	0.793	0.831

Stone	0.872	0.738	0.757	0.701	0.832	0.638

Three architectures, same pattern: ROI improves Cyst (+0.12 to +0.15 F1) and Tumor, degrades Stone (−0.06 to −0.19).

Five findings to discuss

    Patient-level leakage in the Kaggle CT Kidney Dataset inflates accuracy from honest ~38% to apparent 99%. Documented via cross-dataset evaluation on KiTS.

    Multi-source training improves performance substantially (v1: 58.5% accuracy / F1 0.55 → v2: 73.6% / F1 0.72) but does not eliminate source-shortcut learning by itself.

    Region focusing (RQ1) is a class-dependent trade-off across all three architectures:

        Cyst F1: +0.12 to +0.15 (intra-kidney pathology benefits)

        Tumor F1: +0.01 to +0.04 (modest consistent gain)

        Stone F1: −0.06 to −0.19 (stones often sit in the calyces/ureter, cropped out by kidney-bounded ROI)

        Anatomically interpretable; consistent across three substantially different architectures.

    Architectural complexity does not help on this dataset. Plain ResNet50 outperforms both CBAM-ResNet50 (attention) and ViT-Base (transformer) on mean accuracy, F1, and AUC. Plain ResNet50 also has the tightest cross-fold std (1.4% vs 4.6–5.1%).

    Source-shortcut is architecture-independent. The dominant source-class mapping (Mendeley→Normal/Stone, KAUH→Cyst, TCGA→Tumor) is essentially identical across all three architectures. Suggests source-bias is a property of the data structure, not the model.

Questions for the meeting

    Framing of finding 4: should I frame "simpler model wins" as the headline empirical result, or as a secondary observation?

    Stone ROI degradation: do I treat this as a class-specific limitation, or do I propose and test an alternative ROI strategy (e.g. dilated bounding box) before writing? Estimated 2 days for the latter.

    Source-shortcut as future work: would you prefer the dissertation to (a) document it as a finding and propose adversarial domain adaptation as future work, or (b) actually run a DANN-style experiment now? Estimated 1 week for the latter.

    Dissertation structure: one chapter per finding (5 chapters), or conventional Introduction / Related Work / Methods / Results / Discussion / Conclusion structure?

Proposed timeline (to end-August deadline)

Week	Activity

W1	Outline, Introduction, Methodology (data + segmentation)

W2	Methodology (classifiers), Related Work

W3	Results chapter — heavy figures work

W4	Discussion, Conclusion

W5	Polish, supervisor review pass 1

W6	Final revisions, formatting, buffer

Repository

github.com/AbhisriRavi/Kidney-Abnormality-Detection

All code, results, and figures committed and pushed