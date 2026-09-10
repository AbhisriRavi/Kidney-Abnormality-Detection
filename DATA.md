# Data sources

No imaging data is redistributed through this repository. Every source must be
obtained directly from its provider under that provider's own terms. This file
records what was used, under what conditions, and how to acquire it.

The corrected primary dataset (v3c) comprises four sources: KiTS23, KAUH,
TCGA-KIRC and Abdalla. Two further collections were acquired and are documented
here because they appear in the dissertation, but neither forms part of v3c.

## Licensing and conditions

| Source | Licence or basis | Conditions observed |
|---|---|---|
| KiTS23 | CC BY-NC-SA 4.0 | Non-commercial academic use and attribution. |
| KAUH | Research dataset associated with publication | Attribution to the source publication; no redistribution through this repository. |
| TCGA-KIRC | TCIA data-use terms | Attribution to TCIA and the TCGA Research Network; no re-identification. |
| Abdalla | Open research dataset associated with data article | Attribution to the dataset publication; no redistribution through this repository. |
| Mendeley-labelled copy | Same underlying cohort as Abdalla | Excluded after the duplication finding. |
| Kaggle CT Kidney | Public research dataset | Used only for the leakage investigation; not part of v3c. |

This table reproduces Table A.1 of the dissertation.

## Acquisition

Acquisition dates are recorded per source below. Where a date is marked TBC it
should be filled in from the author's acquisition records rather than from file
timestamps.

### KiTS23

Kidney Tumour Segmentation Challenge. Obtained from github.com/neheller/kits23.
Acquired: TBC.

Run `scripts/download_kits.sh`, then `scripts/extract_kits_masks.sh` to derive
the segmentation masks used for ROI extraction.

### KAUH

King Abdullah University Hospital kidney CT dataset, described by Alzu'bi et al.
Publication DOI 10.1155/2022/3861161. Obtained from the repository associated
with that publication. Acquired: TBC.

Downloaded manually under the source access conditions. No script is provided,
because access is not automatable.

### TCGA-KIRC

The Cancer Genome Atlas Kidney Renal Clear Cell Carcinoma collection, distributed
through The Cancer Imaging Archive. Acquired: TBC.

Run `scripts/download_tcga_kirc.sh` followed by `scripts/convert_tcga.sh`, which
converts the DICOM series to PNG. Download manifests are tracked under
`metadata/tcga/`; per-series metadata is regenerable via
`src/data/tcga/download_full.py` and is deliberately not committed, since it
carries per-patient demographic columns supplied by TCIA.

### Abdalla

Axial CT imaging dataset for kidney stone detection. Publication DOI
10.1016/j.dib.2025.111446, which links to the dataset. Acquired: TBC.

Run `scripts/extract_abdalla.sh`.

### Mendeley kidney-stone copy (excluded)

Acquired during the project and subsequently excluded. Provenance checking showed
it to be another processed representation of the Abdalla Normal/Stone cohort
rather than an independent source, overlapping on 201 of 201 patients. Treating
the two as independent produced an apparent improvement of +8.7 percentage
points; after exclusion the same comparison gives +1.4 points and is not
statistically significant.

Exact Mendeley record: TBC.

This source is not present in the v3c manifest. Results generated before the
correction are retained under `dissertation/figures/` for history only.

### Kaggle CT Kidney dataset (leakage analysis only)

Provider: Kaggle user nazmul0087, dataset "CT KIDNEY DATASET: Normal-Cyst-Tumor
and Stone". 12,446 images. Acquired: TBC.

Used solely to quantify patient-level leakage, where image-level partitioning
produced above 99% accuracy against roughly 38% on independently acquired
patients. Not part of the corrected primary dataset.

## Patient identifiers

Patient identifiers in the manifests are pseudonymous labels constructed from
source directory structures or filenames, for example `kits_00123` or
`abdalla_P037`. They exist to enforce patient-grouped partitioning and provide no
route back to an identified person. All imaging was released in de-identified
form by its original providers; no re-identification was attempted.

## After acquisition

Build the manifests with `src/data/build_v3_corrected.py` and
`src/data/build_v3c_roi_manifest.py`, then verify with `pytest tests/ -v`. The
tests check patient-disjoint folds, manifest schema, and that the resulting
dataset matches the sizes reported in the dissertation: 8,924 images from 561
patients for the full-image condition, 5,769 from 463 after ROI extraction.
