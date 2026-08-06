"""
Inspect patients whose slices carry more than one class label.

WHY THIS MATTERS
----------------
Patient-level evaluation assumes one label per patient. 25 patients in the v2
manifest violate that. There are two very different explanations and they need
different responses:

  LEGITIMATE  A patient genuinely presents two findings -- a cyst and a stone,
              say -- and different slices were annotated accordingly. In that
              case single-label patient-level evaluation is a simplification
              that must be stated in the methodology, and multi-label
              evaluation is the honest alternative.

  BUG         Patient IDs collide across sources, so two different people are
              being treated as one. That would be a leakage-adjacent problem:
              slices from two patients sharing an ID could straddle a fold
              boundary, which is precisely the failure mode this project set out
              to eliminate.

The distinguishing test is whether the affected patients span more than one
source. A single patient legitimately has slices from one acquisition; an ID
collision typically shows up as the same ID appearing under two source names.

Usage
-----
python -m src.evaluation.check_multilabel_patients \
    --manifest $SCRATCH/kidney-data/processed/unified_v2/manifest_with_folds.csv
"""
import argparse
from pathlib import Path

import pandas as pd

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--show", type=int, default=25,
                    help="How many affected patients to list")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"Manifest: {args.manifest}")
    print(f"  {len(df)} images | {df['patient_id'].nunique()} patients | "
          f"sources {sorted(df['source'].unique())}\n")

    g = df.groupby("patient_id").agg(
        n_slices=("label", "size"),
        n_labels=("label", "nunique"),
        n_sources=("source", "nunique"),
        n_folds=("fold", "nunique"),
        labels=("label", lambda s: ", ".join(sorted(s.unique()))),
        sources=("source", lambda s: ", ".join(sorted(s.unique()))),
        folds=("fold", lambda s: ", ".join(str(x) for x in sorted(s.unique()))),
    )

    multi = g[g["n_labels"] > 1].sort_values("n_slices", ascending=False)
    print(f"Patients with more than one label: {len(multi)}\n")
    if len(multi):
        print(multi.head(args.show).to_string())

    # ---- Diagnosis 1: do any span multiple sources? ----
    cross_source = multi[multi["n_sources"] > 1]
    print(f"\n--- Diagnosis ---")
    print(f"Multi-label patients spanning >1 SOURCE: {len(cross_source)}")
    if len(cross_source):
        print("  => Likely PATIENT ID COLLISION across datasets. Two different "
              "people share an ID. Fix by prefixing patient_id with the source "
              "name in the manifest builder, then rebuild folds.")
        print(cross_source.to_string())
    else:
        print("  => No cross-source IDs. Consistent with genuine multi-pathology "
              "patients rather than an ID collision.")

    # ---- Diagnosis 2: does any patient straddle a fold boundary? ----
    straddle = g[g["n_folds"] > 1]
    print(f"\nPatients appearing in >1 FOLD: {len(straddle)}")
    if len(straddle):
        print("  => PATIENT LEAKAGE. Slices from one patient are on both sides "
              "of a fold boundary. This must be fixed before any result is "
              "reported.")
        print(straddle.to_string())
    else:
        print("  => Zero. Patient grouping is intact; no fold leakage.")

    # ---- Label co-occurrence, for the methodology chapter ----
    if len(multi):
        print("\n--- Label pairs among multi-label patients ---")
        pairs = multi["labels"].value_counts()
        print(pairs.to_string())
        print("\nClinically plausible combinations (e.g. Cyst + Stone, "
              "Normal + Stone) support treating these as genuine "
              "multi-pathology cases. Implausible ones (e.g. Tumor + Stone "
              "across different sources) point at a manifest problem.")

    affected = int(multi["n_slices"].sum()) if len(multi) else 0
    print(f"\nSlices affected: {affected} of {len(df)} "
          f"({100 * affected / len(df):.2f}%)")
    print(f"Patients affected: {len(multi)} of {df['patient_id'].nunique()} "
          f"({100 * len(multi) / df['patient_id'].nunique():.2f}%)")


if __name__ == "__main__":
    main()
