"""
Slice-level vs patient-level evaluation.

WHY THIS MATTERS
----------------
Every metric reported so far treats each axial slice as an independent sample.
Patients contribute very different numbers of slices, so pooled slice-level
accuracy is implicitly weighted by scan length: one patient with 200 slices
influences macro-F1 more than twenty patients with 10 slices each.

Patients, not slices, are the independent unit of analysis. This script
aggregates the per-slice probability vectors produced by dump_predictions.py
into one prediction per patient and reports both levels side by side.

Three aggregation rules are provided:
    mean    -- average the softmax vectors across the patient's slices (default;
               the standard choice, robust to a handful of noisy slices)
    max     -- take the highest abnormality probability across slices, mirroring
               clinical practice where one convincing slice is enough to call a
               finding. Implemented as per-class max, then renormalise.
    vote    -- majority vote over per-slice argmax predictions

Usage
-----
python -m src.evaluation.patient_level_eval \
    --predictions v3c_rn50_full v3c_rn50_roi \
    --agg mean
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
PROB_COLS = [f"p_{c}" for c in CLASSES]


def aggregate_to_patient(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Collapse per-slice predictions to one row per (fold, patient_id)."""
    if how == "vote":
        def _vote(g):
            counts = np.bincount(g["y_pred"].values, minlength=len(CLASSES))
            probs = counts / counts.sum()
            return pd.Series(probs, index=PROB_COLS)
        agg = df.groupby(["fold", "patient_id"], sort=False).apply(
            _vote, include_groups=False
        )
    elif how == "max":
        agg = df.groupby(["fold", "patient_id"], sort=False)[PROB_COLS].max()
        agg = agg.div(agg.sum(axis=1).clip(lower=1e-8), axis=0)
    elif how == "mean":
        agg = df.groupby(["fold", "patient_id"], sort=False)[PROB_COLS].mean()
    else:
        raise ValueError(f"Unknown aggregation '{how}'")

    meta = df.groupby(["fold", "patient_id"], sort=False).agg(
        source=("source", "first"),
        y_true=("y_true", "first"),
        n_slices=("y_true", "size"),
        label_consistent=("y_true", lambda s: s.nunique() == 1),
    )

    out = meta.join(agg).reset_index()
    out["y_pred"] = out[PROB_COLS].values.argmax(axis=1)

    inconsistent = (~out["label_consistent"]).sum()
    if inconsistent:
        print(f"  WARNING: {inconsistent} patients have slices with more than one "
              f"label. Using the first label. Check your manifest construction "
              f"if this number is not zero.")
    return out


def score(df: pd.DataFrame, level: str) -> dict:
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    probs = df[PROB_COLS].values

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pcl = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    aucs = {}
    for i, c in enumerate(CLASSES):
        binary = (y_true == i).astype(int)
        aucs[c] = float(roc_auc_score(binary, probs[:, i])) if binary.sum() and \
            binary.sum() < len(binary) else None
    valid = [v for v in aucs.values() if v is not None]
    macro_auc = float(np.mean(valid)) if valid else float("nan")

    return {
        "level": level,
        "n_units": len(df),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        **{f"f1_{c}": pcl[2][i] for i, c in enumerate(CLASSES)},
        "_cm": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))),
    }


def per_fold_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Fold-wise metrics, so downstream tests still have paired fold values."""
    rows = []
    for fold, g in df.groupby("fold"):
        s = score(g, level=f"fold_{fold}")
        s.pop("_cm")
        s["fold"] = fold
        rows.append(s)
    return pd.DataFrame(rows).sort_values("fold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True,
                    help="Prediction CSV stems under kidney-results/predictions/")
    ap.add_argument("--agg", default="mean", choices=["mean", "max", "vote"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    pred_dir = scratch / "kidney-results/predictions"
    outdir = Path(args.outdir) if args.outdir else \
        scratch / "kidney-results/comparison/patient_level"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for stem in args.predictions:
        path = pred_dir / f"{stem}.csv"
        if not path.exists():
            print(f"MISSING: {path} -- run dump_predictions.py for this condition first")
            continue

        print(f"\n=== {stem} ===")
        df = pd.read_csv(path)

        slice_scores = score(df, "slice")
        pat_df = aggregate_to_patient(df, how=args.agg)
        patient_scores = score(pat_df, f"patient_{args.agg}")

        n_pat = pat_df["patient_id"].nunique()
        slices_per = pat_df["n_slices"]
        print(f"  {len(df)} slices -> {n_pat} patients "
              f"(median {slices_per.median():.0f} slices/patient, "
              f"range {slices_per.min()}-{slices_per.max()})")

        for s in (slice_scores, patient_scores):
            print(f"  {s['level']:<16} n={s['n_units']:5d}  "
                  f"acc={s['accuracy']:.4f}  macroF1={s['macro_f1']:.4f}  "
                  f"macroAUC={s['macro_auc']:.4f}")
            print("      per-class F1: " + "  ".join(
                f"{c}={s[f'f1_{c}']:.3f}" for c in CLASSES))
            row = {k: v for k, v in s.items() if not k.startswith("_")}
            row["condition"] = stem
            summary_rows.append(row)

        # Fold-wise patient-level metrics, for paired statistical tests later.
        fold_tbl = per_fold_scores(pat_df)
        fold_tbl.insert(0, "condition", stem)
        fold_tbl.to_csv(outdir / f"{stem}_patient_foldwise.csv", index=False)

        pat_df.to_csv(outdir / f"{stem}_patient_predictions.csv", index=False)

        # Delta table -- the headline number for the dissertation
        d_acc = patient_scores["accuracy"] - slice_scores["accuracy"]
        d_f1 = patient_scores["macro_f1"] - slice_scores["macro_f1"]
        print(f"  Δ patient − slice:  acc {d_acc:+.4f}   macroF1 {d_f1:+.4f}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        cols = ["condition", "level", "n_units", "accuracy", "macro_f1", "macro_auc"] + \
               [f"f1_{c}" for c in CLASSES]
        summary = summary[cols]
        out_csv = outdir / f"slice_vs_patient_{args.agg}.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\n\n=== Combined summary ===")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
