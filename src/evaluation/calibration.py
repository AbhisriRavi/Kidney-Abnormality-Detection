"""
Calibration analysis: reliability diagrams, ECE, MCE, Brier score, and
optional temperature scaling.

WHY THIS BELONGS IN THE DISSERTATION
------------------------------------
Accuracy alone does not tell you whether a model is clinically usable. A model
that is 74% accurate and well calibrated supports triage -- you can threshold
its confidence and route uncertain cases to a radiologist. A model that is 74%
accurate but confidently wrong cannot be used that way at all.

Given that the headline accuracies in this project sit in the low 70s, the
calibration question is arguably more decision-relevant than another
percentage point of accuracy, and it costs no GPU time: everything here is
computed from the saved probability vectors.

Metrics
-------
ECE  Expected Calibration Error -- confidence-weighted mean gap between
     predicted confidence and empirical accuracy, over equal-width bins.
MCE  Maximum Calibration Error -- worst bin gap. Catches a model that is
     well calibrated on average but badly overconfident in one region.
Brier (multiclass) -- mean squared error between the one-hot label and the
     full probability vector. A proper scoring rule, so it rewards both
     discrimination and calibration.

Usage
-----
python -m src.evaluation.calibration \
    --conditions v3c_rn50_full v3c_rn50_roi --level patient --temperature-scale
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.patient_level_eval import aggregate_to_patient, CLASSES, PROB_COLS


def expected_calibration_error(conf, correct, n_bins=15):
    """Returns (ECE, MCE, per-bin dataframe)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin_low": lo, "bin_high": hi, "count": 0,
                         "avg_conf": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(avg_conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        rows.append({"bin_low": lo, "bin_high": hi, "count": cnt,
                     "avg_conf": avg_conf, "accuracy": acc, "gap": gap})
    return ece, mce, pd.DataFrame(rows)


def multiclass_brier(probs, y_true, n_classes=4):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(probs, y_true, grid=None):
    """
    Fit a single temperature T on log-probabilities by grid search on NLL.

    Note the caveat for the write-up: temperature should properly be fitted on
    a held-out validation split. Fitting on the same predictions you then report
    gives an optimistic ECE. This function is provided so you can quantify the
    *headroom* -- "calibration error falls from X to Y under an oracle
    temperature" -- which is a legitimate and clearly-labelled statement.
    """
    if grid is None:
        grid = np.linspace(0.5, 5.0, 91)
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        scaled = logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p /= p.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y_true)), y_true], 1e-12, 1.0)))
        if nll < best_nll:
            best_nll, best_t = nll, t
    scaled = logits / best_t
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    p = np.exp(scaled)
    p /= p.sum(axis=1, keepdims=True)
    return best_t, p


def analyse(df, name, n_bins, outdir, temperature_scale=False):
    probs = df[PROB_COLS].values.astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    y_true = df["y_true"].values
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)

    ece, mce, bins = expected_calibration_error(conf, correct, n_bins)
    brier = multiclass_brier(probs, y_true)
    result = {"condition": name, "n": len(df), "accuracy": float(correct.mean()),
              "mean_confidence": float(conf.mean()), "ECE": ece, "MCE": mce,
              "Brier": brier, "overconfidence": float(conf.mean() - correct.mean())}

    if temperature_scale:
        t, p_scaled = fit_temperature(probs, y_true)
        conf_s = p_scaled.max(axis=1)
        correct_s = (p_scaled.argmax(axis=1) == y_true).astype(float)
        ece_s, mce_s, _ = expected_calibration_error(conf_s, correct_s, n_bins)
        result.update({"oracle_T": t, "ECE_after_T": ece_s, "MCE_after_T": mce_s,
                       "Brier_after_T": multiclass_brier(p_scaled, y_true)})

    bins.to_csv(outdir / f"{name}_reliability_bins.csv", index=False)

    # ---- reliability diagram ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.2, 6.2), height_ratios=[3, 1], sharex=True
    )
    valid = bins.dropna(subset=["avg_conf"])
    centres = (valid["bin_low"] + valid["bin_high"]) / 2

    ax1.plot([0, 1], [0, 1], "--", color="0.5", lw=1, label="Perfect calibration")
    ax1.bar(centres, valid["accuracy"], width=1.0 / n_bins * 0.9,
            edgecolor="black", lw=0.6, color="#4C72B0", label="Observed accuracy")
    ax1.plot(centres, valid["avg_conf"], "o-", color="#C44E52", ms=4,
             lw=1.3, label="Mean confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"{name}\nECE={ece:.4f}  MCE={mce:.4f}  Brier={brier:.4f}")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.bar(centres, valid["count"], width=1.0 / n_bins * 0.9,
            color="0.6", edgecolor="black", lw=0.5)
    ax2.set_xlabel("Predicted confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(outdir / f"{name}_reliability.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- selective-prediction curve: the clinically useful figure ----
    order = np.argsort(-conf)
    acc_curve = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(coverage, acc_curve, lw=1.6, color="#4C72B0")
    ax.axhline(correct.mean(), ls="--", color="0.5", lw=1,
               label=f"Full-coverage accuracy = {correct.mean():.3f}")
    ax.set_xlabel("Coverage (fraction of cases auto-reported, most confident first)")
    ax.set_ylabel("Accuracy on retained cases")
    ax.set_title(f"{name} — selective prediction")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"{name}_selective_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy at a few operating points, for the discussion chapter.
    for cov in (0.25, 0.50, 0.75):
        k = max(int(cov * len(order)), 1)
        result[f"acc_at_{int(cov*100)}pct_coverage"] = float(acc_curve[k - 1])

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--level", default="patient", choices=["patient", "slice"])
    ap.add_argument("--agg", default="mean", choices=["mean", "max", "vote"])
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--temperature-scale", action="store_true")
    args = ap.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    pred_dir = scratch / "kidney-results/predictions"
    outdir = scratch / "kidney-results/comparison/calibration"
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for stem in args.conditions:
        path = pred_dir / f"{stem}.csv"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        df = pd.read_csv(path)
        if args.level == "patient":
            df = aggregate_to_patient(df, how=args.agg)
        name = f"{stem}_{args.level}"
        print(f"Analysing {name} ({len(df)} units)...")
        results.append(analyse(df, name, args.n_bins, outdir, args.temperature_scale))

    if results:
        out = pd.DataFrame(results)
        out_csv = outdir / f"calibration_summary_{args.level}.csv"
        out.to_csv(out_csv, index=False)
        print("\n=== Calibration summary ===")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved: {out_csv}")
        print(f"Figures in: {outdir}")


if __name__ == "__main__":
    main()
