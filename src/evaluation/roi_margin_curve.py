"""
Plot per-class F1 against ROI dilation margin.

This produces the single figure that turns the Stone/ROI trade-off from an
observation into a tested mechanism. If the anatomical explanation is right,
Stone F1 should rise monotonically with margin (calyceal and pelvic stones are
progressively brought back inside the crop) while Cyst F1 falls slowly (the
crop reverts toward a full-image view and reintroduces background), giving a
crossing point somewhere in the middle.

Report the crossing margin as the empirical answer to "how tight should the
kidney ROI be?" -- a concrete, transferable recommendation.

Usage
-----
python -m src.evaluation.roi_margin_curve \
    --tags roi_margin_000 roi_margin_015 roi_margin_025 roi_margin_040 \
    --margins 0.00 0.15 0.25 0.40
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
COLOURS = {"Normal": "#4C72B0", "Cyst": "#DD8452",
           "Tumor": "#55A868", "Stone": "#C44E52"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--margins", nargs="+", type=float, required=True)
    args = ap.parse_args()
    if len(args.tags) != len(args.margins):
        raise SystemExit("--tags and --margins must be the same length")

    scratch = Path(os.environ["SCRATCH"])
    outdir = scratch / "kidney-results/comparison/roi_margin"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, margin in zip(args.tags, args.margins):
        path = scratch / "kidney-results/kfold" / tag / "summary.json"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        with open(path) as f:
            s = json.load(f)
        agg = s["aggregate"]
        folds = s["fold_results"]
        row = {"tag": tag, "margin": margin,
               "accuracy": agg["accuracy_mean"], "accuracy_std": agg["accuracy_std"],
               "macro_f1": agg["macro_f1_mean"], "macro_f1_std": agg["macro_f1_std"]}
        for c in CLASSES:
            vals = [fr["per_class"][c]["f1"] for fr in folds]
            row[f"f1_{c}"] = float(np.mean(vals))
            row[f"f1_{c}_std"] = float(np.std(vals))
        rows.append(row)

    if not rows:
        raise SystemExit("No results found.")

    df = pd.DataFrame(rows).sort_values("margin").reset_index(drop=True)
    df.to_csv(outdir / "roi_margin_sweep.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for c in CLASSES:
        ax1.errorbar(df["margin"], df[f"f1_{c}"], yerr=df[f"f1_{c}_std"],
                     marker="o", ms=5, capsize=3, lw=1.6,
                     color=COLOURS[c], label=c)
    ax1.set_xlabel("ROI dilation margin (fraction of bounding-box side)")
    ax1.set_ylabel("F1 (5-fold mean ± SD)")
    ax1.set_title("Per-class F1 vs ROI margin")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.errorbar(df["margin"], df["macro_f1"], yerr=df["macro_f1_std"],
                 marker="s", ms=5, capsize=3, lw=1.8, color="black",
                 label="Macro-F1")
    ax2.errorbar(df["margin"], df["accuracy"], yerr=df["accuracy_std"],
                 marker="^", ms=5, capsize=3, lw=1.8, color="0.5",
                 label="Accuracy")
    ax2.set_xlabel("ROI dilation margin")
    ax2.set_ylabel("Score (5-fold mean ± SD)")
    ax2.set_title("Overall performance vs ROI margin")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / "roi_margin_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- locate the Stone/Cyst crossing, if there is one ----
    stone, cyst, m = df["f1_Stone"].values, df["f1_Cyst"].values, df["margin"].values
    diff = stone - cyst
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    print()
    if len(sign_change):
        i = sign_change[0]
        # linear interpolation between the bracketing margins
        x = m[i] + (m[i + 1] - m[i]) * abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
        print(f"Stone/Cyst F1 crossing at margin ≈ {x:.3f}")
    else:
        print("No Stone/Cyst crossing within the swept range. Either extend the "
              "sweep or report that the trade-off does not invert.")

    best = df.loc[df["macro_f1"].idxmax()]
    print(f"Best macro-F1 at margin {best['margin']:.2f} "
          f"({best['macro_f1']:.4f} ± {best['macro_f1_std']:.4f})")
    print(f"Stone F1 recovery from margin 0: "
          f"{df['f1_Stone'].max() - df['f1_Stone'].iloc[0]:+.4f}")
    print(f"Cyst F1 cost over the same range:  "
          f"{df['f1_Cyst'].iloc[-1] - df['f1_Cyst'].iloc[0]:+.4f}")
    print(f"\nSaved: {outdir / 'roi_margin_curve.png'}")


if __name__ == "__main__":
    main()
