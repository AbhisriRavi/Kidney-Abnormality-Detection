"""
Pairwise statistical significance testing across the 6-condition experimental matrix.

For each pair of conditions and each metric:
  - paired t-test (5 paired fold-level values)
  - mean diff, t-statistic, p-value
  - Bonferroni-corrected significance at alpha=0.05

Output: per-metric CSV files in $SCRATCH/kidney-results/comparison/significance/
"""
import json
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
OUT = SCRATCH / "kidney-results/comparison/significance"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("RN50_full",  "v2_run1"),
    ("RN50_roi",   "v2_roi_run1"),
    ("CBAM_full",  "cbam_full_run1"),
    ("CBAM_roi",   "cbam_roi_run1"),
    ("ViT_full",   "vit_full_run1"),
    ("ViT_roi",    "vit_roi_run1"),
]
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

# Load fold-level results for every condition
fold_data = {}  # name -> dict of metric_name -> [5 values]
for name, tag in CONDITIONS:
    path = SCRATCH / "kidney-results/kfold" / tag / "summary.json"
    with open(path) as f:
        s = json.load(f)
    fr = s["fold_results"]
    data = {
        "accuracy": [r["test_accuracy"] for r in fr],
        "macro_f1": [r["test_macro_f1"] for r in fr],
        "macro_auc": [r["test_macro_auc"] for r in fr],
    }
    for c in CLASSES:
        data[f"f1_{c}"] = [r["per_class"][c]["f1"] for r in fr]
    fold_data[name] = data

# Sanity check
for name in fold_data:
    for m, vals in fold_data[name].items():
        assert len(vals) == 5, f"{name} {m} has {len(vals)} folds, expected 5"

# Pairwise comparisons
pairs = list(combinations([n for n, _ in CONDITIONS], 2))
n_pairs = len(pairs)
print(f"Running paired t-tests for {n_pairs} pairs × {3 + len(CLASSES)} metrics\n")

# Bonferroni-corrected alpha for n_pairs tests at family-wise 0.05
ALPHA = 0.05
BONF_ALPHA = ALPHA / n_pairs
print(f"Alpha (uncorrected): {ALPHA}")
print(f"Alpha (Bonferroni-corrected for {n_pairs} comparisons): {BONF_ALPHA:.5f}\n")

METRICS = ["accuracy", "macro_f1", "macro_auc"] + [f"f1_{c}" for c in CLASSES]

all_rows = []
for metric in METRICS:
    rows = []
    print(f"=== Metric: {metric} ===")
    for a, b in pairs:
        va = np.array(fold_data[a][metric])
        vb = np.array(fold_data[b][metric])
        diffs = va - vb
        mean_diff = float(np.mean(diffs))
        # paired t-test
        t_stat, p_val = stats.ttest_rel(va, vb)
        sig_raw = p_val < ALPHA
        sig_bonf = p_val < BONF_ALPHA
        rows.append({
            "condition_A": a,
            "condition_B": b,
            "mean_A": float(np.mean(va)),
            "mean_B": float(np.mean(vb)),
            "mean_diff (A-B)": mean_diff,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "sig_at_0.05": "yes" if sig_raw else "no",
            "sig_after_Bonferroni": "yes" if sig_bonf else "no",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"sig_{metric}.csv", index=False)

    # Print a compact summary
    sig_pairs_raw = df[df["sig_at_0.05"] == "yes"]
    sig_pairs_bonf = df[df["sig_after_Bonferroni"] == "yes"]
    print(f"  Pairs significant at α=0.05:        {len(sig_pairs_raw)}/{n_pairs}")
    print(f"  Pairs significant after Bonferroni: {len(sig_pairs_bonf)}/{n_pairs}")
    if len(sig_pairs_bonf) > 0:
        print(f"  Bonferroni-significant pairs:")
        for _, r in sig_pairs_bonf.iterrows():
            direction = ">" if r["mean_diff (A-B)"] > 0 else "<"
            print(f"    {r['condition_A']} {direction} {r['condition_B']} "
                  f"(mean_diff={r['mean_diff (A-B)']:+.4f}, p={r['p_value']:.4f})")
    print()
    all_rows.extend([{"metric": metric, **r} for r in rows])

# One master file
master = pd.DataFrame(all_rows)
master.to_csv(OUT / "all_pairwise_significance.csv", index=False)
print(f"Saved per-metric files and master CSV to: {OUT}")
print(f"  Master: all_pairwise_significance.csv ({len(master)} rows)")