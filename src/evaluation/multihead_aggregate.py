"""
Aggregate multi-head results across 3 seeds × 5 folds per variant.
Compare each variant to v3 baseline using Welch's t-test.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import os

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"

BASELINE_TAGS = [
    f"v3_rn50_full_run1_seed{s}" for s in [42, 123, 456, 789, 1011]
]
MH_VARIANTS = {
    "kauh": [f"v3_mh_kauh_seed{s}"  for s in [42, 123, 456]],
    "kits": [f"v3_mh_kits_seed{s}"  for s in [42, 123, 456]],
    "both": [f"v3_mh_both_seed{s}"  for s in [42, 123, 456]],
}

def gather(tags):
    """Return list of dicts, one per fold, with metrics."""
    rows = []
    for tag in tags:
        s = KFOLD / tag / "summary.json"
        if not s.exists():
            continue
        summary = json.load(open(s))
        for fr in summary["fold_results"]:
            rows.append({
                "tag": tag,
                "fold": fr["fold"],
                "accuracy": fr["test_accuracy"],
                "macro_f1": fr["test_macro_f1"],
                "cyst_f1": fr["per_class"]["Cyst"]["f1"],
                "stone_f1": fr["per_class"]["Stone"]["f1"],
                "normal_f1": fr["per_class"]["Normal"]["f1"],
                "tumor_f1": fr["per_class"]["Tumor"]["f1"],
            })
    return pd.DataFrame(rows)

print("=" * 80)
print("AGGREGATE COMPARISON: v3 baseline vs multi-head variants")
print("=" * 80)

baseline = gather(BASELINE_TAGS)
print(f"\nBaseline: n={len(baseline)} fold observations from {len(BASELINE_TAGS)} seeds")

variants = {}
for name, tags in MH_VARIANTS.items():
    variants[name] = gather(tags)
    print(f"Multi-head {name}: n={len(variants[name])} fold observations from {len(tags)} seeds")

metrics = ["accuracy", "macro_f1", "cyst_f1", "normal_f1", "tumor_f1", "stone_f1"]

# Count total tests for Bonferroni
n_tests = len(metrics) * len(variants)
alpha_bonf = 0.05 / n_tests
print(f"\nBonferroni: {n_tests} tests, α_corrected = {alpha_bonf:.4f}")

print(f"\n{'Variant':<8} {'Metric':<10} {'Baseline':>18} {'Variant':>18} {'Δ':>10} {'p (Welch)':>12} {'sig':>6}")
print("-" * 90)
for vname, vdf in variants.items():
    for m in metrics:
        b_vals = baseline[m].values
        v_vals = vdf[m].values
        b_mean, b_sd = b_vals.mean(), b_vals.std(ddof=1)
        v_mean, v_sd = v_vals.mean(), v_vals.std(ddof=1)
        delta = v_mean - b_mean
        t, p = stats.ttest_ind(v_vals, b_vals, equal_var=False)
        sig = ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        bonf = "BONF" if p < alpha_bonf else ""
        print(f"{vname:<8} {m:<10} {b_mean:.4f}±{b_sd:.4f}  {v_mean:.4f}±{v_sd:.4f}  {delta:+.4f}  {p:.4f}  {sig:>3} {bonf}")
    print()
