"""Corrected v3 vs v2 headline comparison, Welch's t-test, Bonferroni."""
import json, os
from pathlib import Path
import numpy as np
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"

V2_TAGS = ["v2_run1", "v2_run1_seed123", "v2_run1_seed456"]  # seeds 42, 123, 456
V3C_TAGS = ["v3c_rn50_baseline_seed42", "v3c_rn50_baseline_seed123", "v3c_rn50_baseline_seed456"]

def gather(tags, label):
    rows = []
    for tag in tags:
        p = KFOLD / tag / "summary.json"
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        s = json.load(open(p))
        for fr in s["fold_results"]:
            rows.append({
                "accuracy": fr["test_accuracy"],
                "macro_f1": fr["test_macro_f1"],
                "cyst_f1": fr["per_class"]["Cyst"]["f1"],
                "normal_f1": fr["per_class"]["Normal"]["f1"],
                "tumor_f1": fr["per_class"]["Tumor"]["f1"],
                "stone_f1": fr["per_class"]["Stone"]["f1"],
            })
    print(f"{label}: n={len(rows)}")
    return rows

v2 = gather(V2_TAGS, "v2 baseline")
v3c = gather(V3C_TAGS, "corrected v3")

metrics = ["accuracy", "macro_f1", "cyst_f1", "normal_f1", "tumor_f1", "stone_f1"]
n_tests = len(metrics)
alpha_bonf = 0.05 / n_tests
print(f"\nBonferroni: {n_tests} tests, alpha_corrected = {alpha_bonf:.4f}\n")
print(f"{'Metric':<12} {'v2 mean':>14} {'v3c mean':>14} {'Delta':>10} {'p':>10} {'sig':>6}")
print("-" * 76)
for m in metrics:
    v2v = np.array([r[m] for r in v2])
    v3v = np.array([r[m] for r in v3c])
    t, p = stats.ttest_ind(v3v, v2v, equal_var=False)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    bonf = "BONF" if p < alpha_bonf else ""
    print(f"{m:<12} {v2v.mean():>10.4f}±{v2v.std():.4f}  {v3v.mean():>10.4f}±{v3v.std():.4f}  {v3v.mean()-v2v.mean():>+.4f}  {p:>8.4f}  {sig:>3} {bonf}")
