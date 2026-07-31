"""
Full corrected v3 comparison: baseline vs 6 mitigation interventions.
Welch's t-test with Bonferroni across 6 configs × 6 metrics = 36 tests.
"""
import json, os
from pathlib import Path
import numpy as np
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"

BASELINE_TAGS = [f"v3c_rn50_baseline_seed{s}" for s in [42, 123, 456]]
CONFIGS = {
    "focal":     [f"v3c_focal_seed{s}" for s in [42, 123, 456]],
    "dann":      [f"v3c_dann_seed{s}"  for s in [42, 123, 456]],
    "mh_kauh":   [f"v3c_mh_kauh_seed{s}" for s in [42, 123, 456]],
    "mh_kits":   [f"v3c_mh_kits_seed{s}" for s in [42, 123, 456]],
    "mh_both":   [f"v3c_mh_both_seed{s}" for s in [42, 123, 456]],
}
# TTA is in a different location
TTA_SUMMARY = SCRATCH / "kidney-results/v3c_tta/tta_summary.json"

def gather_from_tags(tags):
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
    return rows

def gather_tta():
    s = json.load(open(TTA_SUMMARY))
    return [{
        "accuracy": fr["test_accuracy"],
        "macro_f1": fr["test_macro_f1"],
        "cyst_f1": fr["per_class"]["Cyst"]["f1"],
        "normal_f1": fr["per_class"]["Normal"]["f1"],
        "tumor_f1": fr["per_class"]["Tumor"]["f1"],
        "stone_f1": fr["per_class"]["Stone"]["f1"],
    } for fr in s["fold_results"]]

baseline = gather_from_tags(BASELINE_TAGS)
print(f"Baseline: n={len(baseline)}")

variants = {name: gather_from_tags(tags) for name, tags in CONFIGS.items()}
variants["tta"] = gather_tta()
for name, rows in variants.items():
    print(f"{name}: n={len(rows)}")

metrics = ["accuracy", "macro_f1", "cyst_f1", "normal_f1", "tumor_f1", "stone_f1"]
n_tests = len(metrics) * len(variants)
alpha_bonf = 0.05 / n_tests
print(f"\nBonferroni: {n_tests} tests total (6 configs x 6 metrics), alpha_corrected = {alpha_bonf:.4f}")

print(f"\n{'Config':<10} {'Metric':<12} {'Baseline':>16} {'Variant':>16} {'Delta':>10} {'p':>10} {'sig':>6}")
print("-" * 90)
for vname, vrows in variants.items():
    for m in metrics:
        b = np.array([r[m] for r in baseline])
        v = np.array([r[m] for r in vrows])
        t, p = stats.ttest_ind(v, b, equal_var=False)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        bonf = " BONF" if p < alpha_bonf else ""
        print(f"{vname:<10} {m:<12} {b.mean():>10.4f}±{b.std(ddof=1):.4f}  {v.mean():>10.4f}±{v.std(ddof=1):.4f}  {v.mean()-b.mean():>+.4f}  {p:>8.4f}  {sig:>3}{bonf}")
    print()