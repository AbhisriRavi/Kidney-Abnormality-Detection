"""
v4 mitigation comparison + v3c ROI vs full-image + v4 vs v3c corrected.
"""
import json, os
from pathlib import Path
import numpy as np
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"

def gather(tags):
    rows = []
    for tag in tags:
        p = KFOLD / tag / "summary.json"
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        s = json.load(open(p))
        for fr in s["fold_results"]:
            r = {
                "accuracy": fr["test_accuracy"],
                "macro_f1": fr["test_macro_f1"],
                "cyst_f1": fr["per_class"]["Cyst"]["f1"],
                "normal_f1": fr["per_class"]["Normal"]["f1"],
                "tumor_f1": fr["per_class"]["Tumor"]["f1"],
                "stone_f1": fr["per_class"]["Stone"]["f1"],
            }
            rows.append(r)
    return rows

# v3c baseline for comparison
v3c_bl = gather([f"v3c_rn50_baseline_seed{s}" for s in [42, 123, 456]])

# v4 configs
v4_configs = {
    "baseline": [f"v4_rn50_baseline_seed{s}" for s in [42, 123, 456]],
    "focal": [f"v4_focal_seed{s}" for s in [42, 123, 456]],
    "dann": [f"v4_dann_seed{s}" for s in [42, 123, 456]],
    "mh_kauh": [f"v4_mh_kauh_seed{s}" for s in [42, 123, 456]],
    "mh_kits": [f"v4_mh_kits_seed{s}" for s in [42, 123, 456]],
    "mh_both": [f"v4_mh_both_seed{s}" for s in [42, 123, 456]],
}
v4 = {k: gather(v) for k, v in v4_configs.items()}

# v3c ROI baseline
v3c_roi = gather([f"v3c_roi_baseline_seed{s}" for s in [42, 123, 456]])

metrics = ["accuracy", "macro_f1", "cyst_f1", "normal_f1", "tumor_f1", "stone_f1"]

print("=" * 90)
print("PART A: v4 mitigations vs v4 baseline (Bonferroni)")
print("=" * 90)
n_tests = 5 * len(metrics)  # 5 comparisons x 6 metrics
alpha_bonf = 0.05 / n_tests
print(f"Bonferroni: {n_tests} tests, alpha_corrected = {alpha_bonf:.4f}")

v4_baseline = v4["baseline"]
for name in ["focal", "dann", "mh_kauh", "mh_kits", "mh_both"]:
    v = v4[name]
    print(f"\n--- v4_{name} vs v4_baseline (n={len(v)} vs {len(v4_baseline)}) ---")
    for m in metrics:
        b = np.array([r[m] for r in v4_baseline])
        vv = np.array([r[m] for r in v])
        t, p = stats.ttest_ind(vv, b, equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        bonf = "BONF" if p < alpha_bonf else ""
        print(f"  {m:<12}: base={b.mean():.4f}±{b.std(ddof=1):.4f}  var={vv.mean():.4f}±{vv.std(ddof=1):.4f}  d={vv.mean()-b.mean():+.4f}  p={p:.4f} {sig} {bonf}")

print("\n" + "=" * 90)
print("PART B: v4 baseline vs v3c baseline (does real Mendeley help?)")
print("=" * 90)
n_tests = 6
alpha_bonf = 0.05 / n_tests
print(f"Bonferroni: {n_tests} tests, alpha_corrected = {alpha_bonf:.4f}")
for m in metrics:
    b = np.array([r[m] for r in v3c_bl])
    v = np.array([r[m] for r in v4_baseline])
    t, p = stats.ttest_ind(v, b, equal_var=False)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    bonf = "BONF" if p < alpha_bonf else ""
    print(f"  {m:<12}: v3c={b.mean():.4f}±{b.std(ddof=1):.4f}  v4={v.mean():.4f}±{v.std(ddof=1):.4f}  d={v.mean()-b.mean():+.4f}  p={p:.4f} {sig} {bonf}")

print("\n" + "=" * 90)
print("PART C: v3c ROI vs v3c full-image baseline (does ROI help on corrected data?)")
print("=" * 90)
n_tests = 6
alpha_bonf = 0.05 / n_tests
print(f"Bonferroni: {n_tests} tests, alpha_corrected = {alpha_bonf:.4f}")
for m in metrics:
    b = np.array([r[m] for r in v3c_bl])
    r_ = np.array([r[m] for r in v3c_roi])
    t, p = stats.ttest_ind(r_, b, equal_var=False)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    bonf = "BONF" if p < alpha_bonf else ""
    print(f"  {m:<12}: full={b.mean():.4f}±{b.std(ddof=1):.4f}  ROI={r_.mean():.4f}±{r_.std(ddof=1):.4f}  d={r_.mean()-b.mean():+.4f}  p={p:.4f} {sig} {bonf}")
