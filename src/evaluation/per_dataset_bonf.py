"""
Per-dataset architecture comparison with Bonferroni correction.
For each dataset, compares 3 architectures pairwise on multiple metrics.
"""
import json, os
from pathlib import Path
import numpy as np
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
PD_ROOT = SCRATCH / "kidney-results/per_dataset"

DATASETS_CLASSES = {
    "mendeley": ["Normal", "Cyst", "Tumor", "Stone"],
    "abdalla": ["Normal", "Stone"],
    "kits": ["Normal", "Cyst", "Tumor"],
}
ARCHES = ["resnet50", "efficientnet_b0", "densenet121"]
SEEDS = [42, 123, 456]

def gather(dataset, arch):
    rows = []
    for seed in SEEDS:
        tag = f"pd_{dataset}_{arch}_seed{seed}"
        p = PD_ROOT / tag / "summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        for fr in s["fold_results"]:
            r = {"accuracy": fr["test_accuracy"], "macro_f1": fr["test_macro_f1"]}
            for c in DATASETS_CLASSES[dataset]:
                r[f"{c.lower()}_f1"] = fr["per_class"][c]["f1"]
            rows.append(r)
    return rows

for dataset in ["mendeley", "abdalla", "kits"]:
    classes = DATASETS_CLASSES[dataset]
    metrics = ["accuracy", "macro_f1"] + [f"{c.lower()}_f1" for c in classes]
    print("\n" + "=" * 78)
    print(f"DATASET: {dataset.upper()} ({len(classes)}-class)")
    print("=" * 78)

    data = {arch: gather(dataset, arch) for arch in ARCHES}
    for arch in ARCHES:
        print(f"  {arch}: n={len(data[arch])} fold observations")

    # Aggregate table
    print(f"\n{'Metric':<12} " + "  ".join(f"{a:>22}" for a in ARCHES))
    for m in metrics:
        line = f"{m:<12} "
        for arch in ARCHES:
            vals = np.array([r[m] for r in data[arch]])
            line += f" {vals.mean():.4f}±{vals.std(ddof=1):.4f}   "
        print(line)

    # Pairwise Welch t-test with Bonferroni
    pairs = [("resnet50", "efficientnet_b0"), ("resnet50", "densenet121"),
             ("efficientnet_b0", "densenet121")]
    n_tests = len(pairs) * len(metrics)
    alpha_bonf = 0.05 / n_tests
    print(f"\nBonferroni: {n_tests} tests, alpha_corrected = {alpha_bonf:.4f}")
    print(f"\n{'Comparison':<32} {'Metric':<12} {'Delta':>10} {'p':>10} {'sig':>6}")
    print("-" * 80)
    for a, b in pairs:
        for m in metrics:
            va = np.array([r[m] for r in data[a]])
            vb = np.array([r[m] for r in data[b]])
            t, p = stats.ttest_ind(va, vb, equal_var=False)
            delta = va.mean() - vb.mean()
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            bonf = "BONF" if p < alpha_bonf else ""
            print(f"{a} vs {b:<20} {m:<12} {delta:>+.4f}  {p:>8.4f}  {sig:>3} {bonf}")
        print()

    # Per-class support summary
    print(f"\nPer-class F1 mean +- SD (n=15 per arch):")
    hdr = f"  {'Class':<10}"
    for arch in ARCHES:
        hdr += f" {arch:>22}"
    print(hdr)
    for c in classes:
        line = f"  {c:<10}"
        for arch in ARCHES:
            key = f"{c.lower()}_f1"
            vals = np.array([r[key] for r in data[arch]])
            line += f" {vals.mean():.4f}±{vals.std(ddof=1):.4f}   "
        print(line)
