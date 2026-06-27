"""
Aggregate fold-level results across multiple seeds for each of the 6 conditions.

Each condition originally had 5 fold-level results (seed 42).
After multi-seed runs, we have 3 seeds × 5 folds = 15 paired observations.

Output:
  $SCRATCH/kidney-results/multi_seed_aggregate/
    multi_seed_summary.csv         per-condition aggregate stats (n=15)
    multi_seed_significance.csv    pairwise t-tests on n=15
    per_class_f1_n15.csv           per-class F1 means/stds at n=15
"""
import os
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
OUT = SCRATCH / "kidney-results/multi_seed_aggregate"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
SEEDS = [42, 123, 456]

# Map condition -> { seed_42_tag, seed_123_tag, seed_456_tag }
CONDITIONS = {
    "RN50_full":  {42: "v2_run1",         123: "v2_run1_seed123",         456: "v2_run1_seed456"},
    "RN50_roi":   {42: "v2_roi_run1",     123: "v2_roi_run1_seed123",     456: "v2_roi_run1_seed456"},
    "CBAM_full":  {42: "cbam_full_run1",  123: "cbam_full_run1_seed123",  456: "cbam_full_run1_seed456"},
    "CBAM_roi":   {42: "cbam_roi_run1",   123: "cbam_roi_run1_seed123",   456: "cbam_roi_run1_seed456"},
    "ViT_full":   {42: "vit_full_run1",   123: "vit_full_run1_seed123",   456: "vit_full_run1_seed456"},
    "ViT_roi":    {42: "vit_roi_run1",    123: "vit_roi_run1_seed123",    456: "vit_roi_run1_seed456"},
}


def load_fold_metrics(tag):
    """Returns list of dicts (one per fold) with per-fold metrics."""
    path = SCRATCH / "kidney-results/kfold" / tag / "summary.json"
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        s = json.load(f)
    out = []
    for fr in s["fold_results"]:
        row = {
            "accuracy": fr["test_accuracy"],
            "macro_f1": fr["test_macro_f1"],
            "macro_auc": fr["test_macro_auc"],
        }
        for c in CLASSES:
            row[f"f1_{c}"] = fr["per_class"][c]["f1"]
        out.append(row)
    return out


# Load all fold-level data
print("Loading fold-level results for all conditions × seeds...\n")
per_condition_metrics = {}  # condition -> list of 15 dicts (5 folds × 3 seeds)
missing = []
for cond, seed_tags in CONDITIONS.items():
    rows = []
    for seed in SEEDS:
        tag = seed_tags[seed]
        folds = load_fold_metrics(tag)
        if folds is None:
            missing.append((cond, seed, tag))
            continue
        for fold_idx, fold_data in enumerate(folds):
            rows.append({"seed": seed, "fold": fold_idx, **fold_data})
    per_condition_metrics[cond] = rows
    print(f"  {cond:10s}: {len(rows)} fold-level results")

if missing:
    print(f"\nWARNING: {len(missing)} missing runs:")
    for cond, seed, tag in missing:
        print(f"  {cond} seed={seed} → {tag}")
    print("Some conditions will have fewer than 15 observations.\n")

# Per-condition summary: mean ± std over all available fold-level results
METRICS = ["accuracy", "macro_f1", "macro_auc"] + [f"f1_{c}" for c in CLASSES]

summary_rows = []
for cond, rows in per_condition_metrics.items():
    if not rows:
        continue
    df = pd.DataFrame(rows)
    summary_row = {"condition": cond, "n_observations": len(df)}
    for m in METRICS:
        summary_row[f"{m}_mean"] = float(df[m].mean())
        summary_row[f"{m}_std"] = float(df[m].std(ddof=1))
    summary_rows.append(summary_row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "multi_seed_summary.csv", index=False)

print("\n" + "=" * 70)
print("MULTI-SEED SUMMARY (n=15 per condition where complete)")
print("=" * 70)
print(f"{'Condition':<12} {'n':>4} {'Acc (mean±std)':>20} {'F1 (mean±std)':>20} {'AUC (mean±std)':>20}")
print("-" * 76)
for _, row in summary_df.iterrows():
    print(f"{row['condition']:<12} {int(row['n_observations']):>4} "
          f"{row['accuracy_mean']*100:>8.2f}% ± {row['accuracy_std']*100:>5.2f}%  "
          f"{row['macro_f1_mean']:>8.3f} ± {row['macro_f1_std']:>5.3f}    "
          f"{row['macro_auc_mean']:>8.3f} ± {row['macro_auc_std']:>5.3f}")

# Pairwise significance with n_observations samples
pairs = list(combinations(list(CONDITIONS.keys()), 2))
n_pairs = len(pairs)
ALPHA = 0.05
BONF_ALPHA = ALPHA / n_pairs
print(f"\nPaired t-tests on aggregated multi-seed results")
print(f"  Alpha (uncorrected): {ALPHA}")
print(f"  Alpha (Bonferroni for {n_pairs} comparisons): {BONF_ALPHA:.5f}\n")

all_sig_rows = []
for metric in METRICS:
    rows = []
    for a, b in pairs:
        if not per_condition_metrics.get(a) or not per_condition_metrics.get(b):
            continue
        da = pd.DataFrame(per_condition_metrics[a]).sort_values(["seed", "fold"])
        db = pd.DataFrame(per_condition_metrics[b]).sort_values(["seed", "fold"])
        # Pair by (seed, fold) — that's the same patient-grouped CV partition
        # used for both conditions
        merged = da.merge(db, on=["seed", "fold"], suffixes=("_a", "_b"))
        if len(merged) < 3:
            continue
        va = merged[f"{metric}_a"].values
        vb = merged[f"{metric}_b"].values
        diffs = va - vb
        if np.allclose(diffs, 0):
            continue
        t_stat, p_val = stats.ttest_rel(va, vb)
        rows.append({
            "metric": metric,
            "A": a, "B": b,
            "n_pairs": len(merged),
            "mean_A": float(va.mean()),
            "mean_B": float(vb.mean()),
            "mean_diff": float(diffs.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "sig_uncorrected": "yes" if p_val < ALPHA else "no",
            "sig_bonferroni": "yes" if p_val < BONF_ALPHA else "no",
        })
    if rows:
        df_metric = pd.DataFrame(rows)
        sig_count_raw = (df_metric["sig_uncorrected"] == "yes").sum()
        sig_count_bonf = (df_metric["sig_bonferroni"] == "yes").sum()
        print(f"  {metric:>14}: {sig_count_raw:>2}/{len(df_metric)} at α=0.05 | "
              f"{sig_count_bonf:>2}/{len(df_metric)} after Bonferroni")
        if sig_count_bonf > 0:
            for _, r in df_metric[df_metric["sig_bonferroni"] == "yes"].iterrows():
                direction = ">" if r["mean_diff"] > 0 else "<"
                print(f"      {r['A']} {direction} {r['B']} "
                      f"(diff={r['mean_diff']:+.4f}, p={r['p_value']:.5f})")
        all_sig_rows.extend(rows)

if all_sig_rows:
    pd.DataFrame(all_sig_rows).to_csv(OUT / "multi_seed_significance.csv", index=False)
    print(f"\nSaved: {OUT / 'multi_seed_significance.csv'}")

print(f"\nFiles saved to: {OUT}")
print(f"  - multi_seed_summary.csv")
print(f"  - multi_seed_significance.csv")