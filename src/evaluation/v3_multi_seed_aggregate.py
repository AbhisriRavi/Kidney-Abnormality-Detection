"""
Aggregate v3 RN50-full results across 5 seeds × 5 folds = 25 observations,
and run paired significance tests against the corresponding v2 results.

Uses seeds [42, 123, 456, 789, 1011] where available. Missing seeds are skipped
and n_observations is reported honestly.

Outputs:
  $SCRATCH/kidney-results/v3_analysis/
    v3_multi_seed_summary.csv         per-condition aggregates
    v3_vs_v2_significance.csv         paired t-tests, Bonferroni-corrected
    v3_vs_v2_headline.txt             human-readable summary
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"
OUT = SCRATCH / "kidney-results/v3_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
SEEDS = [42, 123, 456, 789, 1011]

# Map: condition -> {seed: run_tag}
CONDITIONS = {
    "v2_rn50_full": {
        42:  "v2_run1",
        123: "v2_run1_seed123",
        456: "v2_run1_seed456",
    },
    "v3_rn50_full": {
        42:   "v3_rn50_full_run1_seed42",
        123:  "v3_rn50_full_run1_seed123",
        456:  "v3_rn50_full_run1_seed456",
        789:  "v3_rn50_full_run1_seed789",
        1011: "v3_rn50_full_run1_seed1011",
    },
}


def load_fold_metrics(tag):
    path = KFOLD / tag / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        s = json.load(f)
    if not s.get("fold_results"):
        return None
    out = []
    for fr in s["fold_results"]:
        row = {
            "accuracy": fr["test_accuracy"],
            "macro_f1": fr["test_macro_f1"],
            "macro_auc": fr["test_macro_auc"],
        }
        for c in CLASSES:
            row[f"f1_{c}"] = fr["per_class"][c]["f1"]
            row[f"auc_{c}"] = fr["per_class"][c]["auc"]
        out.append(row)
    return out


# Load all rows
print("=" * 70)
print("Loading fold-level results...")
print("=" * 70)
per_condition = {}
for cond, seed_tags in CONDITIONS.items():
    rows = []
    for seed, tag in seed_tags.items():
        folds = load_fold_metrics(tag)
        if folds is None:
            print(f"  {cond:20s} seed={seed}: MISSING ({tag})")
            continue
        for fold_idx, data in enumerate(folds):
            rows.append({"seed": seed, "fold": fold_idx, **data})
        print(f"  {cond:20s} seed={seed}: OK ({len(folds)} folds)")
    per_condition[cond] = rows

# ----------------------------------------------------------------------
# Per-condition summary
# ----------------------------------------------------------------------
METRICS = ["accuracy", "macro_f1", "macro_auc"] + \
          [f"f1_{c}" for c in CLASSES] + [f"auc_{c}" for c in CLASSES]

summary_rows = []
for cond, rows in per_condition.items():
    if not rows:
        continue
    df = pd.DataFrame(rows)
    summary_row = {"condition": cond, "n_observations": len(df)}
    for m in METRICS:
        vals = df[m].dropna()
        if len(vals) > 0:
            summary_row[f"{m}_mean"] = float(vals.mean())
            summary_row[f"{m}_std"] = float(vals.std(ddof=1))
    summary_rows.append(summary_row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "v3_multi_seed_summary.csv", index=False)

print()
print("=" * 70)
print("PER-CONDITION SUMMARY")
print("=" * 70)
print(f"{'Condition':<20} {'n':>4} {'Acc (mean±std)':>22} "
      f"{'F1 (mean±std)':>22} {'AUC (mean±std)':>22}")
print("-" * 92)
for _, row in summary_df.iterrows():
    print(f"{row['condition']:<20} {int(row['n_observations']):>4} "
          f"{row['accuracy_mean']*100:>10.2f}% ± {row['accuracy_std']*100:>5.2f}%  "
          f"{row['macro_f1_mean']:>10.4f} ± {row['macro_f1_std']:>5.4f}  "
          f"{row['macro_auc_mean']:>10.4f} ± {row['macro_auc_std']:>5.4f}")

print()
print("Per-class F1 (mean ± std):")
print(f"{'Condition':<20} {'Normal':>18} {'Cyst':>18} {'Tumor':>18} {'Stone':>18}")
print("-" * 96)
for _, row in summary_df.iterrows():
    parts = [f"{row['condition']:<20}"]
    for c in CLASSES:
        parts.append(f"{row[f'f1_{c}_mean']:>7.4f} ± {row[f'f1_{c}_std']:>5.4f}")
    print(" ".join(parts))

# ----------------------------------------------------------------------
# Paired t-tests: v3 vs v2 (unpaired since different fold assignments per seed)
# ----------------------------------------------------------------------
print()
print("=" * 70)
print("v3 vs v2 SIGNIFICANCE TESTS (Welch's t-test, unequal samples)")
print("=" * 70)
print("(Independent samples: v2 and v3 use different manifests, so paired testing")
print(" would be invalid. Using Welch's t-test which handles unequal variances")
print(" and unequal sample sizes.)")

if "v2_rn50_full" in per_condition and "v3_rn50_full" in per_condition \
        and per_condition["v2_rn50_full"] and per_condition["v3_rn50_full"]:

    v2 = pd.DataFrame(per_condition["v2_rn50_full"])
    v3 = pd.DataFrame(per_condition["v3_rn50_full"])

    n_metrics_tested = len(METRICS)
    BONF_ALPHA = 0.05 / n_metrics_tested
    print(f"\nAlpha (uncorrected): 0.05")
    print(f"Alpha (Bonferroni for {n_metrics_tested} metrics): {BONF_ALPHA:.5f}\n")

    sig_rows = []
    for metric in METRICS:
        va = v2[metric].dropna().values
        vb = v3[metric].dropna().values
        if len(va) < 3 or len(vb) < 3:
            continue
        # Welch's t-test (unpaired, unequal variances)
        t_stat, p_val = stats.ttest_ind(va, vb, equal_var=False)
        diff = float(vb.mean() - va.mean())
        sig = "yes" if p_val < 0.05 else "no"
        sig_bonf = "yes" if p_val < BONF_ALPHA else "no"
        direction = "↑ v3 better" if diff > 0 else "↓ v3 worse"
        print(f"  {metric:<12} v2={va.mean():.4f} v3={vb.mean():.4f} "
              f"Δ={diff:+.4f} p={p_val:.5f} "
              f"{'*' if sig == 'yes' else ' '}"
              f"{'**' if sig_bonf == 'yes' else '  '}  {direction}")
        sig_rows.append({
            "metric": metric,
            "v2_mean": float(va.mean()),
            "v2_std": float(va.std(ddof=1)),
            "v3_mean": float(vb.mean()),
            "v3_std": float(vb.std(ddof=1)),
            "n_v2": len(va),
            "n_v3": len(vb),
            "delta": diff,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "sig_uncorrected": sig,
            "sig_bonferroni": sig_bonf,
        })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(OUT / "v3_vs_v2_significance.csv", index=False)

    print()
    print("Legend: *  = p < 0.05 uncorrected")
    print("        ** = p < 0.05 Bonferroni-corrected")
else:
    print("Skipping (need both v2 and v3 data).")

# ----------------------------------------------------------------------
# Headline summary
# ----------------------------------------------------------------------
headline_path = OUT / "v3_vs_v2_headline.txt"
with open(headline_path, "w") as f:
    f.write("v3 (5-source) vs v2 (4-source) - RN50 Full-Image\n")
    f.write("=" * 70 + "\n\n")
    for _, row in summary_df.iterrows():
        f.write(f"{row['condition']} (n={int(row['n_observations'])}):\n")
        f.write(f"  Accuracy:  {row['accuracy_mean']*100:.2f}% ± {row['accuracy_std']*100:.2f}%\n")
        f.write(f"  Macro-F1:  {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}\n")
        f.write(f"  Macro-AUC: {row['macro_auc_mean']:.4f} ± {row['macro_auc_std']:.4f}\n")
        for c in CLASSES:
            f.write(f"  {c:<8} F1: {row[f'f1_{c}_mean']:.4f} ± {row[f'f1_{c}_std']:.4f}\n")
        f.write("\n")

print(f"\nSaved:")
print(f"  {OUT / 'v3_multi_seed_summary.csv'}")
print(f"  {OUT / 'v3_vs_v2_significance.csv'}")
print(f"  {headline_path}")
