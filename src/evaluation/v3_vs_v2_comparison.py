"""
v2 vs v3 headline comparison for the dissertation.

Generates:
  1. Bar chart comparing v2 and v3 per-class F1 (mean ± std)
  2. Bar chart comparing v2 and v3 overall accuracy / F1 / AUC
  3. A clean summary table (CSV + human-readable txt)

Reads the summary CSV produced by v3_multi_seed_aggregate.py, so run that first.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(os.environ["SCRATCH"])
V3_ANALYSIS = SCRATCH / "kidney-results/v3_analysis"
SUMMARY_CSV = V3_ANALYSIS / "v3_multi_seed_summary.csv"

if not SUMMARY_CSV.exists():
    raise SystemExit(f"Missing {SUMMARY_CSV}. Run v3_multi_seed_aggregate.py first.")

df = pd.read_csv(SUMMARY_CSV)
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

v2 = df[df["condition"] == "v2_rn50_full"].iloc[0] if \
     (df["condition"] == "v2_rn50_full").any() else None
v3 = df[df["condition"] == "v3_rn50_full"].iloc[0] if \
     (df["condition"] == "v3_rn50_full").any() else None

if v2 is None or v3 is None:
    raise SystemExit("Need both v2 and v3 rows in summary CSV.")

# ----------------------------------------------------------------------
# Figure 1: Per-class F1 comparison
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(CLASSES))
w = 0.35

v2_means = [v2[f"f1_{c}_mean"] for c in CLASSES]
v2_stds  = [v2[f"f1_{c}_std"] for c in CLASSES]
v3_means = [v3[f"f1_{c}_mean"] for c in CLASSES]
v3_stds  = [v3[f"f1_{c}_std"] for c in CLASSES]

b1 = ax.bar(x - w/2, v2_means, w, yerr=v2_stds, capsize=4,
            label=f"v2 (4-source, n={int(v2['n_observations'])})",
            color="#6B7280", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, v3_means, w, yerr=v3_stds, capsize=4,
            label=f"v3 (5-source, n={int(v3['n_observations'])})",
            color="#1E40AF", edgecolor="black", linewidth=0.5)

for bars, vals in [(b1, v2_means), (b2, v3_means)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1.05)
ax.set_title("Per-class F1: v2 (4-source) vs v3 (5-source + Abdalla), RN50 full-image")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(V3_ANALYSIS / "v2_vs_v3_per_class_f1.png", dpi=130, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Figure 2: Overall metrics comparison
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ["accuracy", "macro_f1", "macro_auc"]
labels = ["Accuracy", "Macro-F1", "Macro-AUC"]
x = np.arange(len(metrics))
w = 0.35

v2_o_means = [v2[f"{m}_mean"] for m in metrics]
v2_o_stds  = [v2[f"{m}_std"] for m in metrics]
v3_o_means = [v3[f"{m}_mean"] for m in metrics]
v3_o_stds  = [v3[f"{m}_std"] for m in metrics]

b1 = ax.bar(x - w/2, v2_o_means, w, yerr=v2_o_stds, capsize=4,
            label=f"v2 (n={int(v2['n_observations'])})",
            color="#6B7280", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, v3_o_means, w, yerr=v3_o_stds, capsize=4,
            label=f"v3 (n={int(v3['n_observations'])})",
            color="#1E40AF", edgecolor="black", linewidth=0.5)

for bars, vals in [(b1, v2_o_means), (b2, v3_o_means)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Overall: v2 vs v3 headline metrics, RN50 full-image")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(V3_ANALYSIS / "v2_vs_v3_overall.png", dpi=130, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------
rows = []
for m in metrics + [f"f1_{c}" for c in CLASSES]:
    v2_m, v2_s = v2[f"{m}_mean"], v2[f"{m}_std"]
    v3_m, v3_s = v3[f"{m}_mean"], v3[f"{m}_std"]
    delta = v3_m - v2_m
    rows.append({
        "metric": m,
        "v2_mean": v2_m,
        "v2_std": v2_s,
        "v3_mean": v3_m,
        "v3_std": v3_s,
        "delta": delta,
        "pct_change": 100 * delta / v2_m if v2_m > 0 else np.nan,
    })

tbl = pd.DataFrame(rows)
tbl.to_csv(V3_ANALYSIS / "v2_vs_v3_comparison_table.csv", index=False)

# Human-readable
readable = V3_ANALYSIS / "v2_vs_v3_comparison_table.txt"
with open(readable, "w") as f:
    f.write("v2 (4-source) vs v3 (5-source with Abdalla) - RN50 Full-Image\n")
    f.write(f"v2 n_observations = {int(v2['n_observations'])}\n")
    f.write(f"v3 n_observations = {int(v3['n_observations'])}\n\n")
    f.write(f"{'Metric':<15} {'v2 (mean±std)':>18} {'v3 (mean±std)':>18} "
            f"{'Δ':>10} {'% change':>10}\n")
    f.write("-" * 75 + "\n")
    for _, r in tbl.iterrows():
        f.write(f"{r['metric']:<15} "
                f"{r['v2_mean']:.4f} ± {r['v2_std']:.4f}   "
                f"{r['v3_mean']:.4f} ± {r['v3_std']:.4f}   "
                f"{r['delta']:+.4f}   "
                f"{r['pct_change']:+.1f}%\n")

print("Comparison table:\n")
print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved to: {V3_ANALYSIS}")
print(f"  - v2_vs_v3_per_class_f1.png")
print(f"  - v2_vs_v3_overall.png")
print(f"  - v2_vs_v3_comparison_table.csv")
print(f"  - v2_vs_v3_comparison_table.txt")
