"""
Master 6-condition comparison: builds the canonical dissertation table and figures.

Inputs: summary.json files from all 6 runs
Outputs:
  - master_comparison_table.csv       (the main results table)
  - per_class_f1_grouped_bar.png      (one bar chart, 3 architectures x 2 inputs, by class)
  - accuracy_macroF1_overall.png      (accuracy + macro-F1 side by side, all 6 conditions)
  - confusion_matrices_6panel.png     (one confusion matrix per condition, 2x3 grid)
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRATCH = Path(os.environ["SCRATCH"])
OUT = SCRATCH / "kidney-results/comparison"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

CONDITIONS = [
    ("Plain ResNet50", "full", "v2_run1"),
    ("Plain ResNet50", "roi", "v2_roi_run1"),
    ("CBAM-ResNet50", "full", "cbam_full_run1"),
    ("CBAM-ResNet50", "roi", "cbam_roi_run1"),
    ("ViT-Base", "full", "vit_full_run1"),
    ("ViT-Base", "roi", "vit_roi_run1"),
]

def load(tag):
    path = SCRATCH / "kidney-results/kfold" / tag / "summary.json"
    if not path.exists():
        raise SystemExit(f"Missing: {path}")
    with open(path) as f:
        return json.load(f)

summaries = {tag: load(tag) for _, _, tag in CONDITIONS}

# Master table
rows = []
for arch, region, tag in CONDITIONS:
    s = summaries[tag]
    a = s["aggregate"]
    row = {
        "Architecture": arch,
        "Region": region.upper(),
        "Accuracy": f"{a['accuracy_mean']*100:.1f}% ± {a['accuracy_std']*100:.1f}%",
        "Macro-F1": f"{a['macro_f1_mean']:.3f} ± {a['macro_f1_std']:.3f}",
        "Macro-AUC": f"{a['macro_auc_mean']:.3f} ± {a['macro_auc_std']:.3f}",
    }
    for c in CLASSES:
        pc = a["per_class_f1"][c]
        row[f"F1 {c}"] = f"{pc['mean']:.3f} ± {pc['std']:.3f}"
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(OUT / "master_comparison_table.csv", index=False)
print("=== MASTER COMPARISON TABLE ===\n")
print(table.to_string(index=False))

# Numeric arrays for plotting
arch_labels = [a for a, _, _ in CONDITIONS]
region_labels = [r for _, r, _ in CONDITIONS]
acc_means = [summaries[t]["aggregate"]["accuracy_mean"] for _, _, t in CONDITIONS]
acc_stds = [summaries[t]["aggregate"]["accuracy_std"] for _, _, t in CONDITIONS]
f1_means = [summaries[t]["aggregate"]["macro_f1_mean"] for _, _, t in CONDITIONS]
f1_stds = [summaries[t]["aggregate"]["macro_f1_std"] for _, _, t in CONDITIONS]
cond_labels = [f"{a}\n({r})" for a, r, _ in CONDITIONS]

# Plot 1: Overall accuracy + macro-F1 side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2E86AB', '#A23B72'] * 3  # alternating full / roi

for ax, vals, stds, label in zip(axes,
                                  [np.array(acc_means)*100, f1_means],
                                  [np.array(acc_stds)*100, f1_stds],
                                  ["Accuracy (%)", "Macro-F1"]):
    bars = ax.bar(range(len(CONDITIONS)), vals, yerr=stds, capsize=4,
                   color=colors, edgecolor='black')
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(cond_labels, rotation=15, ha='right')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*0.01,
                f"{val:.2f}" if isinstance(val, float) and val < 1 else f"{val:.1f}",
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "accuracy_macroF1_overall.png", dpi=130, bbox_inches="tight")
plt.close()

# Plot 2: per-class F1 grouped bar
per_class = pd.DataFrame({
    "Condition": [f"{a} {r}" for a, r, _ in CONDITIONS],
    **{c: [summaries[t]["aggregate"]["per_class_f1"][c]["mean"] for _, _, t in CONDITIONS]
       for c in CLASSES},
})
per_class_stds = pd.DataFrame({
    "Condition": [f"{a} {r}" for a, r, _ in CONDITIONS],
    **{c: [summaries[t]["aggregate"]["per_class_f1"][c]["std"] for _, _, t in CONDITIONS]
       for c in CLASSES},
})
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(CONDITIONS))
width = 0.2
class_colors = {'Normal': '#1f77b4', 'Cyst': '#ff7f0e', 'Tumor': '#2ca02c', 'Stone': '#d62728'}
for i, c in enumerate(CLASSES):
    vals = per_class[c].values
    stds = per_class_stds[c].values
    ax.bar(x + i*width - 1.5*width, vals, width, label=c, yerr=stds, capsize=3,
            color=class_colors[c], edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(cond_labels, rotation=15, ha='right')
ax.set_ylabel("F1 score")
ax.set_title("Per-class F1 across all six conditions")
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(OUT / "per_class_f1_grouped_bar.png", dpi=130, bbox_inches="tight")
plt.close()

# Plot 3: Confusion matrices, one per condition, 2 rows × 3 cols (arch × region)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for col, (arch, region, tag) in enumerate(CONDITIONS):
    s = summaries[tag]
    cm_total = np.zeros((4, 4), dtype=int)
    for fr in s["fold_results"]:
        cm_total += np.array(fr["confusion_matrix"])
    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True) * 100
    row = 0 if region == "full" else 1
    col_idx = ["Plain ResNet50", "CBAM-ResNet50", "ViT-Base"].index(arch)
    ax = axes[row, col_idx]
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                vmin=0, vmax=100, cbar=(col_idx == 2))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{arch}\n({region}-image, % per row)")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrices_6panel.png", dpi=130, bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to: {OUT}")
print(f"  - master_comparison_table.csv")
print(f"  - accuracy_macroF1_overall.png")
print(f"  - per_class_f1_grouped_bar.png")
print(f"  - confusion_matrices_6panel.png")