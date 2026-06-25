"""Side-by-side overall confusion matrices: v2-full vs v2-ROI."""
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
FULL_SUMMARY = SCRATCH / "kidney-results/kfold/v2_run1/summary.json"
ROI_SUMMARY = SCRATCH / "kidney-results/kfold/v2_roi_run1/summary.json"
OUT = SCRATCH / "kidney-results/comparison"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

with open(FULL_SUMMARY) as f:
    full = json.load(f)
with open(ROI_SUMMARY) as f:
    roi = json.load(f)

# Aggregate confusion matrix across all folds
def aggregate_cm(summary):
    cm = np.zeros((4, 4), dtype=int)
    for fr in summary["fold_results"]:
        cm += np.array(fr["confusion_matrix"])
    return cm

cm_full = aggregate_cm(full)
cm_roi = aggregate_cm(roi)

# Normalize per row to see recall pattern clearly
cm_full_norm = cm_full.astype(float) / cm_full.sum(axis=1, keepdims=True) * 100
cm_roi_norm = cm_roi.astype(float) / cm_roi.sum(axis=1, keepdims=True) * 100

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
for ax, (cm, title) in zip(axes[0], [(cm_full, "v2-FULL (all folds aggregated)"),
                                       (cm_roi, "v2-ROI (all folds aggregated)")]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title + "\n(absolute counts)")
for ax, (cm, title) in zip(axes[1], [(cm_full_norm, "v2-FULL"), (cm_roi_norm, "v2-ROI")]):
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                vmin=0, vmax=100, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title + "\n(% of true row)")
plt.tight_layout()
plt.savefig(OUT / "confusion_comparison.png", dpi=120, bbox_inches="tight")

print("=== AGGREGATED CONFUSION MATRICES ===\n")
print("v2-FULL (counts):")
print(pd.DataFrame(cm_full, index=CLASSES, columns=CLASSES).to_string())
print("\nv2-ROI (counts):")
print(pd.DataFrame(cm_roi, index=CLASSES, columns=CLASSES).to_string())
print("\nv2-FULL (% recall per row):")
print(pd.DataFrame(cm_full_norm, index=CLASSES, columns=CLASSES).round(1).to_string())
print("\nv2-ROI (% recall per row):")
print(pd.DataFrame(cm_roi_norm, index=CLASSES, columns=CLASSES).round(1).to_string())
print(f"\nSaved plot: {OUT / 'confusion_comparison.png'}")