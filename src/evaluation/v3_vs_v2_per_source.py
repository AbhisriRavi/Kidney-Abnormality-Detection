"""
v2 vs v3 per-source comparison.

For each source common to v2 and v3 (kits, mendeley, kauh, tcga), computes
per-source accuracy on that source's test-fold images using the best-fold
model of each configuration. Adds abdalla as a v3-only source.

This visualises the trade-off: v3 gains on most sources (especially the new
Abdalla), but collapses on KiTS due to Cyst weight compensation.

Outputs:
  $SCRATCH/kidney-results/v3_analysis/
    v2_vs_v3_per_source_accuracy.png
    v2_vs_v3_per_source.csv
    v2_vs_v3_per_source_class_breakdown.csv
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(os.environ["SCRATCH"])
KFOLD = SCRATCH / "kidney-results/kfold"
V2_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
V3_MANIFEST = SCRATCH / "kidney-data/processed/unified_v3/manifest_with_folds.csv"
OUT = SCRATCH / "kidney-results/v3_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONFIGS = {
    "v2": {
        "tag": "v2_run1",
        "manifest": V2_MANIFEST,
    },
    "v3": {
        "tag": "v3_rn50_full_run1_seed42",
        "manifest": V3_MANIFEST,
    },
}


def pick_best_fold(tag):
    with open(KFOLD / tag / "summary.json") as f:
        s = json.load(f)
    return int(np.argmax([fr["test_macro_f1"] for fr in s["fold_results"]]))


eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class KidneyDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        return eval_tf(im), LABEL2IDX[row["label"]], idx


def predict_test_fold(tag, manifest_path):
    best_fold = pick_best_fold(tag)
    ckpt_path = KFOLD / tag / f"fold_{best_fold}" / "best.pth"
    print(f"  [{tag}] best fold = {best_fold}, ckpt = {ckpt_path}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()

    df = pd.read_csv(manifest_path)
    test_df = df[df["fold"] == best_fold].reset_index(drop=True)
    loader = DataLoader(KidneyDataset(test_df), batch_size=64,
                        shuffle=False, num_workers=4, pin_memory=True)

    preds = np.zeros(len(test_df), dtype=int)
    with torch.no_grad():
        for imgs, ys, idxs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(imgs)
            p = out.argmax(1).cpu().numpy()
            for i, ix in zip(p, idxs.numpy()):
                preds[ix] = i

    test_df["prediction"] = [IDX2LABEL[p] for p in preds]
    test_df["correct"] = test_df["prediction"] == test_df["label"]
    return test_df


print("=" * 70)
print("Loading v2 and v3 best-fold predictions...")
print("=" * 70)
v2_preds = predict_test_fold(CONFIGS["v2"]["tag"], CONFIGS["v2"]["manifest"])
v3_preds = predict_test_fold(CONFIGS["v3"]["tag"], CONFIGS["v3"]["manifest"])

# ----------------------------------------------------------------------
# Per-source accuracy comparison
# ----------------------------------------------------------------------
V2_SOURCES = sorted(v2_preds["source"].unique())
V3_SOURCES = sorted(v3_preds["source"].unique())
ALL_SOURCES = sorted(set(V2_SOURCES) | set(V3_SOURCES))

rows = []
for src in ALL_SOURCES:
    v2_sub = v2_preds[v2_preds["source"] == src]
    v3_sub = v3_preds[v3_preds["source"] == src]
    v2_acc = float(v2_sub["correct"].mean()) if len(v2_sub) > 0 else np.nan
    v3_acc = float(v3_sub["correct"].mean()) if len(v3_sub) > 0 else np.nan
    v2_f1 = f1_score([LABEL2IDX[l] for l in v2_sub["label"]],
                     [LABEL2IDX[l] for l in v2_sub["prediction"]],
                     labels=range(len(CLASSES)), average="macro",
                     zero_division=0) if len(v2_sub) > 0 else np.nan
    v3_f1 = f1_score([LABEL2IDX[l] for l in v3_sub["label"]],
                     [LABEL2IDX[l] for l in v3_sub["prediction"]],
                     labels=range(len(CLASSES)), average="macro",
                     zero_division=0) if len(v3_sub) > 0 else np.nan
    rows.append({
        "source": src,
        "v2_n": len(v2_sub),
        "v2_accuracy": v2_acc,
        "v2_macro_f1": v2_f1,
        "v3_n": len(v3_sub),
        "v3_accuracy": v3_acc,
        "v3_macro_f1": v3_f1,
        "delta_accuracy": v3_acc - v2_acc if not np.isnan(v2_acc) else np.nan,
    })

per_source_df = pd.DataFrame(rows)
per_source_df.to_csv(OUT / "v2_vs_v3_per_source.csv", index=False)

print()
print("Per-source accuracy (test fold, best-fold model):")
print(per_source_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ----------------------------------------------------------------------
# Per-source class breakdown (both configs)
# ----------------------------------------------------------------------
class_rows = []
for cfg_name, preds_df in [("v2", v2_preds), ("v3", v3_preds)]:
    for src in sorted(preds_df["source"].unique()):
        sub = preds_df[preds_df["source"] == src]
        for c in CLASSES:
            true_c = (sub["label"] == c).sum()
            pred_c = (sub["prediction"] == c).sum()
            correct_c = ((sub["label"] == c) & (sub["prediction"] == c)).sum()
            class_rows.append({
                "config": cfg_name,
                "source": src,
                "class": c,
                "n_true": int(true_c),
                "n_predicted": int(pred_c),
                "n_correct": int(correct_c),
                "recall": float(correct_c / true_c) if true_c > 0 else np.nan,
                "precision": float(correct_c / pred_c) if pred_c > 0 else np.nan,
            })
class_df = pd.DataFrame(class_rows)
class_df.to_csv(OUT / "v2_vs_v3_per_source_class_breakdown.csv", index=False)

# ----------------------------------------------------------------------
# Bar chart: per-source accuracy comparison
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(ALL_SOURCES))
w = 0.35

v2_accs = [row["v2_accuracy"] for _, row in per_source_df.iterrows()]
v3_accs = [row["v3_accuracy"] for _, row in per_source_df.iterrows()]
v2_ns   = [row["v2_n"] for _, row in per_source_df.iterrows()]
v3_ns   = [row["v3_n"] for _, row in per_source_df.iterrows()]

# Handle NaN for sources not present in v2 (abdalla)
v2_display = [a if not np.isnan(a) else 0 for a in v2_accs]

b1 = ax.bar(x - w/2, v2_display, w, label="v2 (4-source)",
            color="#6B7280", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, v3_accs, w, label="v3 (5-source + Abdalla)",
            color="#1E40AF", edgecolor="black", linewidth=0.5)

# Mark v2-absent (abdalla) with hatching
for i, a in enumerate(v2_accs):
    if np.isnan(a):
        b1[i].set_hatch("///")
        b1[i].set_facecolor("white")
        b1[i].set_edgecolor("#6B7280")
        ax.text(x[i] - w/2, 0.02, "not in v2",
                ha="center", va="bottom", fontsize=7, style="italic",
                color="#6B7280", rotation=90)

for bars, vals, ns in [(b1, v2_accs, v2_ns), (b2, v3_accs, v3_ns)]:
    for bar, val, n in zip(bars, vals, ns):
        if not np.isnan(val) and val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in ALL_SOURCES])
ax.set_ylabel("Accuracy on test fold")
ax.set_ylim(0, 1.15)
ax.set_title("Per-source accuracy: v2 vs v3 (best-fold model, seed 42)")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Highlight KiTS collapse
kits_idx = ALL_SOURCES.index("kits") if "kits" in ALL_SOURCES else None
if kits_idx is not None:
    v2_kits = v2_accs[kits_idx]
    v3_kits = v3_accs[kits_idx]
    if not np.isnan(v2_kits) and not np.isnan(v3_kits):
        delta = v3_kits - v2_kits
        if delta < -0.1:
            ax.annotate(f"Δ = {delta:+.2f}",
                        xy=(kits_idx + w/2, v3_kits),
                        xytext=(kits_idx + 1.0, v3_kits + 0.15),
                        fontsize=10, color="#B91C1C", weight="bold",
                        arrowprops=dict(arrowstyle="->",
                                        color="#B91C1C", lw=1))

plt.tight_layout()
plt.savefig(OUT / "v2_vs_v3_per_source_accuracy.png",
            dpi=130, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print()
print("=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)

kits_row = per_source_df[per_source_df["source"] == "kits"].iloc[0]
if not np.isnan(kits_row["v2_accuracy"]) and not np.isnan(kits_row["v3_accuracy"]):
    print(f"\nKiTS accuracy: v2={kits_row['v2_accuracy']*100:.1f}% "
          f"→ v3={kits_row['v3_accuracy']*100:.1f}% "
          f"({kits_row['delta_accuracy']*100:+.1f} pp)")

    kits_cyst_v2 = class_df[(class_df["config"] == "v2") &
                             (class_df["source"] == "kits") &
                             (class_df["class"] == "Cyst")].iloc[0]
    kits_cyst_v3 = class_df[(class_df["config"] == "v3") &
                             (class_df["source"] == "kits") &
                             (class_df["class"] == "Cyst")].iloc[0]
    print(f"KiTS Cyst predictions: v2={kits_cyst_v2['n_predicted']} "
          f"(of {kits_cyst_v2['n_true']} true) → v3={kits_cyst_v3['n_predicted']} "
          f"(of {kits_cyst_v3['n_true']} true)")

for src in ["mendeley", "kauh", "tcga", "abdalla"]:
    row = per_source_df[per_source_df["source"] == src]
    if not len(row):
        continue
    row = row.iloc[0]
    if np.isnan(row["v2_accuracy"]):
        print(f"\n{src.upper()}: v3={row['v3_accuracy']*100:.1f}% (not in v2)")
    else:
        print(f"\n{src.upper()}: v2={row['v2_accuracy']*100:.1f}% "
              f"→ v3={row['v3_accuracy']*100:.1f}% "
              f"({row['delta_accuracy']*100:+.1f} pp)")

print()
print(f"Files saved to: {OUT}")
print(f"  - v2_vs_v3_per_source.csv")
print(f"  - v2_vs_v3_per_source_class_breakdown.csv")
print(f"  - v2_vs_v3_per_source_accuracy.png")
