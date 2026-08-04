"""
Visualise U-Net predictions on Mendeley images at three quality tiers:
top (score>=90), medium (60-70), and bottom (score=0).

Shows: original + predicted mask overlay for 6 samples per tier.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(os.environ["SCRATCH"])
SCORES = SCRATCH / "kidney-data/processed/mendeley_pseudo_labels/scores.csv"
OUT = SCRATCH / "kidney-data/processed/mendeley_pseudo_labels"

scores = pd.read_csv(SCORES)

# Pick 6 samples from each tier
top = scores[scores["score"] >= 90].sample(n=min(6, (scores["score"] >= 90).sum()), random_state=42)
mid = scores[(scores["score"] >= 60) & (scores["score"] < 80)].sample(
    n=min(6, ((scores["score"] >= 60) & (scores["score"] < 80)).sum()), random_state=42)
bot = scores[scores["score"] == 0].sample(n=min(6, (scores["score"] == 0).sum()), random_state=42)

def overlay(orig_path, mask_path):
    orig = np.array(Image.open(orig_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    rgb = np.stack([orig, orig, orig], axis=-1).astype(float) / 255.0
    rgb[mask] = rgb[mask] * 0.4 + np.array([0.85, 0.1, 0.1]) * 0.6
    return rgb

# Build figure: 3 rows (tiers) × 6 columns (samples)
fig, axes = plt.subplots(3, 6, figsize=(20, 10))
for col, (_, row) in enumerate(top.iterrows()):
    rgb = overlay(row["img_path"], row["mask_path"])
    axes[0, col].imshow(rgb)
    axes[0, col].set_title(f"HIGH (score={row['score']:.0f})\n{row['label']}", fontsize=10)
    axes[0, col].axis("off")
for col, (_, row) in enumerate(mid.iterrows()):
    rgb = overlay(row["img_path"], row["mask_path"])
    axes[1, col].imshow(rgb)
    axes[1, col].set_title(f"MED (score={row['score']:.0f})\n{row['label']}", fontsize=10)
    axes[1, col].axis("off")
for col, (_, row) in enumerate(bot.iterrows()):
    orig = np.array(Image.open(row["img_path"]).convert("L"))
    axes[2, col].imshow(orig, cmap="gray")
    axes[2, col].set_title(f"ZERO (no detection)\n{row['label']}", fontsize=10)
    axes[2, col].axis("off")

axes[0, 0].set_ylabel("HIGH-score predictions\n(candidates for pseudo-labelling)", fontsize=11)
axes[1, 0].set_ylabel("MED-score predictions\n(borderline)", fontsize=11)
axes[2, 0].set_ylabel("ZERO-score (no kidney found)\n(originals, no mask to overlay)", fontsize=11)

plt.suptitle("Mendeley U-Net predictions at three quality tiers", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "score_tiers_visualisation.png", dpi=130, bbox_inches="tight")
print(f"Saved: {OUT / 'score_tiers_visualisation.png'}")