"""
Grad-CAM visualisation for v3 RN50-full model.

Loads the best-fold model from v3_rn50_full_run1_seed42 and produces a
6-panel Grad-CAM figure across 6 representative images spanning classes
and sources. Same layout as the v2 CBAM Grad-CAM figure for direct
comparison.

Sample selection:
  - 1 KiTS Cyst  (persistent KiTS-Cyst confusion probe)
  - 1 KAUH Cyst  (single-source-shortcut probe)
  - 1 KiTS Tumor (multi-class source probe)
  - 1 TCGA Tumor (single-source-shortcut probe)
  - 1 Mendeley Stone (was v2's failure case)
  - 1 Abdalla Stone (new source — does attention transfer?)
"""
import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

SCRATCH = Path(os.environ["SCRATCH"])
V3_MANIFEST = SCRATCH / "kidney-data/processed/unified_v3_corrected/manifest_with_folds.csv"
V3_KFOLD = SCRATCH / "kidney-results/kfold/v3c_rn50_baseline_seed42"
OUT = SCRATCH / "kidney-results/v3_analysis/gradcam"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Pick best fold by test macro-F1
with open(V3_KFOLD / "summary.json") as f:
    s = json.load(f)
best_fold = int(np.argmax([fr["test_macro_f1"] for fr in s["fold_results"]]))
best_ckpt = V3_KFOLD / f"fold_{best_fold}" / "best.pth"
print(f"Using best fold {best_fold} (test F1 = "
      f"{s['fold_results'][best_fold]['test_macro_f1']:.4f})")

# Load plain RN50
print("Loading v3 RN50-full...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE).eval()

target_layer = model.layer4[-1]
print(f"Grad-CAM target layer: {target_layer.__class__.__name__}")
cam = GradCAM(model=model, target_layers=[target_layer])

# Sample selection — same test fold as best model
df = pd.read_csv(V3_MANIFEST)
test_df = df[df["fold"] == best_fold].reset_index(drop=True)

sample_specs = [
    ("Cyst",   "kits"),
    ("Cyst",   "kauh"),
    ("Tumor",  "kits"),
    ("Tumor",  "tcga"),
    ("Stone",  "mendeley"),
    ("Stone",  "abdalla"),
]

samples = []
for cls, src in sample_specs:
    sub = test_df[(test_df["label"] == cls) & (test_df["source"] == src)]
    if len(sub) == 0:
        print(f"  WARNING: no {cls} from {src} in fold {best_fold}")
        continue
    samples.append(sub.iloc[0])

print(f"\nSelected {len(samples)} samples for Grad-CAM")

# Figure: 6 rows × 3 cols (original / gradcam / probs)
n = len(samples)
fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
if n == 1:
    axes = axes[None, :]

for i, sample in enumerate(samples):
    img_path = sample["path"]
    true_label = sample["label"]
    src = sample["source"]

    orig_pil = Image.open(img_path).convert("L").convert("RGB")
    orig_arr = np.array(orig_pil.resize((224, 224))) / 255.0

    input_tensor = eval_tf(orig_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(logits.argmax(1).item())
    pred_label = IDX2LABEL[pred_idx]

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(pred_idx)])[0]
    cam_overlay = show_cam_on_image(orig_arr, grayscale_cam,
                                     use_rgb=True, image_weight=0.55)

    axes[i, 0].imshow(orig_arr)
    axes[i, 0].set_title(f"Original\nTrue: {true_label} ({src})", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(cam_overlay)
    correct = "✓" if pred_label == true_label else "✗"
    axes[i, 1].set_title(f"v3 RN50 Grad-CAM\nPred: {pred_label} "
                          f"({probs[pred_idx]:.2f}) {correct}", fontsize=10)
    axes[i, 1].axis("off")

    axes[i, 2].barh(CLASSES, probs,
                     color=['gray' if c != pred_label else 'tab:orange'
                            for c in CLASSES])
    axes[i, 2].set_xlim(0, 1)
    axes[i, 2].set_title("Class probabilities", fontsize=10)
    axes[i, 2].invert_yaxis()
    for j, p in enumerate(probs):
        axes[i, 2].text(p + 0.02, j, f"{p:.2f}", va='center', fontsize=8)

plt.suptitle(f"v3c RN50-full: Grad-CAM attention maps across classes and sources "
             f"(fold {best_fold}, seed 42)", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "v3_gradcam.png", dpi=140, bbox_inches="tight")
print(f"Saved: {OUT / 'v3_gradcam.png'}")