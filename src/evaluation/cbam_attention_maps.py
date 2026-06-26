"""
Visualise CBAM attention via Grad-CAM on a sample of images per class.

Uses pytorch-grad-cam to hook into the last bottleneck block of the CBAM-ResNet50
and produce a heatmap showing what the model attends to when making its prediction.

Output: per-class panels showing original / heatmap overlay / prediction.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local import for CBAM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cbam import add_cbam_to_resnet

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
CBAM_KFOLD = SCRATCH / "kidney-results/kfold/cbam_full_run1"
OUT = SCRATCH / "kidney-results/cbam_attention_maps"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same eval transforms as training
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# Pick representative samples — 2 per class, varied source where possible
df = pd.read_csv(MANIFEST)
sample_specs = [
    ("Cyst",   "kits"),     # KiTS cyst
    ("Cyst",   "kauh"),     # KAUH cyst (source-shortcut class)
    ("Tumor",  "kits"),     # KiTS tumor
    ("Tumor",  "tcga"),     # TCGA tumor (source-shortcut class)
    ("Stone",  "mendeley"), # Mendeley stone
    ("Normal", "mendeley"), # Mendeley normal
]

samples = []
for cls, src in sample_specs:
    sub = df[(df["label"] == cls) & (df["source"] == src) & (df["fold"] == 0)]
    if len(sub) > 0:
        samples.append(sub.iloc[0])

print(f"Selected {len(samples)} sample images for visualisation")

# Load CBAM-ResNet50 from fold 0
print("Loading CBAM-ResNet50 from fold 0...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = add_cbam_to_resnet(model, reduction=16)
ckpt = torch.load(CBAM_KFOLD / "fold_0" / "best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model = model.to(DEVICE).eval()

# Target the last Bottleneck of layer4 — deepest spatial features
target_layer = model.layer4[-1]
print(f"Grad-CAM target layer: {target_layer.__class__.__name__}")
cam = GradCAM(model=model, target_layers=[target_layer])

# Build figure
n = len(samples)
fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
if n == 1:
    axes = axes[None, :]

for i, sample in enumerate(samples):
    img_path = sample["path"]
    true_label = sample["label"]
    src = sample["source"]

    # Load original for display
    orig_pil = Image.open(img_path).convert("L").convert("RGB")
    orig_arr = np.array(orig_pil.resize((224, 224))) / 255.0

    # Predict + grad-cam
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

    # Column 1: original
    axes[i, 0].imshow(orig_arr)
    axes[i, 0].set_title(f"Original\nTrue: {true_label} ({src})", fontsize=10)
    axes[i, 0].axis("off")

    # Column 2: Grad-CAM heatmap
    axes[i, 1].imshow(cam_overlay)
    correct = "✓" if pred_label == true_label else "✗"
    axes[i, 1].set_title(f"CBAM Grad-CAM\nPred: {pred_label} ({probs[pred_idx]:.2f}) {correct}",
                          fontsize=10)
    axes[i, 1].axis("off")

    # Column 3: confidence bar chart
    axes[i, 2].barh(CLASSES, probs, color=['gray' if c != pred_label else 'tab:orange' for c in CLASSES])
    axes[i, 2].set_xlim(0, 1)
    axes[i, 2].set_title("Class probabilities", fontsize=10)
    axes[i, 2].invert_yaxis()
    for j, p in enumerate(probs):
        axes[i, 2].text(p + 0.02, j, f"{p:.2f}", va='center', fontsize=8)

plt.suptitle("CBAM-ResNet50: Grad-CAM attention maps across classes and sources",
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "cbam_attention_maps.png", dpi=140, bbox_inches="tight")
print(f"Saved: {OUT / 'cbam_attention_maps.png'}")