"""
Visualise Stone failure cases: Mendeley images where v2-full classified correctly
but v2-ROI got it wrong. Shows original + U-Net mask + ROI crop side by side.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
from monai.networks.nets import UNet
import cv2
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(os.environ["SCRATCH"])
V2_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
ROI_MANIFEST = SCRATCH / "kidney-data/processed/unified_v2_roi/manifest_with_folds.csv"
V2_KFOLD = SCRATCH / "kidney-results/kfold/v2_run1"
ROI_KFOLD = SCRATCH / "kidney-results/kfold/v2_roi_run1"
SEGMENTER_CKPT = SCRATCH / "kidney-results/segmenter/best.pth"

OUT = SCRATCH / "kidney-results/stone_failure_cases"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load classifier from fold 0 (any fold works for visualisation) ---
def load_classifier(ckpt_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# --- Load segmenter ---
seg_model = UNet(
    spatial_dims=2, in_channels=1, out_channels=2,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch", dropout=0.1,
).to(DEVICE)
seg_ckpt = torch.load(SEGMENTER_CKPT, map_location=DEVICE, weights_only=False)
seg_model.load_state_dict(seg_ckpt["model_state"])
seg_model.eval()

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify(model, img_path):
    img = Image.open(img_path).convert("L").convert("RGB")
    t = eval_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), autocast():
        out = model(t)
    return IDX2LABEL[int(out.argmax(1).item())], torch.softmax(out, dim=1)[0].cpu().numpy()

def predict_mask(img_path):
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    if img.shape != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    inp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(inp.astype(np.float32) / 255.0)[None, None].to(DEVICE)
    with torch.no_grad(), autocast():
        logits = seg_model(t)
    pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    mask = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_NEAREST)
    return img, mask

def clean_mask(mask):
    if mask.sum() == 0:
        return mask
    labels = measure.label(mask, connectivity=2)
    sizes = np.bincount(labels.flat); sizes[0] = 0
    return (labels == sizes.argmax()).astype(np.uint8)

# --- Find Stone failure cases ---
print("Loading manifests and models...")
v2_df = pd.read_csv(V2_MANIFEST)
roi_df = pd.read_csv(ROI_MANIFEST)
roi_stems = set(roi_df["stem"].tolist())

# Filter to Mendeley Stone images that exist in both v2 and ROI
mendeley_stones = v2_df[
    (v2_df["source"] == "mendeley") &
    (v2_df["label"] == "Stone") &
    (v2_df["stem"].isin(roi_stems))
].reset_index(drop=True)
print(f"Mendeley Stone images in matched subset: {len(mendeley_stones)}")

# We'll evaluate fold by fold (each image belongs to one fold)
failures = []
for fold in range(5):
    print(f"\nFold {fold}: checking failures...")
    fold_df = mendeley_stones[mendeley_stones["fold"] == fold].reset_index(drop=True)
    if len(fold_df) == 0:
        continue
    full_model = load_classifier(V2_KFOLD / f"fold_{fold}" / "best.pth")
    roi_model = load_classifier(ROI_KFOLD / f"fold_{fold}" / "best.pth")

    fold_roi_df = roi_df[roi_df["fold"] == fold].reset_index(drop=True)
    roi_by_stem = {row["stem"]: row["path"] for _, row in fold_roi_df.iterrows()}

    for _, row in fold_df.iterrows():
        full_pred, _ = classify(full_model, row["path"])
        if full_pred != "Stone":
            continue  # not a "full got it right" case
        roi_path = roi_by_stem.get(row["stem"])
        if roi_path is None:
            continue
        roi_pred, _ = classify(roi_model, roi_path)
        if roi_pred != "Stone":
            failures.append({
                "stem": row["stem"],
                "full_path": row["path"],
                "roi_path": roi_path,
                "full_pred": full_pred,
                "roi_pred": roi_pred,
                "fold": fold,
            })

print(f"\nFound {len(failures)} cases where full got it right but ROI got it wrong")

# Pick first 6 for visualisation
n_show = min(6, len(failures))
print(f"\nVisualising first {n_show} cases...")

fig, axes = plt.subplots(n_show, 3, figsize=(12, n_show * 3.5))
if n_show == 1:
    axes = axes[None, :]

for i, case in enumerate(failures[:n_show]):
    # Column 1: original image
    orig, mask = predict_mask(case["full_path"])
    mask_clean = clean_mask(mask)
    n_kid = mask_clean.sum()

    axes[i, 0].imshow(orig, cmap="gray")
    axes[i, 0].set_title(f"Original (Stone, Mendeley)\nFull-image pred: {case['full_pred']} ✓")
    axes[i, 0].axis("off")

    # Column 2: original + mask overlay
    overlay = np.stack([orig, orig, orig], axis=-1).astype(float) / 255.0
    overlay[mask_clean > 0] = overlay[mask_clean > 0] * 0.5 + np.array([0.8, 0, 0]) * 0.5
    axes[i, 1].imshow(overlay)
    axes[i, 1].set_title(f"U-Net kidney mask ({n_kid} px)")
    axes[i, 1].axis("off")

    # Column 3: ROI crop (what the ROI classifier sees)
    roi_img = np.array(Image.open(case["roi_path"]).convert("L"))
    axes[i, 2].imshow(roi_img, cmap="gray")
    axes[i, 2].set_title(f"ROI crop fed to model\nROI pred: {case['roi_pred']} ✗")
    axes[i, 2].axis("off")

plt.suptitle("Stone Failure Cases: Mendeley images where Full → Stone, ROI → wrong class",
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "stone_failure_cases.png", dpi=130, bbox_inches="tight")
plt.close()

# Save raw case list
pd.DataFrame(failures).to_csv(OUT / "stone_failure_list.csv", index=False)

print(f"\nSaved to: {OUT}")
print(f"  - stone_failure_cases.png ({n_show} cases visualised)")
print(f"  - stone_failure_list.csv (all {len(failures)} failure cases)")