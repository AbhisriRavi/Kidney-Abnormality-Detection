"""
Score U-Net predictions on Mendeley images to find pseudo-labelling candidates.

For each Mendeley image:
  1. Run U-Net to get a kidney mask
  2. Count connected components, sizes, positions
  3. Compute a "good prediction" score
  4. Save mask + score for later filtering

Output:
  $SCRATCH/kidney-data/processed/mendeley_pseudo_labels/
    masks/<filename>.png        (predicted masks for all Mendeley images)
    scores.csv                  (filename, n_components, total_px, score, etc.)
"""
import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from monai.networks.nets import UNet
import cv2
from PIL import Image
from skimage import measure
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST = SCRATCH / "kidney-data/processed/unified_v2/manifest_with_folds.csv"
SEGMENTER_CKPT = SCRATCH / "kidney-results/segmenter/best.pth"
OUT_ROOT = SCRATCH / "kidney-data/processed/mendeley_pseudo_labels"
MASK_OUT = OUT_ROOT / "masks"
SCORES_OUT = OUT_ROOT / "scores.csv"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MASK_OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
SEG_INPUT_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load segmenter
print("Loading U-Net segmenter...")
seg_model = UNet(
    spatial_dims=2, in_channels=1, out_channels=2,
    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    num_res_units=2, norm="batch", dropout=0.1,
).to(DEVICE)
ckpt = torch.load(SEGMENTER_CKPT, map_location=DEVICE, weights_only=False)
seg_model.load_state_dict(ckpt["model_state"])
seg_model.eval()


def predict_mask(img_path):
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    inp = cv2.resize(img, (SEG_INPUT_SIZE, SEG_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(inp.astype(np.float32) / 255.0)[None, None].to(DEVICE)
    with torch.no_grad(), autocast():
        logits = seg_model(t)
    pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST), img


def score_prediction(mask):
    """
    Higher score = more likely to be a correct kidney segmentation.
    Heuristics:
      - 2 components (bilateral) = best
      - Reasonable total pixel count (kidneys ~500-3000 px on 224x224)
      - Roughly symmetric around horizontal center
    Returns: dict with components and overall score [0, 100]
    """
    total_px = int(mask.sum())
    if total_px == 0:
        return {"n_components": 0, "total_px": 0, "main_px": 0,
                "x_split_balance": 0, "score": 0.0}

    labels = measure.label(mask, connectivity=2)
    components = measure.regionprops(labels)
    # Keep only components > 50 px (filter noise)
    components = [c for c in components if c.area > 50]
    n_comp = len(components)

    # Bilateral score: 2 components is best, 1 is OK, 3+ is bad
    if n_comp == 2:
        bilat_score = 50
    elif n_comp == 1:
        bilat_score = 25
    else:
        bilat_score = 10

    # Pixel count score: kidneys typically 500-3000 px
    if 500 <= total_px <= 3000:
        size_score = 30
    elif 300 <= total_px < 500 or 3000 < total_px <= 4000:
        size_score = 20
    else:
        size_score = 10

    # Symmetry score: components should sit on opposite sides of center
    x_centroids = sorted([c.centroid[1] for c in components])
    if n_comp == 2:
        # both x_centroids; check if one is left-of-centre and other right
        if x_centroids[0] < IMG_SIZE/2 and x_centroids[1] > IMG_SIZE/2:
            sym_score = 20
        else:
            sym_score = 5
        x_split_balance = abs((x_centroids[0] + x_centroids[1]) / 2 - IMG_SIZE/2)
    else:
        sym_score = 5
        x_split_balance = -1

    score = bilat_score + size_score + sym_score
    main_px = int(max([c.area for c in components])) if components else 0

    return {
        "n_components": n_comp,
        "total_px": total_px,
        "main_px": main_px,
        "x_split_balance": float(x_split_balance) if x_split_balance != -1 else None,
        "score": float(score),
    }


# Filter manifest to Mendeley
df = pd.read_csv(MANIFEST)
mendeley = df[df["source"] == "mendeley"].reset_index(drop=True)
print(f"Processing {len(mendeley)} Mendeley images...")

rows = []
for _, row in tqdm(mendeley.iterrows(), total=len(mendeley)):
    img_path = row["path"]
    mask, _ = predict_mask(img_path)
    metrics = score_prediction(mask)
    # Save mask
    mask_name = Path(img_path).stem + ".png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(MASK_OUT / mask_name)
    rows.append({
        "img_path": img_path,
        "mask_path": str(MASK_OUT / mask_name),
        "label": row["label"],
        "patient_id": row["patient_id"],
        "stem": row["stem"],
        **metrics,
    })

scores_df = pd.DataFrame(rows)
scores_df = scores_df.sort_values("score", ascending=False).reset_index(drop=True)
scores_df.to_csv(SCORES_OUT, index=False)

print(f"\nProcessed {len(scores_df)} images")
print(f"Score distribution:")
print(scores_df["score"].describe().to_string())
print(f"\nN images with score >= 90: {(scores_df['score'] >= 90).sum()}")
print(f"N images with score >= 80: {(scores_df['score'] >= 80).sum()}")
print(f"N images with score >= 70: {(scores_df['score'] >= 70).sum()}")
print(f"N images with score >= 60: {(scores_df['score'] >= 60).sum()}")
print(f"\nN images with 2 components: {(scores_df['n_components'] == 2).sum()}")
print(f"N images with 1 component:  {(scores_df['n_components'] == 1).sum()}")
print(f"N images with 0 components: {(scores_df['n_components'] == 0).sum()}")
print(f"\nSaved: {SCORES_OUT}")
print(f"Masks: {MASK_OUT}")