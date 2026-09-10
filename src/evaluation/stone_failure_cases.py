"""
Visualise Stone failure cases: Abdalla images where v3c-full classified
correctly but v3c-ROI did not. Shows original + U-Net mask + ROI crop
side by side.

Emits two figures:
  stone_failure_main.png      - the rows listed in MAIN_ROWS (main text)
  stone_failure_appendix.png  - every failure case (appendix)

Also writes stone_failure_list.csv with per-case mask diagnostics, so the
main-text rows can be chosen on evidence rather than manifest order.

Run once with the default MAIN_ROWS, read the printed table, then set the
indices you want and rerun.
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

# ---------------------------------------------------------------------------
# Which failure cases appear in the main-text figure.
# Indices refer to the order printed by this script and stored in the CSV.
# The appendix figure always shows every case regardless of this setting.
# ---------------------------------------------------------------------------
MAIN_ROWS = [0, 1, 4]

SCRATCH = Path(os.environ["SCRATCH"])
FULL_MANIFEST = SCRATCH / "kidney-data/processed/unified_v3_corrected/manifest_with_folds.csv"
ROI_MANIFEST = SCRATCH / "kidney-data/processed/unified_v3c_roi/manifest_with_folds.csv"
FULL_KFOLD = SCRATCH / "kidney-results/kfold/v3c_rn50_baseline_seed42"
ROI_KFOLD = SCRATCH / "kidney-results/kfold/v3c_roi_baseline_seed42"
SEGMENTER_CKPT = SCRATCH / "kidney-results/segmenter/best.pth"

OUT = SCRATCH / "kidney-results/stone_failure_cases"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Load classifier from a given fold ---
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


def mask_diagnostics(mask):
    """Return largest-component mask plus counts needed to distinguish
    'segmenter found one kidney' from 'post-processing discarded one'."""
    raw_area = int(mask.sum())
    if raw_area == 0:
        return mask, {"raw_area": 0, "kept_area": 0, "n_components": 0,
                      "second_area": 0}
    labels = measure.label(mask, connectivity=2)
    sizes = np.bincount(labels.flat)
    sizes[0] = 0
    n_components = int((sizes > 0).sum())
    order = np.argsort(sizes)[::-1]
    kept = (labels == order[0]).astype(np.uint8)
    second_area = int(sizes[order[1]]) if n_components > 1 else 0
    return kept, {"raw_area": raw_area, "kept_area": int(kept.sum()),
                  "n_components": n_components, "second_area": second_area}


# --- Find Stone failure cases ---
print("Loading manifests and models...")
full_df = pd.read_csv(FULL_MANIFEST)
roi_df = pd.read_csv(ROI_MANIFEST)
for _df in (full_df, roi_df):
    if "stem" not in _df.columns:
        _df["stem"] = _df["path"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
roi_stems = set(roi_df["stem"].tolist())

abdalla_stones = full_df[
    (full_df["source"] == "abdalla") &
    (full_df["label"] == "Stone") &
    (full_df["stem"].isin(roi_stems))
].reset_index(drop=True)
print(f"Abdalla Stone images in matched subset: {len(abdalla_stones)}")

failures = []
for fold in range(5):
    print(f"\nFold {fold}: checking failures...")
    fold_df = abdalla_stones[abdalla_stones["fold"] == fold].reset_index(drop=True)
    if len(fold_df) == 0:
        continue
    full_model = load_classifier(FULL_KFOLD / f"fold_{fold}" / "best.pth")
    roi_model = load_classifier(ROI_KFOLD / f"fold_{fold}" / "best.pth")

    fold_roi_df = roi_df[roi_df["fold"] == fold].reset_index(drop=True)
    roi_by_stem = {row["stem"]: row["path"] for _, row in fold_roi_df.iterrows()}

    for _, row in fold_df.iterrows():
        full_pred, _ = classify(full_model, row["path"])
        if full_pred != "Stone":
            continue
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

n_total = len(failures)
print(f"\nFound {n_total} cases where full got it right but ROI got it wrong")

# --- Precompute masks and diagnostics once, reused by both figures ---
panels = []
for case in failures:
    orig, mask = predict_mask(case["full_path"])
    kept, diag = mask_diagnostics(mask)
    roi_img = np.array(Image.open(case["roi_path"]).convert("L"))
    panels.append({"case": case, "orig": orig, "mask": kept,
                   "roi_img": roi_img, **diag})

print("\nPer-case mask diagnostics")
print(f"{'idx':>3}  {'stem':<28} {'kept':>6} {'raw':>6} {'comp':>4} "
      f"{'2nd':>6}  {'roi_pred':<8}")
for i, p in enumerate(panels):
    print(f"{i:>3}  {p['case']['stem']:<28} {p['kept_area']:>6} "
          f"{p['raw_area']:>6} {p['n_components']:>4} {p['second_area']:>6}  "
          f"{p['case']['roi_pred']:<8}")
print("\nkept = largest component (what is drawn and cropped); raw = full "
      "predicted mask.\nWhere comp > 1 the segmenter found more than one "
      "region and post-processing\ndiscarded the rest, which is not the same "
      "as the segmenter missing a kidney.")


def draw(selected, out_name, dpi):
    """Render a figure from the given panel list."""
    n = len(selected)
    if n == 0:
        print(f"Nothing to draw for {out_name}")
        return
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
    if n == 1:
        axes = axes[None, :]

    for i, p in enumerate(selected):
        case = p["case"]
        orig, mask_clean = p["orig"], p["mask"]

        axes[i, 0].imshow(orig, cmap="gray")
        axes[i, 0].set_title(
            f"Original (Stone, Abdalla)\nFull-image pred: {case['full_pred']} \u2713")
        axes[i, 0].axis("off")

        overlay = np.stack([orig, orig, orig], axis=-1).astype(float) / 255.0
        overlay[mask_clean > 0] = overlay[mask_clean > 0] * 0.5 + np.array([0.8, 0, 0]) * 0.5
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"U-Net kidney mask ({p['kept_area']} px)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(p["roi_img"], cmap="gray")
        axes[i, 2].set_title(
            f"ROI crop fed to model\nROI pred: {case['roi_pred']} \u2717")
        axes[i, 2].axis("off")

    # No suptitle: the LaTeX caption carries this information.
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_name} ({n} cases)")


main_rows = [i for i in MAIN_ROWS if 0 <= i < n_total]
dropped = [i for i in MAIN_ROWS if i not in main_rows]
if dropped:
    print(f"\nWarning: MAIN_ROWS indices out of range and skipped: {dropped}")

draw([panels[i] for i in main_rows], "stone_failure_main.png", dpi=200)
draw(panels, "stone_failure_appendix.png", dpi=130)

diag_df = pd.DataFrame([
    {**p["case"], "kept_area": p["kept_area"], "raw_area": p["raw_area"],
     "n_components": p["n_components"], "second_area": p["second_area"],
     "in_main_figure": i in main_rows}
    for i, p in enumerate(panels)
])
diag_df.to_csv(OUT / "stone_failure_list.csv", index=False)

kept_areas = [p["kept_area"] for p in panels]
if kept_areas:
    print(f"\nCaption check: kept-component areas span {min(kept_areas)} to "
          f"{max(kept_areas)} px across all {n_total} cases.")
    main_areas = [panels[i]["kept_area"] for i in main_rows]
    if main_areas:
        print(f"Within the main-text rows they span {min(main_areas)} to "
              f"{max(main_areas)} px.")

print(f"\nSaved to: {OUT}")
print(f"  - stone_failure_main.png ({len(main_rows)} cases)")
print(f"  - stone_failure_appendix.png (all {n_total} cases)")
print(f"  - stone_failure_list.csv (all {n_total} cases with diagnostics)")