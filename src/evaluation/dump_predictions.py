"""
Re-run inference from already-saved fold checkpoints and dump per-image
predictions to a tidy CSV.

WHY THIS EXISTS
---------------
The training scripts save only aggregate metrics (metrics.json) plus a
confusion matrix. That is enough for slice-level accuracy but not enough for:
  - patient-level aggregation
  - bootstrap confidence intervals resampled over patients
  - McNemar tests between conditions
  - calibration / reliability diagrams
  - per-source error breakdowns

All of those need the raw per-image probability vector alongside patient_id
and source. This script produces exactly that from checkpoints you already
have, so NO RETRAINING IS REQUIRED.

Output: $SCRATCH/kidney-results/predictions/<tag>.csv with columns
    fold, path, patient_id, source, label, y_true, y_pred,
    p_Normal, p_Cyst, p_Tumor, p_Stone

Usage
-----
python -m src.evaluation.dump_predictions \
    --tag v3c_rn50_full --arch resnet50 --manifest v3c

python -m src.evaluation.dump_predictions \
    --tag cbam_roi_run1 --arch cbam --manifest v2_roi
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.models.factory import build_model, forward_class_logits, probs_from_logits

CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MANIFEST_DIRS = {
    "v2": "kidney-data/processed/unified_v2",
    "v2_roi": "kidney-data/processed/unified_v2_roi",
    "v3": "kidney-data/processed/unified_v3",
    "v3_roi": "kidney-data/processed/unified_v3_roi",
    "v3c": "kidney-data/processed/unified_v3_corrected",
    "v3c_roi": "kidney-data/processed/unified_v3c_roi",
    "v4": "kidney-data/processed/unified_v4",
    "v4_roi": "kidney-data/processed/unified_v4_roi",
    "v4_roi_dilated": "kidney-data/processed/unified_v4_roi_dilated",
}


class KidneyEvalDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        return self.transform(im), LABEL2IDX[row["label"]], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Training run tag, i.e. the directory under kidney-results/kfold/")
    ap.add_argument("--arch", required=True,
                    choices=["resnet50", "cbam", "vit", "multihead", "dann"])
    ap.add_argument("--manifest", required=True, choices=list(MANIFEST_DIRS))
    ap.add_argument("--seed-suffix", default="")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-name", default=None,
                    help="Override output filename stem (defaults to --tag)")
    args = ap.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    manifest = scratch / MANIFEST_DIRS[args.manifest] / \
        f"manifest_with_folds{args.seed_suffix}.csv"
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    run_dir = scratch / "kidney-results/kfold" / args.tag
    out_dir = scratch / "kidney-results/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(manifest)
    n_sources = df["source"].nunique()
    print(f"Manifest: {manifest.name} | {len(df)} images | "
          f"{df['patient_id'].nunique()} patients | {n_sources} sources")

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_rows = []
    for fold_idx in range(args.n_folds):
        ckpt_path = run_dir / f"fold_{fold_idx}" / "best.pth"
        if not ckpt_path.exists():
            print(f"  fold {fold_idx}: checkpoint missing, skipping ({ckpt_path})")
            continue

        test_df = df[df["fold"] == fold_idx].reset_index(drop=True)

        model = build_model(args.arch, n_classes=len(CLASSES), n_sources=n_sources)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  fold {fold_idx}: {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys (expected for aux heads)")
        model = model.to(device).eval()

        loader = DataLoader(
            KidneyEvalDataset(test_df, eval_tf),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        probs_buf = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
        with torch.no_grad():
            for img, _, idxs in loader:
                img = img.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = forward_class_logits(model, args.arch, img)
                p = probs_from_logits(logits.float(), args.arch).cpu().numpy()
                probs_buf[idxs.numpy()] = p

        fold_out = test_df[["path", "patient_id", "source", "label"]].copy()
        fold_out.insert(0, "fold", fold_idx)
        fold_out["y_true"] = fold_out["label"].map(LABEL2IDX)
        fold_out["y_pred"] = probs_buf.argmax(axis=1)
        for i, c in enumerate(CLASSES):
            fold_out[f"p_{c}"] = probs_buf[:, i]

        acc = (fold_out["y_true"] == fold_out["y_pred"]).mean()
        print(f"  fold {fold_idx}: {len(fold_out):5d} images, slice-level acc = {acc:.4f}")
        all_rows.append(fold_out)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not all_rows:
        raise SystemExit("No folds produced predictions. Check --tag and checkpoint paths.")

    out = pd.concat(all_rows, ignore_index=True)
    stem = args.out_name or args.tag
    out_path = out_dir / f"{stem}.csv"
    out.to_csv(out_path, index=False)

    overall = (out["y_true"] == out["y_pred"]).mean()
    print(f"\nPooled slice-level accuracy: {overall:.4f} over {len(out)} images")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
