"""
Build ROI crops with a configurable dilation margin around the kidney
bounding box.

HYPOTHESIS BEING TESTED
-----------------------
Across all three architectures, ROI cropping improves Cyst F1 (+0.12 to +0.15)
and Tumor F1 (+0.01 to +0.04) but degrades Stone F1 (-0.06 to -0.19). The
proposed explanation is anatomical: renal calculi frequently sit in the calyces,
renal pelvis, and proximal ureter, structures that fall on or just outside the
boundary of a tight kidney-parenchyma mask. A tight crop therefore removes the
very evidence the Stone class depends on.

If that explanation is correct, dilating the bounding box should recover Stone
F1 while preserving most of the Cyst gain, and there should be a margin at
which the two curves cross. Sweeping the margin turns an observed limitation
into a tested causal mechanism -- a substantially stronger discussion chapter
than reporting the trade-off alone.

Recommended sweep: --margin 0.00 0.15 0.25 0.40
(0.00 reproduces the existing tight-crop condition as a control.)

INPUT MODES
-----------
--mask-dir     Use pre-computed binary kidney masks (KiTS ground truth, or
               masks already written by src/data/extract_kits_seg_masks.py).
               Mask filenames must match image filenames.
--segmenter    Run the trained MONAI U-Net from train_segmenter.py to predict a
               mask on the fly. Use this for sources without ground-truth masks.

Usage
-----
python -m src.data.build_roi_dilated \
    --in-manifest  $SCRATCH/kidney-data/processed/unified_v3_corrected/manifest_with_folds.csv \
    --out-root     $SCRATCH/kidney-data/processed/roi_dilated \
    --segmenter    $SCRATCH/kidney-results/segmenter/best.pth \
    --margin 0.00 0.15 0.25 0.40
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

SEG_IMG_SIZE = 256          # matches train_segmenter.py
OUT_SIZE = 224              # matches the classifier input
MIN_BOX_PX = 24             # reject degenerate masks


def load_segmenter(ckpt_path, device):
    from monai.networks.nets import UNet
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    channels = tuple(cfg.get("channels", (32, 64, 128, 256, 512)))
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=2,
        channels=channels, strides=(2,) * (len(channels) - 1),
        num_res_units=2,
    )
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval(), cfg.get("img_size", SEG_IMG_SIZE)


@torch.no_grad()
def predict_mask(model, img_pil, img_size, device):
    arr = np.array(img_pil.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].to(device)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear",
                      align_corners=False)
    # match the segmenter's normalisation: per-image z-score
    t = (t - t.mean()) / t.std().clamp_min(1e-6)
    logits = model(t)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred


def largest_components_bbox(mask, keep=2):
    """
    Bounding box over the `keep` largest connected components.

    Keeping the two largest components (rather than all foreground) suppresses
    spurious speckle while retaining both kidneys when both are visible.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask > 0)
    if n == 0:
        return None
    sizes = ndimage.sum(mask > 0, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1][:keep]
    sel = np.isin(lab, order + 1)
    if sel.sum() == 0:
        return None
    ys, xs = np.where(sel)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate_box(box, margin, w, h):
    """
    Expand a box by `margin` (fraction of box side) on each side, clipped to
    the image. margin=0.25 grows a 100px box to 150px.
    """
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    dx, dy = bw * margin, bh * margin
    return (max(int(round(x0 - dx)), 0),
            max(int(round(y0 - dy)), 0),
            min(int(round(x1 + dx)), w),
            min(int(round(y1 + dy)), h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--margin", nargs="+", type=float, default=[0.0, 0.15, 0.25, 0.40])
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--segmenter", default=None)
    ap.add_argument("--square", action="store_true", default=True,
                    help="Pad the box to a square before cropping, so the "
                         "aspect ratio is not distorted by the final resize")
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap image count")
    args = ap.parse_args()

    if not args.mask_dir and not args.segmenter:
        raise SystemExit("Provide either --mask-dir or --segmenter")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.in_manifest)
    if args.limit:
        df = df.head(args.limit)
    print(f"Input manifest: {len(df)} images, "
          f"{df['patient_id'].nunique()} patients, "
          f"sources {sorted(df['source'].unique())}")

    model, seg_size = (None, SEG_IMG_SIZE)
    if args.segmenter:
        model, seg_size = load_segmenter(args.segmenter, device)
        print(f"Loaded segmenter (input {seg_size}px) from {args.segmenter}")

    out_root = Path(args.out_root)
    manifests = {m: [] for m in args.margin}
    n_fail = 0

    for n, row in enumerate(df.itertuples(index=False)):
        img = Image.open(row.path).convert("L")
        W, H = img.size

        # ---- obtain a binary kidney mask ----
        mask = None
        if args.mask_dir:
            mpath = Path(args.mask_dir) / Path(row.path).name
            if mpath.exists():
                mask = (np.array(Image.open(mpath).convert("L")) > 127).astype(np.uint8)
        if mask is None and model is not None:
            mask = predict_mask(model, img, seg_size, device)

        box = largest_components_bbox(mask) if mask is not None else None

        if box is not None and mask.shape != (H, W):
            # rescale box from mask resolution back to image resolution
            sy, sx = H / mask.shape[0], W / mask.shape[1]
            box = (int(box[0] * sx), int(box[1] * sy),
                   int(box[2] * sx), int(box[3] * sy))

        if box is None or (box[2] - box[0]) < MIN_BOX_PX or (box[3] - box[1]) < MIN_BOX_PX:
            # Fallback: central crop. Recorded in the manifest so the failure
            # rate can be reported honestly in the methodology chapter.
            n_fail += 1
            side = int(min(W, H) * 0.7)
            cx, cy = W // 2, H // 2
            box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
            fallback = True
        else:
            fallback = False

        for margin in args.margin:
            x0, y0, x1, y1 = dilate_box(box, margin, W, H)
            if args.square:
                bw, bh = x1 - x0, y1 - y0
                side = max(bw, bh)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                x0 = max(cx - side // 2, 0)
                y0 = max(cy - side // 2, 0)
                x1 = min(x0 + side, W)
                y1 = min(y0 + side, H)

            crop = img.crop((x0, y0, x1, y1)).resize(
                (OUT_SIZE, OUT_SIZE), Image.BILINEAR)

            tag = f"m{int(round(margin * 100)):03d}"
            dest_dir = out_root / f"margin_{tag}" / row.source / row.label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(row.path).name
            crop.save(dest)

            manifests[margin].append({
                "path": str(dest),
                "orig_path": row.path,
                "label": row.label,
                "patient_id": row.patient_id,
                "source": row.source,
                "fold": row.fold,
                "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1,
                "roi_margin": margin,
                "mask_fallback": fallback,
            })

        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(df)} images processed "
                  f"({n_fail} mask failures so far)")

    print(f"\nMask-detection failures: {n_fail}/{len(df)} "
          f"({100 * n_fail / max(len(df), 1):.2f}%) -- report this figure in "
          f"the methodology chapter")

    for margin, rows in manifests.items():
        tag = f"m{int(round(margin * 100)):03d}"
        mdir = out_root / f"margin_{tag}"
        mdir.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(rows)
        out_path = mdir / "manifest_with_folds.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  margin {margin:.2f}: {len(out_df)} images -> {out_path}")

    print("\nNext step: train one classifier per margin, e.g.")
    print("  for M in 000 015 025 040; do")
    print("    python -m src.training.train_unified --arch resnet50 \\")
    print("      --manifest-path $OUT_ROOT/margin_m$M/manifest_with_folds.csv \\")
    print("      --tag roi_margin_$M --loss ce --balanced-sampler")
    print("  done")
    print("Then plot Stone F1 and Cyst F1 against margin. A crossing point is "
          "the result you are looking for.")


if __name__ == "__main__":
    main()
