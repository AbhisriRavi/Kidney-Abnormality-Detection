"""
Leave-One-Source-Out (LOSO) training and evaluation.

WHAT QUESTION THIS ANSWERS
--------------------------
The dissertation's central diagnostic claim is that the network exploits a
structural covariance between source identity and class label. The evidence so
far is indirect: near-perfect per-source scores (KAUH Cyst F1 = 0.996,
TCGA Tumor F1 = 1.000) alongside near-chance performance on KiTS, the only
source carrying three classes.

LOSO makes the claim direct. Train on all sources but one, test on the held-out
source. This is the standard external-validation protocol in medical imaging
and answers the question a clinician actually cares about: does the model work
at a hospital it has never seen?

Two predictions follow from the shortcut hypothesis, and the script reports
both:
  1. Held-out-source performance collapses relative to within-distribution CV.
  2. Errors on the held-out source are systematically biased toward the class
     that source is *absent* from in training -- i.e. the model falls back on
     appearance priors learned from other scanners.

A "shortcut margin" is also reported: within-distribution accuracy minus
held-out-source accuracy. A large margin quantifies exactly how much of the
headline number is source recognition rather than pathology recognition.

IMPORTANT CAVEAT (state this in the write-up)
---------------------------------------------
Because sources do not each contain all four classes, held-out-source macro-F1
is not comparable across folds. Metrics are therefore reported over the classes
actually present in the held-out source, and the number of evaluable classes is
printed alongside every result.

Usage
-----
python -m src.training.train_loso --manifest v3c --tag loso_rn50 --arch resnet50
python -m src.training.train_loso --manifest v3c --tag loso_rn50 --arch resnet50 \
       --hold-out kits          # single source, for quick turnaround
"""
import os
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
    "v3c_roi": "kidney-data/processed/unified_v3_corrected_roi",
    "v4": "kidney-data/processed/unified_v4",
    "v4_roi": "kidney-data/processed/unified_v4_roi",
    "v4_roi_dilated": "kidney-data/processed/unified_v4_roi_dilated",
}


class KidneyDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        return self.transform(im), LABEL2IDX[row["label"]], idx


def build_train_transforms(strength):
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if strength == "mild":
        base += [transforms.RandomHorizontalFlip(0.5)]
    elif strength == "moderate":
        base += [transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomRotation(10),
                 transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    elif strength == "strong":
        base += [transforms.RandomHorizontalFlip(0.5),
                 transforms.RandomRotation(10),
                 transforms.ColorJitter(brightness=0.1, contrast=0.1),
                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                         scale=(0.9, 1.1)),
                 transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]
    base += [transforms.ToTensor(),
             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(base)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, choices=list(MANIFEST_DIRS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="resnet50",
                    choices=["resnet50", "cbam", "vit", "multihead"])
    ap.add_argument("--hold-out", default="all",
                    help="Source name to hold out, or 'all' to loop over every source")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--aug-strength", default="strong",
                    choices=["mild", "moderate", "strong"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-balanced-sampler", action="store_true", default=True)
    args = ap.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    manifest = scratch / MANIFEST_DIRS[args.manifest] / "manifest_with_folds.csv"
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    out_root = scratch / "kidney-results/loso" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(manifest)
    sources = sorted(df["source"].unique())
    print(f"Manifest {args.manifest}: {len(df)} images, "
          f"{df['patient_id'].nunique()} patients, sources = {sources}")

    # ---- Source x class structure: print it, it belongs in the dissertation ----
    xtab = pd.crosstab(df["source"], df["label"])
    xtab = xtab.reindex(columns=CLASSES, fill_value=0)
    print("\nSource x class image counts:")
    print(xtab.to_string())
    xtab.to_csv(out_root / "source_class_crosstab.csv")

    held_out_list = sources if args.hold_out == "all" else [args.hold_out]
    for s in held_out_list:
        if s not in sources:
            raise SystemExit(f"Unknown source '{s}'. Available: {sources}")

    train_tf = build_train_transforms(args.aug_strength)
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    all_results = []
    for held in held_out_list:
        print(f"\n{'=' * 70}\nHELD-OUT SOURCE: {held}\n{'=' * 70}")
        fold_out = out_root / f"heldout_{held}"
        fold_out.mkdir(exist_ok=True)

        test_df = df[df["source"] == held].reset_index(drop=True)
        pool_df = df[df["source"] != held].reset_index(drop=True)

        present = sorted(test_df["label"].unique(), key=lambda c: LABEL2IDX[c])
        present_idx = [LABEL2IDX[c] for c in present]
        absent = [c for c in CLASSES if c not in present]
        print(f"  Classes present in held-out source: {present}")
        print(f"  Classes absent:                     {absent}")

        # Patient-grouped train/val split within the remaining sources.
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10,
                                     random_state=args.seed)
        tr_idx, va_idx = next(splitter.split(
            pool_df.index.values, groups=pool_df["patient_id"].values))
        train_df = pool_df.iloc[tr_idx].reset_index(drop=True)
        val_df = pool_df.iloc[va_idx].reset_index(drop=True)

        print(f"  Train: {len(train_df):6d} img / {train_df['patient_id'].nunique()} pat "
              f"({sorted(train_df['source'].unique())})")
        print(f"  Val:   {len(val_df):6d} img / {val_df['patient_id'].nunique()} pat")
        print(f"  Test:  {len(test_df):6d} img / {test_df['patient_id'].nunique()} pat "
              f"(source '{held}', unseen)")

        counts = train_df["label"].value_counts()
        total = len(train_df)
        class_weights = torch.tensor(
            [total / (len(CLASSES) * counts.get(c, 1)) for c in CLASSES],
            dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(args.seed)

        train_ds = KidneyDataset(train_df, train_tf)
        if args.use_balanced_sampler:
            w = torch.tensor([1.0 / counts.get(l, 1) for l in train_df["label"]],
                             dtype=torch.float)
            sampler = WeightedRandomSampler(w, num_samples=len(train_df),
                                            replacement=True, generator=g)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True,
                                      worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True,
                                      worker_init_fn=seed_worker, generator=g)

        val_loader = DataLoader(KidneyDataset(val_df, eval_tf),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(KidneyDataset(test_df, eval_tf),
                                 batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        model = build_model(args.arch, n_classes=len(CLASSES)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        log_path = fold_out / "log.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                    "val_acc", "val_macro_f1", "lr"])

        best_val_f1 = -1.0
        for epoch in range(args.epochs):
            model.train()
            tl = tc = tn = 0
            for img, y, _ in train_loader:
                img, y = img.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    out = forward_class_logits(model, args.arch, img)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tl += loss.item() * img.size(0)
                tc += (out.argmax(1) == y).sum().item()
                tn += img.size(0)
            tl /= tn
            ta = tc / tn

            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for img, y, _ in val_loader:
                    img = img.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        out = forward_class_logits(model, args.arch, img)
                    vp.extend(out.argmax(1).cpu().numpy())
                    vt.extend(y.numpy())
            va = accuracy_score(vt, vp)
            vf1 = f1_score(vt, vp, average="macro", zero_division=0)
            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    ep {epoch+1:02d}/{args.epochs} tr_acc={ta:.3f} "
                  f"val_acc={va:.3f} val_F1={vf1:.3f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, f"{tl:.6f}", f"{ta:.6f}",
                                        f"{va:.6f}", f"{vf1:.6f}", f"{lr_now:.2e}"])
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                            "val_macro_f1": vf1, "held_out": held},
                           fold_out / "best.pth")

        # ---------------- Evaluation on the unseen source ----------------
        ckpt = torch.load(fold_out / "best.pth", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        probs = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
        with torch.no_grad():
            for img, _, idxs in test_loader:
                img = img.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = forward_class_logits(model, args.arch, img)
                probs[idxs.numpy()] = probs_from_logits(
                    logits.float(), args.arch).cpu().numpy()

        y_true = test_df["label"].map(LABEL2IDX).values
        y_pred = probs.argmax(axis=1)

        # Within-distribution reference: same model on its own validation split.
        vprobs, vtrue = [], []
        with torch.no_grad():
            for img, y, _ in val_loader:
                img = img.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = forward_class_logits(model, args.arch, img)
                vprobs.append(probs_from_logits(logits.float(), args.arch).cpu().numpy())
                vtrue.append(y.numpy())
        vprobs = np.concatenate(vprobs)
        vtrue = np.concatenate(vtrue)
        indist_acc = accuracy_score(vtrue, vprobs.argmax(1))

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_present = f1_score(y_true, y_pred, labels=present_idx,
                              average="macro", zero_division=0)
        pcl = precision_recall_fscore_support(y_true, y_pred,
                                              labels=range(len(CLASSES)),
                                              zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

        # Fraction of predictions falling into classes the held-out source
        # cannot contain. Pure shortcut-failure signal.
        absent_idx = [LABEL2IDX[c] for c in absent]
        leak_to_absent = float(np.isin(y_pred, absent_idx).mean()) if absent_idx else 0.0

        aucs = {}
        for i, c in enumerate(CLASSES):
            b = (y_true == i).astype(int)
            aucs[c] = float(roc_auc_score(b, probs[:, i])) if 0 < b.sum() < len(b) else None

        # Chance baselines for context.
        maj = pd.Series(y_true).value_counts(normalize=True).iloc[0]

        print(f"\n  --- Held-out '{held}' results ---")
        print(f"    accuracy                   = {acc:.4f}")
        print(f"    balanced accuracy          = {bal_acc:.4f}")
        print(f"    macro-F1 (present classes) = {f1_present:.4f}  "
              f"({len(present)} classes)")
        print(f"    majority-class baseline    = {maj:.4f}")
        print(f"    in-distribution val acc    = {indist_acc:.4f}")
        print(f"    SHORTCUT MARGIN            = {indist_acc - acc:+.4f}")
        print(f"    predictions into absent classes = {leak_to_absent:.4f}")

        result = {
            "held_out_source": held,
            "arch": args.arch, "manifest": args.manifest,
            "n_test_images": int(len(test_df)),
            "n_test_patients": int(test_df["patient_id"].nunique()),
            "classes_present": present, "classes_absent": absent,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1_present_classes": float(f1_present),
            "majority_class_baseline": float(maj),
            "in_distribution_val_accuracy": float(indist_acc),
            "shortcut_margin": float(indist_acc - acc),
            "frac_predicted_absent_class": leak_to_absent,
            "per_class": {c: {"precision": float(pcl[0][i]),
                              "recall": float(pcl[1][i]),
                              "f1": float(pcl[2][i]),
                              "support": int(pcl[3][i]),
                              "auc": aucs[c]} for i, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist(),
            "best_val_macro_f1": float(best_val_f1),
            "best_epoch": int(ckpt["epoch"]),
        }
        with open(fold_out / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        pred_out = test_df[["path", "patient_id", "source", "label"]].copy()
        pred_out["y_true"] = y_true
        pred_out["y_pred"] = y_pred
        pred_out["fold"] = 0
        for i, c in enumerate(CLASSES):
            pred_out[f"p_{c}"] = probs[:, i]
        pred_out.to_csv(fold_out / "predictions.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"LOSO — held-out '{held}'\nacc={acc:.3f}  "
                     f"balanced acc={bal_acc:.3f}")
        plt.tight_layout()
        plt.savefig(fold_out / "confusion_matrix.png", dpi=130, bbox_inches="tight")
        plt.close()

        all_results.append(result)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------------- Cross-source summary ----------------
    if all_results:
        summ = pd.DataFrame([{
            "held_out": r["held_out_source"],
            "n_images": r["n_test_images"],
            "n_patients": r["n_test_patients"],
            "n_classes_present": len(r["classes_present"]),
            "accuracy": r["accuracy"],
            "balanced_acc": r["balanced_accuracy"],
            "macro_f1_present": r["macro_f1_present_classes"],
            "majority_baseline": r["majority_class_baseline"],
            "in_dist_val_acc": r["in_distribution_val_accuracy"],
            "shortcut_margin": r["shortcut_margin"],
            "pred_into_absent": r["frac_predicted_absent_class"],
        } for r in all_results])
        summ.to_csv(out_root / "loso_summary.csv", index=False)
        with open(out_root / "loso_all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'=' * 70}\nLOSO SUMMARY ({args.arch}, {args.manifest})\n{'=' * 70}")
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nMean shortcut margin: {summ['shortcut_margin'].mean():+.4f}")
        print(f"Saved: {out_root / 'loso_summary.csv'}")


if __name__ == "__main__":
    main()
