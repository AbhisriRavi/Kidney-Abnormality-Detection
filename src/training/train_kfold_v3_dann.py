"""
DANN training on v3 with gradient-reversal adversarial source head.

Backbone: ResNet-50 (ImageNet-pretrained, IMAGENET1K_V2).
Class head: 4-way classification (Normal/Cyst/Tumor/Stone).
Source head: 5-way source classification (kits/mendeley/kauh/tcga/abdalla),
             preceded by a gradient reversal layer with scalar lambda.

Loss:  L = L_class(class_head(features)) + lambda * L_source(source_head(GRL(features)))
"""
import os
import csv
import json
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------
# Gradient Reversal Layer (Ganin & Lempitsky, 2015)
# ----------------------------------------------------------------------
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)


# ----------------------------------------------------------------------
# DANN Model wrapper
# ----------------------------------------------------------------------
class DANNResNet50(nn.Module):
    def __init__(self, n_classes, n_sources, dropout=0.1):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Everything except the final fc becomes the shared feature extractor
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # includes avgpool
        feat_dim = base.fc.in_features  # 2048

        # Class head
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )
        # Source (adversarial) head — 2-layer MLP is standard
        self.source_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_sources),
        )

    def forward(self, x, lambda_=1.0):
        feats = self.backbone(x)  # [B, 2048, 1, 1]
        class_logits = self.class_head(feats)
        source_logits = self.source_head(grad_reverse(feats, lambda_))
        return class_logits, source_logits


# ----------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tag", type=str, default="v3_dann_run1")
parser.add_argument("--seed-suffix", type=str, default="")
parser.add_argument("--folds-to-run", type=str, default="all")
parser.add_argument("--aug-strength", type=str, default="strong",
                    choices=["mild", "moderate", "strong"])
parser.add_argument("--dann-lambda", type=float, default=0.1,
                    help="Fixed lambda for gradient reversal")
parser.add_argument("--dann-schedule", type=str, default="fixed",
                    choices=["fixed", "linear_ramp"],
                    help="'fixed' uses --dann-lambda; 'linear_ramp' ramps 0->lambda over training")
args = parser.parse_args()

SCRATCH = Path(os.environ["SCRATCH"])
V3_DIR = SCRATCH / "kidney-data/processed/unified_v3"
manifest_name = f"manifest_with_folds{args.seed_suffix}.csv"
MANIFEST = V3_DIR / manifest_name
if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")
print(f"Using manifest: {MANIFEST}")
OUT = SCRATCH / "kidney-results/kfold" / args.tag
OUT.mkdir(parents=True, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
N_FOLDS = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"Device: {DEVICE} | Output: {OUT}")
print(f"HP: lr={args.lr}, wd={args.weight_decay}, bs={args.batch_size}, "
      f"aug={args.aug_strength}, folds={args.folds_to_run}")
print(f"DANN: lambda={args.dann_lambda}, schedule={args.dann_schedule}")

df = pd.read_csv(MANIFEST)
SOURCES = sorted(df["source"].unique())
SOURCE2IDX = {s: i for i, s in enumerate(SOURCES)}
print(f"Sources: {SOURCES}")
print(f"Loaded {len(df)} images, {df['patient_id'].nunique()} patients")


class KidneyDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("L").convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, LABEL2IDX[row["label"]], SOURCE2IDX[row["source"]]


def build_train_transforms(strength):
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if strength == "mild":
        base += [transforms.RandomHorizontalFlip(0.5)]
    elif strength == "moderate":
        base += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    elif strength == "strong":
        base += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(base)


train_tf = build_train_transforms(args.aug_strength)
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

fold_metrics = []

if args.folds_to_run == "all":
    folds_to_run = list(range(N_FOLDS))
else:
    folds_to_run = [int(x) for x in args.folds_to_run.split(",")]

for fold_idx in folds_to_run:
    print(f"\n{'=' * 60}\nFOLD {fold_idx} / {N_FOLDS - 1}\n{'=' * 60}")
    fold_out = OUT / f"fold_{fold_idx}"
    fold_out.mkdir(exist_ok=True)

    test_df = df[df["fold"] == fold_idx].reset_index(drop=True)
    trainval_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(
        trainval_df.index.values, groups=trainval_df["patient_id"].values
    ))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    print(f"  Train: {len(train_df):5d} / Val: {len(val_df):5d} / Test: {len(test_df):5d}")

    class_counts = train_df["label"].value_counts()
    total = len(train_df)
    class_weights = torch.tensor(
        [total / (len(CLASSES) * class_counts.get(c, 1)) for c in CLASSES],
        dtype=torch.float32
    ).to(DEVICE)
    source_counts = train_df["source"].value_counts()
    source_weights = torch.tensor(
        [total / (len(SOURCES) * source_counts.get(s, 1)) for s in SOURCES],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"  Class weights: {dict(zip(CLASSES, [round(w, 3) for w in class_weights.tolist()]))}")
    print(f"  Source weights: {dict(zip(SOURCES, [round(w, 3) for w in source_weights.tolist()]))}")

    train_loader = DataLoader(KidneyDataset(train_df, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(KidneyDataset(val_df, eval_tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(KidneyDataset(test_df, eval_tf),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = DANNResNet50(n_classes=len(CLASSES), n_sources=len(SOURCES),
                          dropout=0.1).to(DEVICE)

    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    source_criterion = nn.CrossEntropyLoss(weight=source_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    log_path = fold_out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_class_loss", "train_source_loss", "train_class_acc",
            "train_source_acc", "val_class_loss", "val_class_acc",
            "val_class_f1", "val_source_acc", "lambda", "lr"
        ])

    best_val_f1 = 0.0
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        tcl, tsl, tcc, tsc, tn = 0.0, 0.0, 0, 0, 0
        for img, y, s in train_loader:
            img = img.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            s = s.to(DEVICE, non_blocking=True)

            # Lambda schedule
            if args.dann_schedule == "linear_ramp":
                p = global_step / max(total_steps, 1)
                current_lambda = args.dann_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            else:
                current_lambda = args.dann_lambda

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                class_logits, source_logits = model(img, lambda_=current_lambda)
                l_class = class_criterion(class_logits, y)
                l_source = source_criterion(source_logits, s)
                loss = l_class + l_source  # source loss already scaled by lambda via GRL
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tcl += l_class.item() * img.size(0)
            tsl += l_source.item() * img.size(0)
            tcc += (class_logits.argmax(1) == y).sum().item()
            tsc += (source_logits.argmax(1) == s).sum().item()
            tn += img.size(0)
            global_step += 1
        tcl /= tn; tsl /= tn; tca = tcc / tn; tsa = tsc / tn

        model.eval()
        vcl, vcc, vn, vp, vt = 0.0, 0, 0, [], []
        vsc = 0
        with torch.no_grad():
            for img, y, s in val_loader:
                img = img.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                s = s.to(DEVICE, non_blocking=True)
                with autocast():
                    class_logits, source_logits = model(img, lambda_=0.0)
                    l_class = class_criterion(class_logits, y)
                vcl += l_class.item() * img.size(0)
                p = class_logits.argmax(1)
                vcc += (p == y).sum().item(); vn += img.size(0)
                vsc += (source_logits.argmax(1) == s).sum().item()
                vp.extend(p.cpu().numpy()); vt.extend(y.cpu().numpy())
        vcl /= vn; vca = vcc / vn; vsa = vsc / vn
        val_f1 = f1_score(vt, vp, average="macro", zero_division=0)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Ep{epoch+1:02d}/{args.epochs} | "
              f"tr_cls_acc={tca:.3f} tr_src_acc={tsa:.3f} "
              f"| val_cls_acc={vca:.3f} val_F1={val_f1:.3f} val_src_acc={vsa:.3f} "
              f"| lam={current_lambda:.3f} lr={lr_now:.2e}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, f"{tcl:.6f}", f"{tsl:.6f}", f"{tca:.6f}",
                f"{tsa:.6f}", f"{vcl:.6f}", f"{vca:.6f}",
                f"{val_f1:.6f}", f"{vsa:.6f}",
                f"{current_lambda:.4f}", f"{lr_now:.2e}"
            ])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                        "val_macro_f1": val_f1, "fold": fold_idx,
                        "lambda": args.dann_lambda},
                       fold_out / "best.pth")

    # Test evaluation on best checkpoint
    print(f"  Loading best checkpoint (val_F1={best_val_f1:.4f})...")
    ckpt = torch.load(fold_out / "best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    t_p, t_t, t_pr, t_s = [], [], [], []
    with torch.no_grad():
        for img, y, s in test_loader:
            img = img.to(DEVICE, non_blocking=True)
            with autocast():
                class_logits, _ = model(img, lambda_=0.0)
            t_pr.extend(torch.softmax(class_logits, dim=1).cpu().numpy())
            t_p.extend(class_logits.argmax(1).cpu().numpy())
            t_t.extend(y.numpy())
            t_s.extend(s.numpy())
    t_p = np.array(t_p); t_t = np.array(t_t); t_pr = np.array(t_pr); t_s = np.array(t_s)

    acc = accuracy_score(t_t, t_p)
    macro_f1 = f1_score(t_t, t_p, average="macro", zero_division=0)
    pcl = precision_recall_fscore_support(t_t, t_p, labels=range(len(CLASSES)), zero_division=0)
    aucs = {}
    for i, c in enumerate(CLASSES):
        try:
            aucs[c] = float(roc_auc_score((t_t == i).astype(int), t_pr[:, i]))
        except ValueError:
            aucs[c] = None
    macro_auc = float(np.mean([v for v in aucs.values() if v is not None]))
    cm = confusion_matrix(t_t, t_p, labels=range(len(CLASSES)))

    # Per-source test accuracy
    per_source = {}
    for src_name, src_idx in SOURCE2IDX.items():
        mask = t_s == src_idx
        if mask.sum() == 0:
            continue
        per_source[src_name] = {
            "n_images": int(mask.sum()),
            "accuracy": float((t_p[mask] == t_t[mask]).mean()),
            "macro_f1": float(f1_score(t_t[mask], t_p[mask],
                                        labels=range(len(CLASSES)),
                                        average="macro", zero_division=0)),
        }

    fold_result = {
        "fold": fold_idx,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(best_val_f1),
        "dann_lambda": args.dann_lambda,
        "dann_schedule": args.dann_schedule,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "test_macro_auc": macro_auc,
        "per_class": {c: {
            "precision": float(pcl[0][i]), "recall": float(pcl[1][i]),
            "f1": float(pcl[2][i]), "support": int(pcl[3][i]), "auc": aucs[c]
        } for i, c in enumerate(CLASSES)},
        "per_source": per_source,
        "confusion_matrix": cm.tolist(),
    }
    with open(fold_out / "metrics.json", "w") as f:
        json.dump(fold_result, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold_idx} | acc={acc:.3f} F1={macro_f1:.3f} λ={args.dann_lambda}")
    plt.tight_layout()
    plt.savefig(fold_out / "confusion_matrix.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"  Fold {fold_idx} test: acc={acc:.4f}, F1={macro_f1:.4f}, AUC={macro_auc:.4f}")
    print(f"  Per-source acc: {dict((k, round(v['accuracy'], 3)) for k, v in per_source.items())}")
    fold_metrics.append(fold_result)

# Aggregate
print(f"\n{'=' * 60}\nAGGREGATE ({len(fold_metrics)} folds)\n{'=' * 60}")
accs = [m["test_accuracy"] for m in fold_metrics]
f1s = [m["test_macro_f1"] for m in fold_metrics]
aucs = [m["test_macro_auc"] for m in fold_metrics]
print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Macro-F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Macro-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
for c in CLASSES:
    pc_f1 = [m["per_class"][c]["f1"] for m in fold_metrics]
    print(f"  {c:<8} F1={np.mean(pc_f1):.4f} ± {np.std(pc_f1):.4f}")

summary = {
    "n_folds": len(fold_metrics),
    "epochs": args.epochs, "batch_size": args.batch_size,
    "lr": args.lr, "weight_decay": args.weight_decay,
    "aug_strength": args.aug_strength, "seed": args.seed,
    "manifest": "v3", "seed_suffix": args.seed_suffix,
    "dann_lambda": args.dann_lambda,
    "dann_schedule": args.dann_schedule,
    "folds_run": folds_to_run,
    "fold_results": fold_metrics,
    "aggregate": {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "macro_auc_mean": float(np.mean(aucs)), "macro_auc_std": float(np.std(aucs)),
        "per_class_f1": {c: {
            "mean": float(np.mean([m["per_class"][c]["f1"] for m in fold_metrics])),
            "std": float(np.std([m["per_class"][c]["f1"] for m in fold_metrics])),
        } for c in CLASSES},
    },
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)