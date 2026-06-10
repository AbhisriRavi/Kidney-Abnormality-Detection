"""
Train a 4-class kidney abnormality classifier on the Kaggle axial subset.
--mode full : trains on full-image axial slices
--mode roi  : trains on segmenter-cropped kidney ROIs

Both modes use the same manifest, same split, same architecture, same hyperparams.
Only the input image differs. This is the RQ1 comparison.
"""
import os, csv, json, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, accuracy_score, precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["full", "roi"], required=True)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

SCRATCH = Path(os.environ["SCRATCH"])
MANIFEST = SCRATCH / "kidney-data/processed/classification_manifest.csv"
OUT = SCRATCH / "kidney-results/classification" / args.mode
OUT.mkdir(parents=True, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}
IMG_SIZE = 224
print(f"Mode={args.mode}  Device={DEVICE}  Output={OUT}")

df = pd.read_csv(MANIFEST)
PATH_COL = "full_image_path" if args.mode == "full" else "roi_path"
train_df = df[df.split == "train"].reset_index(drop=True)
val_df = df[df.split == "val"].reset_index(drop=True)
test_df = df[df.split == "test"].reset_index(drop=True)
print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")

# Class weights (inverse frequency, normalised)
class_counts = train_df["label"].value_counts()
total = len(train_df)
class_weights = torch.tensor(
    [total / (4 * class_counts[c]) for c in CLASSES], dtype=torch.float32
).to(DEVICE)
print(f"Class weights: {dict(zip(CLASSES, [round(w, 3) for w in class_weights.tolist()]))}")

class KidneyClassDataset(Dataset):
    def __init__(self, df, path_col, transform=None):
        self.df = df
        self.path_col = path_col
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row[self.path_col]).convert("L").convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, LABEL2IDX[row["label"]]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader = DataLoader(KidneyClassDataset(train_df, PATH_COL, train_tf),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(KidneyClassDataset(val_df, PATH_COL, eval_tf),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(KidneyClassDataset(test_df, PATH_COL, eval_tf),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

# ResNet50 with ImageNet pretrained weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scaler = GradScaler()

log_path = OUT / "log.csv"
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"])

best_macro_f1 = 0.0
for epoch in range(args.epochs):
    # Train
    model.train()
    tl, tc, tn = 0, 0, 0
    for img, y in train_loader:
        img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model(img)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tl += loss.item() * img.size(0); tc += (out.argmax(1) == y).sum().item(); tn += img.size(0)
    tl /= tn; ta = tc / tn

    # Val
    model.eval()
    vl, vc, vn, vp, vt = 0, 0, 0, [], []
    with torch.no_grad():
        for img, y in val_loader:
            img, y = img.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with autocast():
                out = model(img); loss = criterion(out, y)
            vl += loss.item() * img.size(0)
            p = out.argmax(1)
            vc += (p == y).sum().item(); vn += img.size(0)
            vp.extend(p.cpu().numpy()); vt.extend(y.cpu().numpy())
    vl /= vn; va = vc / vn
    v_macro_f1 = f1_score(vt, vp, average="macro")
    scheduler.step()
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{args.epochs} | train_loss={tl:.4f} train_acc={ta:.4f} | "
          f"val_loss={vl:.4f} val_acc={va:.4f} val_macroF1={v_macro_f1:.4f} | lr={lr_now:.2e}")
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, f"{tl:.6f}", f"{ta:.6f}", f"{vl:.6f}",
                                 f"{va:.6f}", f"{v_macro_f1:.6f}", f"{lr_now:.2e}"])
    if v_macro_f1 > best_macro_f1:
        best_macro_f1 = v_macro_f1
        torch.save({"epoch": epoch + 1, "model_state": model.state_dict(),
                    "val_macro_f1": v_macro_f1, "mode": args.mode}, OUT / "best.pth")
        print(f"  ✓ New best (macro-F1={v_macro_f1:.4f})")

# Final test eval using best checkpoint
print(f"\nLoading best checkpoint (val macro-F1={best_macro_f1:.4f})...")
ckpt = torch.load(OUT / "best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

t_preds, t_targets, t_probs = [], [], []
with torch.no_grad():
    for img, y in test_loader:
        img = img.to(DEVICE, non_blocking=True)
        with autocast():
            out = model(img)
        t_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
        t_preds.extend(out.argmax(1).cpu().numpy())
        t_targets.extend(y.numpy())
t_preds = np.array(t_preds); t_targets = np.array(t_targets); t_probs = np.array(t_probs)

test_acc = accuracy_score(t_targets, t_preds)
test_macro_f1 = f1_score(t_targets, t_preds, average="macro")
pcl = precision_recall_fscore_support(t_targets, t_preds, labels=range(4), zero_division=0)
auc_pc = {}
for i, c in enumerate(CLASSES):
    try:
        auc_pc[c] = float(roc_auc_score((t_targets == i).astype(int), t_probs[:, i]))
    except ValueError:
        auc_pc[c] = None
macro_auc = float(np.mean([v for v in auc_pc.values() if v is not None]))
cm = confusion_matrix(t_targets, t_preds)

metrics = {
    "mode": args.mode, "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_macro_f1), "test_macro_auc": macro_auc,
    "per_class": {c: {"precision": float(pcl[0][i]), "recall": float(pcl[1][i]),
                       "f1": float(pcl[2][i]), "support": int(pcl[3][i]),
                       "auc": auc_pc[c]} for i, c in enumerate(CLASSES)},
    "confusion_matrix": cm.tolist(), "classes": CLASSES,
    "best_val_macro_f1": float(best_macro_f1), "best_epoch": int(ckpt["epoch"]),
}
with open(OUT / "final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

pred_df = test_df.copy()
pred_df["predicted_label"] = [IDX2LABEL[p] for p in t_preds]
for i, c in enumerate(CLASSES):
    pred_df[f"prob_{c}"] = t_probs[:, i]
pred_df.to_csv(OUT / "predictions.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (mode={args.mode}, acc={test_acc:.3f})")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=110, bbox_inches="tight")

print(f"\n=== Final results (mode={args.mode}) ===")
print(f"Test accuracy:  {test_acc:.4f}")
print(f"Test macro-F1:  {test_macro_f1:.4f}")
print(f"Test macro-AUC: {macro_auc:.4f}")
print("\nPer-class:")
for c in CLASSES:
    pc = metrics["per_class"][c]
    auc_str = f"{pc['auc']:.4f}" if pc["auc"] is not None else "N/A"
    print(f"  {c:<8} P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f} AUC={auc_str} (n={pc['support']})")
print(f"\nSaved to {OUT}")