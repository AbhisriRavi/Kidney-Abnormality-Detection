"""
Train a 2D U-Net for binary kidney segmentation on KiTS23 axial slices.

Outputs:
  results/segmenter/best.pth        - best checkpoint by validation Dice
  results/segmenter/log.csv         - per-epoch metrics
  results/segmenter/sample_preds.png - validation predictions visualised
"""
import os
import csv
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

import monai
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    EnsureChannelFirstd, ToTensord
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRATCH = Path(os.environ["SCRATCH"])
SLICES = SCRATCH / "kidney-data/processed/kits_axial_slices"
OUT = SCRATCH / "kidney-results/segmenter"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 50
IMG_SIZE = 256  # downsample from 512 to fit batch size on GPU comfortably

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Slices directory: {SLICES}")

# ---------------------------------------------------------------------------
# Case-level train/val split
# ---------------------------------------------------------------------------
img_files = sorted(SLICES.glob("*_img.npy"))
all_cases = sorted({f.name.split("_slice")[0] for f in img_files})
print(f"Total cases: {len(all_cases)}, total slices: {len(img_files)}")

train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=SEED)
train_cases, val_cases = set(train_cases), set(val_cases)

train_files = [f for f in img_files if f.name.split("_slice")[0] in train_cases]
val_files = [f for f in img_files if f.name.split("_slice")[0] in val_cases]
print(f"Train: {len(train_files)} slices from {len(train_cases)} cases")
print(f"Val:   {len(val_files)} slices from {len(val_cases)} cases")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class KiTSSliceDataset(Dataset):
    def __init__(self, img_files, transform=None):
        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = img_path.parent / img_path.name.replace("_img.npy", "_mask.npy")
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        # Resize to IMG_SIZE x IMG_SIZE
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        img_t = torch.nn.functional.interpolate(img_t, size=IMG_SIZE, mode="bilinear", align_corners=False)
        mask_t = torch.nn.functional.interpolate(mask_t, size=IMG_SIZE, mode="nearest")
        sample = {"image": img_t.squeeze(0), "mask": mask_t.squeeze(0).long()}

        if self.transform:
            sample = self.transform(sample)
        return sample

train_tf = Compose([
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "mask"], prob=0.5),
    RandAffined(keys=["image", "mask"], prob=0.5,
                rotate_range=0.1, scale_range=0.1, padding_mode="zeros"),
    RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
])

train_ds = KiTSSliceDataset(train_files, train_tf)
val_ds = KiTSSliceDataset(val_files, None)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ---------------------------------------------------------------------------
# Model, loss, optimiser
# ---------------------------------------------------------------------------
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
    dropout=0.1,
).to(DEVICE)

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler()
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
log_path = OUT / "log.csv"
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "lr"])

best_dice = 0.0
for epoch in range(EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    n = 0
    for batch in train_loader:
        img = batch["image"].to(DEVICE, non_blocking=True)
        mask = batch["mask"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(img)
            loss = loss_fn(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * img.size(0)
        n += img.size(0)
    train_loss = epoch_loss / n

    # Validate
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(img)
            pred = logits.argmax(dim=1, keepdim=True)
            pred_onehot = monai.networks.one_hot(pred, num_classes=2)
            mask_onehot = monai.networks.one_hot(mask, num_classes=2)
            dice_metric(y_pred=pred_onehot, y=mask_onehot)
    val_dice = dice_metric.aggregate().item()

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_dice={val_dice:.4f} | lr={current_lr:.2e}")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, f"{train_loss:.6f}", f"{val_dice:.6f}", f"{current_lr:.2e}"])

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "val_dice": val_dice,
            "config": {"img_size": IMG_SIZE, "channels": (32, 64, 128, 256, 512)},
        }, OUT / "best.pth")
        print(f"  ✓ New best, saved checkpoint (Dice={val_dice:.4f})")

print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")
print(f"Checkpoint: {OUT / 'best.pth'}")
print(f"Log: {log_path}")

# ---------------------------------------------------------------------------
# Save a few example predictions
# ---------------------------------------------------------------------------
print("Generating example predictions...")
ckpt = torch.load(OUT / "best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
val_iter = iter(val_loader)
batch = next(val_iter)
img = batch["image"].to(DEVICE)
mask = batch["mask"]
with torch.no_grad():
    logits = model(img)
pred = logits.argmax(dim=1).cpu().numpy()
img_np = img.cpu().numpy()
mask_np = mask.numpy()

for i in range(4):
    axes[i, 0].imshow(img_np[i, 0], cmap="gray")
    axes[i, 0].set_title("Image")
    axes[i, 1].imshow(img_np[i, 0], cmap="gray")
    axes[i, 1].imshow(mask_np[i, 0], cmap="Reds", alpha=0.4)
    axes[i, 1].set_title("Ground truth")
    axes[i, 2].imshow(img_np[i, 0], cmap="gray")
    axes[i, 2].imshow(pred[i], cmap="Greens", alpha=0.4)
    axes[i, 2].set_title("Prediction")
    for j in range(3):
        axes[i, j].axis("off")
plt.tight_layout()
plt.savefig(OUT / "sample_preds.png", dpi=110, bbox_inches="tight")
print(f"Saved sample predictions to {OUT / 'sample_preds.png'}")