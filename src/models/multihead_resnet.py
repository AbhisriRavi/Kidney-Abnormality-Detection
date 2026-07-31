"""
MultiHeadResNet50 — architectural innovation for kidney CT classification.

- ResNet50 backbone (ImageNet-pretrained)
- 4 independent binary classification heads (Normal, Cyst, Tumor, Stone)
  Each head produces a single logit; sigmoid at inference; argmax across
  the 4 sigmoid probabilities gives the final class prediction.
- Auxiliary segmentation decoder (mini U-Net-style upsampling with skip
  connections from ResNet50 stages 2, 3, 4).
  Outputs a single-channel [B, 1, 224, 224] kidney mask logit.

Design notes:
- Independent binary heads (rather than a shared 4-way softmax) allow
  per-class source-gated training — e.g. the Cyst head can train only
  on KiTS or KAUH samples while other heads train on the full dataset.
- Segmentation head serves as a learn-to-look auxiliary regulariser.
  Trained only on KiTS samples (only source with ground-truth masks).
"""
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """Basic Conv-BN-ReLU block for the segmentation decoder."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SegDecoder(nn.Module):
    """
    Segmentation decoder — small U-Net-style upsampler.
    Takes ResNet50 feature maps from stages 2, 3, 4 (spatial 28/14/7)
    and progressively upsamples to 224x224 single-channel mask logits.
    """
    def __init__(self, ch2=512, ch3=1024, ch4=2048):
        super().__init__()
        # Stage 4 (7x7, 2048) → 14x14, 512
        self.up1 = nn.ConvTranspose2d(ch4, 512, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(512 + ch3, 512)  # skip from stage 3 (14x14, 1024)

        # 14x14, 512 → 28x28, 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256 + ch2, 256)  # skip from stage 2 (28x28, 512)

        # 28x28, 256 → 56x56, 128 (no skip — layer 1 output shape differs)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 128)

        # 56x56, 128 → 112x112, 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(64, 64)

        # 112x112, 64 → 224x224, 32
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec5 = ConvBlock(32, 32)

        # Final 1x1 conv to single-channel logit
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, f2, f3, f4):
        # f2: [B, 512, 28, 28]
        # f3: [B, 1024, 14, 14]
        # f4: [B, 2048, 7, 7]
        x = self.up1(f4)                              # [B, 512, 14, 14]
        x = torch.cat([x, f3], dim=1)                 # [B, 512+1024, 14, 14]
        x = self.dec1(x)                              # [B, 512, 14, 14]

        x = self.up2(x)                               # [B, 256, 28, 28]
        x = torch.cat([x, f2], dim=1)                 # [B, 256+512, 28, 28]
        x = self.dec2(x)                              # [B, 256, 28, 28]

        x = self.up3(x)                               # [B, 128, 56, 56]
        x = self.dec3(x)

        x = self.up4(x)                               # [B, 64, 112, 112]
        x = self.dec4(x)

        x = self.up5(x)                               # [B, 32, 224, 224]
        x = self.dec5(x)

        return self.final(x)                          # [B, 1, 224, 224]


class MultiHeadResNet50(nn.Module):
    """
    ResNet50 backbone + 4 binary classification heads + segmentation decoder.

    Args:
        n_classes: number of classes for the classification heads. Fixed at 4
                   for this project but parameterised for flexibility.
        dropout:   dropout rate applied before each classification head.
        seg_enabled: whether to include the segmentation decoder. False disables it
                     entirely (for ablation studies).
    """
    CLASS_NAMES = ["Normal", "Cyst", "Tumor", "Stone"]

    def __init__(self, n_classes: int = 4, dropout: float = 0.2, seg_enabled: bool = True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Split ResNet50 into stem + 4 stages so we can grab intermediate features
        self.stem = nn.Sequential(
            base.conv1,       # [B, 64,  112, 112]
            base.bn1,
            base.relu,
            base.maxpool,     # [B, 64,  56, 56]
        )
        self.layer1 = base.layer1                          # [B, 256, 56, 56]
        self.layer2 = base.layer2                          # [B, 512, 28, 28]
        self.layer3 = base.layer3                          # [B, 1024, 14, 14]
        self.layer4 = base.layer4                          # [B, 2048, 7, 7]
        self.avgpool = base.avgpool                        # [B, 2048, 1, 1]

        # 4 independent binary classification heads
        feat_dim = 2048
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(feat_dim, 1) for _ in range(n_classes)
        ])

        self.seg_enabled = seg_enabled
        if seg_enabled:
            self.seg_decoder = SegDecoder(ch2=512, ch3=1024, ch4=2048)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor [B, 3, 224, 224]
        Returns:
            class_logits: [B, n_classes] — raw logits per binary head
            seg_logits:   [B, 1, 224, 224] or None if seg_enabled=False
        """
        x = self.stem(x)
        f1 = self.layer1(x)                          # [B, 256, 56, 56]
        f2 = self.layer2(f1)                         # [B, 512, 28, 28]
        f3 = self.layer3(f2)                         # [B, 1024, 14, 14]
        f4 = self.layer4(f3)                         # [B, 2048, 7, 7]

        # Classification path
        pooled = self.avgpool(f4).flatten(1)         # [B, 2048]
        pooled = self.dropout(pooled)
        class_logits = torch.cat([h(pooled) for h in self.heads], dim=1)  # [B, n_classes]

        # Segmentation path
        seg_logits = None
        if self.seg_enabled:
            seg_logits = self.seg_decoder(f2, f3, f4)  # [B, 1, 224, 224]

        return class_logits, seg_logits

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.
        Runs forward, applies sigmoid to each head, returns argmax.
        """
        class_logits, _ = self.forward(x)
        probs = torch.sigmoid(class_logits)          # [B, n_classes]
        return probs.argmax(dim=1)                   # [B]


def _sanity_check():
    """Quick shape check — run this file directly to verify."""
    print("Building MultiHeadResNet50...")
    model = MultiHeadResNet50(n_classes=4, dropout=0.2, seg_enabled=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    dummy = torch.randn(2, 3, 224, 224)
    class_logits, seg_logits = model(dummy)
    print(f"\nForward pass with [2, 3, 224, 224] input:")
    print(f"  class_logits shape: {tuple(class_logits.shape)}  (expect (2, 4))")
    print(f"  seg_logits shape:   {tuple(seg_logits.shape)}  (expect (2, 1, 224, 224))")

    preds = model.predict_class(dummy)
    print(f"  predict_class:      {tuple(preds.shape)}  (expect (2,))")

    # Test with seg disabled
    model_no_seg = MultiHeadResNet50(n_classes=4, seg_enabled=False)
    class_logits, seg_logits = model_no_seg(dummy)
    print(f"\nWith seg_enabled=False:")
    print(f"  class_logits shape: {tuple(class_logits.shape)}  (expect (2, 4))")
    print(f"  seg_logits:         {seg_logits}  (expect None)")


if __name__ == "__main__":
    _sanity_check()