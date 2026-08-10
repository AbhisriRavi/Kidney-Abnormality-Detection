"""
Single place where every architecture in the project is constructed.

Before this module, each of the 15 train_kfold_*.py scripts built its model
inline, which meant an evaluation script could not reload a checkpoint without
duplicating the construction logic. Everything now goes through `build_model`.

Architectures:
    resnet50    -- plain ResNet50, 4-way softmax head
    cbam        -- ResNet50 with CBAM inserted in every Bottleneck
    vit         -- timm ViT-Base/16-224
    multihead   -- MultiHeadResNet50 (4 binary heads + aux seg decoder)
    dann        -- ResNet50 backbone + class head + GRL source head
"""
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

ARCHS = ["resnet50", "cbam", "vit", "multihead", "dann"]


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)


class DANNResNet50(nn.Module):
    """ResNet50 backbone with a class head and a gradient-reversed source head."""

    def __init__(self, n_classes: int = 4, n_sources: int = 5, dropout: float = 0.2):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(feat_dim, n_classes)
        self.source_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_sources),
        )

    def forward(self, x, lambda_: float = 1.0):
        feats = self.dropout(self.backbone(x))
        class_logits = self.class_head(feats)
        source_logits = self.source_head(grad_reverse(feats, lambda_))
        return class_logits, source_logits


def build_model(
    arch: str,
    n_classes: int = 4,
    n_sources: int = 5,
    dropout: float = 0.2,
    seg_enabled: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Construct one of the project's architectures by name."""
    arch = arch.lower()

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model

    if arch == "cbam":
        from src.models.cbam import add_cbam_to_resnet
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model = add_cbam_to_resnet(model)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model

    if arch == "vit":
        import timm
        return timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=n_classes
        )

    if arch == "multihead":
        from src.models.multihead_resnet import MultiHeadResNet50
        return MultiHeadResNet50(
            n_classes=n_classes, dropout=dropout, seg_enabled=seg_enabled
        )

    if arch == "dann":
        from src.models.dann import DANNResNet50 as _D
        return _D(n_classes=n_classes, n_sources=n_sources, dropout=dropout)

    raise ValueError(f"Unknown arch '{arch}'. Expected one of {ARCHS}.")


def forward_class_logits(model: nn.Module, arch: str, x: torch.Tensor) -> torch.Tensor:
    """
    Uniform forward pass that returns [B, n_classes] class logits regardless of
    which architecture is in use. Lets evaluation code stay architecture-agnostic.
    """
    arch = arch.lower()
    if arch == "multihead":
        class_logits, _ = model(x)
        return class_logits
    if arch == "dann":
        class_logits, _ = model(x, lambda_=0.0)
        return class_logits
    return model(x)


def probs_from_logits(logits: torch.Tensor, arch: str) -> torch.Tensor:
    """
    Convert logits to a probability vector over classes.

    The multi-head model uses independent binary heads, so its outputs are
    sigmoids that do not sum to 1. They are renormalised here so that the same
    downstream metric code (AUC, calibration, patient-level averaging) applies.
    """
    if arch.lower() == "multihead":
        p = torch.sigmoid(logits)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.softmax(logits, dim=1)
