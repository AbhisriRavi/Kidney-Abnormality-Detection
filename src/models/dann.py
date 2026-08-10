import torch
import torch.nn as nn
from torchvision import models

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)

class DANNResNet50(nn.Module):
    """Matches the checkpoint layout written by train_kfold_*_dann.py."""
    def __init__(self, n_classes=4, n_sources=4, dropout=0.2, pretrained=False):
        super().__init__()
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=w)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.class_head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2048, n_classes))
        self.source_head = nn.Sequential(nn.Flatten(), nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, n_sources))
    def forward(self, x, lambda_=1.0):
        f = self.backbone(x)
        return self.class_head(f), self.source_head(grad_reverse(f, lambda_))
