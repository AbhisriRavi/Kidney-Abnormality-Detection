"""
CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).
Inserted between the residual sum and the final ReLU of each ResNet Bottleneck.
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


def add_cbam_to_resnet(model, reduction=16):
    """
    Wrap each Bottleneck's forward so CBAM is applied to the conv branch
    BEFORE the residual add. This preserves the skip connection identity.
    """
    from torchvision.models.resnet import Bottleneck
    import types

    def make_forward(cbam_module):
        def cbam_forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            out = cbam_module(out)  # CBAM applied before residual add
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            out = self.relu(out)
            return out
        return cbam_forward

    for module in model.modules():
        if isinstance(module, Bottleneck):
            channels = module.bn3.num_features
            cbam = CBAM(channels, reduction=reduction)
            module.add_module("cbam", cbam)
            module.forward = types.MethodType(make_forward(module.cbam), module)
    return model