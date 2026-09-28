"""The two CNNs the assignment requires: one from scratch, one by transfer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class ConvBlock(nn.Module):
    """two 3x3 convolutions + batch-norm + ReLU, then 2x2 max-pooling (slide 17)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.body(x)


class ScratchCNN(nn.Module):
    """Model 1 -- trained from scratch, ~1.2 M parameters.

    Four blocks at 32 -> 64 -> 128 -> 256 channels, then global average pooling
    and a single fully connected layer. Small on purpose: with 960 training
    clips a larger network memorises the actors instead of the emotions.
    """

    def __init__(self, n_classes: int = C.N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128), ConvBlock(128, 256)
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)                      # (B, 256, 4, 18)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # global average pooling
        return self.head(x)


class TransferResNet(nn.Module):
    """Model 2 -- ResNet-18 pretrained on ImageNet (slide 18).

    The spectrogram is a single-channel 64x301 image; ImageNet features expect
    three channels at roughly 224x224, so we repeat the channel and resize.
    `freeze` reproduces the frozen-backbone setup from the slide; unfreezing
    layer4 is reported alongside it, because a fully frozen ImageNet trunk
    transfers poorly to spectrograms -- the domain gap the slide warns about.
    """

    def __init__(
        self,
        n_classes: int = C.N_CLASSES,
        freeze: bool = True,
        unfreeze_layer4: bool = False,
        resize: int = 224,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.resize = resize
        # At inference the checkpoint carries every weight, so skip the ImageNet
        # download -- otherwise the deployed server stalls on first start.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_layer4:
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = True

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, n_classes))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        if self.resize:
            x = F.interpolate(x, size=(self.resize, self.resize),
                              mode="bilinear", align_corners=False)
        return self.head(self.backbone(x))

    def train(self, mode: bool = True):
        """Keep frozen BatchNorm layers in eval mode.

        A frozen backbone still updates its running statistics in train mode,
        which shifts the features the head was fitted to and makes validation
        drift away from training for no visible reason.
        """
        super().train(mode)
        if mode:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(
                    p.requires_grad for p in m.parameters(recurse=False)
                ):
                    m.eval()
        return self


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "scratch":
        return ScratchCNN(**kwargs)
    if name == "transfer":
        return TransferResNet(freeze=True, unfreeze_layer4=False, **kwargs)
    if name == "transfer_ft":
        return TransferResNet(freeze=True, unfreeze_layer4=True, **kwargs)
    raise ValueError(f"unknown model {name!r}")


#: Human-readable descriptions, shown in the app and the README.
MODEL_DESCRIPTIONS = {
    "scratch": "CNN trained from scratch (4 conv blocks, 1.2 M parameters)",
    "transfer": "ResNet-18 pretrained on ImageNet, backbone frozen",
    "transfer_ft": "ResNet-18 pretrained on ImageNet, layer4 fine-tuned",
}


MODEL_NAMES = ["scratch", "transfer", "transfer_ft"]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
