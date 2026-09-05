"""ConvNeXt-Base with a regression head that predicts a single scalar
(building height in metres) from one satellite tile."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Base_Weights


class ConvNeXtRegression(nn.Module):
    """Pretrained ConvNeXt feature extractor + a small MLP regression head.

    The head pools the feature map to a single vector and narrows it
    1024 -> 512 -> 128 -> 1, with dropout that decreases with depth (0.3 /
    0.2 / 0.1): heavier regularisation near the large pretrained features,
    lighter near the output where the signal is already compressed.
    """

    def __init__(self, backbone: nn.Module, num_features: int):
        super().__init__()
        self.features = backbone.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(-1) so the output is (B,) rather than (B, 1), matching the
        # shape of the target tensor.
        return self.head(self.features(x)).squeeze(-1)


def build_model(pretrained: bool = True) -> ConvNeXtRegression:
    """Builds the regression model.

    pretrained=False skips the ImageNet weight download, which is what the
    tests use -- the architecture and output shape can be checked without
    pulling ~350 MB.
    """
    weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    backbone = models.convnext_base(weights=weights)

    # Channel count out of the feature extractor. It does not depend on the
    # input's spatial size, so probe with a small tile rather than a full
    # 384x384 one -- same answer, far less compute.
    with torch.no_grad():
        num_features = backbone.features(torch.zeros(1, 3, 64, 64)).shape[1]

    return ConvNeXtRegression(backbone, num_features)


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable
