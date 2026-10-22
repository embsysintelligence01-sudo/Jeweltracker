from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from session_dataset import CLASSES


class TemporalMobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.pool = base.avgpool
        feature_count = base.classifier[0].in_features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(feature_count * 5, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(256, len(CLASSES)),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.pool(features).flatten(1)

    def forward(self, triplets: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = triplets.shape
        encoded = self.encode(
            triplets.reshape(batch * steps, channels, height, width)
        ).reshape(batch, steps, -1)
        baseline = encoded[:, 0]
        transition = encoded[:, 1]
        final = encoded[:, 2]
        combined = torch.cat((
            baseline,
            transition,
            final,
            torch.abs(baseline - transition),
            torch.abs(baseline - final),
        ), dim=1)
        return self.classifier(combined)

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self


class SharedTemporalBackbone(nn.Module):
    def __init__(self, architecture: str, pretrained: bool = True):
        super().__init__()
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.pool = base.avgpool
            feature_count = base.fc.in_features
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            self.pool = base.avgpool
            feature_count = base.classifier[-1].in_features
        else:
            raise ValueError(f"Unsupported shared architecture: {architecture}")
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(feature_count * 5, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(256, len(CLASSES)),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(images)).flatten(1)

    def forward(self, triplets: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = triplets.shape
        encoded = self.encode(
            triplets.reshape(batch * steps, channels, height, width)
        ).reshape(batch, steps, -1)
        baseline = encoded[:, 0]
        transition = encoded[:, 1]
        final = encoded[:, 2]
        return self.classifier(torch.cat((
            baseline,
            transition,
            final,
            torch.abs(baseline - transition),
            torch.abs(baseline - final),
        ), dim=1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self


class EarlyFusionMobileNetV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        original = base.features[0][0]
        replacement = nn.Conv2d(
            15,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight.repeat(1, 5, 1, 1) / 5.0)
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
        base.features[0][0] = replacement
        self.features = base.features
        self.pool = base.avgpool
        feature_count = base.classifier[0].in_features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for parameter in replacement.parameters():
            parameter.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(feature_count, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(256, len(CLASSES)),
        )

    def forward(self, triplets: torch.Tensor) -> torch.Tensor:
        baseline = triplets[:, 0]
        transition = triplets[:, 1]
        final = triplets[:, 2]
        fused = torch.cat((
            baseline,
            transition,
            final,
            torch.abs(baseline - transition),
            torch.abs(baseline - final),
        ), dim=1)
        return self.classifier(self.pool(self.features(fused)).flatten(1))

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


ARCHITECTURES = (
    "shared_mobilenet_v3_small",
    "early_fusion_mobilenet_v3_small",
    "shared_resnet18",
    "shared_efficientnet_b0",
)


def make_temporal_model(architecture: str, pretrained: bool = True) -> nn.Module:
    if architecture == "shared_mobilenet_v3_small":
        return TemporalMobileNetV3(pretrained=pretrained)
    if architecture == "early_fusion_mobilenet_v3_small":
        return EarlyFusionMobileNetV3(pretrained=pretrained)
    if architecture == "shared_resnet18":
        return SharedTemporalBackbone("resnet18", pretrained=pretrained)
    if architecture == "shared_efficientnet_b0":
        return SharedTemporalBackbone("efficientnet_b0", pretrained=pretrained)
    raise ValueError(f"Unsupported temporal architecture: {architecture}")
