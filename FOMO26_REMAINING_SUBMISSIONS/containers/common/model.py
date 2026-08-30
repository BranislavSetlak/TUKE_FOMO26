"""Inference-only TUKE SwinUNETR models used by FOMO26 containers."""

from __future__ import annotations

import torch
from monai.networks.nets import SwinUNETR
from torch import nn


ARCHITECTURE = {
    "depths": (2, 2, 2, 2),
    "num_heads": (3, 6, 12, 24),
    "feature_size": 48,
}


def _backbone(in_channels: int, out_channels: int) -> SwinUNETR:
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        depths=ARCHITECTURE["depths"],
        num_heads=ARCHITECTURE["num_heads"],
        feature_size=ARCHITECTURE["feature_size"],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        use_v2=False,
    )


class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = _backbone(in_channels, out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ClsRegModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        full = _backbone(in_channels, out_channels)
        self.encoder = full.swinViT
        self.normalize = bool(full.normalize)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.LayerNorm(ARCHITECTURE["feature_size"] * 16),
            nn.Dropout(p=0.1),
            nn.Linear(ARCHITECTURE["feature_size"] * 16, out_channels),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(image, self.normalize)
        return self.head(hidden[4])


class MultiScaleEmbeddingModel(nn.Module):
    """Pool the four semantic Swin stages into one fixed 1-D representation."""

    embedding_dim = 96 + 192 + 384 + 768

    def __init__(self) -> None:
        super().__init__()
        full = _backbone(1, 1)
        self.encoder = full.swinViT
        self.normalize = bool(full.normalize)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(image, self.normalize)
        pooled = [stage.float().mean(dim=(2, 3, 4)) for stage in hidden[1:5]]
        embedding = torch.cat(pooled, dim=1)
        return torch.nn.functional.normalize(embedding, p=2, dim=1, eps=1e-8)
