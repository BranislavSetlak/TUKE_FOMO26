"""Downstream heads for the 3D DINO teacher backbone.

The attribute name ``teacher_backbone`` is deliberate: it makes the backbone
keys identical to those in the self-supervised Lightning checkpoint, so the
existing Asparagus checkpoint loader can transfer the EMA-teacher weights.
"""

from math import prod
from typing import Sequence

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets.vit import ViT
from torch import nn

from asparagus.modules.networks.vision_transformer import MaskedVisionTransformer


def _as_tuple3(value: Sequence[int]) -> tuple[int, int, int]:
    value = tuple(int(v) for v in value)
    if len(value) != 3:
        raise ValueError(f"Expected three spatial values, got {value}")
    return value


def _make_backbone(input_channels: int, img_size, patch_size, hidden_size: int):
    return MaskedVisionTransformer(
        vit=ViT(
            in_channels=input_channels,
            img_size=_as_tuple3(img_size),
            patch_size=_as_tuple3(patch_size),
            hidden_size=hidden_size,
            mlp_dim=3072,
            num_layers=8,
            num_heads=12,
            proj_type="conv",
            classification=True,
            spatial_dims=3,
        )
    )


class DINO3DClassifierRegressor(nn.Module):
    """Full fine-tuning model for 3D classification or regression."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(output_channels)
        self.patch_size = _as_tuple3(patch_size)
        self.teacher_backbone = _make_backbone(
            input_channels=input_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_classes),
        )
        self.stem_weight_name = "teacher_backbone.vit.patch_embedding.patch_embeddings.weight"

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_backbone(x, mask=None)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.forward_features(x))

    def freeze_backbone(self) -> None:
        for parameter in self.teacher_backbone.parameters():
            parameter.requires_grad = False


class DINO3DSegmenter(nn.Module):
    """ViT encoder with a lightweight 3D upsampling segmentation decoder."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size: int = 384,
    ):
        super().__init__()
        self.num_classes = int(output_channels)
        self.patch_size = _as_tuple3(patch_size)
        if self.patch_size != (16, 16, 16):
            raise ValueError("The supplied decoder is designed for a 16x16x16 patch embedding.")

        self.teacher_backbone = _make_backbone(
            input_channels=input_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, 192, kernel_size=2, stride=2),
            nn.GroupNorm(12, 192),
            nn.GELU(),
            nn.ConvTranspose3d(192, 96, kernel_size=2, stride=2),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.ConvTranspose3d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.ConvTranspose3d(48, 24, kernel_size=2, stride=2),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv3d(24, self.num_classes, kernel_size=1),
        )
        self.stem_weight_name = "teacher_backbone.vit.patch_embedding.patch_embeddings.weight"

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.teacher_backbone(x, mask=None)[:, 1:]
        grid = tuple(int(size // patch) for size, patch in zip(x.shape[2:], self.patch_size))
        if tokens.shape[1] != prod(grid):
            raise RuntimeError(f"Patch-token count {tokens.shape[1]} does not match grid {grid}.")
        return tokens.transpose(1, 2).reshape(x.shape[0], tokens.shape[2], *grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(self.forward_features(x))
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits

    def sliding_window_predict(self, data: torch.Tensor, patch_size, overlap: float = 0.5) -> torch.Tensor:
        return sliding_window_inference(
            inputs=data,
            roi_size=tuple(int(v) for v in patch_size),
            sw_batch_size=1,
            predictor=self.forward,
            overlap=float(overlap),
        )

    def freeze_backbone(self) -> None:
        for parameter in self.teacher_backbone.parameters():
            parameter.requires_grad = False


def dino3d_clsreg(input_channels: int, output_channels: int, **kwargs):
    return DINO3DClassifierRegressor(input_channels, output_channels, **kwargs)


def dino3d_seg(input_channels: int, output_channels: int, **kwargs):
    return DINO3DSegmenter(input_channels, output_channels, **kwargs)
