"""Minimal inference-only TUKE SwinUNETR infarct classifier."""

from __future__ import annotations

import torch
from monai.networks.nets import SwinUNETR
from torch import nn


class InfarctSwinClassifier(nn.Module):
    """Swin encoder and downstream head used by the fine-tuning run.

    The reconstruction decoder is intentionally discarded after construction.
    This preserves the trained encoder architecture while keeping inference
    memory and exported checkpoints substantially smaller.
    """

    def __init__(self) -> None:
        super().__init__()
        full_model = SwinUNETR(
            in_channels=3,
            out_channels=2,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            spatial_dims=3,
            use_v2=False,
        )
        self.encoder = full_model.swinViT
        self.normalize = bool(full_model.normalize)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.LayerNorm(48 * 16),
            nn.Dropout(p=0.1),
            nn.Linear(48 * 16, 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder(image, self.normalize)
        return self.head(hidden_states[4])

