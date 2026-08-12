"""3D SwinUNETR with masked reconstruction and MRI-sequence heads."""

from typing import Sequence

import torch
from monai.networks.nets import SwinUNETR
from torch import nn


class SwinUNETRHybrid(nn.Module):
    """MONAI SwinUNETR with an auxiliary classifier on its deepest features.

    ``forward`` remains reconstruction-only so the network behaves like a
    conventional SwinUNETR. ``forward_with_features`` exposes the bottleneck
    and sequence logits expected by :class:`SelfSupervisedModule`.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_sequence_classes: int = 14,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        sequence_dropout_rate: float = 0.0,
        use_checkpoint: bool = True,
        use_v2: bool = False,
    ):
        super().__init__()
        if num_sequence_classes <= 1:
            raise ValueError("num_sequence_classes must be greater than one")
        if feature_size % 12 != 0:
            raise ValueError("MONAI SwinUNETR feature_size must be divisible by 12")

        self.num_sequence_classes = int(num_sequence_classes)
        self.feature_size = int(feature_size)
        self.swin_unetr = SwinUNETR(
            in_channels=input_channels,
            out_channels=output_channels,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            use_v2=use_v2,
        )
        self.sequence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Dropout(p=sequence_dropout_rate),
            nn.Linear(feature_size * 16, num_sequence_classes),
        )

    @property
    def encoder(self):
        """Expose the Swin encoder for downstream checkpoint consumers."""

        return self.swin_unetr.swinViT

    def _hidden_states(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.swin_unetr.swinViT(x, self.swin_unetr.normalize)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deepest Swin encoder feature map without decoding."""

        return self._hidden_states(x)[4]

    def _decode(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states[0])
        enc2 = self.swin_unetr.encoder3(hidden_states[1])
        enc3 = self.swin_unetr.encoder4(hidden_states[2])
        dec4 = self.swin_unetr.encoder10(hidden_states[4])
        dec3 = self.swin_unetr.decoder5(dec4, hidden_states[3])
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        decoded = self.swin_unetr.decoder1(dec0, enc0)
        return self.swin_unetr.out(decoded)

    def forward_with_features(self, x: torch.Tensor):
        hidden_states = self._hidden_states(x)
        bottleneck = hidden_states[4]
        reconstruction = self._decode(x, hidden_states)
        sequence_logits = self.sequence_head(bottleneck)
        return reconstruction, bottleneck, sequence_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, _, _ = self.forward_with_features(x)
        return reconstruction

    def freeze_backbone(self) -> None:
        """Freeze only the Swin encoder; decoder and auxiliary head stay trainable."""

        for parameter in self.swin_unetr.swinViT.parameters():
            parameter.requires_grad = False
        self.swin_unetr.swinViT.eval()


def swinunetr_hybrid(**kwargs) -> SwinUNETRHybrid:
    """Hydra-friendly constructor."""

    return SwinUNETRHybrid(**kwargs)
