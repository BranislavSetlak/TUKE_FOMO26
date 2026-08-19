"""Frozen-feature downstream heads for the 3D DINO teacher backbone.

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


def _init_channel_average(linear: nn.Linear, input_channels: int, hidden_size: int) -> None:
    """Initialize a channel mixer as an identity-preserving modality average."""
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        identity = torch.eye(hidden_size, dtype=linear.weight.dtype, device=linear.weight.device)
        for channel in range(input_channels):
            start = channel * hidden_size
            linear.weight[:, start : start + hidden_size].copy_(identity / input_channels)


class _DINO3DChannelEncoder(nn.Module):
    """Run the single-channel pretrained encoder per modality, then mix tokens.

    The DINO checkpoint was pretrained on one scan/sequence at a time.  For a
    multi-modal downstream case, averaging modalities before the transformer
    destroys modality identity.  Instead each modality is encoded with the
    shared pretrained backbone and a small trainable projection mixes the
    resulting tokens.
    """

    def _configure_channel_encoder(
        self,
        input_channels: int,
        img_size,
        patch_size,
        hidden_size: int,
    ) -> None:
        self.input_channels = int(input_channels)
        self.hidden_size = int(hidden_size)
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")

        # Keep the pretrained convolution exactly single-channel.  This also
        # makes checkpoint loading exact instead of repeating/averaging its
        # stem weights for multi-modal tasks.
        self.teacher_backbone = _make_backbone(
            input_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )
        if self.input_channels == 1:
            self.channel_mixer = nn.Identity()
        else:
            self.channel_mixer = nn.Linear(self.input_channels * hidden_size, hidden_size)
            _init_channel_average(self.channel_mixer, self.input_channels, hidden_size)

        self.stem_weight_name = "teacher_backbone.vit.patch_embedding.patch_embeddings.weight"

    def _intermediate_channel_tokens(self, x: torch.Tensor, n_last_blocks: int):
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {x.shape[1]}")

        batch_size, channels = x.shape[:2]
        flat_modalities = x.reshape(batch_size * channels, 1, *x.shape[2:])
        layer_tokens = self.teacher_backbone.get_intermediate_layers(
            flat_modalities,
            n=n_last_blocks,
            mask=None,
        )

        mixed_layers = []
        for tokens in layer_tokens:
            # B*C,N,D -> B,N,C*D -> B,N,D
            tokens = tokens.reshape(batch_size, channels, tokens.shape[1], tokens.shape[2])
            tokens = tokens.permute(0, 2, 1, 3).reshape(
                batch_size,
                tokens.shape[2],
                channels * tokens.shape[3],
            )
            mixed_layers.append(self.channel_mixer(tokens))
        return tuple(mixed_layers)

    def freeze_backbone(self) -> None:
        for parameter in self.teacher_backbone.parameters():
            parameter.requires_grad = False


class DINO3DClassifierRegressor(_DINO3DChannelEncoder):
    """DINOv2-style linear readout for 3D classification or regression."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size: int = 384,
        dropout: float = 0.1,
        n_last_blocks: int = 4,
        use_avgpool: bool = True,
    ):
        super().__init__()
        self.num_classes = int(output_channels)
        self.patch_size = _as_tuple3(patch_size)
        self.n_last_blocks = int(n_last_blocks)
        self.use_avgpool = bool(use_avgpool)
        if self.n_last_blocks < 1 or self.n_last_blocks > 8:
            raise ValueError("n_last_blocks must be between 1 and 8")

        self._configure_channel_encoder(
            input_channels=input_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
        )
        feature_size = hidden_size * (self.n_last_blocks + int(self.use_avgpool))
        decoder_layers = []
        if dropout > 0:
            decoder_layers.append(nn.Dropout(dropout))
        linear = nn.Linear(feature_size, self.num_classes)
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear.bias)
        decoder_layers.append(linear)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        layers = self._intermediate_channel_tokens(x, self.n_last_blocks)
        features = [tokens[:, 0] for tokens in layers]
        if self.use_avgpool:
            features.append(layers[-1][:, 1:].mean(dim=1))
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.forward_features(x))

class DINO3DSegmenter(_DINO3DChannelEncoder):
    """Frozen multi-layer patch-token readout for 3D segmentation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size: int = 384,
        n_last_blocks: int = 4,
    ):
        super().__init__()
        self.num_classes = int(output_channels)
        self.patch_size = _as_tuple3(patch_size)
        self.n_last_blocks = int(n_last_blocks)
        if self.n_last_blocks < 1 or self.n_last_blocks > 8:
            raise ValueError("n_last_blocks must be between 1 and 8")

        self._configure_channel_encoder(
            input_channels=input_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
        )
        # Two point-wise projections are the 3D equivalent of a lightweight
        # linear/MLP dense probe.  Spatial upsampling is interpolation, so the
        # decoder measures information already present in DINO patch tokens.
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_size * self.n_last_blocks, hidden_size, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden_size, self.num_classes, kernel_size=1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        layers = self._intermediate_channel_tokens(x, self.n_last_blocks)
        grid = tuple(int(size // patch) for size, patch in zip(x.shape[2:], self.patch_size))
        feature_maps = []
        for layer in layers:
            patch_tokens = layer[:, 1:]
            if patch_tokens.shape[1] != prod(grid):
                raise RuntimeError(f"Patch-token count {patch_tokens.shape[1]} does not match grid {grid}.")
            feature_maps.append(
                patch_tokens.transpose(1, 2).reshape(x.shape[0], patch_tokens.shape[2], *grid)
            )
        return torch.cat(feature_maps, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(self.forward_features(x))
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits

    def sliding_window_predict(self, data: torch.Tensor, patch_size, overlap: float = 0.75) -> torch.Tensor:
        return sliding_window_inference(
            inputs=data,
            roi_size=tuple(int(v) for v in patch_size),
            sw_batch_size=1,
            predictor=self.forward,
            overlap=float(overlap),
        )

def dino3d_clsreg(input_channels: int, output_channels: int, **kwargs):
    return DINO3DClassifierRegressor(input_channels, output_channels, **kwargs)


def dino3d_seg(input_channels: int, output_channels: int, **kwargs):
    return DINO3DSegmenter(input_channels, output_channels, **kwargs)
