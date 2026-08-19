"""Global intensity non-linear (GIN) augmentation for 3-D batches.

The implementation follows Ouyang et al. (IEEE TMI 2022): a shallow random
convolutional network with Leaky-ReLU activations, interpolation with the
original image, and per-sample Frobenius-norm matching.  New random kernels are
sampled on every call and are never optimized.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class Torch_GIN(nn.Module):
    """Apply GIN to ``data_dict["image"]`` while leaving targets unchanged."""

    def __init__(
        self,
        probability: float = 1.0,
        intermediate_channels: int = 2,
        kernel_sizes: Sequence[int] = (1, 3),
        n_layers: int = 4,
        negative_slope: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if intermediate_channels < 1:
            raise ValueError("intermediate_channels must be positive")
        if n_layers < 2:
            raise ValueError("n_layers must be at least two")
        if not kernel_sizes or any(int(k) < 1 or int(k) % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers")

        self.probability = float(probability)
        self.intermediate_channels = int(intermediate_channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.n_layers = int(n_layers)
        self.negative_slope = float(negative_slope)
        self.eps = float(eps)

    @staticmethod
    def _random_group_conv(x: torch.Tensor, out_channels: int, kernel_size: int) -> torch.Tensor:
        """Use one independently sampled convolutional kernel bank per sample."""
        batch_size, in_channels, depth, height, width = x.shape
        weight = torch.randn(
            batch_size * out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            kernel_size,
            device=x.device,
            dtype=x.dtype,
        )
        bias = torch.randn(
            1,
            batch_size * out_channels,
            1,
            1,
            1,
            device=x.device,
            dtype=x.dtype,
        )
        grouped_input = x.reshape(1, batch_size * in_channels, depth, height, width)
        output = F.conv3d(
            grouped_input,
            weight,
            bias=None,
            stride=1,
            padding=kernel_size // 2,
            groups=batch_size,
        )
        output = output + bias
        return output.reshape(batch_size, out_channels, depth, height, width)

    def _random_network(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        output = image
        for layer_index in range(self.n_layers):
            out_channels = channels if layer_index == self.n_layers - 1 else self.intermediate_channels
            kernel_index = int(torch.randint(len(self.kernel_sizes), (), device=image.device).item())
            output = self._random_group_conv(output, out_channels, self.kernel_sizes[kernel_index])
            if layer_index != self.n_layers - 1:
                output = F.leaky_relu(output, negative_slope=self.negative_slope)
        return output

    def forward(self, data_dict: dict) -> dict:
        image = data_dict["image"]
        if image.ndim != 5:
            raise ValueError(f"GIN expects [B,C,D,H,W], got {tuple(image.shape)}")
        if not image.is_floating_point():
            raise TypeError("GIN expects a floating-point image tensor")
        if self.probability == 0.0:
            return data_dict

        random_output = self._random_network(image)
        batch_size = image.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=image.device, dtype=image.dtype)
        mixed = alpha * random_output + (1.0 - alpha) * image

        input_norm = torch.linalg.vector_norm(image.flatten(start_dim=1), dim=1)
        mixed_norm = torch.linalg.vector_norm(mixed.flatten(start_dim=1), dim=1)
        scale = (input_norm / mixed_norm.clamp_min(self.eps)).reshape(batch_size, 1, 1, 1, 1)
        augmented = mixed * scale

        if self.probability < 1.0:
            apply = torch.rand(batch_size, 1, 1, 1, 1, device=image.device) < self.probability
            augmented = torch.where(apply, augmented, image)

        data_dict["image"] = augmented
        transforms_applied = data_dict.setdefault("transforms_applied", {})
        transforms_applied["gin"] = True
        return data_dict
