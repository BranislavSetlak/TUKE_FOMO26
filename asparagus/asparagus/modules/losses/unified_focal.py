"""PyTorch implementation of asymmetric Unified Focal loss.

The formulation follows the authors' reference implementation while extending
its foreground branch from binary to multiclass softmax segmentation.
"""

import torch
import torch.nn.functional as F
from torch import nn


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Asymmetric focal Tversky plus asymmetric focal cross entropy.

    ``delta`` weights false negatives and foreground cross entropy. Values
    above 0.5 therefore favor recall. ``weight`` is the paper's lambda and
    mixes focal Tversky with focal cross entropy.
    """

    def __init__(
        self,
        weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if not 0.0 <= delta <= 1.0:
            raise ValueError("delta must be in [0, 1]")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.weight = float(weight)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    @staticmethod
    def _target_indices(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 4:
            raise ValueError(f"Expected dense logits [B,C,...], got {tuple(logits.shape)}")
        if logits.shape[1] < 2:
            raise ValueError("Unified Focal loss expects background plus at least one foreground class")
        if target.ndim == logits.ndim and target.shape[1] == 1:
            target = target[:, 0]
        elif target.ndim != logits.ndim - 1:
            raise ValueError(
                f"Target shape {tuple(target.shape)} is incompatible with logits {tuple(logits.shape)}"
            )
        return target.long()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_index = self._target_indices(logits, target)

        # Compute the loss in float32 even during bf16 mixed precision.
        probabilities = logits.float().softmax(dim=1).clamp(self.smooth, 1.0 - self.smooth)
        one_hot = F.one_hot(target_index, num_classes=logits.shape[1]).movedim(-1, 1).float()
        spatial_dims = tuple(range(2, logits.ndim))

        true_positive = torch.sum(probabilities * one_hot, dim=spatial_dims)
        false_negative = torch.sum((1.0 - probabilities) * one_hot, dim=spatial_dims)
        false_positive = torch.sum(probabilities * (1.0 - one_hot), dim=spatial_dims)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.delta * false_negative
            + (1.0 - self.delta) * false_positive
            + self.smooth
        )

        background_tversky_loss = 1.0 - tversky[:, 0]
        foreground_tversky_loss = torch.pow(
            (1.0 - tversky[:, 1:]).clamp_min(self.smooth),
            1.0 - self.gamma,
        ).mean(dim=1)
        asymmetric_focal_tversky = torch.mean(
            0.5 * (background_tversky_loss + foreground_tversky_loss)
        )

        cross_entropy = -one_hot * torch.log(probabilities)
        background_focal_ce = (
            (1.0 - self.delta)
            * torch.pow(1.0 - probabilities[:, 0], self.gamma)
            * cross_entropy[:, 0]
        )
        foreground_ce = self.delta * torch.sum(cross_entropy[:, 1:], dim=1)
        asymmetric_focal_ce = torch.mean(background_focal_ce + foreground_ce)

        return (
            self.weight * asymmetric_focal_tversky
            + (1.0 - self.weight) * asymmetric_focal_ce
        )
