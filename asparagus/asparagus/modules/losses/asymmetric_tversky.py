"""Segmentation loss with an explicit false-positive/false-negative trade-off."""

import torch
import torch.nn.functional as F
from torch import nn


class AsymmetricTverskyCrossEntropyLoss(nn.Module):
    """Foreground Tversky plus multiclass cross entropy.

    ``alpha`` multiplies false positives and ``beta`` multiplies false
    negatives.  Therefore alpha > beta is the deliberate setting for an
    over-segmenting model.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        tversky_weight: float = 1.0,
        cross_entropy_weight: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("alpha and beta must be non-negative with a positive sum")
        if tversky_weight < 0 or cross_entropy_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if tversky_weight + cross_entropy_weight == 0:
            raise ValueError("at least one loss weight must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tversky_weight = float(tversky_weight)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 4:
            raise ValueError(f"Expected dense logits [B,C,...], got {tuple(logits.shape)}")
        if target.ndim == logits.ndim and target.shape[1] == 1:
            target_index = target[:, 0]
        elif target.ndim == logits.ndim - 1:
            target_index = target
        else:
            raise ValueError(
                f"Target shape {tuple(target.shape)} is incompatible with logits {tuple(logits.shape)}"
            )
        target_index = target_index.long()

        ce = F.cross_entropy(logits, target_index)
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target_index, num_classes=logits.shape[1]).movedim(-1, 1)
        one_hot = one_hot.to(dtype=probabilities.dtype)

        # Background has a qualitatively different scale and is handled by CE.
        # Tversky focuses on each foreground class, including classes absent in
        # a patch (where any predicted foreground is a false positive).
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]
        reduce_dims = (0, *range(2, probabilities.ndim))
        true_positive = torch.sum(probabilities * one_hot, dim=reduce_dims)
        false_positive = torch.sum(probabilities * (1.0 - one_hot), dim=reduce_dims)
        false_negative = torch.sum((1.0 - probabilities) * one_hot, dim=reduce_dims)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        tversky_loss = 1.0 - tversky.mean()
        return self.tversky_weight * tversky_loss + self.cross_entropy_weight * ce
