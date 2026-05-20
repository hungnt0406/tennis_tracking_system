"""Loss functions for TrackNetV2/V3."""

import torch
import torch.nn as nn


class WBCELoss(nn.Module):
    """Focal-style weighted BCE from the official TrackNetV2 repo."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(self.eps, 1 - self.eps)
        loss = -(
            (1 - pred) ** 2 * target * torch.log(pred)
            + pred ** 2 * (1 - target) * torch.log(1 - pred)
        )
        return loss.mean()
