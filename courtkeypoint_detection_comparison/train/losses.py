"""Heatmap losses for keypoint detection."""

import torch
import torch.nn as nn


class FocalHeatmapLoss(nn.Module):
    """CenterNet penalty-reduced focal loss for Gaussian-heatmap regression.

    Inputs are expected in [0, 1] (apply ``torch.sigmoid`` to model logits
    before calling this loss). Target peaks must be exactly 1.0; surrounding
    pixels are penalty-reduced by ``(1 - gt) ** beta``.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        neg_weights = torch.pow(1.0 - gt, self.beta)

        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * pos_mask
        neg_loss = (
            torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_mask
        )

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos
