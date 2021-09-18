from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import weighted_loss
from ..builder import LOSSES


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Wrapper of mse losses."""
    return F.mse_loss(pred, target, reduction="none")


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss with weights.

    Args:
        reduction: The method that reduces the losses to a scalar. Options are "none", "mean" and "sum".
        loss_weight: The weight of the losses.
    """

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
    ) -> Tensor:
        """Forward function of losses.

        Args:
            pred: The prediction.
            target: The learning target of the prediction.
            weight: Weight of the losses for each prediction.
            avg_factor: Average factor that is used to average the losses.

        Returns:
            torch.Tensor: The calculated losses
        """
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor,
        )
        return loss
