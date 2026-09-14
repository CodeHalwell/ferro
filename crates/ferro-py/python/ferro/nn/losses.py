from .module import Module
from .functional import mse_loss


class MSELoss(Module):
    reduction: str = 'mean'

    def build(self):
        if self.reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def forward(self, prediction, target):
        return mse_loss(prediction, target, self.reduction)
