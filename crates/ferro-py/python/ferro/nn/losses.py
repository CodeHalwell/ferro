from .module import Module
from . import functional as F
from .functional import mse_loss


class MSELoss(Module):
    reduction: str = 'mean'

    def build(self):
        if self.reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def forward(self, prediction, target):
        return mse_loss(prediction, target, self.reduction)


class _Loss(Module):
    """Native losses are mean-reduced; 'sum' rescales, 'none' is unsupported."""
    reduction: str = 'mean'
    _reductions = ('mean', 'sum')

    def build(self):
        if self.reduction not in self._reductions:
            raise ValueError(f'reduction must be one of {self._reductions}')


class L1Loss(_Loss):
    def forward(self, input, target):
        return F.l1_loss(input, target, self.reduction)


class CrossEntropyLoss(_Loss):
    def forward(self, input, target):
        return F.cross_entropy(input, target, self.reduction)


class NLLLoss(_Loss):
    def forward(self, input, target):
        return F.nll_loss(input, target, self.reduction)


class BCELoss(_Loss):
    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)


class HuberLoss(_Loss):
    delta: float = 1.0

    def forward(self, input, target):
        return F.huber_loss(input, target, self.reduction, self.delta)


class SmoothL1Loss(_Loss):
    beta: float = 1.0

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, self.reduction, self.beta)


class KLDivLoss(_Loss):
    _reductions = ('mean', 'sum', 'batchmean')

    def forward(self, input, target):
        return F.kl_div(input, target, self.reduction)


class PoissonNLLLoss(_Loss):
    def forward(self, input, target):
        return F.poisson_nll_loss(input, target, reduction=self.reduction)


class SoftMarginLoss(_Loss):
    def forward(self, input, target):
        return F.soft_margin_loss(input, target, self.reduction)


class HingeEmbeddingLoss(_Loss):
    margin: float = 1.0

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, self.margin, self.reduction)


class MarginRankingLoss(_Loss):
    margin: float = 0.0

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.reduction)


class CosineEmbeddingLoss(_Loss):
    margin: float = 0.0

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.reduction)


class TripletMarginLoss(_Loss):
    margin: float = 1.0
    p: float = 2.0
    eps: float = 1e-6

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p, self.eps, reduction=self.reduction)
