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
    """Native tensor compositions with explicit elementwise or scalar reductions."""
    reduction: str = 'mean'
    _reductions = ('none', 'mean', 'sum')

    def build(self):
        if self.reduction not in self._reductions:
            raise ValueError(f'reduction must be one of {self._reductions}')


class L1Loss(_Loss):
    def forward(self, input, target):
        return F.l1_loss(input, target, self.reduction)


class _ClassificationLoss(Module):
    """Classification scalar configuration plus a persistent class-weight buffer."""
    _builtin_init = True
    reduction: str = 'mean'
    ignore_index: int = -100

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        weight, reduction, _ = F._classification_options(weight, ignore_index, reduction)
        F._classification_weight(weight)
        self._initialize()
        object.__setattr__(self, 'reduction', reduction)
        object.__setattr__(self, 'ignore_index', ignore_index)
        if weight is None:
            self.weight = None
        else:
            self.register_buffer('weight', weight)


class CrossEntropyLoss(_ClassificationLoss):
    """See functional.cross_entropy; weights are saved as persistent buffers."""
    _builtin_init = True
    label_smoothing: float = 0.0

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        weight, reduction, smoothing = F._classification_options(weight, ignore_index, reduction, label_smoothing)
        super().__init__(weight, ignore_index, reduction)
        object.__setattr__(self, 'label_smoothing', smoothing)

    def forward(self, input, target):
        return F.cross_entropy(input, target, self.weight, self.ignore_index, self.reduction, self.label_smoothing)


class NLLLoss(_ClassificationLoss):
    def forward(self, input, target):
        return F.nll_loss(input, target, self.weight, self.ignore_index, self.reduction)


class BCELoss(_Loss):
    _builtin_init = True

    def __init__(self, weight=None, reduction='mean'):
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self._initialize()
        object.__setattr__(self, 'reduction', reduction)
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(BCELoss):
    _builtin_init = True

    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super().__init__(weight, reduction)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, self.weight, reduction=self.reduction, pos_weight=self.pos_weight)


class HuberLoss(_Loss):
    delta: float = 1.0

    def forward(self, input, target):
        return F.huber_loss(input, target, self.reduction, self.delta)


class SmoothL1Loss(_Loss):
    beta: float = 1.0

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, self.reduction, self.beta)


class KLDivLoss(_Loss):
    _reductions = ('none', 'mean', 'sum', 'batchmean')
    log_target: bool = False

    def forward(self, input, target):
        return F.kl_div(input, target, self.reduction, self.log_target)


class PoissonNLLLoss(_Loss):
    log_input: bool = True
    full: bool = False
    eps: float = 1e-8

    def forward(self, input, target):
        return F.poisson_nll_loss(input, target, self.log_input, self.full, self.eps, self.reduction)


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
    swap: bool = False

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p, self.eps, self.swap, self.reduction)
