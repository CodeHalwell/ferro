"""Graph normalization adapters over canonical native BatchNorm state."""
import math

from ._native import Tensor, cat
from ._graph_conv import _configure, _features, _flags, _positive, _reset_linear
from ._graph_conditioned import _eager
from ._graph_norm import _device
from ._graph_utils import _capacity, _cpu_tensor, _index_ids
from .nn import BatchNorm1d, Linear, Module
from .nn import functional as F


class BatchNorm(Module):
    """Graph-node BatchNorm with PyG's optional singleton evaluation path.

    Rank-two CPU float32 input uses the registered BatchNorm1d's native affine,
    running-statistic and int64 counter state. With allow_single_element=True,
    zero/one-node batches use saved statistics without advancing the counter.
    """
    _builtin_init = True
    in_channels: int
    allow_single_element: bool

    def __init__(self, in_channels, eps=1e-5, momentum=.1, affine=True, track_running_stats=True,
                 allow_single_element=False, device=None):
        _positive(in_channels, 'in_channels')
        _flags(affine=affine, track_running_stats=track_running_stats, allow_single_element=allow_single_element)
        _device(device)
        if allow_single_element and not track_running_stats:
            raise ValueError('allow_single_element requires track_running_stats=True')
        _capacity((in_channels,))
        _configure(self, in_channels=in_channels, allow_single_element=allow_single_element)
        self.module = BatchNorm1d(in_channels, eps, momentum, affine, track_running_stats, device=device)

    def reset_running_stats(self):
        self.module.reset_running_stats()

    def reset_parameters(self):
        self.module.reset_parameters()

    def forward(self, x):
        _eager()
        _features(x, 'x', self.in_channels)
        _capacity(x.shape)
        if self.allow_single_element and x.shape[0] <= 1:
            weight = None if self.module.weight is None else self.module.weight.tensor()
            bias = None if self.module.bias is None else self.module.bias.tensor()
            return F.batch_norm(x, self.module.running_mean, self.module.running_var, weight, bias, False, 0., self.module.eps)
        return self.module(x)


def _distance_sum(left, right):
    rows, width = left.shape
    columns = right.shape[0]
    _capacity((rows, columns, width))
    if rows > 25 or columns > 25:
        _capacity((max(rows, columns), width + 2))
        first = cat([left * -2., left.square().sum_dim(1, True), Tensor.ones([rows, 1])], 1)
        second = cat([right, Tensor.ones([columns, 1]), right.square().sum_dim(1, True)], 1)
        return first.matmul(second.transpose(0, 1)).clamp(0., float('inf')).sqrt().sum().item()
    return (left.unsqueeze(1) - right.unsqueeze(0)).square().sum_dim(2, False).sqrt().sum().item()


class DiffGroupNorm(Module):
    """Learn soft group assignments and normalize their weighted node features.

    This CPU float32 eager layer uses canonical native BatchNorm1d and its
    counters, affine parameters and checkpoint state. Singleton training errors
    follow BatchNorm's counter behavior. group_distance_ratio is a scalar
    diagnostic for contiguous nonempty class labels and at least two classes;
    its Euclidean distance path follows Torch's size-based matmul selection.
    """
    _builtin_init = True
    in_channels: int
    groups: int
    lamda: float

    def __init__(self, in_channels, groups, lamda=.01, eps=1e-5, momentum=.1, affine=True,
                 track_running_stats=True, device=None):
        _positive(in_channels, 'in_channels')
        _positive(groups, 'groups')
        _flags(affine=affine, track_running_stats=track_running_stats)
        _device(device)
        if type(lamda) not in (int, float) or not math.isfinite(lamda): raise ValueError('lamda must be finite')
        _capacity((groups, in_channels))
        _configure(self, in_channels=in_channels, groups=groups, lamda=float(lamda))
        self.lin = Linear(in_channels, groups, bias=False)
        self.norm = BatchNorm1d(groups * in_channels, eps, momentum, affine, track_running_stats, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        _reset_linear(self.lin)
        self.norm.reset_parameters()

    def forward(self, x):
        _eager()
        _features(x, 'x', self.in_channels)
        nodes = x.shape[0]
        _capacity((nodes, self.groups, self.in_channels))
        assignments = self.lin(x).softmax(1)
        values = assignments.unsqueeze(2) * x.unsqueeze(1)
        output = self.norm(values.reshape([nodes, self.groups * self.in_channels])).reshape([nodes, self.groups, self.in_channels]).sum_dim(1, False)
        return x + self.lamda * output

    @staticmethod
    def group_distance_ratio(x, y, eps=1e-5):
        _eager()
        _features(x, 'x')
        _cpu_tensor(y, 'y', 'int64')
        _capacity((x.shape[0] + 1,), 8)
        _capacity(x.shape)
        if y.shape != [x.shape[0]]: raise ValueError('y must contain one class label per node')
        if type(eps) not in (int, float) or not math.isfinite(eps) or eps < 0: raise ValueError('eps must be finite and nonnegative')
        ids = _index_ids(y, 'y', x.shape[0])
        classes = sorted(set(ids))
        if len(classes) < 2 or classes[0] != 0 or classes[-1] + 1 != len(classes):
            raise ValueError('group distance requires at least two contiguous nonempty classes')
        groups = [[row for row, label in enumerate(ids) if label == group] for group in classes]
        numerator = denominator = 0.
        for group, selected in enumerate(groups):
            other = [row for row, label in enumerate(ids) if label != group]
            left, right = x.index_select(0, selected), x.index_select(0, other)
            numerator += _distance_sum(left, right) / (len(selected) * len(other))
            denominator += _distance_sum(left, left) / (len(selected) * len(selected))
        return (numerator / (len(classes) - 1) ** 2) / (denominator / len(classes) + eps)
