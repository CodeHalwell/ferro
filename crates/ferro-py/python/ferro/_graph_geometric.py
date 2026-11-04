"""Eager CPU coordinate transforms using native differentiable tensor algebra.

Calls shallow-copy the container and stores, as PyG BaseTransform does; forward
updates the supplied container. Untouched tensors and private batch slice state
remain shared. Center/LinearTransformation visit typed node stores; the other
transforms require homogeneous Data, including already constructed batches.
Coordinates support F32/F64 rank two; LinearTransformation also accepts rank
one. Topology is discrete CPU int64 or native EdgeIndex. No geometric search or
per-graph centering is implied for a Batch: centering uses all stored positions.
"""
import math

from ._native import Tensor, cat, _is_capturing
from ._graph_data import Data, _custom_hooks
from ._graph_hetero import HeteroData
from ._graph_subgraph import _copy, _topology
from ._graph_utils import _capacity, _cpu_tensor
from .linalg import vector_norm


def _floating(value, name, ranks=(2,)):
    _cpu_tensor(value, name, dtype=None)
    if value.dtype not in ('float32', 'float64') or value.ndim not in ranks:
        raise ValueError(name + ' requires CPU float32/float64 with supported rank')
    _capacity(value.shape, 8 if value.dtype == 'float64' else 4)
    return value


def _data(data):
    if _is_capturing():
        raise NotImplementedError('geometric transforms require eager execution')
    if not isinstance(data, (Data, HeteroData)):
        raise TypeError('geometric transforms require Data or HeteroData')
    return data


def _nodes(data):
    _data(data)
    return [data._store] if isinstance(data, Data) else [store._mapping for store in data.node_stores]


def _position_plans(data, ranks=(2,)):
    return [(store, _floating(store['pos'], 'pos', ranks)) for store in _nodes(data) if 'pos' in store]


def _center(pos):
    # Empty positions need no irrelevant reduction workspace.
    return pos.clone() if pos.numel() == 0 else pos - pos.mean_dim(0, True)


class _Transform:
    def __call__(self, data):
        _data(data)
        return self.forward(_copy(data))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Center(_Transform):
    def forward(self, data):
        plans = _position_plans(data)
        for store, pos in plans:
            store['pos'] = _center(pos)
        return data


class NormalizeScale(_Transform):
    def __init__(self):
        self.center = Center()

    def forward(self, data):
        _data(data)
        if not isinstance(data, Data) or 'pos' not in data:
            raise ValueError('NormalizeScale requires homogeneous Data.pos')
        pos = _floating(data.pos, 'pos')
        if pos.numel() == 0:
            raise ValueError('NormalizeScale requires nonempty positions for the global maximum')
        data = self.center(data)
        scale = (1. / data.pos.abs().max()) * .999999
        data._store['pos'] = data.pos * scale
        return data


class LinearTransformation(_Transform):
    def __init__(self, matrix):
        matrix = _floating(matrix, 'matrix')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('transformation matrix must be square')
        self.matrix = matrix.T

    def forward(self, data):
        plans = _position_plans(data, (1, 2))
        matrix = _floating(self.matrix, 'matrix')
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('transformation matrix must remain square')
        for _, pos in plans:
            _capacity(matrix.shape, 8 if pos.dtype == 'float64' else 4)
            width = 1 if pos.ndim == 1 else pos.shape[1]
            if width != matrix.shape[0]:
                raise ValueError('position width and matrix size differ')
        for store, pos in plans:
            points = pos.reshape([pos.shape[0], 1]) if pos.ndim == 1 else pos
            store['pos'] = points @ matrix.to(dtype=pos.dtype)
        return data


def _flag(value, name):
    if type(value) is not bool:
        raise TypeError(name + ' must be bool')
    return value


def _real(value, name):
    if type(value) not in (int, float):
        raise TypeError(name + ' must be a real scalar')
    return float(value)


class _Coordinates(_Transform):
    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm, self.cat = _flag(norm, 'norm'), _flag(cat, 'cat')
        self.max = None if max_value is None else _real(max_value, 'max_value')

    def _plans(self, data, width=None):
        _data(data)
        if not isinstance(data, Data) or _custom_hooks(data):
            raise NotImplementedError('edge coordinate transforms require homogeneous single-space Data')
        if 'pos' not in data or data.edge_index is None:
            raise ValueError('edge coordinate transforms require pos and edge_index')
        pos = _floating(data.pos, 'pos', (1, 2) if isinstance(self, Cartesian) and not isinstance(self, Distance) else (2,))
        if width is not None and pos.shape[1] != width:
            raise ValueError('position width differs from the coordinate transform')
        if pos.shape[0] != data.num_nodes:
            raise ValueError('position rows differ from num_nodes')
        src, dst = _topology(data.edge_index, (data.num_nodes, data.num_nodes))
        _capacity((len(src), 1 if pos.ndim == 1 else pos.shape[1]), 8 if pos.dtype == 'float64' else 4)
        out_width = width if width is not None else 1 if isinstance(self, Distance) or pos.ndim == 1 else pos.shape[1]
        _capacity((len(src), max(out_width, 1)), 8 if pos.dtype == 'float64' else 4)
        pseudo = data._store.get('edge_attr') if self.cat else None
        if pseudo is not None:
            _cpu_tensor(pseudo, 'edge_attr', dtype=None)
            if pseudo.dtype not in ('float32', 'float64', 'int64', 'bool') or pseudo.ndim not in (1, 2) or pseudo.shape[0] != len(src):
                raise ValueError('edge_attr requires aligned rank-one/two CPU fields')
            old_width = 1 if pseudo.ndim == 1 else pseudo.shape[1]
            _capacity((len(src), old_width + out_width), 8 if pseudo.dtype in ('float64', 'int64') or pos.dtype == 'float64' else 4)
        return pos, src, dst, pseudo

    def _write(self, data, value, pseudo, dtype=None):
        if pseudo is not None:
            previous = pseudo.reshape([pseudo.shape[0], 1]) if pseudo.ndim == 1 else pseudo
            value = value.to(dtype=pseudo.dtype if dtype is None else dtype)
            promoted = next(kind for kind in ('float64', 'float32', 'int64', 'bool') if kind in (previous.dtype, value.dtype))
            value = cat([previous.to(dtype=promoted), value.to(dtype=promoted)], -1)
        data._store['edge_attr'] = value
        data._field_roles['edge_attr'] = 'edge'
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(norm={self.norm}, max_value={self.max})'


class Cartesian(_Coordinates):
    def __init__(self, norm=True, max_value=None, cat=True, interval=(0., 1.)):
        super().__init__(norm, max_value, cat)
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError('interval must contain two bounds')
        self.interval = tuple(_real(value, 'interval bound') for value in interval)

    def forward(self, data):
        pos, src, dst, pseudo = self._plans(data)
        value = pos.index_select(0, src) - pos.index_select(0, dst)
        if value.ndim == 1:
            value = value.reshape([len(src), 1])
        if self.norm and value.numel():
            maximum = value.abs().max().detach() if self.max is None else self.max
            length = self.interval[1] - self.interval[0]
            center = (self.interval[0] + self.interval[1]) / 2
            value = length * value / (2 * maximum) + center
        return self._write(data, value, pseudo)


class Distance(Cartesian):
    def forward(self, data):
        pos, src, dst, pseudo = self._plans(data)
        value = vector_norm(pos.index_select(0, dst) - pos.index_select(0, src), 2, -1, True)
        if self.norm and value.numel():
            maximum = value.max().detach() if self.max is None else self.max
            value = (self.interval[1] - self.interval[0]) * (value / maximum) + self.interval[0]
        return self._write(data, value, pseudo)


class Polar(_Coordinates):
    def forward(self, data):
        pos, src, dst, pseudo = self._plans(data, 2)
        cart = pos.index_select(0, dst) - pos.index_select(0, src)
        rho = vector_norm(cart, 2, -1, True)
        theta = cart.narrow(1, 1, 1).atan2(cart.narrow(1, 0, 1))
        theta = theta + (theta < 0).to(dtype=theta.dtype) * (2 * math.pi)
        if self.norm:
            if not rho.numel() and self.max is None:
                raise ValueError('normalized Polar requires nonempty edges or an explicit max_value')
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * math.pi)
        return self._write(data, cat([rho, theta], -1), pseudo, pos.dtype)


class Spherical(_Coordinates):
    def forward(self, data):
        pos, src, dst, pseudo = self._plans(data, 3)
        cart = pos.index_select(0, dst) - pos.index_select(0, src)
        rho = vector_norm(cart, 2, -1, True)
        theta = cart.narrow(1, 1, 1).atan2(cart.narrow(1, 0, 1))
        theta = theta + (theta < 0).to(dtype=theta.dtype) * (2 * math.pi)
        phi = (cart.narrow(1, 2, 1) / rho).acos()
        if self.norm:
            if not rho.numel() and self.max is None:
                raise ValueError('normalized Spherical requires nonempty edges or an explicit max_value')
            rho = rho / (rho.max() if self.max is None else self.max)
            theta, phi = theta / (2 * math.pi), phi / math.pi
        return self._write(data, cat([rho, theta, phi], -1), pseudo, pos.dtype)
