"""Native variance-preserving grouped summation."""
from ._native import Tensor, scatter
from ._graph_aggregation import _Aggregation, _layout
from ._graph_conv import _configure


class VariancePreservingAggregation(_Aggregation):
    """Sum divided by square-root group size, with empty groups equal to zero.

    CPU float32 rank-two features accept unsorted index groups or contiguous CSR
    pointers. Index counts use native float32 scatter; CSR counts use native
    integer pointer differences before casting. Both preserve reference rounding,
    and feature adjoints remain on native operations.
    """
    _builtin_init = True

    def __init__(self):
        _configure(self)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        ids, count = _layout(x, index, ptr, dim_size, dim)
        total = scatter(x, ids, count, 'sum')
        if ptr is None:
            counts = scatter(Tensor.ones([x.shape[0], 1]), ids, count, 'sum')
        else:
            counts = (ptr.narrow(0, 1, count) - ptr.narrow(0, 0, count)).to(dtype='float32').reshape([count, 1])
        return total / counts.sqrt().clamp(1., float('inf'))
