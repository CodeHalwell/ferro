"""Rust tensors with composable Python model and training APIs."""
from ._native import *
from . import nn, optim, graph, recurrent, basis, checkpoint, data, transforms, linalg, func
from .progress import Progress
from ._tensor_api import tensor, as_tensor, empty, zeros, ones, full, arange, eye, rand, randn
from ._ops import (is_tensor, numel, cat, concat, reshape, flatten, transpose, permute,
                  squeeze, unsqueeze, split, chunk, unbind, sum, mean, mm, bmm, matmul,
                  zeros_like, ones_like, full_like, rand_like, randn_like, diagonal, trace)
from ._boolean_api import (eq, ne, lt, le, gt, ge, logical_and, logical_or, logical_xor,
                          logical_not, isnan, isfinite, any, all, nonzero, masked_select, where)
from ._sorting import sort, argsort
from ._arithmetic import add, sub, subtract, mul, multiply, true_divide, div, divide, floor_divide
from .linalg import cholesky_solve, cholesky_inverse, vecdot
from ._math import floor, ceil, trunc, round, frac, rsqrt, sqrt
from ._comparison import equal, isclose, allclose
from ._cumulative import cumsum, cumprod, cummax, cummin

float32 = 'float32'
float64 = 'float64'
int64 = 'int64'
float16 = 'float16'
bfloat16 = 'bfloat16'
float = float32
double = float64
long = int64
bool = 'bool'
preserve_format = 'preserve_format'
contiguous_format = 'contiguous_format'

from ._indexing import take

from ._reductions import prod
