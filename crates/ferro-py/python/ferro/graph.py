"""CPU f32 segment and sparse algebra; first-order autograd, no replay."""
from ._native import COO, CSR
from ._native import segment_sum, segment_mean, segment_max, segment_softmax
