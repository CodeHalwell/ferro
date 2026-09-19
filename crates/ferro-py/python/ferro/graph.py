"""First-order f32 segment and sparse algebra; capture/replay is unsupported.

Sum/softmax support CPU and resident CUDA; mean/max and unprepared sparse algebra are CPU-only.
COO.prepare(device) provides resident SpMM/SDDMM with reusable row/column plans.
COO.batch(graphs) returns block-diagonal COO and cumulative [row, column] offsets.
PreparedSegments.select(x, dim) gathers along any axis; scatter_add(base, src, dim)
adds source slices at the same IDs. Repeated IDs accumulate in both adjoints.
These are one-dimensional index-select semantics, not elementwise torch.gather.
Initialize CUDA with ``ferro.cuda_init()`` before preparing a CUDA topology::

    plan = PreparedSegments([2, 0, 2], num_segments=4, device="cuda:0")
    y = plan.sum(x)       # x has shape [3, ...] on cuda:0
    z = plan.softmax(x)   # reuse with fresh inputs, without repeating IDs

IDs are copied as nonnegative i64 integers and validated at construction. CUDA
uploads topology once; the plan and recorded backward retain its original backend
and allocation owners. Inputs must match that device (foreign contexts reject,
never silently copy to CPU). Free functions prepare topology on every invocation.
CUDA softmax rejects nonfinite logits via one synchronizing four-byte status read
per nonempty forward, not a feature download. Strided f32 CUDA features and backward
seeds materialize on device. Prepared sparse algebra uses O(edges * features)
scratch. GPU checkpoint restore and higher-order derivatives remain unsupported.
"""
from ._native import COO, CSR, PreparedSegments, PreparedCOO
from ._native import segment_sum, segment_mean, segment_max, segment_softmax
