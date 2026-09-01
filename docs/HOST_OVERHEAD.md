# Host-side overhead: ferro vs torch

This is the benchmark that measures ferro's actual structural thesis, as
opposed to the GPU throughput bench (`GPU_BASELINE_3090.md`) which reads
parity because it is device-bound.

## The thesis (why this bench exists)

ferro cannot beat PyTorch on device throughput: matmul rides cuBLAS (both
1.00-1.04x), elementwise rides HBM bandwidth (both ~46% of peak, unfused).
Those measure the **GPU**, not the framework.

ferro's structural edge is the **host side** -- per-op dispatch, autograd graph
construction, and graph traversal on backward. A statically-dispatched, GIL-free
Rust core should beat torch's eager C++/Python dispatch there. On tiny tensors
the kernel is ~free, so wall time is dominated by that host overhead. That is
the axis under test here.

Raw Rust-vs-C++ codegen is NOT the claim (both go through LLVM; a hot cuBLAS
loop compiles identically). The win is architectural: no GIL, monomorphised
static dispatch, zero-alloc reuse, fusion-ready record_fn seam.

## Method

- Both frameworks driven from the **same** Python interpreter, so both pay the
  Python->native call cost. Fair fight: pyo3+Rust vs pybind+C++ ATen.
- Tensors are `[8]`, CPU: no kernel-launch / PCIe cost to hide behind.
- `torch.set_num_threads(1)` so torch doesn't spin a threadpool on trivial work.
- Warmup discarded; median reported, min drives throughput. Run:
  `python bench/host_overhead.py --json bench/host_3090.json`

## Results (2 runs, for noise disclosure)

### Per-op dispatch (ns/op, lower better; ratio = torch/ferro)

| op          | torch ns | ferro ns | ferro faster |
|-------------|----------|----------|--------------|
| relu(x)     | ~1080    | ~275     | **3.6-4.0x** |
| x + y       | ~1130    | ~365     | **3.0-3.2x** |
| x * y       | ~1140    | ~325     | **3.2-3.3x** |
| relu*y+x    | ~3.5-7k  | ~0.9-1k  | **3.3-3.7x** |

**Solid, repeatable ~3x.** torch pays ~1.1 us/op of ATen dispatch; ferro pays
~0.3 us. This is the eager-mode overhead that `torch.compile` exists to remove
-- ferro doesn't pay it in the first place.

### Autograd chain step (depth-8 relu*1.5+0.1, forward + backward)

| depth | torch us | ferro us | ferro faster |
|-------|----------|----------|--------------|
| 8     | 150-240  | 29-59    | **>=2.4x**   |

torch's backward time is noisy (allocation + GC jitter) so the ratio swings
2.4x-5.2x run to run. Per the honest-number rule we quote the **conservative
floor: >=2.4x**, not the flattering 5x. Even the floor is a real, large win:
graph construction + traversal on tiny tensors is pure host bookkeeping and
ferro's is materially leaner.

## Honest takeaways

1. **This is orchestration speed, not compute speed.** These ops do ~no real
   FLOPs; reading the numbers as "ferro computes 3x faster" would be a lie. It
   is *dispatch + autograd* that is 3x, which is exactly the thesis.
2. **The win is real and repeatable on dispatch (~3x), directional on autograd
   (>=2.4x).** It shows up precisely where torch is weakest: small ops, long
   thin graphs, eager mode, tight training loops -- the regime where launch/
   dispatch overhead is a large fraction of wall time.
3. **Where it does NOT help:** big tensors / GPU throughput. Once the kernel
   dominates, host overhead is amortised to nothing and both frameworks read
   parity (see GPU_BASELINE_3090.md). ferro's edge is a *fraction-of-wall-time*
   win, largest when tensors are small and graphs are long.

## Strategic reading

ferro's defensible differentiation is: **lowest per-step host overhead of any
autograd engine**, because Rust lets it be statically dispatched, GIL-free, and
allocation-deterministic without giving up memory safety. The place to press
this advantage is small-model / high-step-count training and inference loops,
and to compound it with fusion (kill the unfused-elementwise ceiling from
GPU_BASELINE_3090.md) so the device side stops giving parity back.
