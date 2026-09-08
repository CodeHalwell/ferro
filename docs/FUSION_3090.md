# Compiled pointwise fusion on RTX 3090

## PR18 correction (2026-09-08)

The earlier compiled-fusion table and its interpretation are withdrawn. That
handle captured the already-computed first unary result, omitted the first
operation on replay, and could freeze computed side branches. Binary-first
chains could panic. A broadcast that expanded the seed could launch with the
wrong element count. The old benchmark calculated an absolute difference but
never asserted correctness, and the ferro input still required gradients while
PyTorch's did not. Those results did not validate full-expression replay.

The earlier `.fuse()` throughput table and the claim that planning overhead was
its entire loss are also withdrawn as evidence for compiled replay. `.fuse()`
and the low-level `FusedChain::resolve` retain their legacy intermediate-seeded
contract; they are not the compile-once API. Historical Rust `bench_chain`
numbers are not revalidated here and must not be presented as Python API speedups.

## What the corrected handle supports

`Tensor.compile_fused()` walks the recorded graph once. `handle.replay()` executes
all its supported operations, including the first unary or binary operation,
over current graph-leaf storage. It returns a detached tensor. The handle is not
a CUDA graph and does not provide a fused backward pass.

Supported: tagged, left-to-right pointwise chains such as `relu(x)*y+z`, `x*y+z`,
`(relu(x)-y)/z`, repeated leaf operands such as `x*x-x`, and right-leaf broadcasts
that preserve the seed shape (`[2,3]` with `[3]`). Operand order is preserved.

Rejected explicitly on CPU and CUDA: seed expansion (`[3]` to `[2,3]`), computed
right-side branches (`relu(x)*relu(y)`), repeated computed intermediates (`u*u`
where `u=relu(x)`), untagged operations/reductions and roots without a tape.
These raise an error rather than capture stale intermediates.

Enable `requires_grad` **before computing every upstream path that must be
replayed**. The compiler can only see the recorded autograd graph. Detached or
no-grad computations have no upstream tape and are opaque input values; it
cannot recover their original leaves. Explicitly detach a computed value only
when treating it as a frozen graph input is intended. Shape/storage identities
are fixed; leaf values can change (optimizer-driven mutations are regression
tested on CPU and CUDA).

The independent counting backend test asserts three compiled steps, one
`chain_dev` call, zero per-op calls, and numeric output. `fusion_launches()` is a
planner estimate, not a runtime launch counter.

## Corrected measurement

Hardware: NVIDIA GeForce RTX 3090 (GDDR6X, not HBM). Windows, Python bindings built
with `maturin develop --release`, PyTorch 2.6.0+cu124. Two separate runs, each with
30 warmups and 100 wall-clock samples per path and size:

```bash
crates/ferro-py/.venv/Scripts/python.exe bench/compiled_fusion.py --iters 100 --warmup 30 --json bench/compiled_fusion_3090.json
crates/ferro-py/.venv/Scripts/python.exe bench/compiled_fusion.py --iters 100 --warmup 30 --json bench/compiled_fusion_3090_run2.json
```

All paths compute **the full `relu(x)*y+z`** with the same f32 input values and
with autograd disabled during timing. Tape construction and compilation occur
outside timing. Each timed call is bracketed by a copy-free device fence;
allocation and dispatch are included. Device-to-host correctness reads are
outside timing. Before timing, shape, detached-output, three-step and operand
count assertions run, followed by `torch.testing.assert_close` against both
ferro eager and PyTorch eager (`rtol=1e-5`, `atol=1e-6`).

| Run | n | ferro eager median us | replay median us | torch eager median us | eager/replay | torch/replay |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2^20 | 54.3 | 33.5 | 68.2 | 1.62x | 2.03x |
| 1 | 2^22 | 180.5 | 92.0 | 191.8 | 1.96x | 2.09x |
| 1 | 2^24 | 652.8 | 320.4 | 660.3 | 2.04x | 2.06x |
| 1 | 2^26 | 2543.3 | 1224.5 | 2503.5 | 2.08x | 2.04x |
| 2 | 2^20 | 60.6 | 35.4 | 68.4 | 1.71x | 1.93x |
| 2 | 2^22 | 173.2 | 90.8 | 192.9 | 1.91x | 2.12x |
| 2 | 2^24 | 651.1 | 318.3 | 641.9 | 2.05x | 2.02x |
| 2 | 2^26 | 2533.3 | 1226.9 | 2503.7 | 2.06x | 2.04x |

At the largest size the conservative two-run floor is 2.06x vs ferro eager and
2.04x vs PyTorch eager (rounded). Small sizes show host/launch noise; their
ratios are not a stable throughput claim. This does not compare to
`torch.compile` and does not establish a language advantage or a training win.

For f32, the unfused chain's ideal physical traffic is 32n bytes (2+3+3 array
passes), vs 16n bytes fused (x,y,z reads and one output write). The benchmark
reports bandwidth using these separate numerators, not a fused numerator for
the unfused path. At 2^26 it measured 844.4/847.7 GB/s eager and 876.9/875.2 GB/s
replay across the two runs. Cache effects and host overhead remain included.

Both runs passed correctness assertions at every size; the largest observed
absolute difference vs ferro eager was 9.5367431640625e-7. This is **not a ULP
measurement** and implies no universal ULP bound. Raw samples, medians, min/min
secondary ratios and correctness differences are in the two JSON files above.
Verification logs and the failing regression evidence are in `verification/pr18/`.
