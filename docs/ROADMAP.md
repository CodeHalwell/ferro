# Roadmap

Phases 0-5 of this plan are now largely implemented; see FUTURE.md for
the current forward-looking master plan that supersedes the later phases.

This is the honest path from today's MVP to the ambitious end goal: replacing
enough of PyTorch's C++ backend (ATen + autograd) that `ferro` can stand in for
it on a real, if narrow, set of workloads.

Be candid up front: full PyTorch parity is a multi-year effort for a large team.
PyTorch's backend spans thousands of operators, many dtypes, dozens of device and
dispatch keys, complex broadcasting/type-promotion rules, and a huge amount of
performance engineering. `ferro` is not trying to match that on a schedule. The
realistic win is a growing, well-tested subset that is numerically validated
against PyTorch and useful as a sidecar. The phases below are ordered so each one
delivers something usable and de-risks the next.

## Phase 0 - MVP core (done)

Status: complete and gradient-checked.

- `Tensor` with `Arc`-shared storage and strided views (shape/stride/offset).
- Zero-copy broadcast (zero strides) and transpose views.
- 2-D matmul.
- Reverse-mode autograd via recorded `Op` nodes and a topological `backward()`.
- Grad accumulation for reused leaves; `unbroadcast` for broadcasted grads.
- Ops: add, sub, mul, div, matmul, sum, mean, relu, exp, sigmoid, neg, reshape,
  transpose.
- Deterministic PRNG; central-difference gradient checks; a converging
  linear-regression training loop.

See `ARCHITECTURE.md`. Everything after this builds on this frozen core.

## Phase 1 - nn / optim + Python bindings (in progress)

Scope: make `ferro` usable for small training loops from both Rust and Python.

- `nn`: `Module` trait, `Linear`, activations, `Sequential` (partially present).
- `optim`: `Sgd`, `Adam` (partially present).
- `ferro-py`: PyO3 bindings (built via `maturin`) exposing `Tensor`, ops,
  `backward`, and enough of `nn`/`optim` to write a training loop in Python.
- Loss functions (MSE, cross-entropy) and a couple of end-to-end examples.

Difficulty: low-to-moderate. This is mostly wiring over the existing core, plus
PyO3 marshaling. No new autograd or kernel design is required.

## Phase 2 - broaden op / dtype coverage and shapes

Scope: turn the MVP into a genuinely useful CPU tensor library.

- More elementwise ops (log, sqrt, tanh, pow, comparisons, clamp, etc.) and their
  backwards.
- Reductions over arbitrary dims and multiple dims, with keepdim; max/min/argmax,
  softmax/log-softmax.
- N-D and batched matmul (generalize `raw_matmul` beyond 2-D), plus `bmm` and
  broadcasting matmul semantics.
- More dtypes behind the existing `Storage` enum (f64, i64, bool at least), with
  type-promotion rules for mixed-dtype ops.
- Indexing / slicing / concatenation / stacking as views or copies.

Difficulty: moderate. The `Storage` enum and the raw-kernel indirection were
designed for this, but each new dtype multiplies the kernel matrix, which
motivates Phase 3.

## Phase 3 - ATen-style dispatcher

Scope: stop hard-wiring autograd into every forward op; route op calls through a
dispatcher keyed on autograd/device/dtype.

- Introduce dispatch keys (Autograd, CPU, CUDA, Meta) and a registry mapping
  (operator, key) to a kernel.
- Separate the autograd layer (records the `Op` node, then re-dispatches to the
  device kernel) from the device kernels themselves.
- Add a Meta backend that computes only shapes/dtypes (no data) for shape
  inference and testing.

Difficulty: moderate-to-high; it is an architectural change, but the current
"raw kernel + record autograd" split already prefigures it. See `DISPATCHER.md`
for the design and an incremental, non-big-bang migration plan.

## Phase 4 - DLPack zero-copy interop

Scope: make `ferro` dogfoodable early by sharing buffers with `torch` and
`numpy` instead of copying.

- Export `ferro` tensors as DLPack capsules and consume DLPack capsules from
  torch/numpy, zero-copy for CPU (and later CUDA) buffers.
- Use this to numerically validate `ferro` kernels against torch element-for-
  element, and to run `ferro` as a sidecar in an otherwise-PyTorch program rather
  than requiring an all-or-nothing switch.

Difficulty: moderate. The main work is on the storage side: today storage is a
contiguous f32 `Vec` behind an `Arc`, and DLPack needs a raw pointer + deleter,
plus dtype/stride mapping. See `DLPACK.md`.

## Phase 5 - device backends

Scope: real performance and hardware support.

- CPU: back matmul and heavy kernels with a BLAS (or a tuned Rust implementation)
  behind the dispatcher's CPU key.
- CUDA: a GPU backend, e.g. via `cudarc`, registered under the CUDA dispatch key,
  reusing the DLPack path for interop with torch CUDA tensors.

Difficulty: high, and open-ended. This is where most of PyTorch's real
engineering investment lives (kernel performance, memory management, streams).
`ferro` will only ever cover a subset here.

## Phase 6 - torch-compatible Python API shim

Scope: let existing PyTorch-style code run against `ferro` with minimal changes.

- A `torch`-shaped Python API surface (tensor creation, ops, `nn.Module`,
  `optim`) mapping onto `ferro`'s bindings.
- Enough compatibility to run selected small models unmodified, validated against
  real PyTorch via the DLPack bridge.

Difficulty: high and never fully "done". The value is in covering a useful
subset, not in chasing 100% API parity.

## Summary

The arc is: solid MVP core (0) -> usable from Python for small models (1) ->
broad CPU op/dtype coverage (2) -> a dispatcher to keep that maintainable (3) ->
zero-copy interop so it can be validated and dogfooded incrementally (4) ->
device backends for performance (5) -> a compatibility shim on top (6). Each
phase is independently useful, and the project can stop at any phase and still be
a working, interesting artifact.
