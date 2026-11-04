# ferro

> Start with the [current capability ledger and next milestone](docs/CURRENT_STATUS.md). The overview below describes the earlier MVP and is not a complete inventory of the current implementation.

`ferro` is a from-scratch, Rust reimplementation of a PyTorch-style tensor and
autograd backend. It is an experiment: how far can a Rust replacement for
PyTorch's C++ core (ATen/autograd) actually go?

This is a long-horizon, for-fun project. The honest goal is not to ship a
drop-in PyTorch replacement tomorrow, but to grow a real, correct, testable
tensor + reverse-mode autograd runtime in Rust starting from a small but working
CPU MVP, and to see how much of PyTorch's backend can be rebuilt from clean
foundations. Full parity with PyTorch is a multi-year, large-team effort; the
realistic near-term win is a growing, well-tested subset that can be validated
against PyTorch numerically.

## Status

What works today (in `ferro-core`, pure Rust, zero dependencies):

- A reference-counted `Tensor` with `Arc`-shared storage, plus shape / stride /
  offset metadata, so transpose and broadcast are zero-copy strided views.
- NumPy/PyTorch broadcasting for elementwise binary ops, implemented as
  zero-stride views.
- 2-D matmul (`(m,k) @ (k,n) -> (m,n)`).
- Reverse-mode autograd: each forward op records a detached graph node; a scalar
  `backward()` runs a topological reverse pass, accumulating gradients for reused
  leaves and reducing broadcasted gradients via `unbroadcast`.
- Ops with backward support, core: `add`, `sub`, `mul`, `div`, `matmul`, `sum`,
  `mean`, `relu`, `exp`, `sigmoid`, `neg`, `reshape`, `transpose`; extended
  (one file per op in `ops_ext/`, recorded via the `record_fn` closure hook):
  `log`, `tanh`, `sqrt`, `abs`, `powf`, `clamp`, `max`, `sum_dim`, `mean_dim`,
  `softmax`, `log_softmax`, `bmm`.
- Repeated `backward()` follows torch retain_graph semantics (leaf grads
  accumulate, interior grads recomputed); the engine asserts gradient arity and
  shape from custom ops, and both the topological sort and graph teardown are
  iterative, so very deep graphs (100k+ ops) neither overflow the stack in
  backward nor on drop.
- A small deterministic PRNG (`Rng`, splitmix64-seeded xorshift128+) for weight
  init and tests, with no external `rand` dependency.
- Every op is gradient-checked against central differences via
  `testkit::grad_check`, and a linear-regression training loop converges with
  plain SGD (see `crates/ferro-core/tests/`).
- `nn` module: `Module` trait, `Linear` (He init), `Relu`, `Sigmoid`,
  `Sequential`; a 2-layer MLP trains.
- `optim` module: `Sgd` (with momentum), bias-corrected `Adam`.
- `ferro-py`: PyO3 Python bindings (built standalone via `maturin`) - training
  from Python on the Rust backend works end to end.
- DLPack interop: ferro tensors exchange with numpy and torch in all four
  directions (`np.from_dlpack` / `torch.from_dlpack` / `ferro.from_dlpack`),
  which is how ferro kernels are validated against torch numerically.

Storage supports f32 (full compute + autograd), native CPU bool (masks/predicates),
f64 and i64 (data/index
tensors with explicit `to_dtype` casts into float math), and opaque
device-resident buffers; float math and autograd remain f32-only.

## Quickstart

The current Python API includes PyTorch-style factories and tensor functions,
RNN/GRU/LSTM modules, weighted classification losses, and PyG-style GCN, SAGE,
GIN, GAT and GATv2 layers with graph batching and pooling. See the
[compatibility matrix](docs/PYTORCH_PYG_COMPATIBILITY.md) for supported options,
examples and explicit limits. Numerical comparisons use PyTorch 2.13.0+cpu and
PyG 2.8.1; Ferro does not require either package at runtime.

The portability stage adds native Bool tensors/masks, more spatial and graph
layers, heterogeneous graph containers, neighborhood/link sampling, optimizer
groups, schedulers and lazy Python data pipelines. NVIDIA CUDA and Apple Metal
are the GPU targets; see [GPU portability and remaining gaps](docs/GPU_PORTABILITY.md)
for current backend limits and the implementation/validation sequence.

Run `python examples/pyg_parity.py` to train GCN, GraphSAGE and GATv2 classifiers
using the new batching/pooling/loss APIs and verify exact checkpoint continuation.

Run the core test suite (tensor + autograd, gradient checks, training loop):

```
cargo test -p ferro-core
```

`ferro-core` has no external dependencies, so this builds with a stock Rust
toolchain and no network access.

### Graph layers from Python

With the Python extension built, the CPU f32 graph API supports directed
topology, disjoint graph batching, scatter reductions and graph convolution:

```python
from ferro import Tensor
from ferro.graph import EdgeIndex, GcnConv

edges = EdgeIndex(3, [0, 1], [1, 0])  # Node 2 is isolated.
layer = GcnConv(in_features=2, out_features=4)
x = Tensor([1., 0., 0., 1., 1., 1.], [3, 2]).requires_grad_(True)
y = layer(x, edges)
y.sum().backward()
```

Weights, bias and immutable layer configuration participate in the ordinary
Module/optimizer/checkpoint APIs. Save the input topology separately using
`edges.to_tensor()` plus `edges.num_nodes`. This graph layer is first-order
CPU f32; resident GPU graph training and full architecture certification remain
open. See `crates/ferro-py/tests/test_gnn.py` for independent dense-reference
gradient checks and checkpoint continuation.

Run the complete synthetic GCN learning and fresh-process restart proof with:

```sh
python examples/train_restart_gcn.py
```

It trains two graph-convolution layers for 240 steps with seeds 7, 11 and 43,
checks a dense PyTorch gradient reference, and requires at least 95% held-out
accuracy. Removing graph edges must reduce accuracy by at least 40 percentage
points. A worker restores the saved graph/data bundle and training checkpoint
at step 117 and must reproduce every remaining loss and complete training state
exactly. This is a bounded CPU f32 fixture; held-out feature samples are disjoint
while graph topology templates may repeat.

### CNN, recurrent and spline training

Three further public CPU f32 examples check learning, independent PyTorch
gradients and fresh-process restart across seeds 7, 11 and 43:

```sh
python examples/train_restart_cnn.py
python examples/train_restart_recurrent.py
python examples/train_restart_kan.py
```

The CNN recognizes bar orientation with identical per-class intensity
histograms. The GRU remembers token order when paired sequences have identical
token multisets and final tokens. Both require at least 95% held-out accuracy
and at least a 40 percentage point loss when the relevant information is
removed. The two-layer KAN fits a nonlinear function and must beat both
training-fit constant and affine baselines by at least 90% in held-out MSE.

Each fixture saves its dataset and cyclic cursor alongside the ordinary
model/AdamW/config/Generator checkpoint. Restart must reproduce the remaining
losses and complete saved state exactly within the same environment. These are
small synthetic fixtures; the GRU restarts between complete optimizer steps,
with zero hidden state per batch. See the
[architecture continuation report](verification/architectures-2026-11-03/README.md)
for results and limits. KAN checkpoints now include immutable knots, degree
and outside-domain policy; `layer.basis` remains readable, and changing it
requires constructing a new layer.

### Training with differentiable derivatives

Two public CPU f32 examples exercise derivatives inside a training objective:

```sh
python examples/train_restart_pinn.py
python examples/train_restart_contractive.py
```

The PINN solves a one-dimensional Poisson problem using a second-coordinate-
derivative residual, with boundary conditions enforced by its model form. The
contractive autoencoder adds a squared encoder-Jacobian penalty to reconstruction.
Both check the resulting parameter gradients against independent references and
require exact fresh-process restart across three seeds. The
[scientific training report](verification/scientific-2026-11-03/README.md) records
learning gates, the autoencoder's batch-order correction, and the tested limits.

`ferro.autograd.grad` and `vjp` leave persistent `.grad` slots unchanged and can
build derivative graphs for the documented smooth subset. They require eager
execution: inference capture rejects functional differentiation to prevent
freezing a derivative at the original input. These fixtures do not establish
arbitrary PDE, higher-order operator or GPU support.

### Variational and adversarial training

Two further CPU f32 examples test stochastic objectives and exact restart:

```sh
python examples/train_restart_vae.py
python examples/train_restart_wgan.py
```

The Gaussian VAE uses reparameterized latent samples, a normalized Gaussian
likelihood and analytic KL. Its held-out evaluation uses a saved, fixed bank of
latent noise. The WGAN-GP learns a one-dimensional Gaussian location and scale,
with live mixed derivatives through its gradient penalty and five critic updates
per generator update. It checks inactive parameters and optimizer state and
restarts midway through that update cycle. Both run seeds 7, 11 and 43, with
independent PyTorch gradient checks and saved data, phase, optimizer and RNG state.
The [generative training report](verification/adversarial-2026-11-03/README.md)
records the fixed gates, successful runs and the failed first WGAN configuration.
These bounded fixtures do not establish general generative-model convergence.

AdamW now advances bias-correction counters only for parameters with gradients.
Its snapshot key `t` is an f32 vector in parameter order; restore also accepts
the old scalar form and expands its recorded count. Old readers cannot load the
new vector form, and old snapshots cannot recover historical skipped updates.
Counters still use f32 serialization, so very large counts can lose precision.
Rust optimizer clones share parameter slots but own independent optimizer state.

### Sparse attention and streaming state

`ferro.graph.edge_attention(q, k, v, edges, edge_bias=None)` computes CPU f32
single-head attention over directed edges. Destination queries attend source
keys and values; optional bias has one entry per edge. Duplicate edges remain
independent softmax entries, and isolated destinations return zeros. Queries
and keys have shape `[nodes, D]`, values `[nodes, Dv]`. The implementation uses
node/edge-sized intermediates and supports eager first-order gradients.

```sh
python examples/train_restart_graph_attention.py
python examples/train_restart_streaming.py
```

The attention example learns routing that depends on both a query/key match and
edge attributes. Its tests compare dense PyTorch gradients, disjoint graph
batches and topology permutations, and check removal/rewiring ablations. The
streaming LSTM carries detached hidden and cell state between training chunks;
its checkpoint saves that actual state so a fresh process can continue without
recomputing the prefix. Each example checks three seeds and exact continuation.
These are bounded synthetic CPU fixtures; the
[graph and sequence report](verification/graph-sequence-2026-11-03/README.md)
records their gates, results and limitations.

## Project layout

```
ferro/
  Cargo.toml              # workspace (members: ferro-core; ferro-py excluded)
  crates/
    ferro-core/           # pure-Rust tensor + autograd runtime (zero deps)
      src/
        tensor.rs         # Tensor, Storage, views, raw compute kernels
        ops.rs            # core forward ops that record autograd nodes
        ops_ext/          # extended ops, one file per op (record_fn pattern)
        autograd.rs       # Op graph node, backward, topological pass
        shape.rs          # strides, numel, broadcasting rules
        params.rs         # Param: trainable parameter slot
        rng.rs            # small deterministic PRNG
        nn.rs             # Module, Linear, Relu/Sigmoid, Sequential
        optim.rs          # Sgd (momentum), Adam
        testkit.rs        # public finite-difference grad_check
        interop.rs        # contiguous buffer surface for DLPack
        error.rs          # Error / Result
      tests/              # per-op gradient checks + training loops
    ferro-py/             # PyO3 Python bindings + DLPack, built via maturin
  docs/
    ARCHITECTURE.md       # crate + Tensor/autograd design
    ROADMAP.md            # MVP -> "replace the C++ backend" plan
    FUTURE.md             # forward master plan: parity + differentiators
    CAPABILITY.md         # the depth plan: theory, designs, acceptance gates
    DISPATCHER.md         # ATen-style dispatcher design sketch
    DLPACK.md             # zero-copy interop plan (torch/numpy)
```

## How to add a new op

New ops are self-contained: one file in `ops_ext/`, one test file, no shared
code touched. `ops_ext/log.rs` is the worked reference:

1. Create `ops_ext/your_op.rs` with an `impl Tensor` block. Compute the value
   with a detached raw kernel (`raw_unary` / `raw_binary` / `raw_sum_dim` from
   `tensor.rs`, or your own loop over `to_vec()`).
2. Attach autograd with `out.record_fn(vec![inputs...], move |g| vec![grads...])`
   - a closure returning one gradient per input, in order. If backward needs the
   output, capture `let y = out.detach_copy();` (never the live output - that
   would create a reference cycle). The engine asserts arity and gradient shapes.
3. Register the module in `ops_ext/mod.rs`.
4. Add `tests/op_your_op.rs` with a value test and a
   `ferro_core::testkit::grad_check` finite-difference check.

See `docs/ARCHITECTURE.md` for the full design, and `docs/DISPATCHER.md` /
`docs/DLPACK.md` for where this is headed.
