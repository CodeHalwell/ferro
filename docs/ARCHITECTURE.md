# Architecture

This document describes how `ferro` is structured today. It is grounded in the
current code, not in aspirations; see `ROADMAP.md`, `DISPATCHER.md`, and
`DLPACK.md` for where things are headed.

## Crate layout

The workspace lives under `ferro/`:

- `crates/ferro-core`: the pure-Rust tensor + reverse-mode autograd runtime. It
  has intentionally zero external dependencies, so the core compute and
  differentiation layer stays trivially portable and auditable. This is the
  authoritative layer.
- `crates/ferro-py`: Python bindings via PyO3 (in progress). Built standalone
  with `maturin`; it is excluded from the workspace so that
  `cargo test -p ferro-core` does not have to resolve Python-binding
  dependencies.
- Future device crates (not present yet): CPU BLAS and CUDA backends would live
  as sibling crates that may take dependencies, keeping `ferro-core` clean.

`ferro-core` exposes its surface through `lib.rs`: modules `tensor`, `ops`,
`autograd` (private), `shape`, `params`, `rng`, `nn`, `optim`, `error`, and
re-exports `Tensor`, `Storage`, `Param`, `Rng`, `Error`, `Result`.

## Tensor and Storage

The central type is `Tensor(Arc<TensorInner>)` (`tensor.rs`). Cloning a `Tensor`
is cheap: it bumps an `Arc` and shares identity. That shared identity is what
lets a value used in several ops accumulate its gradient exactly once.

`TensorInner` holds:

- `id: usize` - a process-unique id (from a global atomic counter) used to
  deduplicate nodes during the autograd topological sort.
- `storage: Arc<Storage>` - the backing buffer, shared across views.
- `shape: Vec<usize>`, `stride: Vec<usize>`, `offset: usize` - the strided view
  descriptor. Multiple tensors can point at the same `Storage` with different
  shape/stride/offset.
- `requires_grad: bool`.
- `op: Option<Op>` - how this tensor was produced, for reverse mode. `None` for
  leaves.
- `grad: Mutex<Option<Tensor>>` - the accumulated gradient slot.

`Storage` carries the element buffer in one of four variants:

```rust
pub enum Storage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Device(Box<dyn DeviceBuffer>),
}
```

Compute kernels and autograd are f32-only: F64/I64 storage carries data
(indices, class targets) through views and materialization, with explicit
`to_dtype` casts as the only route into float math, and `Device` holds an
opaque backend-owned buffer for device-resident tensors (see DISPATCHER.md).
The view and autograd machinery is shared across all variants.

### Views share storage

Several operations produce new `Tensor` handles that share the same `Storage`
and only differ in shape/stride/offset:

- `broadcast_to` (crate-internal): expands to a target shape by inserting
  zero strides for broadcasted dimensions. It is detached; broadcasting's
  gradient is handled in backward by reducing (see `unbroadcast`).
- `transpose_view` / `transpose`: swaps two dims' shape and stride entries.
- `reshape`: if the source is contiguous it re-descriptors in place; if it is a
  strided (non-contiguous) view it first materializes a contiguous copy via
  `to_vec`.

`is_contiguous()` compares the current stride against `default_strides(shape)`
(row-major / C-contiguous) from `shape.rs`.

### Materialization: to_vec

Every compute kernel reads its inputs through `Tensor::to_vec`, which gathers a
possibly-strided/broadcast view into a contiguous, row-major `Vec<f32>` by
walking a multi-dimensional index and applying `offset + sum(idx[d] * stride[d])`.
Because all kernels go through this one path, strided views (transpose,
broadcast) work transparently without special-casing in each op. `item()` is
`to_vec()[0]` for scalar/single-element tensors.

### Raw kernels vs forward ops

`tensor.rs` defines detached raw kernels that never record autograd:

- `raw_binary(op, a, b, f)`: broadcasts both inputs to the common shape (via
  `broadcast_shapes` in `shape.rs`), then applies `f` elementwise.
- `raw_unary(a, f)`: applies `f` elementwise.
- `raw_matmul(a, b)`: 2-D only in the MVP; returns `Error::Unsupported` for
  other ranks. Uses an i-p-j loop with a zero-skip on the left operand.
- `raw_sum_dim(t, dim, keepdim)`: sum reduction over one dim with PyTorch
  keepdim semantics.
- `unbroadcast(g, target)`: reduces a broadcasted gradient back to `target` shape
  by summing over the expanded dims.

These raw kernels are called by both the forward ops (in `ops.rs`) and the
backward pass (in `autograd.rs`), which keeps forward and backward numerically
consistent and avoids duplicated math.

## Autograd

Forward ops in `ops.rs` follow one pattern: compute the value with a detached raw
kernel, then, only when a gradient is needed, attach the autograd node:

```rust
pub fn add(&self, other: &Tensor) -> Result<Tensor> {
    let out = raw_binary("add", self, other, |a, b| a + b)?;
    let rg = self.requires_grad() || other.requires_grad();
    Ok(out.record(rg, || Op::Add(self.clone(), other.clone())))
}
```

`Tensor::record(requires_grad, op)` sets `requires_grad` and installs the `Op`
on the freshly-produced output. It asserts the output is uniquely owned
(`Arc::get_mut`), which holds because it was just computed.

### The Op graph

`Op` (in `autograd.rs`) is an enum, one variant per differentiable op, each
holding the input tensors it needs to differentiate:

```
Add, Sub, Mul, Div, Matmul, Sum, Mean, Relu, Exp, Sigmoid, Neg, Reshape, Transpose
```

Key points:

- The `Op` node holds cloned `Tensor` inputs (cheap `Arc` bumps). Because the
  node is attached to the output and references the inputs, cloning a `Tensor`
  keeps its producing subgraph alive.
- Ops that need their own output in the backward pass (`Exp`, `Sigmoid`) stash a
  detached snapshot of the output (`detach_copy`) inside the `Op`, rather than a
  live handle. This avoids a reference cycle (output -> op -> output).
- `Op::inputs()` returns the input tensors in a fixed order.
- `Op::backward(g)` takes the gradient flowing into the op's output and returns
  the gradient for each input, in `inputs()` order. This is the vector-Jacobian
  product for that op. For example, `Matmul` computes
  `dA = dC @ B^T` and `dB = A^T @ dC` using `raw_matmul` and `transpose_view`;
  binary ops reduce their per-input grads with `unbroadcast`.

### The backward pass

`Tensor::backward()` is meant to be called on a scalar loss:

1. `build_topo` walks the graph from the loss, deduplicating by tensor `id` with
   a `HashSet`, and produces a topological order (`Vec<Tensor>`).
2. The loss's gradient is seeded with ones (`Tensor::ones(shape)`).
3. Iterating the topological order in reverse, for each op node it reads the
   node's accumulated gradient, calls `op.backward(&g)`, and for each input that
   `requires_grad` calls `inp.accumulate_grad(ig)`.

`accumulate_grad` adds into the existing grad slot (`raw_binary` with `+`) when
one is already present, which is what makes reused leaves correct: a leaf
consumed by multiple ops receives contributions from each and sums them. The
test `reused_leaf_accumulates` checks exactly this (`d(x*x)/dx = 2x`).

Grad storage lives on the tensor: `grad()`, `zero_grad()`, and the internal
`set_grad` / `accumulate_grad`, all behind the per-tensor `Mutex`.

## Param abstraction

`Param` (in `params.rs`) is the trainable-parameter abstraction: a shared,
mutable slot holding a leaf tensor with `requires_grad = true`. It is
`Rc<RefCell<Tensor>>` for the single-threaded MVP (a threaded runtime would swap
these for `Arc<Mutex<...>>`).

`Param::new`/`Param::set` take an OWNING copy of their argument, so the slot's
leaf never aliases caller-held storage. Optimizer steps mutate that leaf's
storage in place through the no-grad seams in `inplace.rs` (fused
`sgd_step`/`adamw_step` kernels): the leaf's identity and storage address are
stable across steps, the storage version is bumped each step, and a stale
graph that saved the parameter fails loudly on its next backward instead of
silently reusing old values. See `inplace.rs` for the mutation rules
(whole-contiguous f32 destinations; the public API additionally refuses
tensors with autograd history).

## nn and optim (in progress)

`nn.rs` and `optim.rs` build on the frozen `Tensor` / `Param` API:

- `nn`: a `Module` trait (`forward`, `parameters`) with `Linear`
  (`y = x @ W + b`, He-initialized), `Relu`, `Sigmoid`, and `Sequential`.
- `optim`: `Sgd` (optional heavy-ball momentum) and `Adam`/`AdamW` (with bias
  correction). Optimizer state (velocity, Adam moments) lives as tensors on
  the parameter's device and is mutated in place by the fused step kernels;
  timestep counters stay host-side.

These are described here at a high level because they are still evolving; the MVP
correctness guarantees rest on `tensor.rs`, `ops.rs`, `autograd.rs`, `shape.rs`,
and the `tests/autograd.rs` suite.
