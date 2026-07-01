# ferro

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
- Ops with backward support: `add`, `sub`, `mul`, `div`, `matmul`, `sum`,
  `mean`, `relu`, `exp`, `sigmoid`, `neg`, `reshape`, `transpose`.
- A small deterministic PRNG (`Rng`, splitmix64-seeded xorshift128+) for weight
  init and tests, with no external `rand` dependency.
- Gradient-checked against central differences, and a linear-regression training
  loop that converges with plain SGD (see `crates/ferro-core/tests/autograd.rs`).

In progress (present in-tree, evolving, not yet load-bearing for the MVP
guarantees above):

- `nn` module: `Module` trait, `Linear`, `Relu`, `Sigmoid`, `Sequential`.
- `optim` module: `Sgd` (with momentum), `Adam`.
- `ferro-py`: PyO3-based Python bindings (built standalone via `maturin`).

Storage is f32-only for now, kept behind a `Storage` enum so more dtypes can be
added without disturbing the tensor/view/autograd machinery.

## Quickstart

Run the core test suite (tensor + autograd, gradient checks, training loop):

```
cd rust_backend
cargo test -p ferro-core
```

`ferro-core` has no external dependencies, so this builds with a stock Rust
toolchain and no network access.

## Project layout

```
rust_backend/
  Cargo.toml              # workspace (members: ferro-core; ferro-py excluded)
  crates/
    ferro-core/           # pure-Rust tensor + autograd runtime (zero deps)
      src/
        tensor.rs         # Tensor, Storage, views, raw compute kernels
        ops.rs            # forward ops that record autograd nodes
        autograd.rs       # Op graph node, backward, topological pass
        shape.rs          # strides, numel, broadcasting rules
        params.rs         # Param: trainable parameter slot
        rng.rs            # small deterministic PRNG
        nn.rs             # (in progress) Module, Linear, Sequential
        optim.rs          # (in progress) Sgd, Adam
        error.rs          # Error / Result
      tests/autograd.rs   # gradient checks + training loop
    ferro-py/             # (in progress) PyO3 Python bindings, built via maturin
  docs/
    ARCHITECTURE.md       # crate + Tensor/autograd design
    ROADMAP.md            # MVP -> "replace the C++ backend" plan
    DISPATCHER.md         # ATen-style dispatcher design sketch
    DLPACK.md             # zero-copy interop plan (torch/numpy)
```

## How to add a new op

The pattern is small and consistent. Using an elementwise or reduction op as a
model:

1. Add a detached compute kernel (no autograd) or reuse an existing one in
   `tensor.rs`. Elementwise ops go through `raw_binary` / `raw_unary`;
   reductions and matmul have their own `raw_*` helpers. These are the same
   kernels the backward pass calls, so keep them autograd-free.
2. Add a forward method on `Tensor` in `ops.rs` that computes the value with the
   raw kernel, then calls `out.record(requires_grad, || Op::YourOp(...))` to
   attach the autograd node only when a gradient is needed. If backward needs the
   output (as `exp` / `sigmoid` do), stash a `detach_copy()` snapshot in the `Op`
   to avoid a reference cycle.
3. Add an `Op::YourOp(...)` variant in `autograd.rs`, list its inputs in
   `Op::inputs()`, and implement its vector-Jacobian product in `Op::backward()`.
   Reduce broadcasted gradients back to input shape with `unbroadcast`.
4. Add a gradient check in `tests/autograd.rs` using `grad_check`, which compares
   autograd against central differences.

See `docs/ARCHITECTURE.md` for the full design, and `docs/DISPATCHER.md` /
`docs/DLPACK.md` for where this is headed.
