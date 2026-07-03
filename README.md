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
