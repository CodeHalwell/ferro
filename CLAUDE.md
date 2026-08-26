# ferro: working conventions

ferro is a from-scratch deep learning engine in Rust. Correctness is proven,
not assumed: every operator is gradient-checked, numerics are cross-validated
against PyTorch, and structural claims (device residency, kernel dispatch)
are asserted by counting test backends.

# Build and test

```
cargo test -p ferro-core          # the main suite; must be green before commit
cargo test -p ferro-fastcpu       # optimized CPU backend
cargo check -p ferro-cuda         # must compile WITHOUT CUDA installed
cargo test -p ferro-cuda          # GPU tests are gated on runtime detection
```

Python bindings (standalone maturin crate, excluded from the workspace):

```
cd crates/ferro-py
python3 -m venv .venv && . .venv/bin/activate
pip install -q maturin && maturin develop --release
python ../examples/ops_vs_torch.py    # numeric parity vs torch (pip install torch)
python ../examples/py_regression.py   # binding regression suite
python ../examples/safetensors_vs_python.py  # file-format parity (pip install safetensors)
```

# Architecture invariants (do not break these)

- `ferro-core` has ZERO external dependencies. Device backends and Python
  bindings live in sibling crates that may take dependencies.
- One autograd mechanism: ops record through `Tensor::record_fn` (inputs plus
  a backward closure returning one gradient per input; the engine asserts
  arity and shapes). Never capture the live output in a backward closure -
  capture `out.detach_copy()`.
- Storage is Arc-shared behind a per-cell RwLock; a cell's variant and buffer
  identity NEVER change after construction (values may). In-place mutation
  goes only through the seams in inplace.rs, always bumps the storage
  version, and requires a whole-contiguous f32 destination; the public
  in-place API additionally refuses tensors with autograd history, and
  device tensors with shared storage (device detach_copy shares buffers
  with backward-closure snapshots). Optimizers use the raw no-grad seams.
- A tensor's grad lives on the tensor's device.
- Device kernels see whole contiguous buffers; broadcasting/materialization
  decisions stay in core. Ops without device kernels fall back to host
  compute and visibly return cpu tensors.
- Backends implement the `dispatch::Backend` trait and register per Device;
  core never depends on a backend crate.

# Adding an operator

One file in `crates/ferro-core/src/ops_ext/`, one test file, no shared code
touched. `ops_ext/log.rs` is the worked reference. Register in `ops_ext/
mod.rs`, add `tests/op_<name>.rs` with a value test and a
`ferro_core::testkit::grad_check` finite-difference check. Fallible ops
return `crate::Result`; validate shapes/dims before computing (panicking on
user input is a bug - see the Error enum for the right variant).

# Testing conventions

- Every op with gradients gets a `grad_check` with O(1)-magnitude inputs at
  differentiable points (avoid ties, zeros at kinks, NaN-adjacent regions).
- Numerics of Python-facing ops are compared against torch in
  examples/ops_vs_torch.py - extend it when binding new ops.
- Tests sharing process-global state (backend registries, counters) must
  serialize on a poison-tolerant Mutex; the harness runs tests in parallel.
- Structural claims need structural tests: if you claim residency or
  laziness, count the calls (see tests/device.rs).

# Style

- Minimize comments; code should be self-documenting. Comments carry
  non-obvious context (invariants, coordinate conventions, FFI contracts),
  never restate the code.
- ASCII only in comments and docs. No trivial 1-2 line single-use helpers.
- Prefer single lines over linter-driven wrapping; pick shorter names first.
- Match the style of the file you are editing.

# Git

- Don't commit unless asked. Run the full test suite before any commit.
- Commit messages: explain root cause and fix for bugs, review order for
  large changes, alternatives considered when there was a real choice. Always
  include a Test Plan section with the literal commands run in fenced blocks.
  No bullet lists of individual changes. Disclose AI assistance.
