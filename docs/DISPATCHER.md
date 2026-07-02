# Dispatcher Design Sketch

This is a design sketch for an ATen-style dispatcher for `ferro`, describing
the target design and an incremental path to get there without a big-bang
rewrite. See `ROADMAP.md` Phase 3 for where this fits.

## Status

Implemented so far (phase 1):

- `device.rs`: a `Device` enum (`Cpu`, `Cuda(u32)`) carried on every
  `TensorInner` and inherited by views. Creation defaults to `Cpu`;
  `Tensor::device()` is public; `raw_binary`/`raw_matmul` reject mixed-device
  operands with `Error::DeviceMismatch`.
- `dispatch.rs`: a process-wide kernel table. `matmul` routes through a
  swappable function pointer (`set_matmul_kernel`), seeded with the naive
  reference kernel; the `ferro-fastcpu` crate overrides it from outside core.
- Autograd was already unified behind `record_fn`, so the autograd layer is a
  single wrapping mechanism rather than per-op branching - the "Autograd key"
  in miniature.

Not yet implemented: named elementwise kernels (still inline CPU closures),
per-device kernel tables (the current table is CPU-only), Meta kernels, and
any real second device. The sections below describe that target.

Note: the code snippet immediately below predates the record_fn unification;
today's ops.rs already records via closures. The scaling argument stands.

## Motivation

Today, autograd is hard-wired into every forward op. In `ops.rs`, each method
computes a value with a detached raw kernel and then, inline, decides whether to
record an `Op` node:

```rust
pub fn add(&self, other: &Tensor) -> Result<Tensor> {
    let out = raw_binary("add", self, other, |a, b| a + b)?;
    let rg = self.requires_grad() || other.requires_grad();
    Ok(out.record(rg, || Op::Add(self.clone(), other.clone())))
}
```

This is fine for one device and one dtype, but it does not scale to a matrix of
(operator x device x dtype). As soon as there are CPU and CUDA kernels, or
several dtypes, we do not want each forward op to branch by hand over device,
dtype, and requires_grad. PyTorch solves this with a dispatcher keyed on
"dispatch keys".

## Concept: dispatch keys

Each operator (`add`, `matmul`, ...) has multiple registered kernels. A call is
routed to one of them based on a set of dispatch keys derived from the inputs:

- Autograd: a wrapping layer. If any input requires grad, this key is active. Its
  kernel records the backward node and then re-dispatches (with the Autograd key
  removed) to run the actual compute.
- CPU: the CPU device kernel for the operator (the current `raw_*` functions).
- CUDA: the GPU device kernel (future, e.g. via `cudarc`).
- Meta: a shape/dtype-only kernel that allocates no data. Useful for shape
  inference, tracing, and testing kernels' metadata without running them.

Keys have a precedence order. Autograd sits above the device keys, so a call with
grad enabled first hits Autograd (record), then falls through to the device key
(compute). The device key is chosen from where the input storage lives.

## How a call routes

Conceptually, for `c = a.add(&b)`:

1. Compute the key set from inputs: device key from the storage (CPU/CUDA),
   plus Autograd if any input requires grad.
2. Dispatch to the highest-precedence key: Autograd.
3. The Autograd kernel:
   - re-dispatches the same op with Autograd masked off (this runs the device
     kernel and produces the output value), then
   - records the backward node (`Op::Add(a, b)`) on the output, exactly what
     `Tensor::record` does today.
4. The device kernel (CPU today) runs the raw computation (`raw_binary` with
   `+`) and returns the output tensor.
5. For `Meta`, the device kernel only computes the output shape/dtype.

The autograd layer is thus fully separated from the device kernels: autograd
knows how to build the graph and re-dispatch; device kernels know only how to
compute values (or metadata). Backward, in turn, calls ops, which re-enter the
dispatcher and land on device kernels with Autograd masked off, so higher-order
graph construction stays possible later without special casing.

## Mapping onto the current design

The current code already has the two halves the dispatcher wants to separate;
they are just fused in `ops.rs`:

- Device-kernel half: the `raw_*` functions in `tensor.rs` (`raw_binary`,
  `raw_unary`, `raw_matmul`, `raw_sum_dim`). These are pure compute, no autograd.
  They map directly onto the CPU dispatch key's kernels.
- Autograd half: the `Op` enum plus `Tensor::record` and `backward` in
  `autograd.rs`. `Op::backward` is already the per-op vector-Jacobian product,
  and `Op::inputs` already declares the graph edges. This maps directly onto the
  Autograd dispatch key.

So a dispatcher does not require new math. It requires a layer of indirection:
instead of `ops.rs::add` calling `raw_binary` then `record` directly, it would
issue an operator call into the dispatcher, and registrations would wire
Autograd -> CPU.

## Incremental migration plan

The goal is to introduce the dispatcher without freezing feature work or
rewriting kernels.

1. Define the operator identity and key types. Add an `enum DispatchKey { Autograd, Cpu, Cuda, Meta }`
   and a way to name operators (an enum or interned string). No behavior change.

2. Add a registry and a single generic entry point, e.g.
   `dispatch(op, key_set, args)`, backed by a table from
   `(operator, DispatchKey)` to a kernel function. Initially register only CPU
   kernels that call the existing `raw_*` functions.

3. Wrap autograd as a key, not inline code. Register an Autograd kernel per op
   that (a) re-dispatches with Autograd masked to get the value and (b) calls the
   existing `record`/`Op` logic. This is a mechanical move of the code already in
   `ops.rs`.

4. Route one op through the dispatcher end to end (say `add`) while leaving the
   rest calling `raw_*` directly. Confirm `tests/autograd.rs` still passes. This
   proves the plumbing on a single op.

5. Migrate the remaining ops one at a time. Each migration is: register its CPU
   kernel (its `raw_*` call) and its Autograd kernel (its `Op` variant), then
   change the public `Tensor` method to go through `dispatch`. The `Op` enum and
   `backward` stay as-is.

6. Add the Meta key. Register shape/dtype-only kernels for the migrated ops to
   enable shape inference and cheaper tests.

7. Only then add new device keys (CUDA in Phase 5). At that point new backends
   are additive: register a CUDA kernel per op; the Autograd layer and graph code
   are untouched.

Because each step keeps the gradient-checked test suite green, the dispatcher can
land gradually behind the existing public `Tensor` API rather than as a single
disruptive change.
