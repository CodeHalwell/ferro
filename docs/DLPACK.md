# DLPack Interop

This document describes the plan to give `ferro` zero-copy interop with `torch`
and `numpy` via DLPack. It is not implemented yet; it explains why this matters
and what the current storage design would need to change. See `ROADMAP.md`
Phase 4.

## Why zero-copy interop is the key to dogfooding early

`ferro` is a young backend with a growing subset of PyTorch's functionality. The
fastest way to make it useful and trustworthy long before it is complete is to
let it share buffers with a mature runtime instead of copying data across a
boundary:

- Numerical validation. If a `ferro` tensor and a `torch` tensor can point at the
  same buffer, we can run the same op in both and compare element-for-element.
  That turns "does my kernel match PyTorch?" into a cheap, exact test rather than
  a hand-rolled reference.
- Sidecar usage. In an otherwise-PyTorch program, a specific tensor or op can be
  handed to `ferro` and the result handed back, with no copy. This makes `ferro`
  adoptable incrementally: an all-or-nothing switch is not required to get value
  out of it.
- Ecosystem reach. DLPack is the common protocol already spoken by torch, numpy,
  cupy, JAX, and others, so implementing it once buys interop with all of them.

DLPack is a small, stable C ABI for describing a strided tensor: a data pointer,
device, dtype, shape, strides, byte offset, and a deleter callback. Tensors are
exchanged as Python "capsules" wrapping a `DLManagedTensor`. Ownership is passed
via the capsule's deleter, so neither side has to copy.

## Plan: expose and consume capsules

Two directions, both zero-copy for CPU (and later CUDA):

- Export (`ferro` -> torch/numpy): implement `__dlpack__` /
  `__dlpack_device__` on the Python `Tensor` (in `ferro-py`). This builds a
  `DLManagedTensor` describing the tensor's storage and hands out a capsule.
  `torch.from_dlpack(x)` / `np.from_dlpack(x)` then wrap it without copying. The
  managed tensor keeps `ferro`'s storage alive (via its `Arc`) until the
  consumer's deleter fires.
- Import (torch/numpy -> `ferro`): accept a capsule (from `torch.to_dlpack` or
  any `__dlpack__` producer), read the `DLManagedTensor`, and build a `ferro`
  tensor that borrows the foreign buffer, holding the capsule's deleter so the
  source memory stays valid for as long as `ferro` references it.

The natural home is `ferro-py`, exposing `to_dlpack` / `from_dlpack` and the
`__dlpack__` protocol methods.

## What the storage layer needs to change

Today `ferro`'s storage is deliberately simple (`tensor.rs`):

```rust
pub enum Storage {
    F32(Vec<f32>),
}
```

That is a contiguous, owned, f32 `Vec` behind an `Arc`. DLPack needs more than a
`Vec` can express, so the storage abstraction has to grow:

- Raw pointer + deleter. DLPack is defined over a raw data pointer with an
  associated deleter, not a Rust `Vec`. `Storage` needs a variant that holds a
  borrowed/foreign buffer: a raw pointer plus the DLPack deleter to invoke on
  drop (for imported tensors), and for exported tensors a way to keep the owning
  `Arc<Storage>` alive until the consumer's deleter runs. On export, the current
  `Vec<f32>` can be exposed by its pointer directly; the deleter drops the
  `Arc` clone that was leaked into the managed tensor.

- Dtype mapping. `Storage` is f32-only, but DLPack carries an explicit dtype
  (`DLDataType`: code + bits + lanes). Interop needs a `ferro` dtype enum that
  round-trips to/from `DLDataType`. Importing anything other than f32 requires
  the corresponding `Storage` variant (see `ROADMAP.md` Phase 2 on dtypes), so
  early DLPack support can start f32-only and reject other dtypes explicitly.

- Stride mapping. `ferro` already carries `shape`, `stride`, and `offset` per
  tensor, which lines up well with DLPack's `shape`, `strides`, and
  `byte_offset`. Two adjustments: DLPack strides and offsets are in elements
  relative to the data pointer (convert `ferro`'s element strides/offset to the
  DLPack convention, and account for byte offset if the ABI version in use
  expects bytes), and imported tensors may be non-contiguous, which `ferro`
  already supports because every kernel reads through `to_vec`.

- Device tag. CPU only at first. When a CUDA backend lands (Phase 5), the same
  path carries a CUDA device in the `DLDevice`, enabling zero-copy sharing of GPU
  buffers with torch CUDA tensors.

## Payoff

Once export/import work for CPU f32, `ferro` gains an exact oracle (torch/numpy)
for validating every kernel it adds, and it becomes usable as a sidecar inside
real PyTorch programs. That is the difference between a from-scratch backend that
can only be tested against itself and one that can be checked against, and
embedded alongside, the system it is trying to replace.
