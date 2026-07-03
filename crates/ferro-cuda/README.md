# ferro-cuda

CUDA backend for ferro, implementing ferro-core's `Backend` trait on top of
[`cudarc`](https://docs.rs/cudarc): cuBLAS sgemm for `matmul`, and tiny
nvrtc-compiled CUDA C kernels (generated per `UnaryKind`/`BinaryKind`,
compiled once and cached) for the elementwise ops.

## What works

- `install(ordinal)`: initializes the device and registers the backend for
  `Device::Cuda(ordinal)`. Returns `Err` -- never panics -- when the CUDA
  driver, NVRTC, cuBLAS, or the device itself is missing.
- `is_available()`: cheap probe for the driver library (libcuda). True does
  not guarantee a usable device; `install` performs the real init.
- All `UnaryKind` variants (including parametrized `Powf`/`Clamp`, cached per
  scalar value), all `BinaryKind` variants, and `matmul` via cuBLAS.
- Device-resident storage (dispatcher phase 3): the `alloc_from_host` /
  `copy_to_host` / `unary_dev` / `binary_dev` / `matmul_dev` methods operate
  on `CudaBuf` (a `CudaSlice<f32>` tagged with its device), so on a GPU box
  `x.to_device(Device::Cuda(0))` followed by chained ops keeps the data in
  GPU memory: one upload, N device kernels, one download. Driver failures on
  this path surface as `Err(ferro_core::Error::Unsupported)`, and buffers
  from another backend (or another CUDA ordinal) are rejected the same way.

## Host-slice fallback

The host-slice `Backend` methods (`unary`/`binary`/`matmul` over `&[f32]`)
remain as the path core uses for non-resident tensors -- e.g. broadcasted
binaries, which core materializes on the host before dispatching. They are
thin wrappers over the same device kernels: htod, `*_dev` compute, dtoh.

Remaining gaps:

- No broadcasting on device: binaries with mismatched shapes fall back to the
  host-slice path (core errors on device tensors that would need broadcast).
- Autograd is host-only: `requires_grad_` on a device tensor panics in core;
  train on cpu, run inference on the device.
- Ops without device kernels (softmax, reductions, ...) fall back to host
  compute in core and return cpu tensors.

## Usage on a GPU box

```rust
if let Err(e) = ferro_cuda::install(0) {
    eprintln!("CUDA unavailable: {e}");
}
// Tensors moved to Device::Cuda(0) now stay resident across chained ops.
let y = x.to_device(ferro_core::Device::Cuda(0))?.relu().exp();
```

Requires an NVIDIA driver plus the CUDA runtime libraries (libnvrtc,
libcublas) loadable at runtime. Nothing is needed at build time: cudarc's
`dynamic-loading` feature dlopens the libraries on first use, so this crate
compiles and its host-side tests pass on machines with no CUDA installation.
The bindings are pinned to the CUDA 12.8 API (`cuda-12080` feature); set
`CUDARC_CUDA_VERSION` at build time (e.g. `12060`) to target another version.

## Row-major vs cuBLAS column-major

cuBLAS is column-major while ferro buffers are row-major. `matmul` uses the
identity `C^T = B^T * A^T`: since a row-major buffer reinterpreted
column-major is its transpose, a plain N/N sgemm with the operands swapped
and m/n swapped writes `C^T` column-major, which is byte-identical to `C`
row-major. The mapping is unit-tested against ferro-core's reference matmul
with a pure-host simulation of cuBLAS semantics (no GPU needed).

## Testing

`cargo test -p ferro-cuda` is green without a GPU: it covers kernel source
generation for every op kind, the sgemm layout mapping, a compile-level check
that `CudaBackend` implements the full `Backend` trait (including the `*_dev`
methods) and `CudaBuf` the `DeviceBuffer` trait, and that
`install`/`CudaBackend::new` fail gracefully. The `gpu_end_to_end` test is a
no-op without a driver; on a GPU box it validates the host-slice fallback,
direct `*_dev` round trips with foreign-buffer rejection, and a resident
tensor chain (`to_device` -> relu -> exp -> mul -> matmul -> back to cpu)
against the same chain computed on the cpu.
