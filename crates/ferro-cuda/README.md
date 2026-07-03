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

## The host-slice stopgap

Tensor storage is host-resident until dispatcher phase 3, and the `Backend`
seam passes host slices and returns `Vec<f32>`. Every op therefore round
trips: copy host -> device, compute, copy device -> host. This is correct but
leaves most of the GPU's advantage on the table; tensors cannot yet live on
the GPU. When the dispatcher grows device-resident storage, only the copy
staging here needs to change.

## Usage on a GPU box

```rust
if let Err(e) = ferro_cuda::install(0) {
    eprintln!("CUDA unavailable: {e}");
}
// Tensors on Device::Cuda(0) now dispatch through this backend.
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
generation for every op kind, the sgemm layout mapping, and that
`install`/`CudaBackend::new` fail gracefully. The `gpu_end_to_end` test is a
no-op without a driver and validates real unary/binary/matmul round trips on
a GPU box.
