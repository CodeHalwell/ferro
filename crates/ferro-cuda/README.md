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
  scalar value, and `Gtz`, the relu gradient mask), all `BinaryKind`
  variants, and `matmul` via cuBLAS.
- Device-resident storage (dispatcher phase 3): the `*_dev` methods operate
  on `CudaBuf` (a `CudaSlice<f32>` tagged with its device), so on a GPU box
  `x.to_device(Device::Cuda(0))` followed by chained ops keeps the data in
  GPU memory: one upload, N device kernels, one download. Driver failures on
  this path surface as `Err(ferro_core::Error::Unsupported)`, and buffers
  from another backend (or another CUDA ordinal) are rejected the same way.
- The full extended `Backend` device surface, which is what core needs to run
  autograd resident on the device:
  - `binary_bc_dev`: numpy right-aligned broadcasting. The kernel is
    generated per (kind, rank); the output shape and both padded input
    strides (0 for broadcast dims) are plain `unsigned int` kernel
    parameters, so one compiled function serves every shape of that rank.
    Each thread divmods its flat output index into coordinates and gathers
    the two inputs through the strides.
  - `reduce_dev` (`Sum`/`Mean`): a deliberately simple single-thread loop
    kernel -- correct, serial, unbenchmarked (no GPU here to tune against).
  - `sum_dim_dev`: one thread per keepdim-layout output element, looping the
    reduced dim via outer/inner decomposition.
  - `fill_dev`: device-side fill kernel (the value is a kernel parameter),
    overriding the host-upload default.
  - `matmul_dev` with `ta`/`tb` transpose-storage flags, mapped natively onto
    cuBLAS transa/transb (see below) so backward passes never materialize
    transposes.

With those in place, device tensors support broadcasting binaries,
reductions (`sum`/`mean`/`sum_dim`), and full autograd: a training loop
(forward, MSE, backward, tensor-op SGD) runs entirely resident on the GPU.

## Host-slice fallback

The host-slice `Backend` methods (`unary`/`binary`/`matmul` over `&[f32]`)
remain as the path core uses for non-resident tensors. They are thin wrappers
over the same device kernels: htod, `*_dev` compute, dtoh.

Remaining gaps:

- `ops_ext` composites without device kernels (softmax, conv2d, ...) still
  fall back to host compute in core and return cpu tensors.

## Usage on a GPU box

```rust
if let Err(e) = ferro_cuda::install(0) {
    eprintln!("CUDA unavailable: {e}");
}
// Tensors moved to Device::Cuda(0) now stay resident across chained ops,
// including backward passes.
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
identity `C^T = op(B)^T * op(A)^T`: since a row-major buffer reinterpreted
column-major is its transpose, an sgemm with the operands swapped and m/n
swapped writes `C^T` column-major, which is byte-identical to `C` row-major.
The `ta`/`tb` flags (operand stored transposed) map onto cuBLAS natively: an
unflagged operand's column-major view is already the transpose the identity
needs (`CUBLAS_OP_N`), while a flagged operand's view is the logical matrix
itself, so cuBLAS transposes it (`CUBLAS_OP_T`); leading dims follow the
storage (`lda = tb ? k : n`, `ldb = ta ? m : k`). All four flag combinations
are unit-tested against ferro-core's reference matmul with a pure-host
simulation of cuBLAS semantics (no GPU needed).

## Testing

`cargo test -p ferro-cuda` is green without a GPU: it covers kernel source
generation for every op kind (elementwise including `Gtz`, broadcast binary,
reduce, sum_dim, fill), the broadcast stride/index math against a host
reference, the sgemm layout mapping for all four `(ta, tb)` combinations on
non-square shapes, a compile-level check that `CudaBackend` implements the
full `Backend` trait (including the `*_dev` methods) and `CudaBuf` the
`DeviceBuffer` trait, and that `install`/`CudaBackend::new` fail gracefully.
The `gpu_end_to_end` test is a no-op without a driver; on a GPU box it
validates the host-slice fallback, direct `*_dev` round trips with
foreign-buffer rejection, a resident tensor chain against the cpu, and the
same 40-step resident linear-regression training loop core proves against
its fake backend (`ferro-core/tests/device.rs::training_loop_stays_resident`),
asserting convergence and parity with the cpu loop.
