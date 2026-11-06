# ferro-cuda

For the current Rust kernel-authoring ecosystem and optional compiler plans,
see [Rust CUDA integration research](../../docs/RUST_CUDA.md), checked 2026-11-03.
The capability overview below describes the original backend; the broader
[current ledger](../../docs/CURRENT_STATUS.md) records later additions.

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

- Spatial kernels (`src/spatial.cu`): im2col (`unfold2d_dev`), its
  adjoint (`fold2d_dev`), and max-pool forward/backward. With cuBLAS these
  keep conv2d (groups = 1), max_pool2d and batch_norm resident through
  forward and backward. See [the GPU runbook](../../docs/GPU_RUNBOOK.md) for
  how to check and benchmark them on a card.

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

### Captured chain ownership

`capture_chain` retains the original input allocations and backend, and uses the
same private-stream preparation and cleanup boundary as `StaticPointwiseGraph`.
Dropping caller input handles cannot return those allocations to the pool while
the graph still owns them. Context event tracking stays enabled.

Bind the returned handle as mutable: `let mut captured = backend.capture_chain(...)?`.
`captured.replay()` now takes `&mut self`, so a live output borrow cannot coexist
with a replay that overwrites it. The handle is neither `Send` nor `Sync`.
`output()` panics after a failed replay; `copy_output_to_host` returns an error
for a poisoned graph. Zero-element captured chains are explicitly unsupported.

Input mutation is scoped to the owning backend stream. The older in-place alias
helper still lacks original-allocation write-event bookkeeping for concurrent
foreign-stream readers; the unwind repair does not close that gap. Experimental
whole-training-step capture is separate and retains its existing limitations.
GPU regressions for retained inputs, changed inputs, broadcasts and failed capture
cleanup must execute on real hardware before claiming runtime validation.

### Capturable AdamW

The 2026-11-03 optimizer update gives eager AdamW a counter per parameter.
`AdamW::capturable()` enables its shared device counter only when all parameters
are on one CUDA device and their counters are equal. Unequal counters leave the
optimizer in eager mode. Restore into an existing capturable optimizer rejects
unequal counters before committing state. Calling `capturable()` again preserves
the current device counter and its allocation.

Each host invocation of a capturable step requires gradients for every parameter
or none. With none, weights, moments and the device counter remain unchanged.
A partial set, including a mix of frozen and active parameters, is rejected
before advancing the counter; the current infallible `step()` API panics on that
update error.

These checks run when the host invokes the optimizer. Changing freeze flags or
gradient presence does not change an already-captured CUDA graph; dynamic
freezing during graph replay is unsupported. Whole-training-step capture retains
the separate safety limitations described above.

The new contract has CPU/mock regression and compile coverage. Its GPU regression
is gated and has not executed in this environment; passing host tests does not
prove CUDA execution.

On a CPU-only machine, run:

```sh
cargo check -p ferro-cuda --all-targets
cargo test -p ferro-cuda -- --skip required_cuda_
```

The filter explicitly excludes the five mandatory GPU residency tests in
`strided_seed_residency.rs`. Other GPU-gated bodies return early when CUDA is
unavailable, so a successful harness result is not GPU verification. For full
GPU validation use `FERRO_REQUIRE_CUDA=1 cargo test -p ferro-cuda`; missing CUDA
or required capture support then fails instead of silently returning.

The host-executable tests cover kernel source
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
