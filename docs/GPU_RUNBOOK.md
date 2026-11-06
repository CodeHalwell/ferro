# GPU runbook: checking and benchmarking on the RTX 3090

This is the checklist for running the GPU work on a real card. Everything
here was written and checked on a machine with no GPU, so none of it is GPU
evidence until this runbook has been run. What was checked without a GPU is
listed at the end.

## Setup

1. NVIDIA driver plus the CUDA 12.x runtime libraries `libnvrtc` and
   `libcublas` on the loader path. Nothing is needed at build time: cudarc
   dlopens them. If your toolkit is not 12.8, set `CUDARC_CUDA_VERSION`
   (for example `12060`) before building.
2. Rust (stable).
3. For benchmarks only: a Python with a CUDA build of torch, for example

   ```
   python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

## One command

```
PYTHON=.venv/bin/python scripts/gpu_check.sh             # correctness, then benchmarks
PYTHON=.venv/bin/python scripts/gpu_check.sh --no-bench  # correctness only
```

The script sets `FERRO_REQUIRE_CUDA=1` and `FERRO_REQUIRE_NVRTC=1`, so a
missing driver or library fails the run instead of silently skipping GPU test
bodies. It writes `benchmarks/results/gpu_<date>.{log,md,json}`; please send
those three files back.

What each step proves:

| Step | Command | Proves |
|---|---|---|
| NVRTC | `cargo test -p ferro-cuda --lib every_kernel_source_compiles_for_ampere` | Every kernel source the backend generates compiles for `compute_86` |
| CUDA suite | `cargo test -p ferro-cuda` | Existing GPU tests plus `tests/spatial_resident.rs` (below) |
| Host suites | `cargo test -p ferro-core`, `cargo test --release -p ferro-fastcpu` | Same-commit CPU baseline |
| Emulation | `python3 verification/gpu-staging/emulate_spatial.py` | Kernel indexing logic on the host (also runs without a GPU) |
| Benchmarks | `python benchmarks/compare_gpu.py --python $PYTHON` | ferro vs torch medians on cuda:0 |

## New GPU paths in this change

Before this change conv2d, max_pool2d and batch_norm on a CUDA tensor fell
back to the host, so a CNN training step crossed PCIe on every layer. They
now stay on the device:

- **conv2d** (groups = 1, any stride, padding and dilation): an im2col kernel
  (`unfold2d_dev`), one cuBLAS GEMM over every image of the batch, and a
  layout copy to NCHW. Backward recomputes the columns, runs two GEMMs for dW
  and dcols, and folds with a gather kernel (`fold2d_dev`, one thread per
  input pixel, no atomics, so it is deterministic).
- **max_pool2d**: forward kernel writes the output and the in-plane argmax;
  backward gathers over the windows covering each pixel (no atomics). The
  NaN and tie rules match the CPU exactly, so forward values and gradients
  are expected to be bitwise equal to the CPU.
- **batch_norm** (train and eval): composed from existing resident kernels
  (two `sum_dim` passes per channel statistic, broadcast arithmetic, `pow`),
  so autograd follows from those ops. Running statistics stay on the device.

The kernels are in `crates/ferro-cuda/src/spatial.cu`; the seams are
`Backend::{unfold2d_dev, fold2d_dev, max_pool2d_dev, max_pool2d_backward_dev}`
in `ferro-core/src/dispatch.rs`. Backends without them keep the old host
fallback.

`crates/ferro-cuda/tests/spatial_resident.rs` checks, on the GPU:

- conv2d values and both gradients against the CPU for three geometries,
  with zero device-to-host downloads during forward and backward;
- max_pool2d forward values and gradients bitwise equal to the CPU;
- batch_norm outputs, running statistics and all three gradients against the
  CPU, in train and eval mode;
- a 30-step conv/bn/pool CNN training loop whose only download per step is
  the scalar loss read, and whose loss trajectory tracks the CPU run.

## Reading the benchmark table

`ferro / torch speed` above 1 means ferro is faster. Expectations, so you can
spot something wrong:

- matmul: parity at 4096 (both are cuBLAS), ferro slightly ahead at 1024
  from lower dispatch overhead, as in `docs/GPU_BASELINE_3090.md`.
- conv2d: torch uses cuDNN implicit-GEMM or Winograd kernels; ferro uses
  explicit im2col plus cuBLAS. Expect ferro behind, perhaps 0.3-0.7x. The
  column matrix for the forward case is about 460 MB, so memory traffic, not
  FLOPs, sets ferro's speed.
- max_pool2d: both are memory-bound; expect within 2x.
- batch_norm: torch runs one fused kernel per pass; ferro runs about ten
  generic kernels, so expect it well behind. A fused BN kernel is the obvious
  next step if this shows up in the CNN step.
- cnn_train_step: the headline. Before this change it was dominated by host
  round trips.

The torch side disables TF32 by default because ferro has no TF32 path, so
both sides run true fp32. `--tf32` turns it on to show the gap tensor cores
would close.

## Known limits

- conv2d with groups > 1 still falls back to the host.
- The im2col buffer covers the whole batch: c*kh*kw x n*oh*ow floats. Very
  large batches of large images need a lot of memory (chunking over images
  is the follow-up if this bites).
- avg_pool2d, dropout masks and the other spatial ops still use the host.
- No fp16/bf16 or TF32 compute yet; that is the largest remaining GPU lever.

## Checked here without a GPU

- `cargo check -p ferro-cuda --all-targets` (compiles with no CUDA installed).
- All 50 or so kernel sources compiled to PTX for `compute_86`, `compute_80`
  and `compute_75` with NVRTC 12.9 from the `nvidia-cuda-nvrtc-cu12` wheel:
  `LD_LIBRARY_PATH=<wheel>/nvidia/cuda_nvrtc/lib FERRO_REQUIRE_NVRTC=1 cargo test -p ferro-cuda --lib nvrtc`.
- `spatial.cu` compiled as host C++ and checked against numpy references by
  `verification/gpu-staging/emulate_spatial.py`; a mutated fold or argmax
  rule makes it fail.
- The core routing runs against a mock device in
  `crates/ferro-core/tests/device_spatial.rs`: values and gradients match the
  CPU and a whole conv/bn/pool/linear/cross-entropy step performs zero
  downloads.
- The CNN training body in `spatial_resident.rs` also runs on the CPU
  (`cnn_training_reference_learns_on_cpu`), so the test itself is known to
  train.
