# Rust CUDA integration research

Checked 2026-11-03. These are upstream requirements observed on that date,
not a tested ferro compiler integration. Pin an upstream revision and recheck
its requirements before implementing either authoring track.

Rechecked the cudarc latest-release API, cuda-oxide installation guide and
community compiler pin on 2026-11-04: the versions and baseline requirements
below remain current. The cuTile compatibility check remains dated 2026-11-03.

## Current choices

| Project | Role and observed status | Implication for ferro |
|---|---|---|
| cudarc | Host CUDA driver/library wrapper. Latest published release is 0.19.10, including CUDA 13.4 bindings, borrowed-stream support and pinned-allocation cleanup fixes. | Keep the existing runtime. Current workspace and Python lockfiles resolve the `0.19.8` requirement to 0.19.10; selected bindings remain CUDA 12.8. A newer wrapper does not require selecting a newer CUDA API. |
| NVIDIA cuda-oxide | Rust SIMT kernels compiled ahead of time to PTX. Current docs pin stable Rust 1.99.1 and require Linux, Ampere/sm_80+, CUDA 13+, driver 580+, and LLVM/Clang 21+; Rust's bundled LLVM tools may supply the PTX compiler. Native Windows is unsupported. | Optional Linux experiment, separate from the normal Cargo workspace. Existing Windows/RTX 3090 results do not validate this toolchain. |
| NVIDIA cutile-rs | Rust tile programs, JIT compiled through Tile IR to cubin; stable Rust 1.89+. The current compatibility matrix lists baseline Tile IR 13.2 and sm_80+, including sm_86. Some operations and targets require newer versions. | Keep the tile track alongside SIMT. Check assembler, target, driver and operation requirements together; do not infer compatibility from the GPU name alone. |
| Community Rust-GPU/rust-cuda | Separate experimental project: `rustc_codegen_nvvm`, `cuda_std`, and host wrapper `cust`. Current main pins nightly-2026-04-02. Setup documents CUDA 12+ and LLVM 7, with an optional LLVM 21 path. | A third compiler ecosystem, not a rename of NVIDIA's project or a drop-in cudarc update. Pin compiler and repository revision together if evaluating it. |

Sources: [cudarc 0.19.10 release](https://github.com/chelsea0x3b/cudarc/releases/tag/v0.19.10),
[NVIDIA overview](https://nvidia.github.io/cuda-rust/),
[cuda-oxide installation](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/cuda-oxide-book/getting-started/installation.md),
[cuTile compatibility](https://nvidia.github.io/cuda-rust/cutile/latest/reference/compatibility.html),
[community setup](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html),
[community compiler pin](https://github.com/Rust-GPU/rust-cuda/blob/main/rust-toolchain.toml).

NVIDIA labels both tracks early research with incomplete features and possible
API breakage. Current installation docs supersede launch-era toolchain advice.
Driver 580 is a baseline, not a promise that it can JIT every PTX version emitted
by later toolkits; the cuda-oxide installation guide explicitly documents this
distinction.

The current cuda-oxide guide also documents an optional CuTe translation path
using NVIDIA's CUTLASS 4.7 compiler library. Its managed installer verifies the
archive and compiler-library hashes. That path has its own target restrictions;
it does not make all existing SIMT kernels portable to every CUDA architecture.

The Python bindings have a separate lockfile. It remained on cudarc 0.19.9
through candidate18, while the main workspace used 0.19.10. Both locks now
resolve to 0.19.10, verified by the matching candidate23 wheel and native checks.
No compiler-authoring dependency is added.

## Next experiment

Keep [both roadmap tracks](FOUNDATIONS_AND_ARCHITECTURE_ROADMAP.md#rust-authored-cuda-kernels-support-both-authoring-tracks)
optional. Start each with a separately pinned, out-of-workspace F32 elementwise
kernel and its adjoint. Load into the existing backend's context and stream,
retain allocation/module owners, and reject foreign owners. Verify empty and
non-block-multiple lengths, forward/backward parity, transfer counts and graph
capture before attempting a fused or sparse operator. Compare cold compilation
and warmed execution separately against the current NVRTC path. Keep the normal
CPU build free of experimental compiler requirements.

The review environment exposes no NVIDIA device or driver. It can verify CPU
semantics, CUDA compilation and required-GPU failure behaviour, but cannot
establish kernel correctness, residency, capture compatibility or performance.
No Rust-authored CUDA kernel is enabled by this work.

For a GPU verification machine, use strict execution so missing prerequisites
fail the run:

```sh
FERRO_REQUIRE_CUDA=1 cargo test -p ferro-cuda
```

Without that variable, GPU-gated test bodies may return early. Cargo's harness
can display those returns as passes; they are not evidence of GPU execution.
