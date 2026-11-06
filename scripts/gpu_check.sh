#!/usr/bin/env bash
# One-shot GPU acceptance and benchmark run for a CUDA box (written for an
# RTX 3090, sm_86). Run from anywhere inside the repo:
#
#   scripts/gpu_check.sh            # correctness, then benchmarks
#   scripts/gpu_check.sh --no-bench # correctness only
#
# Needs: an NVIDIA driver, the CUDA runtime libraries (libnvrtc, libcublas)
# on the loader path, and Rust. The benchmark step also needs a Python with
# a CUDA build of torch; set PYTHON=/path/to/python (default: python3).
# Results land in benchmarks/results/gpu_<date>.{json,md} and a log in
# benchmarks/results/gpu_<date>.log.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python3}"
STAMP="$(date -u +%Y%m%d_%H%M)"
OUT="benchmarks/results/gpu_${STAMP}"
mkdir -p benchmarks/results
exec > >(tee "${OUT}.log") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }

step "Machine"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
git log -1 --oneline

# Every gated GPU body fails loudly instead of skipping.
export FERRO_REQUIRE_CUDA=1 FERRO_REQUIRE_NVRTC=1

step "NVRTC: every kernel source compiles for sm_86"
cargo test -p ferro-cuda --lib every_kernel_source_compiles_for_ampere

step "ferro-cuda: full suite on the GPU (includes the new spatial tests)"
cargo test -p ferro-cuda

step "ferro-core and ferro-fastcpu (host suites, for a same-commit baseline)"
cargo test -p ferro-core
cargo test --release -p ferro-fastcpu

step "Spatial kernel host emulation (no GPU needed; cross-checks indexing)"
"$PYTHON" verification/gpu-staging/emulate_spatial.py

if [[ "${1:-}" == "--no-bench" ]]; then
  step "Done (benchmarks skipped)"
  exit 0
fi

step "Benchmarks: ferro vs torch on cuda:0 (TF32 off)"
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'torch has no CUDA'; print(torch.__version__, torch.version.cuda)"
"$PYTHON" benchmarks/compare_gpu.py --python "$PYTHON" --json "${OUT}.json" --markdown "${OUT}.md"

step "Done: ${OUT}.md, ${OUT}.json, ${OUT}.log"
