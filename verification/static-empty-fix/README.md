# Empty static feature-path closure

## Changes

- `crates/ferro-core/src/tensor.rs`: derive row-softmax row count from leading dimensions instead of dividing by the reduced width. Zero-width eager softmax/log-softmax now reaches the existing backend zero-work path; softmax capture keeps its existing semantic metadata.
- `crates/ferro-py/src/dlpack.rs`: allow a null CUDA allocation address only when logical numel is zero. Normal device-view validation, allocation ownership, capsule cleanup, producer stream fence and error guards remain unchanged. No fake pointers or dummy allocations.
- `crates/ferro-core/tests/empty_softmax.rs`: CPU and partial-backend fallback regression over empty shapes and all axes, including invalid-axis rejection.

## Verification

The existing Python acceptance assertions were not changed. Their SHA256 remains `5c4336a4a4c3d2967fef99d1ea1e1d6c75c7823e54eaae643e4866c26f4beff0`.

- Targeted independent Python RED: exit 1, reproduced divide-by-zero and rejected null empty CUDA export (`empty-fix-red.log`).
- Release bindings rebuilt with maturin `develop --release -j2`: exit 0.
- Full independent static coverage: 10 tests passed, including actual Torch consumption of empty CUDA snapshots, arbitrary softmax axes, expanding pointwise seeds, and zero-inner matmul. Final label `empty-fix-final`, exit 0.
- Core empty CPU/partial-backend regression: passed.
- `python verification/dlpack-raw-review/run.py all`: all five phases exit 0. Binding Rust suite: 12 passed under default parallel harness; eager Python: 5 passed; snapshot Python: 1 passed.
- Combined integration: all 13 commands exit 0, including full core/CUDA/fastcpu/tokenizer suites, CUDA check, 26 Python discovery tests, parity/fuzz suites, capture-layout GPU tests, and git diff --check.
- CUDA required with runtime DLL prefix; probe reports Torch and Ferro CUDA available on NVIDIA GeForce RTX 3090.

`results.json` contains exact commands and exits. `sources/`, `sources-sha256.json`, and `combined-wave.patch` preserve the combined source state used for verification. All logs are unfiltered. Existing compiler unused-variable/dead-code warnings and explicitly ignored core tests remain; no test failures. No benchmarks, staging, commits or pushes.
