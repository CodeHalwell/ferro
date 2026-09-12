# Empty static feature-path closure

## Changes

- `crates/ferro-core/src/tensor.rs`: derive row-softmax row count from leading dimensions instead of dividing by the reduced width. Zero-width eager softmax/log-softmax now reaches the existing backend zero-work path; softmax capture keeps its existing semantic metadata.
- `crates/ferro-py/src/dlpack.rs`: allow a null CUDA allocation address only when logical numel is zero. Normal device-view validation, allocation ownership, capsule cleanup, producer stream fence and error guards remain unchanged. No fake pointers or dummy allocations.
- `crates/ferro-core/tests/empty_softmax.rs`: CPU and partial-backend fallback regression over empty shapes and all axes, including invalid-axis rejection.

## Historical stage verification

The existing Python acceptance assertions were not changed. Their SHA256 remains `5c4336a4a4c3d2967fef99d1ea1e1d6c75c7823e54eaae643e4866c26f4beff0`.

- Targeted independent Python RED: exit 1, reproduced divide-by-zero and rejected null empty CUDA export (`empty-fix-red.log`).
- Release bindings rebuilt with maturin `develop --release -j2`: exit 0.
- Full independent static coverage: 10 tests passed, including actual Torch consumption of empty CUDA snapshots, arbitrary softmax axes, expanding pointwise seeds, and zero-inner matmul. Final label `empty-fix-final`, exit 0.
- Core empty CPU/partial-backend regression: passed.
- `python verification/dlpack-raw-review/run.py all`: all five phases exit 0. Binding Rust suite: 12 passed under default parallel harness; eager Python: 5 passed; snapshot Python: 1 passed.
- Combined integration: all 13 commands exit 0, including full core/CUDA/fastcpu/tokenizer suites, CUDA check, 26 Python discovery tests, parity/fuzz suites, capture-layout GPU tests, and git diff --check.
- CUDA required with runtime DLL prefix; probe reports Torch and Ferro CUDA available on NVIDIA GeForce RTX 3090.

`results.json` records historical commands/exits. The old `sources/`,
`sources-sha256.json`, and `combined-wave.patch` are incomplete historical
provenance: the old collector omitted untracked production files outside its
test-directory allowlist. They cannot establish the exact combined tree or
final PR24 source identity. These archives have not been rewritten.

## Current-source collection (not a test rerun)

```bash
python verification/static-empty-fix/collect.py --output-dir verification/static-empty-fix/new-source-snapshot
# Optional read-only evidence copy; no assertion that it ran on current sources:
python verification/static-empty-fix/collect.py --output-dir verification/static-empty-fix/new-with-evidence --evidence-dir verification/static-empty-fix
```

The collector no longer runs tests or builds. It snapshots all existing tracked
and non-ignored untracked files under `crates`, including production adapters,
and includes staged/unstaged changes plus untracked additions in the patch.
Deleted tracked paths are listed in `collection.json`. Paths are resolved from
the script location, so invocation is independent of working directory. Supply
an unused output directory; existing archives are refused. Optional log/JSON
copies are placed in `copied-evidence/` and explicitly remain unattested against
the current source snapshot. Run required-GPU verification separately with fresh
output paths; collection is not GREEN evidence.
