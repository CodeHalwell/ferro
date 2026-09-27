# Reconciled stabilisation publication

This draft layers bounded stabilisation and companion segment/BMM work onto main
`c576e588a8609007c88d21304ce7659a00db0553`. PR26 CPU training/restart and PR28
operators are retained. It does not certify every model, dtype, derivative order,
or GPU architecture. AI assisted implementation, reconciliation and verification.

## Review order and reconciliation

Review core materialization capability/error semantics with CUDA allocation-owner
reads first, then prepared segments and their native/Python tests, independent
unpacked CPU BMM arithmetic, typed Python conversion and optimized-mode restart
gates, and finally historical diagnostic examples and roadmap documents.

The original 54-path allowlist was compared to main and the source HEAD
`dc13a7199c23d92be5dd766102af03a6bfcb4bc3`. Eighteen overlaps were identical.
Eight overlaps differed: bindings and foundations roadmap merged cleanly;
losses.rs, test_architecture_restart.py and the two training entrypoints differed
only in final blank lines and retain main's versions. training_restart.py retains
main's proof while replacing removable assertions with explicit rejection gates.
CURRENT_STATUS retains historical training evidence while superseding the old
companion-out-of-scope label and adding local stabilisation history. No upstream
operator file was changed. These are disjoint additions or superseded scope/
whitespace, not competing operator semantics.

The original working tree, index and archives were not modified. Its 454-input
freeze was rechecked with no mismatches and its index remained empty. Only the
explicit source/test/document allowlist was transplanted; raw archives, wheels,
logs, crash data and environments are not publication content.

## Fresh merged-tree verification

A separate worktree and .venv were used. The release wheel was rebuilt and installed
noneditable; source facade, wheel and installed Python hashes agree, and the wheel
native extension matches the installed binary. Imports resolve inside this
worktree's .venv, and unconditional CUDA initialization succeeded. The 470-input
source freeze and installed-package freeze were unchanged across the full run.
These results supersede original dirty-tree counts for this draft.

| Gate | Actual result |
| --- | --- |
| Python unittest discovery | 114 passed, no skips |
| Core | 811 passed, 2 ignored |
| CUDA | 126 passed, no ignores |
| Fastcpu | 27 passed, 2 ignored |
| Tokenizer | 8 harness passes; optional real-vocab body not exercised |
| Rust Python bindings | 15 passed, no ignores |
| Typed tolist, normal and -O | 5 passed each |
| CUDA registry replacement, normal and -O | 1 passed each |
| Restart gates, normal and -O | 3 passed each |
| CUDA cargo check | exit 0 |

All commands below ran from the reconciled worktree. PY is
`.venv/Scripts/python.exe`; OUT is `verification/reconciled-publication`.
The collector saves expanded argv, cwd, selected environment and raw output in
local receipts, along with sources-before/after.json, installed-before/after.json,
packaging.json, stability.json and summary.json. Those local execution artifacts
and wheels are not shipped with this document. The collector itself is included.

```text
python -m venv C:/Users/DanielHalwell/PythonProjects/ferro-pr-stabilisation/.venv
.venv/Scripts/python.exe -m pip install -r verification/reconciled-publication/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python verification/reconciled-publication/run.py final
python verification/reconciled-publication/run.py command cuda-check cargo check -j2 -p ferro-cuda
```

The initial dependency installation against the default index failed because it
did not supply torch==2.6.0+cu124. Retrying with the public PyTorch cu124 index
installed the original environment's pinned versions. No tests or tolerances were
changed to address setup. The original environment was not replaced.

The collector executes these commands (PY/OUT expanded in local receipts):

```text
nvidia-smi --query-gpu=name,driver_version --format=csv
PY -m maturin build --release -j2 --manifest-path crates/ferro-py/Cargo.toml --out OUT/wheels -i PY
PY -m pip install --force-reinstall --no-deps OUT/wheels/ferro-0.0.1-cp311-abi3-win_amd64.whl
PY -B crates/ferro-py/tests/test_tolist_typed.py -v
PY -B -O crates/ferro-py/tests/test_tolist_typed.py -v
PY -B crates/ferro-py/tests/test_cuda_registry_copy.py -v
PY -B -O crates/ferro-py/tests/test_cuda_registry_copy.py -v
PY -B crates/ferro-py/tests/test_training_restart_gates.py -v
PY -B -O crates/ferro-py/tests/test_training_restart_gates.py -v
PY -B -m unittest discover -s crates/ferro-py/tests -v
cargo test -j2 -p ferro-core
cargo test -j2 -p ferro-cuda
cargo test -j2 -p ferro-fastcpu
cargo test -j2 -p ferro-tokenizer
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib
cargo check -j2 -p ferro-cuda
```

CUDA runtime PATH prepends `%LOCALAPPDATA%/Temp/cuda-rt/nvidia/cuda_nvrtc/bin`
and `%LOCALAPPDATA%/Temp/cuda-rt/nvidia/cublas/bin`, then the isolated Python
Scripts directory. CUDA=1, FERRO_REQUIRE_CUDA=1 and PYO3_PYTHON point at the
isolated interpreter; CARGO_TARGET_DIR is this worktree's target-stabilisation-final.
PYTHONPATH, PYTHONHOME, PYTHONOPTIMIZE and RUST_TEST_THREADS are cleared, except
binding tests require PYTHONHOME set to the interpreter's base_prefix. The normal
parallel Rust harness remains enabled. Explicit -O tests retain their rejection
gates. Neither benchmarks nor ignored performance tests were executed.
FERRO_GPT2_DIR was unset: seven hermetic tokenizer bodies executed, while the
real-vocab conditional returned early. Hosted CI is not certified here.

## Boundaries and unresolved incidents

Historical numerical mismatch and CPU/CUDA native-crash investigations remain
unresolved. Passing these suites does not establish their causes or repair them.
CPU BMM uses independent unpacked tiled arithmetic, not the historical packed
path; explicit nonbatched packed APIs remain separate. No new speed claim is made.

Error::MaterializeUnavailable is a new variant of the public exhaustive Error
enum: downstream exhaustive matches need an arm. Existing public method signatures
are unchanged. Only explicit unsupported capability/layout permits a detached
host fallback; operational errors still panic through the infallible detach API.
Valid CUDA rank greater than 16 may fall back; cross-owner compute/mutation/D2D
remain rejected. This is not universal strided or higher-order GPU residency.

Late CUDA copy/guard-completion error branches remain source-audited rather than
freshly fault-injected. Recorded pending-error tests do not close that coverage
gap. Prepared CUDA sum/softmax are first-order f32; softmax performs a synchronized
four-byte status read. Mean/max remain CPU-only and capture is unsupported.

Earlier verification links and counts in roadmap/audit documents are historical
local provenance, not downloadable evidence or current merged-tree certification.
The final review checked explicit new-file inclusion/dependency closure, staged
whitespace and added-line secret/shell/eval/pickle scans. No independent new reviewer
was available in this one-shot subagent; this draft still requires human review.
