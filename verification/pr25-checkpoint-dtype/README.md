# PR25 checkpoint dtype ownership

Addresses [Codex comment 4009508125](https://github.com/CodeHalwell/ferro/pull/25#discussion_r4009508125)
on reviewed head `f7ed528db68c8dfab9236d40badf0779b277439e`.

## Root cause and scope

The owning copy materialized host/non-whole-device tensors through f32.
The replacement uses the existing same-dtype host gather, raw half-bit
constructors, and typed device downloads/reuploads. Whole f32 device buffers
still use an independent device copy. Placement is preserved in memory;
loading a safetensors checkpoint still produces host tensors.

`try_owned_detach_copy` propagates backend-returned copy/download/upload errors.
The existing infallible API remains an expect wrapper for compatibility:
Checkpoint::clone, with_tensor, from_module and Param construction can still
panic on backend failure. Fallible named-state and optimizer snapshot builders
use the Result path. No public binding API changes are made.

## Reproduction

From the repository root, with Rust and Python installed:

```bash
python verification/pr25-checkpoint-dtype/reproduce.py
python verification/pr25-checkpoint-dtype/verify.py
```

`reproduce.py` extracts the exact reviewed core into a temporary workspace,
adds the current regression test, and asserts five expected failures: F64,
I64, F16, BF16 ownership and device-copy error propagation. It never swaps
production files in the shared worktree. Requires the reviewed Git object.
`verify.py` runs the commands below, checks source stability, and creates
`independent-results.json`. Both scripts generate local logs; raw logs and
historical summarizers are deliberately not committed. No omitted local
scripts or logs are prerequisites.

Literal independently executed verification commands (Git Bash on Windows):

```bash
cargo test -p ferro-core -j2 --test checkpoint_dtype
cargo test -p ferro-core -j2
cargo check -p ferro-cuda -j2
PATH="$PWD/crates/ferro-py/.venv/Lib/site-packages/torch/lib:$PATH" FERRO_REQUIRE_CUDA=1 cargo test -p ferro-cuda -j2 --test checkpoint_dtype
git diff --check
```

The verifier adds the existing project Torch DLL directory to PATH when
present. Otherwise supply CUDA driver, NVRTC and cuBLAS runtime libraries on
PATH. Required CUDA must initialize or the test fails, never silently skips.
No environment installation or upgrade is performed.

Independent results: CPU dtype 8 passed; full core 759 passed, 2 ignored;
CUDA check passed; required CUDA dtype 1 passed, not skipped; diff check clean.
Core counts exclude nested subprocess summaries. Existing compiler warnings
remain; no unrelated files were formatted or repaired. The committed manifest
records exact commands, return codes and hashes of the four source files.

## Review findings and limits

No blocking defects found in the scoped fix. The CPU matrix checks every
supported dtype, large signed i64 values, F64 precision/range and NaN payload,
raw half signaling-NaN/subnormal/negative-zero bits, contiguous/transposed/empty
values, constructor/direct clone/named state/module buffers, file round-trip
and loaded cloning. It checks storage identity and f32 mutation isolation.
CUDA tests check f32 and i64 whole/transposed copies, placement, save/load and
f32 mutation isolation. CUDA f32 samples are integral; payload-bit coverage is
on CPU, not a claim of exhaustive GPU payload tests. Device i64 allocation
independence is also established by the fresh clone_htod allocation path;
its storage-cell identity assertion alone is not a device-pointer proof.

Reviewed async lifetime seams: f32 copy_dev allocates a distinct destination
and uses stream-ordered memcpy_dtod with cudarc device event tracking.
Typed host staging reuses the backend's cudarc 0.19.9 clone_dtoh/clone_htod
paths and ordinary pageable Vec/slice memory. Inspection found those HostSlice
guards return Sync(None), not an explicit stream fence: this fix relies on
the existing safe transfer API and pageable-memory behavior, and does not
claim new explicit fences or redesign asynchronous error reporting. Injected
backend-returned errors cover whole f32 copies, strided f32 downloads, and
whole/strided i64 downloads; to_device upload errors propagate unchanged.

Whole-contiguous CPU f32 remains the transactional restore boundary. No
mixed-dtype compute, typed restore, device model transaction, or F64/F16/BF16
GPU transfer support is added. Existing restore regressions remain green.
The fastcpu 343812-ULP incident remains OPEN, untouched and not retested.
No merge, thread resolution, ready-state change or automatic merge is authorized.

Implementation and independent verification were AI-assisted.
