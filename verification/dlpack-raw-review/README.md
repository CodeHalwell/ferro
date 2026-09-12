# Raw DLPack binding-unit review

## Outcome

Resolved `dlpack::tests::gpu_export_borrows_and_import_htod` without changing production export/import semantics or weakening any assertion. The test called the raw capsule constructor, bypassing the public `__dlpack__` producer fence, then read a pointer produced on a non-blocking stream with synchronous `cuMemcpyDtoH`. Synchronous download completion is not a fence on an unrelated non-blocking producer stream. This was an invalid readiness assumption in the raw-builder test, independently reproduced using only cudarc.

The test now fences its allocation-owned upload stream before calling raw `export_for`. Header, imported device/value, and original-value-after-capsule-deletion assertions remain unchanged. Raw helpers explicitly document caller-owned readiness. The test also fails rather than silently returning on GPU initialization failure when `FERRO_REQUIRE_CUDA=1`.

No modifications to `src/lib.rs`, the public producer fence, or `tests/support/dlpack_recorded_error.rs`. The prior post-guard recorded-error fix remains valid. Acquisition-only errors were already drained by `synchronize -> bind_to_thread -> check_err`; this review makes no contrary claim.

## Root cause evidence

- Original focused test: `red.log`, exit 101, 0 passed / 1 failed. Imported values were `[0,0,0,0,0,0]` versus expected `[0,1,2,3,4,5]`.
- Independent executable: `probe/` depends only on pinned cudarc 0.19.9. No Ferro, Python, capsule, or importer code is linked. It creates a non-blocking stream, uploads the same six values, reads with raw synchronous DtoH, fences the producer stream, and repeats the same download. `cuStreamGetFlags` verifies the stream flag. Actual `probe.log` output:

```
nonblocking_flags=1; unfenced=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; fenced=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
```

The unfenced result is diagnostic, not a guaranteed value: the probe asserts only the fenced result. No host sleep, disabled tracking, mocked data, or changed importer is involved.

- Resolved dependency source excerpts are saved in `cudarc-evidence.txt`: `new_stream` creates `StreamKind::NonBlocking`; `memcpy_htod` uses `memcpy_htod_async`; ordinary host slices return `SyncOnDrop::Sync(None)`; raw `memcpy_dtoh_sync` calls `cuMemcpyDtoH_v2` without the producer stream. DevicePtr guards record/wait events, not host completion.
- Ferro `CudaBackend::new` uses `ctx.new_stream`; `htod` enqueues `memcpy_htod`; `exported_view` returns DevicePtr raw parts without synchronizing. The only production `dlpack::export_for` caller is the public `__dlpack__` method, which already establishes readiness. Thus adding another production fence to the raw builder would duplicate the public fence rather than repair this test's missing precondition.
- Corrected focused test: `green.log`, exit 0, 1 passed. The public error-drain implementation was present throughout this review's RED/GREEN.

## Historical verification (executed at this stage)

Current-source reproduction from repository root (not a historical RED rerun):
`red`/`green` aliases were duplicates and are replaced by `current-test`. Archived
RED failures above require their original pre-fix state; this runner neither
recreates nor validates them. They have not been overwritten. Every invocation
requires an empty output directory; `all` puts each phase in its own subdirectory.
Integration now uses the fresh-output static-perf runner instead of overwriting
continuation evidence. The historical results below remain stage-specific.

```
python verification/dlpack-raw-review/run.py current-test --output-dir verification/dlpack-raw-review/new-focused
python verification/dlpack-raw-review/run.py probe --output-dir verification/dlpack-raw-review/new-probe
python verification/dlpack-raw-review/run.py all --output-dir verification/dlpack-raw-review/new-all
```

`all` runs every phase even if a preceding phase fails, records each real exit code, and exits nonzero if any phase fails. It explicitly includes the standalone Rust binding suite omitted by the original 13-command runner:

```
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib -- --nocapture
```

Final results in `all-results.json`, `all.log`, and phase logs:

- Full Rust binding unit suite: 12 passed, 0 failed, 0 ignored, 0 filtered, default parallel harness. Includes all three recorded-error regression tests.
- Release bindings rebuilt and installed using `maturin develop --release -j2`: exit 0.
- Python eager DLPack: 5 passed, no skips.
- Python snapshot default/non-default stream DLPack: 1 passed, no skips.
- Complete continuation integration: 13/13 commands exited 0, verified programmatically. Its full Python discovery ran 16 tests successfully. Covers core, fastcpu/tokenizer, CUDA, CUDA all-targets check, Python discovery/compiled/fuse/general/ops/safetensors/fuzz, diff check, and capture-layout GPU suite.

`integration-results.json` and `integration-*.log` are scoped copies of the actual continuation runner results; its own output files were refreshed as designed. `source-hashes.json` records the reviewed binding source and preserved regression hashes.

The runner establishes the existing venv and nvrtc/cublas DLL PATH, sets `FERRO_REQUIRE_CUDA=1`, unsets `RUST_TEST_THREADS` for binding units, and sets embedded `PYTHONHOME` from the venv interpreter's actual `sys.base_prefix` only for Rust/Python-embedding tests. The continuation runner itself sets `RUST_TEST_THREADS=1` as before.

## Scope and issues

Changed only `crates/ferro-py/src/dlpack.rs` (test setup and raw-helper documentation) plus this evidence directory. Existing Rust unused/dead-code warnings remain. The edit tool's standalone lint used the wrong edition for existing C string literals; actual Cargo compilation, unit tests, and release rebuild all succeeded. No performance files modified, benchmarks run, staging, commits, or pushes. The GPU skill was updated with the raw-builder/non-blocking-stream diagnostic and complete binding-suite requirement.
