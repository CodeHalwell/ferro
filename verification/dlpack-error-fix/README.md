# DLPack recorded-error fix

## Outcome and corrected diagnosis

`crates/ferro-py/src/lib.rs` now explicitly checks the allocation-owned context after DevicePtr dependency acquisition, saves the dependency/fence result, drops the usage guard, and unconditionally drains the final recorded error before returning any error or entering `dlpack::export_for`. The first error wins if both stages fail. Capsule ownership, event tracking, and backend lookup behavior are unchanged. Test hooks and the export-entry counter exist only under `cfg(test)`; there is no production debug API.

Important correction to the review premise: the resolved dependency is cudarc 0.19.9. In `.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.9/src/driver/safe/core.rs`, DevicePtr records wait errors at lines 1171-1179; stream synchronize at 745-747 calls bind_to_thread, which ALREADY calls check_err at 350-351. Therefore acquisition-only recorded-error injection already rejects export on the original fence. It would be incorrect to claim that this dependency-only case reproduced a swallowed error. The demonstrated gap is the final recorded-error drain after guard release, including cleanup after an earlier error. SyncOnDrop records errors at 1132-1145; check_err drains at 502-509.

## Deterministic RED/GREEN

Test file: `crates/ferro-py/tests/support/dlpack_recorded_error.rs`.

The binding's actual `__dlpack__` method runs against a real CUDA allocation. Test-only hooks inject `CUDA_ERROR_INVALID_VALUE` through the actual allocation context's `record_err`, after dependency acquisition and/or after usage-guard release. Tests assert ValueError with the CUDA error, zero calls into the capsule allocator, a drained context error channel, then a successful clean capsule and unchanged tensor values. Each test runs in its own subprocess so it cannot replace the backend registry used by unrelated binding tests.

This is recorded-error injection, NOT reproduction of a native CUDA wait/event-record failure. The release hook models the error-channel state immediately after Drop; it does not force CUDA's event API to fail.

Exact focused command, with runner-established environment:

```
python verification/dlpack-error-fix/run.py red
python verification/dlpack-error-fix/run.py green
# underlying command for both:
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib dlpack_error_tests -- --nocapture
```

RED ran before implementing the checks, then was repeated after subprocess isolation with the checks temporarily removed. The RED implementation retained test hooks and explicit guard drop but used the original `producer.synchronize()?` early return and no final check. Final source restores the fix.

- `red.log`: exit 101; 1 passed, 2 failed, 9 filtered out.
  - acquisition-only passes, as predicted by bind_to_thread's existing drain.
  - release-only fails: `recorded CUDA error must reject DLPack export before capsule allocation`.
  - acquisition plus release fails because the original early return never reaches explicit release instrumentation (`left: 2`, `right: 0`).
- `green.log`: exit 0; all 3 passed, 9 filtered out; default parallel test harness. The combined case also verifies final draining despite an earlier error.

## Final rebuilt integration

Commands (all use the exact environment and underlying commands recorded in run.py/log headers):

```
python verification/dlpack-error-fix/run.py build
python verification/dlpack-error-fix/run.py eager
python verification/dlpack-error-fix/run.py snapshot
python verification/dlpack-error-fix/run.py integration
```

- Release bindings rebuilt and installed with maturin develop --release -j2: exit 0.
- Eager DLPack: 5 passed, exit 0.
- Snapshot immediate default/nondefault-stream DLPack: 1 passed, exit 0.
- Actual `verification/model-cuda-graphs/continuation/run_checks.py`: 13/13 commands exit 0. Includes core, fastcpu/tokenizer, CUDA tests, CUDA all-targets check, Python capture discovery, compiled/fuse/general/ops/safetensors/fuzz, diff check, and capture-layout GPU tests. Scoped copies are `integration-*.log`; `integration-results.json` preserves original runner rows.
- CUDA runtime prefix explicitly includes `C:/Users/DanielHalwell/AppData/Local/Temp/cuda-rt/nvidia/cuda_nvrtc/bin` and `.../cublas/bin`. FERRO_REQUIRE_CUDA=1. Continuation runner sets RUST_TEST_THREADS=1 itself; focused tests use the default parallel harness.

## Additional issue discovered (not hidden by integration success)

An extra, non-required full Rust binding-unit run (`run.py unit`) returned exit 101: 11 passed, 1 failed. Existing `dlpack::tests::gpu_export_borrows_and_import_htod` at src/dlpack.rs:768 imports zeros instead of [0,1,2,3,4,5]. It calls raw `export_for` directly, bypassing the public binding fence. The same test fails alone (`run.py unit-existing`, exit 101), with no new tests running and with the new production checks temporarily removed. It is therefore not caused by this fix or new-test registry interference. Left unchanged because this task owns lib.rs and the focused regression, not dlpack.rs. See `unit.log` and `unit-existing.log`. The requested 13-command runner does not include Rust ferro-py unit tests, so its success does not imply this supplemental suite passed.

The first embedded-Python attempt failed to find encodings; the runner fixes this by querying the venv Python's sys.base_prefix and setting PYTHONHOME only for Rust unit executables. Existing Rust unused/dead-code warnings remain. The file tool's automatic rustfmt invocation also emitted edition-mismatch parsing errors for existing C string literals; real Cargo builds succeeded.

No performance source files modified; no GPU benchmarks, staging, commits, or pushes. Continuation runner refreshed its own log/JSON outputs as designed. The cudarc skill was updated with the verified call-chain and error-drain lesson.
