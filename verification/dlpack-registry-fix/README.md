# General DLPack export and unit registry lifetime follow-up

Historical docs/STATIC_GRAPH_SAFETY_AUDIT.md and earlier verification logs are
unchanged. This directory records subsequent production fixes and runtime tests.
No commits, staging, benchmarks, or edits to static_graph.rs/model.rs were made.

## Contracts

Python __dlpack__ validates stream tokens before publishing a capsule. CUDA
None/1 (legacy), 2 (per-thread default), and positive pointer-sized stream tokens
all conservatively fence the allocation-owned CudaSlice stream. A DevicePtr read
guard first joins tracked foreign writes. No arbitrary consumer handle is cast
or dereferenced, and neither LAST_BACKEND nor the registry selects the fence.
The optional -1 no-sync extension is explicitly unsupported and rejected, as are
0, other negatives, booleans, non-integers and oversized integers. CPU requires
None and keeps its existing owned-copy capsule path (it was not zero-copy).
CUDA retains its existing zero-copy allocation lifetime/capsule consumption path.
The Array API/DLPack stream documentation fetched via curl is saved here.

The same poison-tolerant REGISTRY_TEST_LOCK is acquired as the first local in
all three audited registry-backed CUDA tests and install_never_panics_without_gpu.
Reverse local destruction keeps the guard through tensor/backend destruction.
The three affected GPU tests now fail on initialization failure when
FERRO_REQUIRE_CUDA is set. Inspection found no other src unit tests using the
core CUDA registry; static graph context tests use direct backends. Integration
executables retain their separate process-local locks and intentional concurrency.

## Historical RED

- eager-red.log: both immediate default and non-default Torch consumers failed
  numerical equality (2,162,688 / 4,194,304 elements mismatched on each).
  The producer queues eight real 2048-square eager GEMMs then ReLU. This is real
  delayed GPU production, not a synthetic delay kernel or host sleep. There is
  no external producer fence, tolist, or download before the consumer clone.
- tokens-red.log: nine invalid-token subtests failed because no error was raised.
- registry-red.log: default-parallel cargo test -j2 -p ferro-cuda --lib failed,
  39 passed / 2 failed. Softmax and mini-block tests hit the concrete foreign
  backend stream/context rejection during binary_dev and matmul_dev.
- registry-source-red.log: strengthened lifetime source check failed before
  the common guard was added. This is source evidence, not runtime proof.

## Historical GREEN

All run.py commands have a 480-second subprocess timeout, required CUDA and the
specified NVRTC/cuBLAS DLL PATH. RUST_TEST_THREADS is explicitly unset; no Rust
run uses --test-threads=1. Logs include exact commands and exit codes.

- registry-required-green-{1,2,3}.log: 41 passed each, 0 failed/ignored/filtered,
  under the default parallel harness. The three target GPU tests cannot skip.
  install_never_panics_without_gpu intentionally returns on GPU-equipped hosts;
  other historical direct-backend tests still retain their old availability gates.
- cuda-full-green.log: cargo test -j2 -p ferro-cuda succeeds across unit,
  integration and doc-test summaries: 80 harness passes, no failures/ignored.
  Harness totals alone should not be interpreted as instrumentation proving that
  every historical availability-gated body executed.
- rebuilt-python-green-{1,2}.log: final rebuilt bindings pass all 5 tests,
  no skips. Each eager consumer performs 12 exact numerical checks plus retained
  export checks after source destruction. Additional tests cover CUDA pointer
  identity, backend reinstall, consumed-capsule rejection, CPU capsule lifetime,
  abandoned capsule destruction and stream validation. Ownership tests supplement
  the RED/GREEN readiness regression; they were not themselves initial failures.
- snapshots-green.log: existing snapshot default/non-default and retained-output
  regression passes, without changing the snapshot production path.
- python-general-green.log: examples/py_regression.py reports all checks passed.
- final-build.log: maturin develop --release -j2 rebuilt/installed successfully.
- final-source.log: all 8 audited source checks pass, no expected failures.
  Only the two fulfilled source contracts were promoted; native failure-injection
  gaps remain untouched. git diff --check passed (existing CRLF notices only).

## Current-source runner semantics

`run.py` is a generic `LABEL COMMAND...` wrapper, not a RED/GREEN phase
implementation. The review description of duplicate built-in phases does not
match this file; nevertheless labels never select or restore pre-fix sources.
Use a new `current-...` label for reruns, for example:

```bash
python verification/dlpack-registry-fix/run.py current-registry cargo test -j2 -p ferro-cuda --lib
```

The exclusive-create log mode refuses existing names. Archived RED/GREEN logs
above were collected at their respective source stages; rerunning the same
command on current source cannot recreate or validate historical failures. No
historical log, timing sample, or metadata has been regenerated for this review.

## Limits

This implements a conservative host fence, not asynchronous consumer event
handoff, concurrent external mutation safety, DLPack import ordering, concurrent
install atomicity, or native graph failure recovery. It adds synchronization cost
to ordinary CUDA export. No performance claim is made. Existing compiler warnings
remain; the generic file-edit linter also reports Rust-edition/format noise, while
actual Cargo and maturin builds succeed. Parent may rerun full workspace/Python
integration after combining ownership areas.
