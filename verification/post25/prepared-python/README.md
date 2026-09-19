# Python reusable segment topology

## Outcome

`ferro.graph.PreparedSegments(ids, num_segments, device="cpu")` is now the native
PyO3 class, also exported by `ferro._native`. It owns the existing core
`PreparedSegments`; `sum(x)` and `softmax(x)` dispatch directly to that retained
plan. No feature conversion, topology re-preparation, fallback, core/kernel edit,
or extra registry lookup was added to the execution wrappers. Registration lives
in the existing architecture registration function; lib.rs was not edited here.
IDs extract as i64 (overflow/floating inputs reject), negative IDs reject before
usize conversion, and the existing core validates bounds and device capability.
The existing convenience functions already worked on CUDA and were left intact.
Graph documentation now qualifies CPU-only mean/max and sparse operations.

## Verified

- RED: `red.log` observes the missing public PreparedSegments class before edits.
- Release noneditable wheel built and installed twice; the final run is in
  `final-build.log` / `final-install.log`.
- `final-contracts.log`: 10 public Python tests passed, including six CUDA tests,
  under required real CUDA initialization. Covers repeated fresh inputs/gradients,
  mutation/drop of source IDs, plan destruction before both adjoints, empty cases,
  integer/shape/dtype/device validation, nonfinite logits and subsequent recovery,
  capture rejection, existing public convenience dispatch and mean/max rejection.
- Registry replacement test runs both sum and softmax with old-context inputs,
  rejects new-context inputs, drops the Python plan, and verifies resident adjoints
  and saved fresh softmax outputs through allocation-owned DLPack/Torch. PyTorch
  is needed for this CUDA lifetime test; there is no dependency added to the API.
- `final-segments.log`: all six existing Rust CUDA segment tests passed unchanged,
  default parallel harness, `FERRO_REQUIRE_CUDA=1`, cargo `-j2`. Native counters
  assert one topology upload at preparation and none across reused fresh steps;
  zero feature uploads/downloads, four operator enqueues, and one 4-byte control
  read/allocation per nonempty softmax forward. These are native-core structural
  measurements, not Python-exposed counter measurements (none were invented).
- `binding-unit.log`: standalone binding lib suite, 15 passed under the default
  parallel harness, followed by final release rebuild/install.
- `general-python.log`: existing general Python regression checks all passed.
- `diff-check.log`: owned tracked production files pass git diff --check.

`verify.py` reproduces the scoped wheel build/install, Python contracts, native
segments, general regression, source identity and installed provenance checks.
Runtime DLL directories are LOCALAPPDATA/Temp/cuda-rt/nvidia/{cuda_nvrtc,cublas}/bin.
`provenance.json` confirms every facade Python source matches its wheel entry and
installed file; native/graph hashes were stable across tests. The Git-visible
crate .rs/.cu/.py/manifest inputs in sources-before/after were unchanged throughout
this scoped run. This is NOT the parent final integration freeze; sibling changes
made later require a new final build/run. No benchmarks, commits or pushes.

## Existing boundary encountered (not changed)

The first extended contract run (`contracts.log`) failed while inspecting an
old-context gradient with generic `tolist()` after registry replacement. The
operation and backward had succeeded; generic tensor copy dispatch consulted the
new backend and rejected the old allocation. `registry_copy_probe.py` isolates
the same failure using only a plain tensor and two cuda_init calls, without any
PreparedSegments. `registry-copy-probe.log` records that failure and verifies
allocation-owned DLPack still returns the original values. The final lifetime
contract uses that allocation-owned inspection path, not a production workaround.
`cuda_init`'s existing idempotence docstring is stale: install really replaces the
registry/stream. Neither shared tensor behavior nor that unrelated docstring was
edited within this ownership scope.

The previously documented general backward strided-seed host round trip remains:
resident claims use contiguous device seeds. Softmax's four-byte control download
is synchronous; zero feature copies must not be described as zero total transfers.

## Owned files

- crates/ferro-py/src/architecture.rs
- crates/ferro-py/python/ferro/graph.py
- crates/ferro-py/tests/test_prepared_segments.py (new)
- verification/post25/prepared-python/ (new probes, runner, reports/logs/provenance)

Existing Rust formatter suggestions for architecture.rs's compact repository style
were not applied as unrelated file-wide formatting. Compiler warnings and Git's
LF/CRLF notices remain; all listed final commands exited zero.
