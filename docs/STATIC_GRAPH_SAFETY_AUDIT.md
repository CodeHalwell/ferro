# Static graph safety audit

> Historical diagnostic/roadmap snapshot. Current scope, limits and fresh merged-tree results: [stabilisation publication](STABILISATION_PUBLICATION.md). Linked local archives are not shipped artifacts.

Native-call ownership follow-up: [failure matrix and executed verification](../verification/native-graph-failures/README.md).
That follow-up distinguishes controlled returned errors from genuine native faults
and documents persistent-failure quarantine. The baseline audit below remains a
historical source inspection, not the current implementation's test report.

## Scope and evidence

Source baseline: c277dcf86862155ecb11dab3ce6abe5ab29023b3. Line references below
refer to that inspected source, not future performance edits. Read-only audit of
production code. No CUDA calls, torch/ferro imports, Rust builds, GPU tests,
package installs, commits, staging or benchmark runs were performed. Existing
untracked verification artifacts were left alone.

Priorities distinguish observed source gaps from unproven failure outcomes.
This is not a claim of native fault recovery or default-parallel runtime success.

## P1: public DLPack export still has no consumer handoff

- `crates/ferro-py/src/lib.rs:769-778` accepts arbitrary `stream` and discards it.
  The comment says transfers are synchronous, but device export is zero-copy.
- `crates/ferro-py/src/dlpack.rs:205-235,268-273` publishes a resident pointer,
  retaining its storage, without a producer fence or consumer event wait.
- `crates/ferro-cuda/src/lib.rs:1923-1934` gets a DevicePtr usage guard on the
  allocation stream and returns the address. This is not an explicit consumer
  stream handoff or host completion fence. Retention solves pointer lifetime,
  not ordering of an immediate foreign-stream read.

An ordinary asynchronous eager tensor export can therefore lack the required
producer-to-consumer dependency. No corrupt output was reproduced in this audit.
The snapshot-specific fix must not be advertised as general DLPack safety:
`static_graph.rs:155-163` copies then synchronizes the backend stream before
publishing its independent allocation. Existing coverage is appropriately narrow:
`static_graph.rs:172-194` puts a delay before replay and queries readiness before
its cleanup fence; `crates/ferro-py/tests/test_static_snapshot_dlpack.py:17-46`
consumes snapshots immediately on default and non-default Torch streams and
retains outputs after source/graph destruction. Neither tests unfenced eager
exports. These tests were inspected, not executed here.

Minimal follow-up: validate stream tokens in a pure helper; either explicitly
fence the allocation's producer stream before ordinary export (conservative), or
record a producer event and make the requested consumer stream wait. Specify
None/default/per-thread/non-default/no-sync sentinel behavior rather than casting
an arbitrary Python object to a handle. Use allocation ownership, not LAST_BACKEND,
for the producer stream. Add delayed eager-output consumption on both Torch stream
kinds, without producer synchronization or tolist before consumption. Also audit
imports: `dlpack.rs:285-286` invokes the producer without a stream argument;
`431-436` describes synchronous DtoH. A synchronous copy does not by itself document
ordering against every foreign non-default producer; this remains a separate
contract question, not a demonstrated import failure.

## P1: default-parallel CUDA unit registry ownership is unguarded

`crates/ferro-core/src/dispatch.rs:666-679` protects individual map operations,
not a test's register/use/drop lifetime. The same unit-test binary has three
writers of Device::Cuda(0), all without a shared test-lifetime mutex:

| Test in crates/ferro-cuda/src/lib.rs | Entry | Registration |
| --- | --- | --- |
| gpu_end_to_end | 2377 | 2436 |
| gpu_softmax_gelu_activations_match_cpu_forward_and_grad | 2538 | 2547 |
| gpu_mini_block_forward_backward_fully_resident_converges | 2656 | 2665 |

An interleaving can allocate on A, replace registry entry with B, then dispatch
an operation on A's buffer through B. `lib.rs:404-407` explicitly rejects a
foreign backend stream. This supports a concrete error/flakiness mechanism;
it does not establish memory corruption. Serial shell commands do not serialize
Rust test threads. `--test-threads=1` masks, rather than fixes, this ownership race.

Exact fix grouping (deferred; no source modifications here):

1. Add one `#[cfg(test)]` crate-visible `REGISTRY_TEST_LOCK: Mutex<()>` and a
   poison-recovering guard helper. Acquire that SAME lock as the first local in
   each of the three tests above, before CUDA/backend setup. Keep it through all
   tensor, closure, graph and backend destruction; never lock only registration.
   Future unit tests that call install/register or consume registry-backed CUDA
   tensors must join this group. Do not weaken the stream-identity rejection.
2. Direct-backend tests, including snapshot, boundary-poison and model failure
   tests, do not use the registry and need not join this group solely for registry
   reasons. This is not a proof that every other shared CUDA state is independent.
3. `tests/gpu_integration.rs:12-29` already returns a poison-tolerant guard with
   its setup backend. Preserve callers' guard lifetime. It is a distinct test
   executable, not the unit binary, so its private mutex cannot protect these
   three unit tests. `tests/model_static_graph.rs:4,8,23,50,69,111,143` similarly
   has its own lifetime guards. Do not replace all locks with unrelated per-test
   locks, and do not attempt cross-process registry serialization.
4. `tests/concurrent_copy_replay.rs:9-17,53-64` has one test, an exact-filtered
   child, and timeout kill/reap. Its internal copy/replay overlap is intentional;
   preserve it, rather than serializing the two workers.

After the performance owner releases the GPU, verify the entire CUDA unit binary
under the default harness with CUDA required, repeatedly, then run integration
binaries. Record actual executed/skipped test counts. This audit did neither.

Adjacent public-state concern: `lib.rs:1887-1888` updates LAST_BACKEND then the
core registry under different locks; concurrent install calls can interleave the
two publications. Direct register_backend calls do not update LAST_BACKEND at
all. `device_synchronize` uses LAST_BACKEND (`1906-1912`). Do not infer that it
always fences every registered allocation stream. This is distinct from the
three-test fix and merits an explicit concurrent-install contract.

## P1 verification gap: native end/instantiate/upload errors are not injected

Current protections and exact limits:

| Site | Existing behavior | Remaining evidence gap |
| --- | --- | --- |
| static_graph.rs:64-74 | Ends capture; disarms session; owns non-null raw graph before instantiate | No test returns an actual error from end/instantiate seam |
| static_graph.rs:65-67 | Sets active=false even when end returns Err | Assumes the attempted end terminates capture; actual status/error classes untested |
| static_graph.rs:68-73 | Null-initialized exec is stored directly in GraphOwner | Destruction of any non-null output after failed instantiate needs explicit API postcondition review |
| static_graph.rs:39-44 | Attempts exec destroy before graph destroy, records errors | No call counts/order assertions or bind/destroy-failure tests |
| static_graph.rs:78-88 | Active unwind calls end, destroys a returned graph | Ordinary panic-after-begin test is not invalidated-capture/native-end-error proof |
| model.rs:194-207 | Warm/enqueue/end/upload stage exits | Hooks are after successful operations; end is after successful instantiation too |
| static_graph.rs:277-280; model.rs:205-208 | Upload followed by stream fence and error check | No upload-call error or upload-completion failure test |
| model.rs:70-75 | Resource drop fences private warmup work | Does not prove successful completion if that fence itself fails |
| static_graph.rs:204-215 | Fences backend, destroys graph/resources before decrement | Recorded fence/destroy failures do not establish completion or safe reuse |

`model.rs:7-22` exercises warm/enqueue/end/upload, checks tracking, backend stream
capture status, zero graph count, then recovers with another graph. It does not
observe the failed private stream's status or native allocation/destruction
counts. Its thread-local injection is reset only after the assertion, not by an
unwind guard. `static_graph.rs:291-303` tests sticky poison using a recorded
context error; it does not force graph launch or completion APIs to fail.
These are useful stage-unwind tests, not exhaustive native failure tests.

Do not automatically retry end after every error: a genuinely invalidated
capture can already be ended by the first call. Conversely, an injected Err
that simply skips cuStreamEndCapture leaves real capture active and tests a
DIFFERENT state. A failure test must specify which native transition occurred.
No actual leak or persistent capture was measured in this audit.

### Minimal call seam proposal (follow-up source owner only)

Create a private `static_graph/native.rs` adapter shared by pointwise and model
preparation. Wrap end_capture, instantiate, upload, exec_destroy and graph_destroy;
include bind/fence/status observation in the test adapter. Production defaults
forward unchanged. Do not add callbacks or counters to steady-state replay.
Prefer a per-session injected adapter, with RAII override restoration if a
thread-local test override is used, rather than a process-global failure flag.

Separate a small driver-free ownership state machine from Arc<CudaContext> and
CudaStream. It should own opaque test handles and delegate destruction through
the same adapter; this makes CPU tests exercise production ownership decisions,
not a Python reimplementation of CUDA. Real-driver tests can then replace the
call result AT the adapter boundary, not exit after the caller's `?` succeeded.
For instantiate, initialize a local exec out-pointer and transfer ownership only
according to documented success/failure out-parameter guarantees. Do not invent
valid handles on an error. For end, expose enough status for tests to distinguish
terminated/invalidated capture from a skipped call retaining active capture.

Required failure matrix and assertions:

- End returns native error after capture termination: no instantiate/upload;
  no null destruction; session cleanup does not accidentally end twice.
- Capture remains active because a call was skipped: explicit cleanup still
  ends it, or a documented poisoned/quarantined path prevents reuse. This case
  is NOT evidence for the native invalidated-capture contract.
- End succeeds with null graph: precise error; no instantiate or destroy-null.
- Instantiate returns Err with null exec and valid owned graph: graph destroyed
  exactly once, exec never destroyed, upload never called, graph-user count zero.
- Instantiate returns success: transfer ownership once; upload Err destroys exec
  before graph exactly once and publishes no graph. Use a defined test-only
  fake handle in CPU tests, never pass fake pointers to CUDA.
- Upload enqueues then fence fails: distinguish from upload-call Err; assert
  error propagation and resource/quarantine policy for uncertain completion.
- Bind/destroy/private fence fail: preserve original failure and record cleanup
  failures without panic; establish policy before claiming storage can be reused.
- Panic at each boundary: cleanup call order/counts, override restoration, user
  count unchanged; next independent preparation works where recovery is promised.

CPU adapter error returns prove host branching/ownership, not that the GPU driver
produces those results. Later native invalidation/recovery tests must run in
bounded child processes after benchmarks finish and verify private stream status,
tracking, subsequent eager work and another preparation. Fatal context errors
must not be relabeled recoverable because a mock succeeded.

Legacy APIs still call cudarc's combined end helper at `lib.rs:953-955,1057`.
The static owner fix is not automatically applied there. The static source's
comment (`static_graph.rs:70-71`) names a cudarc helper leak; the dependency's
implementation was not independently inspected here, so this audit records the
remaining call sites, not a newly established leak finding.

## CPU-only deliverable and validation follow-up

Run from repository root:

```
python -B -m unittest discover -s verification/static-safety-audit -v
```

Actual result: 8 tests, 6 ordinary passes, 2 expected failures, exit 0. Expected
failures expose absent proposed registry guard and ignored public consumer stream;
they do NOT mean those bugs are fixed. These are deliberately labeled source
checks (owner ordering, failure-hook positions, snapshot fence, validation-before-
allocation and registry inventory), not execution of Rust validation or native
cleanup. Text checks may need review after refactors and are not semantic proofs.

Pure model validation is currently inline in `prepare_model_graph` after resident
buffer resolution (`model.rs:79-137`). A copied Python validator would only test
itself. Follow-up: extract that exact production validation into a driver-free
helper taking leaf lengths and StaticRuns and returning validated slot sizes.
Use CPU tests for empty DAG/leaves, forward/invalid slots, checked shape overflow,
zero outputs, u32/i32 boundaries, pointwise operand/arity/broadcast bounds, matmul
and BMM products, layout rank/offset overflow, sum divisibility and normalization
operand sizes. Test valid cases as well as malformed cases; assert failure before
any backend allocation in a counting harness. No extraction/build was attempted
while the sibling owns production changes and GPU performance measurement.
