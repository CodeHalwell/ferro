# Native CUDA graph preparation failures

## Scope and implementation

This follow-up changes only static graph native ownership/preparation and its
verification. It does not change replay dispatch, model lowering, event tracking,
bindings, or the legacy combined cudarc graph APIs. Existing uncommitted work is
preserved. No benchmarks, commits, staging, or pushes were performed.

`crates/ferro-cuda/src/static_graph/native.rs` is a private, preparation-only
adapter. Its generic ownership decisions are exercised with driver-free opaque
handles; those handles never enter CUDA. The production adapter forwards to the
CUDA driver. Test-only, thread-local overrides replace the return at that call
boundary, rather than returning after a caller's successful `?`. Overrides restore
on unwind. Replay still calls the driver directly without adapter dispatch.

Corrections backed by behavioral RED/GREEN logs:

- Own output slots before calling end/instantiate, including unwinding after a
  successful native allocation. Release partial *owned adapter outputs* once.
- On failed end, query the private stream before cleanup. An ended stream is not
  ended again; a skipped end that left capture active is explicitly closed.
  Unknown status is not treated as proof that another end is safe.
- Reject null graph/exec separately from a valid, real zero-node graph.
- Fence uploads even when the upload call returns an error; keep a pending fence
  through unwind and retry completion during graph destruction.
- Retain the complete preparation resource tuple before private warmup begins.
  `Flight` also governs published graph destruction: fence all relevant streams
  before graph/exec, buffers, functions, cuBLAS workspace and owners are released.
- If a cleanup fence and its retry both fail, retain the resources and context
  for process lifetime and retain one legacy-capture exclusion slot. This is
  deliberate quarantine, not successful destruction or recoverable GPU work.
  Repeated failures can grow retained memory without a bound; restart the process
  rather than retrying indefinitely. Cleanup fences run while holding capture
  exclusion, so a stalled fence can also block preparation and legacy capture.
  A pending graph upload with an unsuccessful final fence likewise retains its
  native handles/context. No uncertain-completion buffer reuse is claimed.

The old `Resources` drop fence remains as a local backstop. The encompassing
`Flight` now prevents its callers' destination owners from dropping when completion
cannot be established. Cleanup errors are recorded; they do not replace the
primary preparation error or trigger a panic. Destroy attempts are not blindly
retried: an error is not proof that a native handle remains valid.

## Exact supported failure matrix

| Boundary/state | Evidence | Ownership/recovery assertion |
| --- | --- | --- |
| Begin skipped, returns error | Real GPU, pointwise and model | No end/instantiate/upload; no published user; next graph replays |
| End skipped, capture remains active | Fake + real GPU | Cleanup end is called exactly once more; private status becomes NONE |
| End terminates normally but adapter returns error with owned graph | Fake + real GPU | Exactly one end; graph destroyed once; no instantiate/upload |
| End returns error with null graph after termination | Fake | No null destroy, instantiate or upload; no second end |
| End success with null graph | Fake + real GPU (adapter destroys real graph before returning null) | Precise `null graph` error; no instantiate/destroy-null |
| End unwind after owned output | Fake + real GPU | Output owner destroys once; no second end |
| End during cleanup returns error with owned output | Fake | Error recorded and graph destroy still attempted once |
| Capture status query fails | Fake | Record query error; retain stream/context; no blind second end; encompassing fence/quarantine controls addresses |
| Instantiate skipped, null exec | Fake + real GPU | Graph destroyed once; no exec destroy/upload |
| Instantiate creates exec, then adapter returns error | Fake + real GPU | Exec destroy precedes graph destroy, each once |
| Instantiate success with null exec | Fake | Reject publication and destroy graph only |
| Instantiate unwind after output | Real GPU | Exec then graph destroyed; next preparation/replay works |
| Upload skipped or enqueues then returns error | Fake + real GPU | Completion fence before exec/graph destruction, no published graph |
| Upload enqueues, completion fence skipped with error | Fake + real GPU | Primary error returned; second fence completes before destruction |
| Upload/fence unwind | Real GPU | Pending upload fence runs during Drop; override restored; next replay works |
| Persistent upload-completion failure | Fake only | No unsafe native destruction; retain handles/context |
| Private warmup fence skipped with error | Real model GPU | Cleanup fences backend and private stream before resource release; next model replays |
| One cleanup fence fails, retry succeeds | Fake | Buffers drop only after retry; existing exclusion count preserved |
| Cleanup fence and retry fail | Fake only | No buffer drop; one quarantine exclusion slot retained |
| Bind/exec-destroy/graph-destroy return cleanup errors | Fake only | Original instantiate error preserved; each cleanup attempt/error recorded without panic |
| Valid zero-node graph and zero-reduction destinations | Existing real model tests | Still instantiate/replay, preserve replay count and destination rewrites |

The returned-error GPU matrix asserts exact adapter call order, private stream
status, event tracking, zero published graph users, original leaf reference-count
recovery, and parity from the next actual graph replay. It runs for both pointwise
and model preparation. Panic recovery exercises the shared owner through the
pointwise path; model stage-unwind tests remain covered separately.

## Native API postconditions and limits

Sources inspected (downloaded copies are `stream.html` and `graph.html`):

- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html>
- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html>

The stream API says invalidated capture returns a null graph, and that a stream
in INVALIDATED status must still be terminated with end capture. ACTIVE and
INVALIDATED are both pending capture in the adapter; NONE is terminated.
Instantiation documentation promises an executable output on **success**, not a
valid owned handle for every possible failed native out-parameter.

Accordingly, production native out-parameters are local and null-initialized;
only successful native calls transfer handles to the adapter's owned outputs.
The partial-output tests use known live fake handles or handles from actual
successful native creation, followed by an injected adapter error. They do not
invent valid handles on a genuine failed native call or pass unspecified error
out-parameter bits back to CUDA. Undocumented native partial-output behavior is
not a proven cleanup contract.

**These are controlled returned-error and host-unwind tests, not reproduced
native CUDA faults.** No illegal writes, GPU/context corruption, or real capture
invalidation were induced. In particular, normal end followed by an injected
error is not evidence that the driver itself generated an invalidated-capture
error. Persistent failures/status uncertainty are tested driver-free; process-
lifetime retention is a conservative policy, not demonstrated fatal-context
recovery. Native destroy failures may mean destruction did not complete; tests
prove attempts, order and error preservation, not an impossible success guarantee
for a failing destructor. Restart the process after a persistent quarantine.

## TDD and final verification

The `*-red.log` files contain observed behavioral assertion failures, paired with
`*-green.log` files for the corresponding fixes. Initial owner extraction was a
refactor; existing null/partial instantiate and cleanup branches have additional
characterization tests. The final native module has 16 tests: 13 driver-free and
3 real-GPU tests (the latter iterate multiple boundaries and preparation paths).
Expected, caught panic output appears in `--nocapture` logs. The compile-fail
output-lease doctest also intentionally prints a compiler error and passes.

Final commands, with local temporary NVRTC/cuBLAS DLL paths and required CUDA:

```
python verification/dlpack-raw-review/run.py all
python verification/native-graph-failures/run.py final-cuda-default-parallel --full
python verification/native-graph-failures/collect.py
```

The `all` command ran standalone binding Rust tests under the default parallel
harness, rebuilt the release extension with maturin `-j2`, then ran public Python
interop and all 13 integration commands against that rebuild. The separate full
CUDA run used the default parallel harness; the combined integration runner uses
its existing serial Rust-test setting. No timing samples were collected.

Final results (all exit 0):

- Full default-parallel CUDA: 58 unit, 1 concurrency integration, 22 GPU
  integration, 2 layer norm, 2 layout, 11 model static graph, 4 static safety,
  and 1 compile-fail doctest: **101 harness tests passed, zero ignored**. The
  concurrency child additionally executes its one test; it is not double-counted.
- Standalone binding Rust: **12 passed**, default parallel, zero ignored.
- Eager DLPack: **5 passed**; snapshot DLPack: **1 passed**.
- Full Python test discovery, including static coverage: **26 passed**.
- Combined integration: **13/13 commands passed**, including core, CPU/tokenizer,
  full CUDA, all-targets CUDA check, Python suites/examples/fuzz, diff check and
  capture-layout GPU. Existing core suites report 2 ignored tests and the
  CPU/tokenizer group 1 ignored test; these are not claimed as executed passes.

`all-results.json`, `integration-results.json`, `summary.json`, and `final-*.log`
are archived here. `collect.py` parses actual logs and counts the parent concurrency
harness only once. Older intermediate final logs may reflect the immediately
preceding TDD step; `final-cuda-default-parallel.log` and the archived integration
logs/results are authoritative for the final source.
