# Static CUDA graph inference: historical primitive implementation

> Superseded by [the whole-model continuation](MODEL_CUDA_GRAPH_CONTINUATION.md).
> Core lowering, nonpointwise preparation, Python replay and full-model benchmark
> results are now implemented and verified there. The text below preserves the
> original primitive-stage evidence and limitations, not the current status.

## Status

This does **not** complete whole-model CUDA graph inference. It implements and
GPU-verifies the first ownership/tracking gate from `CUDA_GRAPH_NEXT.md`:
a prepared, real multi-kernel pointwise DAG graph. No claim is made that an MLP,
attention block, `CompiledChain`, or Python model now runs as one CUDA graph.
The base revision was `1c72c89` on `feat-model-cuda-graphs`. No commits, staging,
branch changes or pushes were performed.

## Implemented primitive

`CudaBackend::prepare_pointwise_graph(runs, leaves)` builds a
`StaticPointwiseGraph`. Its slots are original Arc-owned device leaves followed
by one distinct owned destination per run. A run can consume any preceding slot;
shared branches are real graph dependencies, not precomputed frozen tails.

- Validate the complete schedule, exact backend stream identity, operand arity,
  buffer lengths and u32 kernel limits before preparing any allocation.
- Support nonempty, equal-length unary/binary pointwise runs. Broadcast, empty
  runs, invalid references and wrong-backend inputs return explicit errors.
- Compile the existing pointwise kernel source and prepare every destination
  before capture. Warm those exact commands and allocations before capture.
- Capture audited scalar device-pointer kernel arguments on a private nondefault
  stream. No CudaSlice aliases are forged; event tracking is never disabled by
  this new path. The private capture stream excludes unrelated backend enqueues.
- Replay on the original validated backend stream. Acquire original leaf read
  and original destination write usage guards before the single graph launch,
  publish them afterwards, and check cudarc's deferred error channel.
- Own original leaf allocations, destinations, CUDA functions/modules and
  backend. Caller leaf handles can be dropped without freeing graph addresses.
- Require mutable host access to replay; borrowing `output()` prevents another
  replay while that borrow is used. `snapshot()` explicitly allocates and copies
  an independent device result. Both are distinct from `CompiledChain::replay`.
- Pair successful begin with end through an exclusive RAII session, including
  unwind. Own raw graphs before fallible instantiation. Instantiate with flags
  zero: there are no graph-managed allocation/free nodes.
- Poison the handle after an uncertain replay boundary, refusing later replay,
  snapshots and output leases. Drop fences pending original-stream work,
  destroys the graph, and releases captured storage before unlocking legacy modes.
- Block legacy step/chain capture while a prepared static graph is alive, because
  those old modes disable event tracking. They are not used to implement this API.

The general f32 `resident()` check now rejects foreign backend streams even when
CUDA ordinals match. This closes an existing ownership hole rather than relying
on device ordinal as a context proof. Core Tensor mutation/version/autograd code
and its dependency-free boundary were not changed. The new API is backend-level;
it does not introduce a Tensor mutation bypass.

## Executed verification

Runtime library prefix:

```
$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin
$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin
```

All cargo commands used `-j2`. Required-GPU runs set `FERRO_REQUIRE_CUDA=1`;
GPU test binaries were serialized with `RUST_TEST_THREADS=1` or
`-- --test-threads=1`. Raw evidence is under `verification/model-cuda-graphs/`.

| Gate | Evidence |
| --- | --- |
| Foreign-stream rejection RED then GREEN | `context-red.log`, `context-green.log` |
| Prepared multi-run API missing RED, actual GPU GREEN | `dag-red.log`, `dag-green.log` |
| Reject legacy tracking disable RED/GREEN | `mode-red.log`, `mode-green.log`, `chain-mode-red.log`, `chain-mode-green.log` |
| Sticky graph failure RED/GREEN | `poison-red.log`, `poison-green-full-cuda.log` |
| Capture unwind, nested begin rejection, tracking enabled | `cleanup-green.log`, final `cuda.log` |
| Borrowed-output compile-fail test | `lease-green.log`, final `cuda.log` |
| Release binding rebuild and editable install | `build.log` |
| Full repository integration matrix plus standalone layout GPU test | `integration-results.json` |

The structural GPU test captures three distinct kernel nodes for
`relu(x) + x*w`, including a shared input and two computed branches. It asserts
one successful CUDA graph launch per replay, no Ferro allocator requests, no
ordinary pointwise host enqueues, and no changes to the existing f32 upload /
download helper counters during the measured replay interval. Those are
instrumented Ferro-level claims, not an opaque CUDA-driver allocation trace.
Input and weight updates both change results. Caller handles are dropped;
equal-size allocation pressure does not corrupt results. An independent output
snapshot survives subsequent replays and graph destruction.

Final combined-tree matrix: **13 commands passed**. Parsed Rust results include
630 core tests, 20 fastcpu/tokenizer tests, 66 CUDA tests including doctests,
and the standalone capture-layout GPU test. Python capture, compiled/fuse,
general, operator parity, safetensors and 200-trial fuzz checks passed. Existing
compiler warnings remain; the tree is not warning-free.

An initial run of the integration matrix with the default parallel Rust harness
failed: global backend-registration tests received foreign-stream buffers, and
legacy capture overlapped other GPU work. Its evidence is retained as
`parallel-first-cuda.log` and `parallel-first-integration-results.json`.
The scoped runner now forces serialized Rust tests and required GPU execution;
the subsequent full matrix passed. This does not claim default parallel GPU
suite compatibility.

## Missing before whole-model delivery

1. Core backend-neutral static schedule lowering from `CompiledChain`, retained
   Tensor identity and globally ordered StorageCell read guards at replay.
   Current core device storage is Box-owned; the new raw-backend Arc-leaf API
   cannot simply be bolted onto Tensor storage by cloning CudaSlice.
2. Prepared destination commands for matmul/bmm, broadcasts, strided layouts,
   reductions, softmax and affine LayerNorm, including their scratch and
   degenerate-shape rules. The existing raw cuBLAS workspace leak/unchecked
   setter is not repaired by this pointwise-only path.
3. Real Python whole-model preparation/replay API and mutation/autograd gates.
   Bindings were rebuilt to verify compatibility, not to expose an unimplemented
   model API.
4. Full MLP/attention updated-input AND updated-weight parity and graph launch /
   transfer / allocation structural gates on the actual model path.
5. Comprehensive injected enqueue/end/instantiate/upload failures and threaded
   mutation/output-use stress. Current failure tests cover post-begin unwind,
   nested capture rejection and a deliberately injected deferred boundary error;
   they are not an exhaustive CUDA failure matrix.
6. Two serial, reversed-order full-model benchmark runs with actual CUDA graph
   replay and separate setup costs. **These were not performed**: the full-model
   graph path does not exist yet. Existing allocating `ferro_compiled` timings or
   this pointwise graph's timings must not be relabeled as that result. No latency
   improvement or end-to-end performance claim is made.

The old `CapturedChain` input-lifetime/cleanup hazards and experimental training
capture remain separate work. The new primitive avoids those mechanisms rather
than presenting them as a completed model graph implementation.
