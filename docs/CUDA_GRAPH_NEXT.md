# Next whole-model static CUDA graph replay

> Historical diagnostic/roadmap snapshot. Current scope, limits and fresh merged-tree results: [stabilisation publication](STABILISATION_PUBLICATION.md). Linked local archives are not shipped artifacts.

## Scope and evidence

Read-only engineering inspection against HEAD `0dd0c1b2a7d01b631e1085e6071876186c6804de`, branch `perf-fused-layernorm`. No production source edits, installs, builds, tests, or GPU execution were performed. A concurrent agent started modifying `dispatch.rs` during inspection; source observations below refer to the inspected baseline, not a review of that work. This is a proposed implementation, NOT an implemented or measured capability.

## What exists, and what is missing

- `crates/ferro-core/src/graph.rs:475-608`: `CompiledChain` is an inference DAG scheduler, not a CUDA graph. It retains leaf `Tensor` clones, computes fused pointwise runs, and stores explicit forward metadata. `replay_inner` clones the leaf map, calls each run, allocates new results, and drops intermediates at return. It already resolves shared branches at their topological tails; preserve this schedule.
- `autograd.rs:9-25`: forward metadata covers MatMul, Bmm, Reshape, Transpose, Softmax, SumDim, and LayerNorm. Metadata delegates to ordinary tensor methods, not prepared destination-writing kernels. `run_chain` in `graph.rs:389-436` silently tries the fallback on a backend error. Static CUDA preparation must instead fail closed; a host fallback inside capture is not acceptable.
- At baseline, resident LayerNorm is a decomposition (`ops_ext/layer_norm.rs:45-69`) including scalar fills, reductions, broadcasts, and pointwise operations. Lower the actual schedule, including constants and layout materializations. Do not assume one LayerNorm node means one device launch. Integrate any separately landing fused LayerNorm seam only after its own tests.
- `ferro-cuda/src/lib.rs:149-175,961-1058`: `CapturedChain` captures just one fused chain kernel. It owns the original output but does NOT retain the borrowed inputs. A safe caller can drop/recycle an input while the graph still references its address. Its extra `buf` is made with `out.clone()`, which is a detached device copy; the correct download method deliberately reads `out`, not that copy. Local cudarc 0.19.9 `driver/safe/core.rs:854-864` confirms clone allocates/copies rather than retaining ownership.
- `lib.rs:876-950` plus `alloc.rs:142-159,241-253`: whole-training-step capture exists experimentally but is not the right wrapper for inference replay. Actual allocator behavior bypasses reuse and frees dropped slices during capture; `_retained` is normally empty. Several comments still say drops are retained and instantiate without auto-free, contrary to code. Capturing `CompiledChain::replay()` through this API leaves its returned output alive beyond capture while intermediate allocations are graph-managed. That mixes lifetimes and does not provide the desired externally owned fixed-output contract.
- Pool hit counts do not prove node-address stability. `alloc.rs` bins by length with LIFO reuse; overlapping work, live output tensors, and drop order can change which address a node receives. A static plan must own original allocations, not depend on the freelist reproducing them.

## Existing safety blockers to resolve rather than copy

1. Both capture entry points toggle context-wide `disable_event_tracking()`. The single-stream comment is insufficient: `resident()` only checks buffer type/device ordinal (`lib.rs:379-396`), not backend/context/stream identity. Other backend instances on the same ordinal are not excluded. `new_stream()` itself enters cudarc multi-stream management. Re-enabling tracking does not retrofit events into slices allocated while it was off (`cudarc core.rs:470-494`). No new path should disable tracking, forge untracked aliases, or replace `new_stream()` with the legacy default stream.
2. An error after begin in `capture_chain` only calls `capture_status()` (`lib.rs:1042-1045`). Querying status does NOT end or abort capture. A subsequent caller can inherit a capturing/invalidated stream. A panic is likewise not scoped by an RAII capture owner.
3. `begin_step_capture` changes allocator/tracking state before checking for an existing capture. A nested attempt can disturb the outer capture even when rejected. `alloc.begin_capture()` clears its retained set before establishing exclusive ownership.
4. cuBLAS workspace is a raw allocation whose pointer is discarded; `cublasSetWorkspace_v2` status is ignored (`lib.rs:343-350`). Add a lifetime-owned allocation and check setter status; tie workspace/handle/stream to the graph owner and serialize use.
5. cudarc 0.19.9 `safe/graph.rs:54-68` ends capture then instantiates with `?`; if instantiation fails, the raw graph has no RAII owner on that branch. A robust local graph wrapper or dependency fix must destroy that raw graph. `CudaGraph::launch()` itself only calls the driver and does no buffer usage bookkeeping (`safe/graph.rs:78-83`). Replays therefore need explicit tracked read/write boundaries.

CUDA supports captured event dependencies and event graph nodes; the claim that event recording is universally illegal during capture is incorrect. The actual constraints concern dependencies, capture membership, synchronization, and invalidation.[6] Existing skill notes conflict with both this documentation and portions of the current allocator; treat them as historical troubleshooting notes, not an implementation specification.

## Minimal safe extension: prepared static inference plan

### 1. Lower the existing schedule, not a new model tracer

Add an additive preparation API beside `CompiledChain::replay`, leaving the existing allocating replay and its output semantics unchanged. Core exports a backend-neutral static plan (operation, input/output slot IDs, shapes/strides/offsets, scalar constants); a new `Backend` preparation seam defaults to Unsupported. Core remains dependency-free. The backend returns an opaque prepared-execution handle; CUDA-specific graph ownership stays in `ferro-cuda`.

Validate the complete plan BEFORE CUDA capture: f32, one backend/context, static shapes, checked ABI dimensions, valid broadcasts/layouts, supported kernels, no host fallback, no autograd/backward, no data-dependent control flow. Keep leaves as `Tensor` handles to their existing `TensorInner`, not `detached_view()` snapshots that increase StorageCell alias counts. Preserve shape-only views as descriptors and explicitly materialize only at whole-buffer consumers.

### 2. Prepare all allocations and launches before capture

Add destination-writing/prepared forms of the existing launch seams: fused pointwise and expanding broadcast, matmul/bmm, stride-copy materialization, sum_dim, row softmax, fills, and the approved LayerNorm path. Existing allocating backend methods should delegate to the same kernels so eager/static numerics cannot drift. Separate allocation and kernel compilation from enqueue.

A `PreparedCudaPlan` owns a fixed allocation for each materialized run output, every hidden scratch buffer and scalar/index constant, the final output, CUDA functions/modules, backend stream, and cuBLAS workspace. Allocate everything before begin; initially do no liveness-based alias reuse. Distinct buffers cost more memory but make address/lifetime proof straightforward. Retain original slices, never `CudaSlice::clone()` copies. Kernel memset/initialization must be replayed when an op reads initial zeros; pre-zeroing once is insufficient for accumulators. Preserve zero-size and k==0 matmul behavior.

Warm these exact prepared commands with these exact allocations, then synchronize outside capture. An enqueue pass must make no allocation, kernel-cache miss, module-load, upload, host-readback, or stream-synchronize calls. The generic size-binned allocator remains unchanged; do not use its training-capture mode for this path.

### 3. Treat the graph as one tracked asynchronous backend operation

Keep context event tracking enabled throughout. The no-shortcut implementation needs a reviewed graph-use boundary, not simply the current `CudaGraph::launch()` wrapper:

- Serialize graph capture/update/replay/drop and workspace use. Pin leaf storage and validate the exact backend/context, not just CUDA ordinal. V1 may reject cross-backend operands explicitly.
- Before each launch, acquire read dependencies for actual leaf buffers and read/write dependencies for the owned output/scratch; after enqueuing the graph, publish completion to those original buffers. Local cudarc provides `DevicePtr`/`DevicePtrMut` and `SyncOnDrop` for this purpose (`core.rs:1151-1269`); retain guards until AFTER launch and check recorded driver errors. Do not publish a read event for a write or track a temporary alias instead of the original buffer.
- A feasible encapsulation is a prepared-command primitive: obtain/validate stable pointers outside the capture window, capture audited raw kernel/cuBLAS enqueues on privately owned scratch, and wrap every resulting graph launch in the full original-buffer tracking described above. Tracking is not switched off: synchronization moves to the graph boundary exactly as for one large kernel. Scratch never escapes; its reuse is protected by the same owner. This requires careful unsafe-code review, not pointer casts around the existing safe builder.
- Alternatively, use tracked launch builders inside capture only after proving every event wait belongs to a legal dependency sequence. Their ordinary pre-existing slice events are not automatically a valid captured-event protocol. If neither route can be proven, return Unsupported; do not hide the failure by disabling tracking.

This tracking/ownership primitive is the first GPU acceptance gate, not an optimization to postpone. It must also coordinate external tensor updates: ordinary CUDA mutation seams use shared host storage guards, so host RwLocks alone do not serialize asynchronous device writes.

### 4. Inputs, weights, and output semantics

The current stable-address copy seams are useful: `inplace.rs:234-280` overwrites values and bumps versions; `copy_into_dev` launches an in-place copy kernel (`lib.rs:1382-1400`), and `write_dev_from_host` calls the same-address host copy. Public mutation still refuses requires-grad/history or independently shared device storage (`inplace.rs:500-519`). Preserve these gates; do not introduce a raw public input-update escape hatch.

Graph leaf `Tensor::clone()` shares TensorInner without itself creating a new StorageCell alias. Existing `capture_layout_updates.rs:40-104` demonstrates retained captured layouts allow current-input copy, while ordinary device views/detach snapshots still block mutation. Preparation must preserve this distinction. Updates to input/weight VALUES at the same storage work; replacing a model field with a newly allocated tensor does not rebind the captured leaf. Reject unsupported rebinding, or require recompilation. Changing scalar hyperparameters baked into kernel arguments also requires a new plan unless represented by an explicit device slot.

Keep `CompiledChain::replay()` returning independent results. For the new static handle, expose a borrowed output lease that prevents another replay while live, plus an explicit copy-out for persistent snapshots. Do not return a normal freely clonable Tensor that silently changes on the next replay. An alternative replay-into API copies to a caller-owned destination; count that extra copy separately. Drop must wait/order pending graph uses before recycling any address and destroy executable/raw graph before releasing its pinned resources. Updates and output downloads occur outside capture with correct stream dependencies.

### 5. Failure ownership

Use an exclusive RAII capture session: validate no active capture before changing anything; every successful begin must pair with end, including enqueue error and unwind. End an invalidated capture to clear stream state, destroy any graph returned on failure, and preserve the original error while reporting cleanup failures. Instantiate/upload failure drops all graph resources but not in-flight storage prematurely. If cleanup cannot prove the stream usable, poison the handle/backend rather than pretending success. Initial static plans have no graph allocation nodes, so they do not need the training API's auto-free memory policy.

## Concrete tests before claiming whole-model replay

- Core mock preparation tests beside `tests/compiled_chain.rs`, `model_replay.rs`, `capture_layout_updates.rs`: unchanged tail scheduling/shared DAG execution; reject unsupported/mixed-device/dynamic metadata before begin; preserve mutation gates and input identity; replay under inference recording does not grow a tape.
- CUDA prepared-op parity tests: eager versus prepared destination-writing commands, dirty recycled buffers, empty dimensions, expanding broadcasts, transpose plus reshape, and all affine LayerNorm combinations.
- CUDA full-model graph test: MLP and attention including bmm/softmax/residual/layout/norm; capture once, then compare multiple distinct input updates AND weight updates against eager at every iteration. Retain original traced root and vary graph-handle/output lifetimes to expose stale snapshots. Verify recorded addresses and canaries, not just output shapes.
- Structural counters: one graph launch per replay; zero fresh allocations, H2D, D2H, index uploads, and ordinary per-op host enqueues within the replay interval. Separate external input copy/output snapshot costs. Capture-time enqueue counters do not count kernels executed on graph relaunch.
- Failure injection at preparation, post-begin enqueue, end, instantiate, upload, replay, and unwind; verify capture status returns NONE or handle is explicitly poisoned; subsequent normal work must remain usable. Assert tracking remains enabled throughout.
- Lifetime/concurrency: drop caller inputs after graph construction, pool pressure of equal sizes, output lease blocking replay, drop while queued, duplicate leaves, same-ordinal foreign backend rejection, and ordered concurrent update/read. Run GPU tests serialized when the baseline measurement agent is finished. No such tests were run in this task.

## torch.compile: existing environment versus viable options

### Read-only local discovery

| Item | Observed |
| --- | --- |
| Native interpreter | `crates/ferro-py/.venv/Scripts/python.exe`, Python 3.11.15 |
| Installed native PyTorch metadata | `torch 2.6.0+cu124` |
| Native Triton import discovery | `find_spec('triton') == None` |
| Native development headers | Base interpreter `Include/Python.h` exists |
| C++ compiler lookup | `cl` absent from current PATH; this does not prove MSVC is uninstalled |
| WSL | Default Ubuntu-24.04 running under WSL2; docker-desktop running; Ubuntu-22.04 stopped |
| Active WSL kernel | `6.18.33.2-microsoft-standard-WSL2` |
| Active WSL Python | `/usr/bin/python3`, 3.12.3; pip present; torch/triton specs None |
| WSL GPU exposure | `/dev/dxg` and `/usr/lib/wsl/lib/libcuda.so.1` present; WSL nvidia-smi executable present |

No CUDA initialization, nvidia-smi execution, compilation, or timing was performed. RTX 3090 is task-provided context; GPU driver version and actual compute readiness remain unverified. WSL package checks cover its default interpreter, not every possible hidden venv. Thus the truthful current result is 'no validated Inductor CUDA environment in the inspected interpreters', not 'torch.compile cannot work on Windows'.

### Option A: native Windows, same PyTorch version

The maintained `triton-lang/triton-windows` project explicitly documents Windows 10/11, torch.compile, Ampere sm86, and a PyTorch 2.6 -> Triton 3.2 compatibility pairing.[1] In a SEPARATE comparison venv, preserve `torch==2.6.0` from the cu124 wheel index and select `triton-windows>=3.2,<3.3`, then record the exact resolved wheel. Do not blindly install the latest Triton into this 2.6 environment. The fork documents bundled minimal CUDA tooling from post11 and TinyCC from post13; newer bundles do not remove separate runtime/header prerequisites, and CPU compilation still requires a C++ compiler.[1] This is a documented compatible installation option, not a smoke-tested result here.

PyTorch's own current Windows tutorial covers CPU/XPU, requires a C++ compiler, and names CPU support from 2.5 onward; it is not documentation of NVIDIA Inductor support.[4] The Windows Triton fork is now under the triton-lang organization but is still a distinct distribution from the main Triton project. Do not conflate its support claims with stock Linux wheel availability.[1][3]

### Option B: already-available WSL2, stock Linux stack

Prefer a fresh Linux venv in the running Ubuntu-24.04 distro, installing the same PyTorch 2.6/cu124 baseline and its matching Linux Triton dependency first. Mainline Triton documents Linux and NVIDIA compute capability 8.0+, which includes the task's Ampere hardware.[3] NVIDIA documents WSL2 CUDA support for Pascal-or-later GeForce in WDDM mode, using the Windows driver; do not install a Linux GPU driver into WSL.[2] Device bridge files already exist locally, but package installation and a CUDA compile smoke test are still needed to call the environment working.

For fair performance claims, run Ferro and PyTorch in the same OS environment. Comparing native Ferro against WSL torch.compile mixes compiler and OS/driver-interface effects. A Linux Ferro build would be a later approved step, not something performed during baseline measurement.

### Acceptance and comparison matrix (later, not executed)

First validate a small fullgraph CUDA function with `backend='inductor', fullgraph=True, dynamic=False`, default mode, fallback suppression disabled. Compare values over changing inputs, inspect emitted code/logs, and require successful first compilation plus repeated execution. Then test the exact MLP/attention workload and weight updates. `backend='eager'` or `aot_eager` is a diagnostic path, not evidence of optimized Inductor CUDA performance.

Measure separate eager, default-Inductor, and `reduce-overhead` cases; PyTorch 2.6 documents reduce-overhead as CUDA-graph based with applicability limits, including input mutation, and says max-autotune enables CUDA graphs by default.[5] Separate compilation/warmup from steady state. For a controlled compiler-only arm explicitly disable cudagraphs; for the replay arm verify graphs actually engaged rather than assuming the mode guarantees it. Match dtype, TF32/matmul precision, shape, inference mode, transfers, output retention, and timing synchronization. No speedup is asserted here.

## Retrieval limitations

Configured web_search/web_extract failed because the Nous gateway was unavailable. Direct PyTorch HTML returned 403; official tutorial source was discovered through the GitHub repository tree and read from raw GitHub instead. NVIDIA official documentation and current Triton project READMEs were fetched directly. One shell-inline retrieval attempt failed quoting and was retried successfully through Python. These were research failures only; no environment modifications were attempted.

## Sources

[1] https://raw.githubusercontent.com/triton-lang/triton-windows/readme/README.md
[2] https://docs.nvidia.com/cuda/wsl-user-guide/index.html
[3] https://raw.githubusercontent.com/triton-lang/triton/main/README.md
[4] https://raw.githubusercontent.com/pytorch/tutorials/main/unstable_source/inductor_windows.rst
[5] https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/__init__.py
[6] https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
