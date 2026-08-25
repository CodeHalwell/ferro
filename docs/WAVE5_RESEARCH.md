# Wave 5 research: launch-overhead elimination and precision paths

This document grounds the Wave 5 plan in published evidence and in the actual
cudarc 0.19.9 API surface. It answers six questions: what CUDA Graphs buy for
many-small-kernel workloads and how to use them from Rust; how pinned-memory
async copies overlap transfer and compute; what torch.compile actually
delivers on small-transformer-shaped configs; where cudarc stands on NCCL /
multi-GPU as of 2026; whether cuBLASLt tensor-core GEMM is reachable from Rust;
and what burn / candle / dfdx do that ferro should steal. Every claim that
matters carries a URL.

## 1. CUDA Graphs from Rust

### 1.1 The mechanism

CUDA Graphs (CUDA 10+) let a sequence of kernels be captured once into a graph
and replayed with a single CPU call. Replay eliminates per-launch driver work:
the kernel parameters, grid shapes, and stream dependencies are fixed at capture
time, so a whole training step becomes one host-side submission instead of N.
NVIDIA frames the problem exactly as ours: "Modern GPUs are so fast that ... the
time taken by each GPU operation is now measured in microseconds. However,
there are overheads associated with the submission of each operation ... which
are now becoming significant."

- NVIDIA, "Getting Started with CUDA Graphs":
  https://developer.nvidia.com/blog/cuda-graphs/
- PyTorch, "Accelerating PyTorch with CUDA Graphs":
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

### 1.2 Published speedups for many-small-kernel workloads

- NVIDIA's canonical benchmark (20 short ~2.9 us kernels per step, V100):
  individual launches + `cudaStreamSynchronize` cost 9.6 us per kernel effective;
  graphs drop this to 3.4 us per kernel - i.e. launch overhead cut from ~230%
  of kernel time down to ~17%. That is a 2.8x end-to-end improvement on pure
  launch-bound code. https://developer.nvidia.com/blog/cuda-graphs/
- PyTorch reports graphs matter most "when using very small batch sizes, where
  CPU overheads are more pronounced", and credits graph execution as
  instrumental to MLPerf records at 4000+ GPU scale.
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- PyGraph (arXiv 2503.19779) measures 18-23% step-time gains from CUDA-graphing
  copy+compute in compiled PyTorch workloads.
  https://arxiv.org/html/2503.19779v2
- Fireworks.ai's write-up on why Python DL stacks need graphs for CPU-side
  speed: https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning

Ferro's profile (individual kernels sub-0.1ms, backward = 80% of step) is the
exact regime where the 9.6 -> 3.4 us-per-kernel effect applies. A transformer
step at batch=8 seq=128 d_model=256 issues hundreds of pointwise/gemm/backward
kernels; if even ~200 launches x ~5 us of exposed host round-trip exist per
step, that alone is ~1 ms - comparable to our entire measured gap.

### 1.3 cudarc support status

No raw sys calls needed. cudarc 0.19.9 ships first-class graph support in the
safe API (`cudarc::driver::safe`):

- `CudaStream::begin_capture()` -> enters capture mode on the stream
- `CudaStream::end_capture()` -> returns a `CudaGraph`
- `CudaStream::capture_status()` -> check capture state
- `CudaGraph::launch(...)` on a stream; `CudaGraph::upload()` instantiates the
  executable graph; `cu_graph()` / `cu_graph_exec()` give raw handles as an
  escape hatch
- Source: https://docs.rs/cudarc/latest/cudarc/driver/safe/struct.CudaGraph.html
  (source at `driver/safe/graph.rs`)

The known sharp edge is issue #501 ("stream-capture-safe API", opened Dec 2025,
still open): the ordinary launch path performs hidden event tracking /
synchronization that can break or pollute stream capture. The maintainer points
at `CudaContext::disable_event_tracking()` as the existing off-switch; PR #594
(Aug 2026) adds capture-scoped graph memory pools.

- Issue: https://github.com/chelsea0x3b/cudarc/issues/501
- disable_event_tracking:
  https://docs.rs/cudarc/latest/cudarc/driver/safe/struct.CudaContext.html#method.disable_event_tracking

Practical recipe for ferro: allocate all step tensors up front (static shapes),
call `disable_event_tracking()`, run one warmup step normally, then wrap
`begin_capture` / [forward+backward+optimizer kernels] / `end_capture`, upload,
and `launch` per step thereafter. Anything shape-dynamic (loss DtoH readback)
stays outside the graph.

## 2. Pinned memory + async copies in cudarc 0.19.x

Background: `cudaMemcpyAsync` from pageable host memory silently degrades to a
synchronous staged copy; only page-locked (pinned) buffers get true async
DMA that overlaps with kernels.

- Stack Overflow on pinned-vs-pageable async semantics:
  https://stackoverflow.com/questions/41287002/cuda-streams-are-blocking-despite-async
- NVIDIA forum thread confirming the same:
  https://forums.developer.nvidia.com/t/cudamemcpyasync-unexpected-behaviour-while-using-cudastreamnonblocking/61719

Concrete 0.19.x safe-API names (verified against
https://docs.rs/cudarc/latest/cudarc/driver/safe/struct.CudaStream.html):

- Host allocation: `CudaContext::alloc_pinned::<T>(n)` and the `PinnedHostSlice`
  type; `clone_htod` / `memcpy_ftod` accept `[T]`, `Vec<T>`, or
  `PinnedHostSlice<T>` (pinned source enables true async HtoD).
- Copies: `stream.memcpy_htod(...)`, `memcpy_dtoh(...)`, `memcpy_dtod(...)`,
  plus allocating `clone_htod` / `clone_dtoh` variants. All are async with
  respect to the host and ordered on the stream.
- Events: `stream.record_event(None)` -> `CudaEvent`; `event.wait(stream)` makes
  another stream wait; `stream.synchronize()` joins the host.
- Streams: `ctx.new_stream()`, `stream.fork()`, `stream.join()`;
  `CudaContext::default_stream()`.

Pattern for ferro: a second non-default stream fed by pinned staging buffers,
with `record_event` after the last forward kernel of step N and `wait` before
DtoH of metrics/losses, so logging traffic never serializes the compute stream.
Batch/label upload at step start uses a double-buffered `PinnedHostSlice`.

## 3. What torch.compile actually delivers on small models

`mode="reduce-overhead"` is literally CUDA Graphs under Inductor (plus fusion);
`fullgraph=True` forbids graph breaks. Concrete published numbers:

- "Accelerating Generative AI with PyTorch II: GPT, Fast": a GPT-style decode
  goes to **107 tok/s** via `torch.compile(mode="reduce-overhead",
  fullgraph=True)` + static KV-cache, dominated by removing CPU launch
  overhead - the same bottleneck ferro has.
  https://pytorch.org/blog/accelerating-generative-ai-2/
- PyTorch 2.0 launch materials report ~30-2x range on TorchBench with training
  averages around +43% on A100, and inference roughly +20% typical on small
  models (third-party measurement: ~22% average inference gain):
  https://pyimagesearch.com/2023/03/27/whats-new-in-pytorch-2-0-torch-compile/
  https://medium.com/@FrancescoZ/is-pytorch-2-0-faster-f4d2256cf2e9
- stas00/ml-engineering collects compile speedup data and notes gains grow as
  models shrink because overhead fraction rises:
  https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md
- PyGraph ablation (above) isolates the CUDA-graph share of compile's win at
  18-23% for small-workload steps: https://arxiv.org/html/2503.19779v2

Takeaway: for a small transformer, the majority of torch.compile's advantage
over eager is (a) elementwise-chain fusion and (b) graph replay removing
per-op host cost. Ferro is already doing (a) in Wave 4; (b) is the Wave 5
lever, and it does not require a compiler - just static shapes and stable
buffer addresses.

## 4. cudarc multi-GPU / NCCL status (2026)

As of cudarc 0.19.9 (Aug 2026), NCCL is fully bound:

- Crate table lists NCCL with dynamic-load, dynamic-link, and static-link
  support; NCCL 2.28.3 bindings; module `cudarc::nccl` re-exports a safe API
  built around `Comm`.
  https://crates.io/crates/cudarc
  https://docs.rs/cudarc/0.19.9/cudarc/nccl/index.html
  https://docs.rs/cudarc/0.19.9/cudarc/nccl/safe/index.html
- Alternative crates: `baracuda-nccl` (safe RAII communicators + collectives)
  exists but is redundant given cudarc's own bindings:
  https://docs.rs/baracuda-nccl
- NCCL collectives are also CUDA-graph capturable (PyTorch/NVIDIA note above;
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html),
  relevant if ferro ever goes data-parallel.

Status for ferro today: single-GPU workload means multi-GPU is not a Wave 5
item, but the capability is present and unblocks future DDP without leaving
cudarc.

## 5. Tensor-core GEMM via cuBLASLt from Rust

cudarc 0.19.x includes a `cublaslt` module with a safe wrapper
(`CudaBlasLT`) exposing `new()`, `matmul(...)`, plus builder-style knobs
`matrix_type`, `compute_type`, `workspace`, `stream`. The `half` feature pulls
in bf16/f16 element types, matching cuBLASLt's `CUDA_R_16BF` paths.

- https://docs.rs/cudarc/0.19.9/cudarc/cublaslt/index.html
- https://docs.rs/cudarc/0.19.9/cudarc/cublaslt/safe/struct.CudaBlasLT.html

What cuBLASLt buys over plain cuBLAS SGEMM (what ferro uses now):

- Tensor cores: fp32 SGEMM cannot hit tensor cores; bf16/fp16 inputs with fp32
  accumulate can. On Ampere (RTX 3090), peak bf16 TC throughput is ~2x fp32
  FFMA peak (~142 TFLOPS vs ~35.6 TFLOPS dense). Ampere whitepaper:
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
  (Ampere GA102 whitepaper for 3090 numbers).
- Heuristics: `cublasLtMatmulAlgoGetHeuristic` picks near-optimal algos per
  shape; results cache well because ferro's shapes are static.
- Fused epilogues: bias add + activation inside the GEMM removes separate
  pointwise launches - synergy with Wave 4 fusion.

Realistic expectation for our shapes ([1024x256]@[256x256] etc.): these GEMMs
are tiny (134 MFLOP), memory-bandwidth-bound, and already sub-0.1 ms. Switching
to bf16 tensor-core matmul will NOT multiply throughput by 2x here - the win is
(a) halved bytes moved through the memory system for weight reads/writes
(activations stored bf16), typically worth 1.3-1.7x on bandwidth-bound GEMMs,
and (b) freeing launch slots. Treat mixed precision as a Wave 5+ item behind
graphs/fusion, and gate it on numerics parity tests (ferro's existing diff-
against-torch harness).

Plain-cuBLAS caveat: classic cublas has no fp32-in/bf16-in mixed modes; Lt is
the route (see https://stackoverflow.com/questions/79481162/ ).

## 6. Prior art: burn, candle, dfdx

- **burn** (Tracel AI): the most directly relevant design. Its Fusion backend
  decorator intercepts streams of eager tensor ops and JIT-compiles fused
  custom kernels ("Optimal Performance without Static Graphs by Fusing Tensor
  Operation Streams" - up to 78x on WGPU for a fused GELU chain):
  https://burn.dev/blog/fusion-tensor-operation-streams . Its CubeCL matmul
  engine reaches near-cuBLAS performance across backends with autotune +
  tensor cores: https://burn.dev/blog/sota-multiplatform-matmul and autotune:
  https://burn.dev/blog/autotune-for-gpu-kernels . Steal: op-stream interception
  at the Backend trait boundary (ferro's named-kernel Backend trait is exactly
  this seam) and autotune caching keyed on shape signature.
- **candle** (Hugging Face): inference-first; thin cudarc usage, no autograd,
  no graph capture - its perf comes from hand-written kernels and quantization.
  https://github.com/huggingface/candle . Steal: nothing for training perf,
  but its kernel organization per-op is clean reading.
- **dfdx**: same author as cudarc; compile-time-shaped tensors feeding nvrtc
  codegen. Largely dormant, but demonstrated Rust-native JIT kernel generation
  from type-level shapes - philosophically what ferro's generated-nvrtc fusion
  already does. https://github.com/coreylowman/dfdx
- None of the three captures CUDA graphs today; ferro would be early here,
  which is fine - cudarc's safe CudaGraph API makes it low-cost to try.

## 7. PyTorch Lightning: training-loop-level optimizations

Lightning's Trainer is worth studying separately from torch.compile because
most of its speed work is loop plumbing, not compiler magic - exactly the
category ferro can adopt backend-agnostically.

### 7.1 Precision plugins

- `precision="16-mixed"` wraps the forward in `torch.autocast` (fp16) and owns
  a `GradScaler`: loss is scaled before backward, gradients are unscaled
  before clipping/optimizer step, and the step goes through
  `scaler.step/scale` so inf/NaN steps are skipped. Docs:
  https://lightning.ai/docs/pytorch/stable/reference/common/precision_basic
- `precision="bf16-mixed"` uses autocast with bfloat16 and NO GradScaler -
  bf16 has fp32-range exponent, so dynamic loss scaling is unnecessary. This
  matches what ferro would do via cuBLASLt (section 5): bf16 storage/compute,
  fp32 master weights and accumulate.
- Lightning claims "up to +3X speedups" from mixed precision on tensor-core
  GPUs (bandwidth + TC effects combined):
  https://lightning.ai/docs/pytorch/stable/reference/advanced/speed
- Semantics to copy into ferro: autocast scope around forward only; grads,
  params, and optimizer state stay fp32; scale/unscale brackets backward and
  step. Skip-step-on-inf behavior belongs in the optimizer, not the autograd.

### 7.2 Gradient accumulation and clipping

- `Trainer(accumulate_grad_batches=K)` runs K forward/backward passes per
  optimizer step and divides the loss by K (documented on LightningModule /
  training tricks: https://lightning.ai/docs/pytorch/1.4.4/advanced/training_tricks.html ).
  Purely a loop-level change: no kernel changes, but it amortizes optimizer
  cost and host sync points over K batches - relevant to ferro because our
  device-resident optimizer still costs launches that accumulation would
  dilute.
- `gradient_clip_val` + `gradient_clip_algorithm` ("norm" or "value") apply
  clipping between unscale and step. Global-norm clip requires an all-reduce
  of grad norms under DDP but is one reduction locally.
  https://lightning.ai/docs/pytorch/stable/common/trainer.html
- Both are training-loop features: applicable to ferro verbatim regardless of
  backend.

### 7.3 torch.compile integration

- Lightning exposes compile through `LightningModule.compile()` / docs page
  "Compile" (wraps the module in torch.compile before fitting; pairs with
  `precision="bf16-mixed"`):
  https://lightning.ai/docs/pytorch/stable/reference/advanced/compile
- The interesting part is ordering: compile happens once before the loop, then
  Lightning never mutates model structure mid-run (no graph breaks from
  framework hooks). Lesson for ferro: graph capture (Wave 5 item 1) requires
  the step function to be structurally frozen - no allocation, no shape
  branching - which is a loop-design constraint, not a CUDA feature.

### 7.4 Dataloader defaults and host-side pitfalls

- Speed guide prescribes `num_workers>0` and `pin_memory=True` for GPUs, and
  preloading data to RAM when the dataset fits:
  https://lightning.ai/docs/pytorch/stable/reference/advanced/speed
- Explicit anti-patterns, all host-roundtrip hazards: never call `.item()`,
  `.numpy()`, or `.cpu()` in the hot loop ("Lightning takes a great deal of
  care to be optimized for this"); don't call `torch.cuda.empty_cache()`;
  avoid re-transferring tensors to device every batch.
  https://lightning.ai/docs/pytorch/stable/reference/advanced/speed#item-numpy-cpu
- `optimizer_zero_grad` override with `set_to_none=True` is default on
  torch>=2.0 - zeroing by writing None avoids a full memset pass over grads.

### 7.5 Classification: loop-level vs torch-specific

| Lightning mechanism | Category | Ferro-applicable? |
|---|---|---|
| GradScaler bracketing of bwd/step | loop-level (CUDA-specific detail) | yes, if we add fp16 |
| bf16 autocast, no scaler | loop-level + backend support | yes (cuBLASLt path, section 5) |
| accumulate_grad_batches | pure loop-level | yes, trivially |
| gradient_clip_val/algorithm | pure loop-level | yes, as a fused nvrtc reduction |
| compile-before-fit ordering | loop discipline | yes - constrains graph capture design |
| pin_memory + num_workers guidance | host-level | yes (section 2 pinned staging) |
| .item()/empty_cache bans | host-level discipline | already ferro policy; enforce in examples |

Note on published numbers: Lightning publishes qualitative claims ("up to
+3X" mixed precision, above) but no small-transformer throughput table of its
own; the concrete tok/s evidence remains the torch.compile sources in
section 3.

## 8. Wave 5 recommendations

Ranked by expected tok/s impact on our workload (backward-dominated,
launch-count-bound, batch=8 seq=128 d_model=256):

1. **CUDA-graph the full training step** (highest impact). Static-shape step
   function; preallocate all activations/gradients/params; warmup; then
   `begin_capture` -> fwd+bwd+optimizer -> `end_capture` -> `upload` ->
   `launch` per step. Requires `disable_event_tracking()` per issue #501.
   Evidence says 2-3x on launch-bound loops (NVIDIA 9.6->3.4 us/kernel);
   conservatively expect >=2x on our 5,981 tok/s baseline if host round-trips
   dominate as profiling indicates. Gate: tok/s delta + identical loss curves
   over N steps vs uncaptured path.
2. **Finish Wave 4 fusion before measuring anything else** - graphs amplify
   fusion (fewer nodes to capture, less capture-time validation) and fusion
   shrinks the captured graph. Sequence them, measure separately.
3. **Pinned-memory async copy lane**: `alloc_pinned` staging + side stream +
   `record_event`/`wait` for batch uploads and metric readbacks so nothing
   host-visible sits on the compute stream. Small absolute win (~us-scale) but
   cheap and it protects the graph path from sync edges.
4. **cuBLASLt bf16 GEMM behind a numerics gate**: swap strided-batched SGEMM
   for `CudaBlasLt::matmul` with bf16 inputs / f32 accumulate, heuristic-cached
   per static shape, optionally BIAS epilogue fused. Expect 1.3-1.7x on the
   bandwidth-bound GEMMs, not 2x; ship only after parity diffs pass.
5. **Steal burn's shape-keyed autotune cache** for our generated nvrtc fusion
   kernels (pick block sizes empirically once per shape, persist).
6. **Gradient accumulation + fused grad-norm clip (Lightning-derived, loop
   level)**: accumulate K micro-batches before the device-resident optimizer
   step and clip via one nvrtc global-norm reduction between unscale and step
   (semantics per section 7.2). No backend change; dilutes optimizer launch
   cost and composes with item 1 (clip+step live inside the captured graph).
7. **Host-side discipline pass (Lightning speed-guide derived, section 7.4)**:
   audit the hot path for any DtoH readback / `.item()`-equivalent / allocator
   churn; route metrics through pinned staging on the copy stream (section 2).
   Mostly enforcement + tests, near-zero risk.
8. **Defer multi-GPU**: cudarc NCCL bindings exist and work; revisit when there
  is a second GPU. No Wave 5 task.
