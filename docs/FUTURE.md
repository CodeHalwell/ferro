# ferro: the road from here to a world-class deep learning engine

This is the forward-looking master plan. It assumes the current state of the
tree (see Status below) and lays out every major workstream between here and
an engine that could credibly compete with PyTorch/JAX - plus the honest
assessment of which axes "best in the world" is actually winnable on.
CAPABILITY.md is the companion depth document: the mathematics and computer
science behind each workstream here, the concrete designs, and the
falsifiable acceptance gates.

## Where we are (baseline for everything below)

- Pure-Rust core: ~40 autograd ops, every one finite-difference checked and
  most cross-validated against torch numerically (values AND gradients).
  The transformer set has landed: gelu, cumsum, argmax/argmin, gather, topk,
  half-split rope, and a composed causal scaled_dot_product_attention.
- Engine hardened: single closure-based autograd mechanism, strict grad
  arity/shape/device contracts, iterative topo sort and graph teardown (100k+
  op chains), torch retain_graph accumulation semantics.
- Dispatcher: named kernels, per-device Backend trait + registry,
  device-resident storage, device broadcasting + reductions, autograd on
  device. A counting fake backend PROVES a full training loop runs with zero
  per-step uploads and two scalar downloads per step.
- Three backends: reference CPU, ferro-fastcpu (register-blocked AVX2+FMA
  matmul, ~6x naive), ferro-cuda (cuBLAS + nvrtc kernels, compiles without
  CUDA, gated GPU tests staged).
- Dtypes: F32 everywhere; F64/I64 storage + casts; I64 index tensors feeding
  embedding / integer-target cross-entropy.
- nn/optim: Linear, LayerNorm, RmsNorm, Embedding, activations (incl. Gelu),
  Sequential, cross_entropy, SGD (+momentum), Adam, AdamW. MLPs and CNNs
  train from Rust and Python.
- ferro-py: full op bindings, DLPack interop with numpy/torch (leak-free),
  training demos, everything validated against torch.

## The honest framing

PyTorch is ~2000 operators, tens of thousands of kernels, a compiler stack
(torch.compile/Inductor), a decade of numerics hardening, and an ecosystem.
Matching it feature-for-feature is a multi-year, large-team effort. "Best in
the world" is therefore two different games, and this plan plays both:

1. Parity game: close the correctness/performance gap on the workloads that
   matter (transformers, convnets), measured continuously against torch.
2. Differentiation game: win outright on axes where a from-scratch Rust
   engine has structural advantages - safety, embeddability, binary size,
   determinism, edge/wasm targets, and a compiler designed in from the start
   rather than bolted on.

Everything below is tagged [P] parity or [D] differentiator, with a rough
size: (S) days, (M) weeks, (L) months, (XL) multi-month/team-scale.

## 1. Run it on a GPU (the single most informative next step)

- [P/S] Validate on real hardware: `cargo test -p ferro-cuda` on a GPU box
  runs the staged end-to-end resident training loop. Everything is written;
  nothing has executed on silicon yet. Expect a debugging round.
- [P/M] GPU benchmark suite vs torch: matmul, elementwise chains, the
  training loop. Establish the honest baseline gap.

## 2. Correctness and semantics parity [P]

- (M) N-D and batched matmul with broadcasting batch dims (bmm exists;
  general einsum-lite semantics do not).
- (L) In-place operations + storage version counters: LANDED 2026-08.
  Version counters gate mutation (backward errors loudly on stale saved
  inputs); storage sits behind a per-cell RwLock; public in-place ops
  (zero_/fill_/add_/sub_/mul_/div_/add_scalar_/mul_scalar_/copy_from) are
  layout- and autograd-gated; optimizers step through fused in-place
  kernels (see 2.5/3 updates below). Remaining from the original scope:
  in-place through strided views, and in-place ops that rebind live graph
  nodes (torch's grad_fn rewrite) - the current public API refuses
  history-carrying tensors instead.
- (L) Autograd maturity: backward(grad) for non-scalar roots; create_graph /
  double backward (backward closures currently compute detached - they must
  optionally compose recorded ops); gradient hooks; anomaly detection mode.
- (M) Dtype completion: f16/bf16 storage + casts LANDED 2026-08 (raw-bit
  Storage::F16/BF16, RNE conversions in the half module, byte-exact
  safetensors IO - the checkpoint-weight path for M3). Remaining: f64
  autograd; integer arithmetic ops; an explicit type-promotion policy
  (currently: strict f32-only math, explicit casts).
- (L) Views with autograd: as_strided family, aliasing semantics, narrow/
  slice/index_put. Today strided views materialize on read and device views
  fall back to host.
- (XL, ongoing) Operator long tail, prioritized by workload: transformer set
  remainder (fused softmax; scatter and exact-erf gelu have landed), vision
  set (conv variants, interpolate; conv2d is already im2col+GEMM riding the
  swappable matmul kernel), then breadth. gelu/rmsnorm/rope/cumsum/
  topk/argmax/argmin/gather and masked causal attention landed 2026-07.
  Each op stays one-file/one-agent parallel work.
- (M) Torch parity fuzzer: property-based random-shape/dtype op tests diffing
  ferro vs torch through DLPack, run in CI. The single highest-leverage
  correctness investment - it turns "validated on examples" into "validated
  on distributions".

## 2.5 CUDA-graph capture of full training steps [P]

Chain-level capture works (CapturedChain, wave 5b): one chain launch
replays in ~4.4 us vs 12-18 us eager for the same kernel (measured,
bench_chain --n 1024..16384). The optimizer half of the step-capture
blocker fell 2026-08: params and SGD/Adam/AdamW state now mutate in place
with stable buffer addresses (one fused kernel per param per step, scalars
as kernel arguments, zero per-step host traffic - proven by counting
backends in tests/optim_device.rs), and `write_dev_from_host`/`copy_into`
overwrite a buffer without moving it (the batch-upload seam capture needs;
copy_into's dtod-clone no-op bug is fixed). Still blocking full
fwd+bwd+optimizer capture: every backward INTERMEDIATE (activations,
grads) gets a fresh address each step - that needs the buffer pool /
static memory planner (CAPABILITY.md 4.2-4.4). Expected win at our
profile (loss_bwd = 72% of step, launch-gap dominated): NVIDIA-published
9.6 -> 3.4 us per kernel effective; our own measurement shows a 3-7x
reduction per chain.

**Capturable AdamW backend primitive LANDED 2026-08 (PR #14):** timestep
lives on-device (`t = [step, bc1, bc2]`), a 1-thread increment kernel bumps
`step` and recomputes bias correction once per step, and the elementwise
AdamW kernel reads the precomputed scalars - so a captured step advances the
correction under replay instead of freezing it (mirrors PyTorch
`capturable=True`). Proven by a graph-replay test (replay 6x, timestep
1->6, params track eager to 1e-5). Backend layer only.

- **(M) G9 - route the public `AdamW` optimiser through the capturable
  path.** Blocker raised in PR #14 review (Codex P1): `ferro-core::AdamW::
  update` still advances host `self.t`, computes host bc1/bc2, and calls the
  frozen `adamw_step_dev`, so production AdamW is NOT capturable end-to-end
  despite the backend primitive existing. Needs: device-resident optimiser
  state (timestep buffer owned by the optimiser), a capture-aware `update`
  that selects `adamw_step_capturable_dev` + `scalar_increment_dev` during a
  capture window, and lifting those methods out of `pub(crate)`. Stretch:
  on-device `lr` so LR scheduling works under replay (currently lr is baked
  host-const, fixed per captured graph - re-capture to change).

  **G9 LANDED 2026-08 (PR #15):** public `AdamW.capturable()` routes the
  production optimiser through `adamw_step_capturable_dev` +
  `scalar_increment_dev`. CUDA-uniform-device guard (CPU/mixed -> silent host
  fallback), device buffer `[step,bc1,bc2]` is timestep authority (seeded from
  `self.t` so warm-up-then-capture preserves the counter), `snapshot`/`restore`
  round-trip through it, and `restore` lands moments on the param device.
  **Known limitation (Codex P2, deferred):** the device timestep is stored f32,
  so `scalar_increment` (`step += 1.0f`) stops advancing past 2^24 (~16.7M
  steps) - the counter and bias correction silently freeze. Fix is an i32 step
  field in the buffer (mixed int/float layout across kernel + seeding +
  snapshot); distant enough to defer but must precede any >16M-step run.

## 3. Performance: CPU [P]

- (M) Memory: host buffer pool LANDED 2026-08 (pool.rs: thread-local
  size-classed freelists recycling storage on drop; an MLP training step
  performs zero fresh host allocations after warmup - CAPABILITY.md 4.2's
  G5 host half, proven by tests/pool_zero_alloc.rs). accumulate_grad adds
  in place when the stored grad is provably unshared, and in-place
  optimizer steps keep params/state storage stable. Remaining: the device
  caching allocator (see 4 below) and pooling the ops_ext host paths.
- (M) Elementwise: SIMD + multithreaded kernels (fastcpu treatment beyond
  matmul); strided kernels that skip materialization.
- (M) Fusion of elementwise chains at the record_fn layer (peephole first,
  compiler later - see 5).
- (M) conv2d is lowered via im2col+GEMM through the swappable matmul
  kernel (so fastcpu accelerates it); remaining here: pooling/reduction
  parallelism and a blocked direct conv for small kernels.
- (S) Continuous benchmarks (criterion) with a torch comparison harness and
  tracked regressions.

## 4. Performance: GPU [P]

- (L) Memory caching allocator (the thing that actually makes GPU training
  fast; cudaMalloc per op is a non-starter at scale).
- (L) Streams and async execution; pinned host staging buffers; overlap of
  transfer and compute. Today every op is synchronous.
- (M) Real reduction kernels (tree/block reductions - current ones are
  correctness-only), occupancy-tuned elementwise, fused epilogues via
  cuBLASLt.
- (L) conv/attention: cuDNN bindings or implicit-GEMM kernels; a fused
  attention kernel is the marquee target.
- (L) Multi-GPU: NCCL bindings, DDP-style gradient all-reduce.
- (M) CUDA graphs for step capture (ferro's immutable graphs are a natural
  fit - potential differentiator).
- (L) [D] Portable backends through the same Backend trait: Metal, ROCm/HIP,
  and wgpu/WebGPU (the wgpu backend doubles as the browser/edge story).

## 5. The compiler layer [P+D] - where "best" is decided

Modern engine performance is decided by graph capture + fusion, not by
hand-written kernels. ferro's advantages: the graph already exists (record_fn
nodes), tensors are immutable (no aliasing analysis nightmares), and Rust is
a good substrate for an IR.

- (M) Meta kernels: shape/dtype-only execution for tracing and shape
  inference (the dispatch enum already reserves the concept).
- (L) Graph capture: a lazy mode where ops build an IR instead of eagerly
  executing; replay with caching keyed on shapes.
- (XL) Fusion compiler: elementwise/reduction fusion into generated kernels
  (nvrtc on GPU, cranelift or generated Rust on CPU). This is the
  torch.compile/Inductor analogue and the largest single win available.
  STATUS: the pointwise-chain engine EXISTS and is measured — `plan_fusion` →
  `FusedChain::resolve` → `chain_dev` runs `relu(x)*y+z` as one nvrtc kernel at
  a proven **2.1× over the unfused 3-kernel path** on the 3090 (docs/FUSION_3090.md,
  `bench_chain`). It is now reachable from Python via `Tensor.fuse()` /
  `Tensor.fusion_launches()` (collapses launches 3→1, numerically exact).
  REMAINING (the actual next task): `.fuse()` re-plans on every call so it is
  currently ~0.68× (slower than eager) despite the 2× kernel — needs a
  **compile-once fused callable** (plan/resolve once, replay the chain, ideally
  over the existing `capture_chain`/`replay` CUDA-graph seam) to expose the
  kernel win at the Python level. Two planner bugs were fixed getting here:
  same-shape elementwise mislabelled as MatMul, and a `run_host` operand
  off-by-one (see docs/FUSION_3090.md).
- (L) Whole-step compilation: capture forward+backward+optimizer as one
  graph; combined with CUDA graphs this can beat eager torch meaningfully.

## 6. Training stack completeness [P]

- (M) nn: MultiHeadAttention (RoPE + causal), a pre-norm TransformerBlock,
  and Dropout (Philox-backed, train/eval mode) have landed - a one-block LM
  trains and greedy-decodes its target in tests. Remaining: Conv2d module
  with bias, parameter initialization registry.
- (M) optim: AdamW, LR schedulers (StepLr/ExponentialLr/CosineWithWarmup
  with the set_lr driving seam), global-norm grad clipping, and
  device-resident optimizer state have all landed; steps are fused and
  in-place as of 2026-08. Remaining here: parameter groups (per-group lr /
  weight decay) and optimizer-state offloading policies.
- (M) Mixed precision: autocast policy + grad scaler once f16/bf16 land.
- (M) Serialization: safetensors read/write and named state_dict save/load
  on the Module trait (strict torch semantics) landed 2026-07, byte-validated
  against the reference implementation - the model-import path for M3 is
  open end to end.
- (M) Data: a minimal DataLoader (batching, shuffling, parallel prefetch).
- (L) Distributed data parallel once NCCL exists.

## 7. Python and ecosystem [P]

- (M) Binding codegen: a macro/table so every core op gets a PyTensor method
  automatically - the hand-written binding lag is already the known failure
  mode.
- (M) Zero-copy DLPack (export without the copy; the storage refactor now
  supports a stable pointer), numpy __array_interface__.
- (L) A torch-shaped shim module (ferro.nn/ferro.optim mirroring torch names)
  so small torch scripts port by changing an import.
- (M) Packaging: manylinux/macos/windows wheels via maturin CI; abi3.
- (M) Expose device API to Python (to_device, ferro.cuda.is_available) with
  ferro-cuda compiled in behind a feature flag.

## 8. Engineering foundations [P]

- (M) CI: GitHub Actions matrix (test, clippy, fmt, docs), GPU runner for the
  gated suites, the torch-parity fuzzer, benchmark tracking with regression
  alerts.
- (M) Unsafe audit: the DLPack and CUDA FFI surfaces under miri/asan where
  applicable; document every invariant.
- (M) rustdoc + an mdbook (architecture, how to add an op, how to add a
  backend - the agent-parallelizable recipes are already written, promote
  them).
- (S) Versioning/release discipline once anything depends on this.

## 9. Differentiators: where ferro can be genuinely best [D]

- Host-side overhead (MEASURED, see docs/HOST_OVERHEAD.md): ferro's leading
  structural edge. On tiny CPU tensors (kernel ~free, so wall time = host
  orchestration) ferro is a repeatable ~3x faster per-op dispatch and >=2.4x
  faster on a depth-8 autograd step than eager torch 2.6. Cause: no GIL,
  monomorphised static dispatch, allocation-deterministic core -- the eager
  overhead torch.compile exists to remove, which ferro never pays. This is the
  thesis: ferro's edge is everything OUTSIDE the kernel (dispatch, autograd
  graph build/traverse, optimiser-step orchestration, capture/replay), where a
  memory-safe Rust core beats torch's C++/Python eager path. It does NOT extend
  to device throughput (matmul/elementwise read parity, GPU_BASELINE_3090.md);
  it is a fraction-of-wall-time win, largest for small-tensor / long-graph /
  high-step-count training and inference loops. Compound it with fusion (5) to
  stop the device side handing parity back.
- Embeddability: no Python, no runtime, one static binary.
  library measured in single-digit MB that links into anything (games,
  robotics, safety-critical). Torch cannot play here.
- wasm/edge: the wgpu backend + wasm32 target = training and inference in the
  browser from the same codebase.
- Safety: memory-safe kernels, no segfaults-by-design, auditable FFI
  boundary. Sell it where correctness certification matters.
- Determinism-by-default: bitwise-reproducible training runs as a first-class
  mode (immutable graphs make this tractable).
- Compile-time shapes: an optional typed-tensor API (const-generic dims) for
  users who want shape errors at compile time - dfdx proved appetite exists.
- Transparency: the whole engine is readable. Keep it that way; it is a
  feature.

## Milestones (sequenced, each independently demonstrable)

- M1: GPU validation - the staged resident training loop passes on real
  hardware. (Blocked only on access to a GPU box.)
- M2: GPU perf floor - caching allocator + streams + real reductions;
  benchmark suite reporting the gap vs torch eager.
- M3: Transformer inference - load a small real LLM (e.g. a TinyStories-class
  model) from safetensors and generate tokens correctly. The prerequisites
  (transformer op set, serialization, attention/block modules) landed
  2026-07, and ferro-fastcpu's char_lm example proves the full pipeline
  (train -> save -> reload -> generate) on a toy model. The prerequisite
  list closed 2026-08: exact-erf gelu, f16/bf16 weight loading,
  grouped-query attention (MultiHeadAttention::with_kv_heads, HF checkpoint
  shapes), learned positional embeddings, and a dependency-free byte-level
  BPE tokenizer (ferro-tokenizer, validated token-for-token against the
  real GPT-2 vocab). Remaining: the end-to-end demo itself - map a real
  checkpoint's names onto the modules and generate. This is the
  credibility milestone.
- M4: Training parity demo - MNIST/CIFAR conv training on GPU within 2-3x of
  torch eager wall-clock.
- M5: Compiler MVP - captured, fused forward+backward for an MLP beating
  ferro's own eager mode >2x; the foundation for everything after.
- M6: The differentiator release - wasm/wgpu inference demo in a browser +
  single-binary embedded inference, published benchmarks and wheels.

Ordering rationale: M1/M2 derisk the platform, M3 buys credibility and forces
the op set, M4 proves training, M5 is the long-term performance play, M6 is
the positioning play. Workstreams 2, 6, 7, 8 run continuously alongside as
parallel, agent-sized tasks - the one-op-per-file and one-trait-per-backend
seams were built precisely so this scales horizontally.
