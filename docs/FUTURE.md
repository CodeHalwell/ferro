# ferro: the road from here to a world-class deep learning engine

This is the forward-looking master plan. It assumes the current state of the
tree (see Status below) and lays out every major workstream between here and
an engine that could credibly compete with PyTorch/JAX - plus the honest
assessment of which axes "best in the world" is actually winnable on.

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
- (L) In-place operations + storage version counters. Everything is immutable
  today; optimizers reallocate every step. Version counters (torch _version)
  are the prerequisite for safe mutation, and must land BEFORE in-place ops.
- (L) Autograd maturity: backward(grad) for non-scalar roots; create_graph /
  double backward (backward closures currently compute detached - they must
  optionally compose recorded ops); gradient hooks; anomaly detection mode.
- (M) Dtype completion: f16/bf16 storage + casts; f64 autograd; integer
  arithmetic ops; an explicit type-promotion policy (currently: strict
  f32-only math, explicit casts).
- (L) Views with autograd: as_strided family, aliasing semantics, narrow/
  slice/index_put. Today strided views materialize on read and device views
  fall back to host.
- (XL, ongoing) Operator long tail, prioritized by workload: transformer set
  remainder (fused softmax, scatter, exact-erf gelu), vision set (conv
  variants, pooling, interpolate), then breadth. gelu/rmsnorm/rope/cumsum/
  topk/argmax/argmin/gather and masked causal attention landed 2026-07.
  Each op stays one-file/one-agent parallel work.
- (M) Torch parity fuzzer: property-based random-shape/dtype op tests diffing
  ferro vs torch through DLPack, run in CI. The single highest-leverage
  correctness investment - it turns "validated on examples" into "validated
  on distributions".

## 3. Performance: CPU [P]

- (M) Memory: arena/pool allocator for tensor buffers; reuse gradient
  buffers across backward passes; in-place optimizer steps (after version
  counters).
- (M) Elementwise: SIMD + multithreaded kernels (fastcpu treatment beyond
  matmul); strided kernels that skip materialization.
- (M) Fusion of elementwise chains at the record_fn layer (peephole first,
  compiler later - see 5).
- (M) conv2d via im2col+GEMM or blocked direct conv (current one is naive
  7-loop); pooling/reduction parallelism.
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
- (L) Whole-step compilation: capture forward+backward+optimizer as one
  graph; combined with CUDA graphs this can beat eager torch meaningfully.

## 6. Training stack completeness [P]

- (M) nn: MultiHeadAttention (RoPE + causal) and a pre-norm TransformerBlock
  landed 2026-07 - a one-block LM trains and greedy-decodes its target in
  tests. Remaining: Dropout (needs RNG plumbing + train/eval mode), Conv2d
  module with bias, parameter initialization registry.
- (M) optim: AdamW landed 2026-07. Remaining: LR schedulers, grad clipping;
  optimizer state on device (currently host Vecs - must move for GPU
  training).
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

- Embeddability: no Python, no runtime, one static binary. Target: inference
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
  (train -> save -> reload -> generate) on a toy model. Remaining: a real
  checkpoint's architecture (learned positions or GQA, exact-erf gelu,
  f16/bf16 weights) plus a tokenizer. This is the credibility milestone.
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
