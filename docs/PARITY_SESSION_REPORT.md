# ferro Parity Implementation — Session Report

*23 August 2026 · 5 parallel agents + integration · all suites green*

## Verification (real command output)

| Suite | Result |
|---|---|
| `cargo test -p ferro-core` | **269 passed, 0 failed** |
| `cargo test -p ferro-fastcpu` | **12 passed, 0 failed** |
| `cargo check -p ferro-cuda` | clean (no CUDA installed, as required) |
| `cargo check ferro-py` (standalone) | clean |

Nothing committed — per CLAUDE.md, commits wait for explicit ask.

## What landed

### Phase 0 — review fixes
- DLPack import validation: negative ndim/dims rejected, numel overflow-checked,
  stride bounds validated before any pointer use (`ferro-py/src/dlpack.rs`)
- Fake-backend registration in `tests/dispatch.rs` under poison-tolerant Mutex
- CUDA host-slice fallbacks return `Result` instead of panicking
- `requires_grad_` leaf-only enforcement with test coverage
- CUDA `reduce_dev` parallelised (multi-block); u32 launch ceiling guarded

### Phase 2 — op coverage (all with grad_check + value tests)
avg_pool2d, batch_norm, group_norm, layer_norm, logsumexp, pad,
scatter_add, silu, tri (+ existing conv2d/max_pool2d/etc. retained)

### Phase 3+4 — modules & optimisation
- New `modules.rs`: Module trait, containers, init schemes
- nn.rs extended; optim.rs extended with AdamW/SGD-nesterov/schedulers/clip

### Phase 1 — GPU bridge groundwork
Device-resident backward seeding path in core (CUDA side seam completed by WS-E).

## One integration bug (mine)
The new `op_silu.rs` test computed sigmoid(-1) as 0.2689 (that's sigmoid(+1));
silu(-1) = −0.731. Library code was correct; fixed the test constant.
Lesson recorded: `-1.0f32.exp()` parses as `-(1.0f32.exp())` — write `f32::exp(-1.0)`.

## Round 2 (same session, 4 agents)

| Suite | Result |
|---|---|
| `cargo test -p ferro-core` | **293 passed, 0 failed** |
| `cargo test -p ferro-fastcpu` | 12 passed |
| `cargo check -p ferro-cuda` | clean |
| ferro-py | cargo check clean; maturin smoke suite 13/13 in isolated venv |

- ferro-py: device API (`.to/.cpu/.cuda`, device kwarg on factories), reflected
  ops + scalar LHS, int/slice/negative/Ellipsis indexing (detached copies),
  negative-dim normalisation across dim-taking ops. Known limits: indexing has
  no autograd; no boolean/advanced indexing; DLPack export still CPU-only.
- amp.rs: f32-master-weights AMP scaffold, bf16 round-to-nearest-even casts,
  quantized_matmul with FD-verifiable backward, straight-through master grads;
  fused_ops.rs: bias_add_activation + residual_layernorm with fused backwards.
  testkit gained grad_check_strict.
- data.rs: Dataset/Sampler/CollateFn/DataLoader with std::thread workers,
  bounded channel backpressure, deterministic id-reordering (multi-worker
  output bit-identical to single-worker given seed).
- checkpoint.rs: safetensors + JSON sidecar, atomic temp+rename writes,
  strict load, version gating; resume-mid-training trajectory tests.

## Round 3 (GPU machine wave)

| Suite | Result |
|---|---|
| `cargo test -p ferro-cuda` on RTX 3090 | **25/25** (19 unit + 6 real-GPU integration) |
| `cargo test -p ferro-core` | 293 passed, 0 failed |
| GPT-2-small example | loss 3.33 -> 0.64 @300 steps; resume works across process |
| CNN classifier example | 100% eval accuracy, loss 0.0029 |
| Benchmark (CPU vs torch CPU) | ferro ~3.7% of torch eager CPU |

- GPU tests had been silently skipping: driver present but no CUDA toolkit
  (nvrtc/cuBLAS DLLs missing). Fixed environmentally via NVIDIA pip wheels in
  %LOCALAPPDATA%/Temp/cuda-rt (ephemeral - persist or install toolkit later).
  Zero CUDA kernel bugs found; parallel reduce_dev correct on hardware.
- benchmarks/: bench_transformer.rs + bench_torch.py twin harness (identical
  param count 1,313,536). Honest result: ferro CPU is far behind torch CPU;
  CUDA path blocked by missing i64 device transfer for token ids - next fix.
- examples/train_gpt2_small.rs + train_classifier_cnn.rs prove the full stack
  end-to-end incl. checkpoint/resume. Known limit: optimizer moment buffers
  are not checkpointed yet.

## Round 4 (i64 device transfer + CUDA end-to-end)

| Suite | Result |
|---|---|
| `cargo test -p ferro-core` | **293 passed, 0 failed** |
| `cargo test -p ferro-cuda` on RTX 3090 | **27/27** (19 unit + 8 GPU integration) |

- i64 index tensors are now device-resident: new `Storage::DeviceI64` variant,
  `Backend::alloc_i64_from_host/copy_i64_to_host/gather_rows_dev` seams, a
  `CudaBufI64` buffer type and an nvrtc gather kernel; embedding runs fully
  on-device (forward bitwise-equal to CPU, grads identical).
- Device-sticky convention for host-composed ops: softmax, log_softmax, bmm,
  gelu, rope, RmsNorm/LayerNorm eps and attention scale now return to the
  input's device instead of silently dropping to cpu. One device test updated
  to the new contract.
- Bench now moves parameters to the target device before training.
- **First real CUDA benchmark numbers** (batch=8 seq=128 d_model=256):
  ferro on RTX 3090 ~2,440-2,630 tok/s vs torch CPU 74,647 tok/s. GPU
  utilisation ~0% during the run - profiling shows embedding+attention
  dominated by per-op host round-trips in still-host-composed ops (softmax,
  gelu, reductions). Matmul itself is fast (20x1024^3 GEMMs = 24 ms via
  cuBLAS); the gap is op-graph overhead, not kernels. Next lever: move
  softmax/gelu/reductions onto nvrtc kernels and batch launches.

## Round 5 (device kernels + optimizer-state checkpoints)

|| Suite | Result |
||---|---|
|| Benchmark (cuda, warmup 10 / timed 30) | ferro **3,554 tok/s** on RTX 3090 |
|| Benchmark (cpu, same config) | ferro 3,411 tok/s vs torch CPU 79,917 |

- softmax/log_softmax now run as two-pass row-statistics nvrtc kernels and
  gelu through the unary device kernel (`ferro-cuda/src/kernels.rs`), with
  GPU forward+grad tests against CPU. This removed the host round-trips that
  capped the CUDA run at ~2,440-2,630 tok/s in Round 4.
- Optimizer state is checkpointed: `OptimizerState` trait implemented by
  Sgd/Adam/AdamW; `Checkpoint::from_module_with_optim` snapshots parameter +
  moment buffers into a separate `optimizer.safetensors`, restored strictly
  via `load_optim_into`. The Round 3 "moments not checkpointed" limit is
  closed.

## Remaining for Tier 1 (per PARITY_ROADMAP.md)
- DLPack CUDA producer (export still CPU-only)
- Indexing autograd (indexing returns detached copies today)
- Distributed DDP, graph compiler prototype (Tier 2)
