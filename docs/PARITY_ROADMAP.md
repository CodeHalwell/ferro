# ferro → PyTorch Feature Parity: Roadmap & Technical Guide

*Prepared August 2026 · Based on the deep-dive review of ferro (v. current state) and the 2026 Rust GPU ecosystem*

---

## 0. Framing: What "Parity" Realistically Means

PyTorch is ~15 years old, thousands of contributor-years. Full literal parity is a
decade of work — so this document defines parity in tiers:

| Tier | Definition | Achievable |
|---|---|---|
| **T1 — Usable** | Train a CNN/transformer on GPU with autograd, optimisers, dataloaders, save/load | ~12 months focused work |
| **T2 — Competitive** | Mixed precision, distributed training, fused kernels, torch.compile-equivalent perf on common models | +18–24 months |
| **T3 — Ecosystem** | torchvision/torchaudio equivalents, hub ecosystem, quantisation, mobile | community-scale effort |

The strategy throughout: **match the API surface users touch, not PyTorch's internals.**

---

## 1. Current State (from the deep-dive review)

**Strengths to build on**
- Clean zero-dep core with enforced invariants (structural tests!)
- Sound single-path autograd (`record_fn`, arity/shape asserted, version counters)
- Iterative topo sort + iterative Drop (deep-graph safe)
- Backend trait registry; CUDA compiles without CUDA installed
- DLPack export path balanced and double-free-safe

**Gaps that block T1**
- No Python→GPU bridge at all (ferro-py is CPU-only)
- CUDA reductions single-threaded (grid `(1,1,1)`) + `.item()` sync per backward
- Missing ops/layers/optimisers/schedulers/dataloader ecosystem
- DLPack import validation hole (must fix before any of this)

---

## 2. The Rust GPU Stack in 2026 — What Changed

Key landscape shift: NVIDIA has entered Rust-GPU officially.

| Project | Approach | Status 2026 | Fit for ferro |
|---|---|---|---|
| **cudarc** | Safe host-side driver bindings | Stable, responsive maintainer | ✅ Already your host layer — keep it |
| **cuda-oxide** (NVIDIA-adjacent) | `rustc` → NVVM IR → PTX; idiomatic safe Rust on-device | Working again after 2-yr hiatus; NVVM layer deprecated upstream though | ⚠️ Watch; PTX output portable to cudarc launch |
| **CUDA-Oxide (Nvidia official)** | Host API in cudarc style + native Rust kernels | Early/WIP | 👀 Watch closely — could become the sanctioned path |
| **Rust-CUDA (rust-gpu org)** | Rebooted, rustc-based PTX codegen | Works, small team | ⚠️ Viable but thin staffing |
| **CubeCL** | Embedded DSL JIT-compiling to CUDA/ROCm/WGPU | Production (powers Burn) | ✅ Best option for **cross-vendor** kernels |
| **wgpu/WGSL** | WebGPU compute, cross-platform incl. browser/Apple | Production-ready for compute | Optional later backend |

**Recommendation:** stay on **cudarc** for host; write new kernels as CUDA C++
(PTX modules) first — boring and fast — while prototyping **CubeCL** for a future
AMD/portable backend. Track Nvidia's official CUDA-Oxide; if it stabilises it
becomes the natural kernel language. Avoid betting the framework on rust-gpu's
on-device Rust today.

---

## 3. Phase Plan to Tier 1

### Phase 0 — Hygiene (1–2 weeks)
Fix the review findings first; everything else builds on them:
1. DLPack import validation (dims ≥ 0, overflow-checked numel, stride bounds)
2. Leaf-only `requires_grad_` assertion
3. Mutex the `tests/dispatch.rs` fake-backend registration
4. Replace CUDA host-slice `expect`s with `Result`
5. Parallelise `reduce_dev`; remove per-backward `.item()` sync (seed grad on device)

### Phase 1 — The GPU Bridge (4–6 weeks)
- `Tensor.to(device)` / `.cpu()` end-to-end through ferro-py
- Async HtoD/DtoH with pinned memory + stream capture basics
- Device-resident scalars: kill remaining implicit syncs
- Benchmark gate: ResNet-ish forward+backward within 3× of PyTorch eager

### Phase 2 — Op Coverage (3–4 months)
Priority order (what 90% of models need):
- **Ops:** conv1d/2d (+transpose), batch_norm/layer_norm/group_norm, embedding,
  attention primitives (SDPA), pooling, padding, advanced indexing,
  reductions with dims/keepdim, einsum
- **Autograd extensions:** higher-order grads where cheap (double-backward for common ops);
  hooks (tensor/module gradient hooks)
- Every op: value test vs torch + finite-difference grad_check (existing convention)
- Consider cuDNN via FFI for conv/bnorm rather than hand-rolling — everyone does

### Phase 3 — nn.Module Equivalent (6–8 weeks)
```rust
#[derive(Module)]
struct Transformer {
    attn: MultiHeadAttention,
    ffn: Linear,
}
```
- Derive-macro based modules: parameter registration via field reflection
- `Module::train()/eval()` mode propagation, parameter groups
- State dict save/load (safetensors already proven in your examples)
- Python-side mirror API so the bindings feel like `torch.nn`

### Phase 4 — Optimisers & Training Loop (4–6 weeks)
SGD(+momentum/Nesterov), Adam/AdamW, LR schedulers (cosine, step, warmup),
grad clipping, grad accumulation, AMP scaffolding (bf16/fp16 casts),
checkpoint/resume.

### Phase 5 — Data & Ergonomics (ongoing)
- Dataset/DataLoader with worker processes (rayon locally, IPC sharding for scale)
- Collate functions, samplers, prefetching
- TensorBoard/W&B logging via simple file formats
- Error messages: shape-mismatch diagnostics that name the op (your `Result` culture helps)

### Gate to declare T1
Train GPT-2-small and ResNet-50 from the public repo READMEs, unmodified,
with throughput ≥ 50% of PyTorch eager on identical hardware.

---

## 4. Path to Tier 2 (the hard part)

### 4.1 Mixed precision
- bf16 storage dtype end-to-end (easier than fp16 — no loss scaling needed first)
- TensorCore paths: cuBLASLt via FFI for matmul/conv with fp8/bf16 epilogues
- Autocast-style context: dtype promotion rules table mirroring torch's

### 4.2 Kernel fusion
This is where modern frameworks win. Options ranked by fit:
1. **Hand-fused kernels** for the canonical patterns: bias+activation,
   norm+residual, SDPA (flash-attention style tiling)
2. **CubeCL DSL** for maintainable fused ops across vendors
3. Long-term: a small "graph compiler" — trace the record_fn tape, pattern-match
   fusible chains, emit one kernel (a mini torch.compile). Your single-autograd-
   path design makes tracing *easier* for you than for PyTorch.

### 4.3 Distributed training
Realistic order:
1. **DDP equivalent**: gradient all-reduce over NCCL via FFI (cudarc doesn't cover
   NCCL; use `nccl-sys` bindings or C++ shim crate)
2. ZeRO-style optimizer state sharding later; FSDP last
3. Single-node multi-GPU first; multi-node needs an elastic launcher story

### 4.4 Compilation/caching
- Kernel cache keyed on (arch, shapes-bucket, dtype)
- CUDA graphs for static-shaped inference loops
- Memory planner: arena allocator reusing buffers across the tape (biggest real-world
  speedup per unit effort — PyTorch's caching allocator is why it feels fast)

---

## 5. What NOT to Copy

- **torch's legacy quirks**: `view` vs `reshape` ambiguity, in-place op footguns.
  Your immutability-first design is *better* — keep it and document why.
- **Eager-everything**: consider making device execution lazy-by-default earlier
  than PyTorch did; it enables fusion without a second compile mode.
- **Python-first API design**: design the Rust API idiomatically, then make the
  Python binding match torch where it reduces user friction (naming), diverge where
  torch is unsafe (in-place mutation).

## 6. Suggested Milestone Timeline (solo + occasional contributors)

| Quarter | Milestone |
|---|---|
| Q1 | Phase 0–1 done; MNIST/CIFAR on GPU from Python |
| Q2 | Phase 2 core ops; transformer block trains |
| Q3 | Phase 3–4; GPT-2-small trains; parity benchmark published |
| Q4 | AMP + fused SDPA/norm kernels; T1 declared |
| Y2 | Distributed DDP, graph compiler prototype, AMD via CubeCL |

## 7. Key References

- cudarc: github.com/coreylowman/cudarc
- cuda-oxide ecosystem map: nvlabs.github.io/cuda-oxide/appendix/ecosystem.html
- CubeCL: github.com/tracel-ai/cubecl (Burn's compute backend)
- Burn (reference architecture for Rust DL): github.com/tracel-ai/burn
- Candle (minimalist Rust tensor lib): github.com/huggingface/candle
- Rust-GPU org status (Rust All Hands 2026): news.ycombinator.com/item?id=49143096
