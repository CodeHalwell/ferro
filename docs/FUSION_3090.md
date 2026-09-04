# Elementwise Fusion: measured win on RTX 3090

The fusion engine (core `Graph::plan_fusion` → `FusedChain::resolve` → `run` →
`Backend::chain_dev`, emitting one nvrtc kernel via `kernels::chain_source`)
already exists and is correctness-tested (`gpu_integration.rs`:
`fused_relu_mul_add_chain_matches_cpu`, `core_fusion_seam_runs_one_gpu_launch_and_matches_eager`).
This records its **throughput** on the 3090 — the win the GPU baseline's
unfused elementwise ceiling was leaving on the table.

## Why fusion helps (the arithmetic, not a vibe)

The chain `relu(x) * y + z` unfused runs three kernels, each round-tripping
DRAM:

- `t1 = relu(x)`   : read x, write t1              = 2n
- `t2 = t1 * y`    : read t1, read y, write t2      = 3n
- `out = t2 + z`   : read t2, read z, write out     = 3n
- **total = 8n array-passes = 32n bytes**

Fused into one kernel the intermediates `t1,t2` never touch DRAM — they live in
registers:

- read x, y, z, write out = **4n array-passes = 16n bytes**

So fusion halves DRAM traffic (32n → 16n). On a bandwidth-bound elementwise
chain that predicts a **~2× speedup**, and the measurement confirms it.

## Measured (bench_chain.rs, chain = gelu(x)*y+z, 200 iters, device-resident)

Run: `cargo build -p ferro-cuda --release --example bench_chain` then
`target/release/examples/bench_chain.exe --n <N>`. Each timed loop ends in a
device sync; a correctness anchor asserts fused == unfused == graph-replay
before timing.

| n     | fused 1-launch | unfused 3-launch | **speedup** | graph replay | graph vs unfused |
|-------|----------------|------------------|-------------|--------------|------------------|
| 2²⁰   | 22.4 µs/iter   | 52.1 µs/iter     | **2.33×**   | 21.8 µs      | 2.39×            |
| 2²²   | 79.1 µs/iter   | 168.2 µs/iter    | **2.13×**   | 78.4 µs      | 2.15×            |
| 2²⁴   | 305 µs/iter    | 649 µs/iter      | **2.13×**   | 304 µs       | 2.14×            |
| 2²⁶   | 1.208 ms/iter  | 2.523 ms/iter    | **2.09×**   | 1.207 ms     | 2.09×            |

## Honest reading

1. **The 2× is real and matches theory.** At large n (2²⁶) the chain is pure
   DRAM traffic; halving traffic halves time → 2.09×. This is an *algorithm*
   win (fewer bytes moved), available to any framework that fuses — not a
   Rust-vs-C++ language win. torch gets the same class of win from
   `torch.compile`/nvFuser. ferro's angle is that the fusion seam is
   compile-in-from-the-start (record_fn), not an opt-in JIT.
2. **CUDA-graph replay adds little here** (1.00–1.03× over eager-fused) because
   one fused launch is already cheap relative to the kernel; graph capture pays
   off most when there are many small launches. At 2²⁰ its 2.39×-vs-unfused
   edge is the launch-overhead component still visible before the chain goes
   fully bandwidth-bound.
3. **Small-n does NOT beat 2×** by much and never explodes — no inflated
   headline. The range is a tight 2.09–2.33×, largest at the smallest size
   (launch overhead saved on top of traffic saved), settling to the pure-traffic
   2.09× floor.

## Python fusion: eager `.fuse()` vs compiled `.compile_fused()`

ferro-py exposes two entry points:

- `Tensor.fuse()` — one-shot: capture graph, plan, run one fused `chain_dev`
  launch. Correct (launches 3→1, CPU bit-exact, GPU ~1 ULP) but **re-plans every
  call**, so on a memory-bound chain the host cost swamps the traffic saving.
- `Tensor.compile_fused()` — returns a `FusedChain` handle: **plan/resolve ONCE**,
  then `handle.replay()` re-runs the single fused kernel with no tape walk and
  no re-planning. This is the one that banks the win.

### `.fuse()` alone is a loss (bench/eager_fusion.py) — plans every call

| n     | ferro eager (µs) | ferro `.fuse()` (µs) | speedup |
|-------|------------------|----------------------|---------|
| 2²⁰   | 55.8             | 78.0                 | 0.72×   |
| 2²²   | 174.5            | 253.0                | 0.69×   |
| 2²⁴   | 652.1            | 953.3                | 0.68×   |
| 2²⁶   | 2561.5           | 3741.9               | 0.68×   |

The per-call planning overhead (re-`from_root` + re-`plan_fusion` + re-`resolve`
+ re-alloc) is the whole loss; the fused kernel underneath is fine.

### `.compile_fused()` banks it (bench/compiled_fusion.py, 100 iters/30 warmup)

Plan once, `replay()` in the timed loop. Each iteration ends in a device sync;
a correctness anchor asserts replay == eager before timing.

| n     | eager ferro (µs) | compiled replay (µs) | **fused/eager** | torch eager (µs) | **fused/torch** | max\|Δ\| |
|-------|------------------|----------------------|-----------------|------------------|-----------------|---------|
| 2²⁰   | 55.4             | 28.5                 | **1.94×**       | 51.6             | **1.81×**       | 9.5e-7  |
| 2²²   | 174.7            | 85.3                 | **2.05×**       | 175.5            | **2.06×**       | 9.5e-7  |
| 2²⁴   | 671.0            | 322.4                | **2.08×**       | 662.1            | **2.05×**       | 9.5e-7  |
| 2²⁶   | 2541.3           | 1226.6               | **2.07×**       | 2504.1           | **2.04×**       | 9.5e-7  |

**This lands on the Rust `bench_chain` silicon numbers (2.07–2.09× at large n),
independently validating the whole path from Python.** And because torch eager
runs the same unfused 3-launch chain, the compiled handle beats **torch** by
~2.04–2.06× at the same time — the fusion algorithm win, now reachable from
Python without a JIT warmup dance.

## Honest reading of the compiled result

1. **~2.05× is the algorithm win, not a language win.** It comes from moving
   half the bytes (32n → 16n), available to any framework that fuses. torch
   reaches the same class of win via `torch.compile`/nvFuser after a trace/warmup;
   ferro's angle is that the fusion seam is compile-in-from-record (`record_fn`)
   and the compiled handle is a plain object, no JIT guard machinery.
2. **The 2²⁰ case is 1.94×, not higher** — at the smallest size the launch and
   host residue is still a visible fraction, so it sits *below* the large-n
   floor rather than exploding above it. No inflated headline; the range is a
   tight 1.94–2.08×.
3. **`ratio_best` (min/min) tracks the median** (1.93–2.12×), so the headline is
   not resting on one lucky iteration.
4. **What this handle is NOT yet:** it recomputes over the operands' *current*
   storage each replay (correct for repeated forwards with changing leaf
   values), but it does not yet capture a CUDA graph of the launch — the
   `capture_chain`/`replay` seam in ferro-cuda would shave the residual launch
   overhead further at small n. Left as future work; the memory-bound win is
   already banked without it.

### Bugs fixed while wiring this up

- **Classifier mislabel (graph.rs `classify`):** kind was inferred from shapes
  alone, so a same-shape elementwise binary (e.g. 512×512 * 512×512) satisfied
  the matmul shape contract and was tagged `MatMul` (not fusible), silently
  breaking every square-tensor pointwise chain. Fixed by reconciling against the
  op TAG (matmul is untagged; pointwise ops carry `OpTag::Binary/Unary`).
- **Off-by-one in `FusedChain::run_host`:** operand values were keyed at
  `slot+1` while `resolve` emits `other == operand_index`, so the host fallback
  panicked with "no entry found for key". Fixed to key by operand index.
