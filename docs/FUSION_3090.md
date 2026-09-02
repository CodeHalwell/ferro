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

## The gap: eager Python fusion exists but is not yet a speed win

ferro-py now exposes `Tensor.fuse()` (and `Tensor.fusion_launches()`): it
captures the pointwise graph rooted at the tensor, runs the planner, and
executes each chain as one fused `chain_dev` launch. **Correctness and the
structural win are proven** — `(x.relu()*y+z).fuse()` collapses launches 3→1
and matches eager to f32 tolerance (CPU bit-exact, GPU ~1 ULP).

**But `.fuse()` is currently SLOWER than eager** (measured, `bench/eager_fusion.py`):

| n     | ferro eager (µs) | ferro `.fuse()` (µs) | speedup | fused GB/s |
|-------|------------------|----------------------|---------|------------|
| 2²⁰   | 55.8             | 78.0                 | 0.72×   | 215        |
| 2²²   | 174.5            | 253.0                | 0.69×   | 265        |
| 2²⁴   | 652.1            | 953.3                | 0.68×   | 282        |
| 2²⁶   | 2561.5           | 3741.9               | 0.68×   | 287        |

Why: the eager path already runs its 3 kernels at ~838 GB/s (HBM-bound). The
`.fuse()` wrapper re-captures the graph (`from_root`), re-runs `plan_fusion`,
re-resolves the chain, and re-allocates the output **on every call** — so the
fused kernel effectively runs at ~287 GB/s, 3× below its ceiling, and the
per-call planning overhead swamps the 2× traffic saving. The Rust `bench_chain`
gets the real 2.1× because it resolves the chain ONCE and replays `chain_dev`
in the loop.

**Next step (FUTURE.md §5): a compile-once fused callable.** Plan/resolve once,
return a handle that replays the fused chain (ideally over a CUDA graph, which
`capture_chain`/`replay` already implement at the Rust level) so repeated
forwards pay the planning cost zero times. That is what turns the proven 2.1×
kernel win into a Python-visible speedup. The engine, the kernel, and the
correctness are done; only the caching wrapper remains.

### Bugs fixed while wiring this up

- **Classifier mislabel (graph.rs `classify`):** kind was inferred from shapes
  alone, so a same-shape elementwise binary (e.g. 512×512 * 512×512) satisfied
  the matmul shape contract and was tagged `MatMul` (not fusible), silently
  breaking every square-tensor pointwise chain. Fixed by reconciling against the
  op TAG (matmul is untagged; pointwise ops carry `OpTag::Binary/Unary`).
- **Off-by-one in `FusedChain::run_host`:** operand values were keyed at
  `slot+1` while `resolve` emits `other == operand_index`, so the host fallback
  panicked with "no entry found for key". Fixed to key by operand index.
