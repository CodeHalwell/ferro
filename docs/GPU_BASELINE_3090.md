# GPU Baseline: ferro vs torch on RTX 3090

First honest silicon measurement (roadmap FUTURE.md §1 — "establish the honest
baseline gap"). Both frameworks run on the same card, f32, in one process via
ferro-py.

- **Hardware:** NVIDIA GeForce RTX 3090 (FP32 peak ~35.6 TFLOP/s, HBM ~936 GB/s)
- **torch:** 2.6.0+cu124 · **ferro:** ferro-py (release) · driver CUDA 13.1
- **Harness:** `bench/gpu_vs_torch.py` — warmup discarded, **MEDIAN** is the
  headline statistic (min captured as `ratio_best` but never headlined: a lucky
  min can hide a real median regression). Both sides timed with a pure stream
  fence (`torch.cuda.synchronize()` / `ferro.cuda_synchronize()`), NOT a
  device→host copy. Run `python bench/gpu_vs_torch.py --json out.json`.

## matmul (GFLOP/s, higher better, median)

| shape              | torch  | ferro  | ferro/torch |
|--------------------|--------|--------|-------------|
| 512×512 @ 512×512   |  ~5.8k |  ~8.7k | ~1.5×       |
| 1024³              | ~14.7k | ~17.2k | ~1.18×      |
| 2048³              | ~22.3k | ~22.4k | 1.00×       |
| 4096³              | ~25.1k | ~25.3k | 1.00×       |
| 2048×8192 @ 8192×2048 | ~25.4k | ~25.5k | 1.00×    |

Big compute-bound shapes are **exact parity** — both dispatch to cuBLAS, so
parity is the expected and correct result. The small-shape edge (512³ ~1.5×,
1024³ ~1.18×) is real on median and repeatable: it is ferro's lower host
dispatch overhead showing through while the kernel is still small enough for
per-call cost to matter (see HOST_OVERHEAD.md for the isolated measurement).
It shrinks to nothing as the kernel grows, exactly as that model predicts.

## elementwise relu(x)*y+z (effective GB/s, higher better, median)

Effective bandwidth on the **real unfused traffic** (32n bytes: relu 2 passes,
each binary op 3 passes = 8 array-passes × 4 B), so it is comparable to the
936 GB/s HBM peak.

| n        | torch | ferro | ferro/torch |
|----------|-------|-------|-------------|
| 2²⁰      | ~634  | ~556  | 0.88×       |
| 2²²      | ~702  | ~739  | 1.05×       |
| 2²⁴      | ~815  | ~838  | 1.03×       |
| 2²⁶      | ~859  | ~843  | 0.98×       |

At large n both saturate **~840–860 GB/s ≈ 90% of the 3090's 936 GB/s HBM
peak** — the legitimate ceiling for this unfused 3-kernel chain, hit by both.
At the smallest size (2²⁰) ferro trails (0.88×): the chain is launch-bound
there and torch's kernels edge it; not hidden. Mid sizes are within noise.

## Honest takeaways

1. **matmul: parity on big shapes, small edge on small shapes.** No headroom to
   beat cuBLAS on compute-bound work, and none claimed. The small-shape win is
   host-overhead, not kernel speed.
2. **elementwise: parity, both at ~90% HBM peak.** The remaining gap to peak is
   the unfused chain re-reading/re-writing DRAM three times. Fusing relu*y+z
   into one kernel cuts 32n → 16n traffic and is the first real differentiation
   target (FUTURE.md §5).
3. No fabricated deltas: median is headlined, the one sub-1× row is shown, and
   the small-shape matmul edge is attributed to host overhead, not magic.

## Method notes (two corrections made under review, Codex PR #16)

- **Report median, not min.** An earlier draft headlined the fastest sample,
  which on one 2²⁶ run hid a ~2× median slowdown behind a 0.99× min. The
  harness now headlines the median (the statistic it claims to report).
- **Effective bandwidth on real traffic.** An earlier draft used the fused-ideal
  16n-byte numerator against the HBM peak, understating utilisation as ~46%.
  The unfused chain actually moves 32n bytes → ~90% of peak. Fixed.
- **Sync must fence, not transfer.** The very first draft synced ferro via
  `.cpu()` (full PCIe readback), producing a bogus flat ~12 GB/s. Fixed with a
  real stream fence (`ferro.cuda_synchronize()`).
