# GPU Baseline: ferro vs torch on RTX 3090

First honest silicon measurement (roadmap FUTURE.md §1 — "establish the honest
baseline gap"). Both frameworks run on the same card, f32, in one process via
ferro-py.

- **Hardware:** NVIDIA GeForce RTX 3090 (FP32 peak ~35.6 TFLOP/s, HBM ~936 GB/s)
- **torch:** 2.6.0+cu124 · **ferro:** ferro-py (release) · driver CUDA 13.1
- **Harness:** `bench/gpu_vs_torch.py` — warmup discarded, median reported,
  **min drives the throughput number**. Both sides timed with a pure stream
  fence (`torch.cuda.synchronize()` / `ferro.cuda_synchronize()`), NOT a
  device→host copy. Run `python bench/gpu_vs_torch.py --json out.json`.

## matmul (GFLOP/s, higher better)

| shape              | torch  | ferro  | ferro/torch |
|--------------------|--------|--------|-------------|
| 512×512 @ 512×512   |  ~6.2k |  ~7–9k | ~1.2–1.4×*  |
| 1024³              | ~16.5k | ~17.1k | 1.04×       |
| 2048³              | ~23.1k | ~23.4k | 1.01×       |
| 4096³              | ~25.3k | ~25.3k | 1.00×       |
| 2048×8192 @ 8192×2048 | ~25.7k | ~27k | ~1.05×      |

\* The 512³ ratio swings 1.19×→1.36× run-to-run: at sub-millisecond timings
this is launch-overhead noise, **not a real ferro advantage** — do not quote it
as a win. The load-bearing numbers are the large shapes: **1.00–1.04× parity**.
Both dispatch matmul to cuBLAS, so parity is the expected and correct result.

## elementwise relu(x)*y+z (GB/s, higher better)

| n        | torch | ferro | ferro/torch |
|----------|-------|-------|-------------|
| 2²⁰      | ~310  | ~316  | 1.02×       |
| 2²²      | ~386  | ~394  | 1.02×       |
| 2²⁴      | ~420  | ~418  | 0.99×       |
| 2²⁶      | ~431  | ~426  | 0.99×       |

Both saturate ~420–430 GB/s ≈ **46% of the 3090's 936 GB/s HBM peak**. That is
the legitimate ceiling for an *unfused* 3-read/1-write chain issued as separate
kernels (each op re-reads/re-writes global memory). ferro and torch sit on the
same ceiling because neither fuses this chain by default.

## Honest takeaways

1. **matmul: at parity with torch** on compute-bound shapes — both ride cuBLAS,
   so there was never headroom to "beat" torch here without a custom kernel.
2. **elementwise: at parity, and both leave ~2× on the table** vs HBM peak
   purely from lack of fusion. This is where ferro's record_fn fusion seam
   (FUTURE.md §5, FUSION_SEAM.md) can win outright: fusing relu*y+z into one
   kernel would move 4×n bytes → cut to the single-pass minimum and roughly
   double effective bandwidth. That is the first real differentiation target.
3. No fabricated deltas: sub-ms shapes are disclosed as noise, not results.

## Method note (why the first run was wrong)

An earlier draft synced ferro by pulling the whole result to host (`.cpu()`),
which charged ferro a full PCIe readback torch never paid — producing a bogus
flat ~12 GB/s and a fake 0.04× "loss". Fixed by adding a real stream-fence
primitive (`CudaBackend::synchronize` → `ferro.cuda_synchronize()`) so timing
measures kernel completion on both sides. Lesson: benchmark syncs must fence,
not transfer.
