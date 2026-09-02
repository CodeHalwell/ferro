#!/usr/bin/env python
"""GPU benchmark: ferro vs torch on the same card (roadmap FUTURE.md §1).

Honest-number rules (Daniel's "proof, not vibes"):
  * Every timed region is bracketed by a real device sync. For torch that is
    torch.cuda.synchronize(); for ferro we fence the stream via
    ferro.cuda_synchronize() -- a pure stream fence, NOT a device->host copy
    (a readback would charge ferro PCIe traffic torch never pays).
  * Warmup iterations (kernel compile / cuBLAS handle / allocator warmup) are
    discarded. We report the MEDIAN of timed iters as the headline statistic
    (min is also captured as ratio_best for reference, but never headlined --
    a lucky min can hide a real median regression).
  * We never quote a delta smaller than run-to-run noise: the summary prints
    ferro/torch MEDIAN ratio AND each side's throughput so a reader can judge.
  * matmul is scored in GFLOP/s (2*M*N*K), elementwise in EFFECTIVE GB/s on the
    real unfused 32n-byte traffic (not the fused-ideal 16n), so numbers are
    comparable across shapes and to the 3090's spec peaks (FP32 ~35.6 TFLOP/s,
    HBM ~936 GB/s).

Usage:
  python bench/gpu_vs_torch.py                 # full suite
  python bench/gpu_vs_torch.py --iters 100     # more timed iters
  python bench/gpu_vs_torch.py --json out.json # machine-readable dump
"""
import argparse
import json
import statistics
import sys
import time

import ferro
import torch

FTensor = ferro.Tensor


def sync_torch():
    torch.cuda.synchronize()


def time_loop(fn, sync, warmup, iters):
    """Return per-iteration seconds (median, min) after warmup, sync-bracketed."""
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


# ---- ferro sync: pure stream fence (no device->host copy) ----
def sync_ferro():
    ferro.cuda_synchronize()


def bench_matmul(sizes, warmup, iters):
    rows = []
    for (m, k, n) in sizes:
        flops = 2.0 * m * k * n
        # torch
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)
        out_box = [None]
        def torch_fn():
            out_box[0] = a @ b
        med_t, min_t = time_loop(torch_fn, sync_torch, warmup, iters)
        # ferro
        fa = FTensor.randn([m, k], device="cuda:0")
        fb = FTensor.randn([k, n], device="cuda:0")
        fbox = [None]
        def ferro_fn():
            fbox[0] = fa.matmul(fb)
        med_f, min_f = time_loop(ferro_fn, sync_ferro, warmup, iters)
        rows.append({
            "shape": f"{m}x{k} @ {k}x{n}",
            "torch_gflops": flops / med_t / 1e9,
            "ferro_gflops": flops / med_f / 1e9,
            "torch_gflops_best": flops / min_t / 1e9,
            "ferro_gflops_best": flops / min_f / 1e9,
            "torch_ms": med_t * 1e3,
            "ferro_ms": med_f * 1e3,
            "ratio_ferro_over_torch": med_t / med_f,
            "ratio_best": min_t / min_f,
        })
    return rows


def bench_elementwise(sizes, warmup, iters):
    """Chain: relu -> *y -> +z, run UNFUSED (three separate kernels).

    Physical DRAM traffic for the unfused chain = 32*n bytes:
      relu(x):      read x, write t1        = 2 arrays
      t1 * y:       read t1, read y, write  = 3 arrays
      t2 + z:       read t2, read z, write  = 3 arrays
    Total 8 array-passes * n * 4 bytes = 32n. We report EFFECTIVE bandwidth on
    this real traffic so it is comparable to the 936 GB/s HBM peak (Codex P2:
    do NOT use the 16n fused-ideal numerator against a hardware peak). Same
    accounting both sides, so the ratio stays apples-to-apples.
    """
    rows = []
    for n in sizes:
        bytes_moved = 32.0 * n
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        c = torch.randn(n, device="cuda", dtype=torch.float32)
        obx = [None]
        def torch_fn():
            obx[0] = torch.relu(a) * b + c
        med_t, min_t = time_loop(torch_fn, sync_torch, warmup, iters)
        fa = FTensor.randn([n], device="cuda:0")
        fb = FTensor.randn([n], device="cuda:0")
        fc = FTensor.randn([n], device="cuda:0")
        fbox = [None]
        def ferro_fn():
            fbox[0] = fa.relu() * fb + fc
        med_f, min_f = time_loop(ferro_fn, sync_ferro, warmup, iters)
        rows.append({
            "n": n,
            "torch_gbps": bytes_moved / med_t / 1e9,
            "ferro_gbps": bytes_moved / med_f / 1e9,
            "torch_ms": med_t * 1e3,
            "ferro_ms": med_f * 1e3,
            "ratio_ferro_over_torch": med_t / med_f,
            "ratio_best": min_t / min_f,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: torch has no CUDA — cannot run honest GPU comparison.")
        sys.exit(1)
    if not ferro.cuda_is_available():
        print("FATAL: ferro has no CUDA device visible.")
        sys.exit(1)
    ferro.cuda_init(0)  # register ferro's CUDA backend for this process
    dev = torch.cuda.get_device_name(0)
    print(f"# device: {dev}")
    print(f"# torch: {torch.__version__}  ferro-py loaded")
    print(f"# warmup={args.warmup} iters={args.iters} (MEDIAN reported; ratio = median torch/ferro)\n")

    matmul_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (2048, 8192, 2048),
    ]
    ew_sizes = [1 << 20, 1 << 22, 1 << 24, 1 << 26]

    mm = bench_matmul(matmul_sizes, args.warmup, args.iters)
    ew = bench_elementwise(ew_sizes, args.warmup, args.iters)

    print("== matmul (GFLOP/s, higher better) ==")
    print(f"{'shape':<22} {'torch':>10} {'ferro':>10} {'ferro/torch':>12}")
    for r in mm:
        print(f"{r['shape']:<22} {r['torch_gflops']:>10.0f} {r['ferro_gflops']:>10.0f} {r['ratio_ferro_over_torch']:>11.2f}x")
    print("\n== elementwise relu*y+z (GB/s, higher better) ==")
    print(f"{'n':<22} {'torch':>10} {'ferro':>10} {'ferro/torch':>12}")
    for r in ew:
        print(f"{r['n']:<22} {r['torch_gbps']:>10.0f} {r['ferro_gbps']:>10.0f} {r['ratio_ferro_over_torch']:>11.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev, "torch": torch.__version__,
                       "matmul": mm, "elementwise": ew}, f, indent=2)
        print(f"\n# wrote {args.json}")


if __name__ == "__main__":
    main()
