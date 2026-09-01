#!/usr/bin/env python
"""Host-overhead benchmark: ferro vs torch (roadmap FUTURE.md, thesis bench).

Why this exists
---------------
The GPU throughput bench (gpu_vs_torch.py) reads parity: matmul and elementwise
are kernel/bandwidth bound, so both frameworks ride the same cuBLAS / same HBM
ceiling. That measures the DEVICE, not ferro.

ferro's structural thesis is the opposite surface: HOST-SIDE overhead --
per-op dispatch, autograd graph construction, and graph traversal on backward.
That is where a statically-dispatched, GIL-free Rust core can beat torch's
eager C++/Python dispatch. On TINY tensors the kernel is ~free, so wall time is
dominated by that host overhead. That is what we measure here.

Honesty rules (Daniel's "proof, not vibes")
--------------------------------------------
  * Both sides are driven from the SAME Python interpreter, so both pay the
    Python->native call cost. This is a fair fight: pyo3+Rust dispatch vs
    pybind+C++ ATen dispatch. We are NOT comparing a Rust binary to Python.
  * Tensors are tiny ([8]) and on CPU: no kernel-launch or PCIe cost to hide
    behind. What is left is dispatch + autograd bookkeeping.
  * Warmup discarded; we report MEDIAN ns/op and ops/sec (min-time driven).
  * We never quote a delta inside run-to-run noise: both sides' medians print.
  * Result is host-throughput (ops/sec), NOT FLOP/s -- these ops do no real
    compute. Reading it as "compute speed" would be a lie; it is orchestration
    speed, which is exactly the axis under test.

Usage:
  python bench/host_overhead.py
  python bench/host_overhead.py --iters 200000 --json bench/host_3090.json
"""
import argparse
import json
import statistics
import sys
import time

import ferro
import torch

FTensor = ferro.Tensor


def time_ops(fn, warmup, iters):
    """Return (median_ns_per_op, min_ns_per_op). fn does ONE op per call."""
    for _ in range(warmup):
        fn()
    # A few timed blocks; each block runs `chunk` ops so per-call timer noise
    # averages out, then we take the best block (min) and median of blocks.
    blocks = 20
    chunk = max(1, iters // blocks)
    per_op = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(chunk):
            fn()
        dt = time.perf_counter() - t0
        per_op.append(dt / chunk * 1e9)
    return statistics.median(per_op), min(per_op)


def bench_dispatch(warmup, iters):
    """Single elementwise op on a tiny tensor: pure dispatch cost."""
    rows = []

    # torch: leaf tensors, no grad -- measures ATen dispatch only.
    tx = torch.randn(8, dtype=torch.float32)
    ty = torch.randn(8, dtype=torch.float32)
    fx = FTensor.randn([8])
    fy = FTensor.randn([8])

    cases = [
        ("relu(x)", lambda: torch.relu(tx), lambda: fx.relu()),
        ("x + y", lambda: tx + ty, lambda: fx + fy),
        ("x * y", lambda: tx * ty, lambda: fx * fy),
        ("relu(x)*y+x", lambda: torch.relu(tx) * ty + tx,
                          lambda: fx.relu() * fy + fx),
    ]
    for name, tfn, ffn in cases:
        med_t, min_t = time_ops(tfn, warmup, iters)
        med_f, min_f = time_ops(ffn, warmup, iters)
        rows.append({
            "op": name,
            "torch_ns": med_t, "ferro_ns": med_f,
            "torch_ops_per_s": 1e9 / min_t, "ferro_ops_per_s": 1e9 / min_f,
            "ratio_ferro_over_torch": min_t / min_f,  # >1 means ferro faster
        })
    return rows


def bench_autograd(warmup, iters, depth):
    """Build a depth-N elementwise autograd chain then backward, per iter.

    Measures graph construction (forward) + traversal (backward) overhead on
    tiny data. This is the training-loop-per-step overhead the thesis targets,
    isolated from any real kernel cost.
    """
    rows = []

    def torch_step():
        x = torch.randn(8, dtype=torch.float32, requires_grad=True)
        h = x
        for _ in range(depth):
            h = torch.relu(h) * 1.5 + 0.1
        h.sum().backward()

    def ferro_step():
        x = FTensor.randn([8])
        x.requires_grad_(True)
        h = x
        for _ in range(depth):
            h = h.relu() * 1.5 + 0.1
        h.sum().backward()

    med_t, min_t = time_ops(torch_step, warmup, iters)
    med_f, min_f = time_ops(ferro_step, warmup, iters)
    rows.append({
        "depth": depth,
        "torch_us": med_t / 1e3, "ferro_us": med_f / 1e3,
        "torch_steps_per_s": 1e9 / min_t, "ferro_steps_per_s": 1e9 / min_f,
        "ratio_ferro_over_torch": min_t / min_f,
    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)  # host-overhead test: don't let torch spin threads
    print("# host-overhead bench (tiny CPU tensors -- dispatch/autograd bound)")
    print(f"# torch: {torch.__version__}  ferro-py loaded  (both Python-driven)")
    print(f"# warmup={args.warmup} iters={args.iters}\n")

    disp = bench_dispatch(args.warmup, args.iters)
    ag = bench_autograd(args.warmup, max(args.iters // 20, 2000), depth=8)

    print("== per-op dispatch (ns/op, LOWER better; ratio>1 = ferro faster) ==")
    print(f"{'op':<14} {'torch ns':>10} {'ferro ns':>10} {'ferro/torch':>12}")
    for r in disp:
        print(f"{r['op']:<14} {r['torch_ns']:>10.0f} {r['ferro_ns']:>10.0f} "
              f"{r['ratio_ferro_over_torch']:>11.2f}x")

    print("\n== autograd chain step (us/step, LOWER better) ==")
    print(f"{'depth':<14} {'torch us':>10} {'ferro us':>10} {'ferro/torch':>12}")
    for r in ag:
        print(f"depth={r['depth']:<8} {r['torch_us']:>10.1f} {r['ferro_us']:>10.1f} "
              f"{r['ratio_ferro_over_torch']:>11.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"torch": torch.__version__,
                       "dispatch": disp, "autograd": ag}, f, indent=2)
        print(f"\n# wrote {args.json}")


if __name__ == "__main__":
    main()
