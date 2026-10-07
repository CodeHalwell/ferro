"""PyTorch twin of examples/bench_suite.rs: same case names, shapes, work
counts and JSON-lines output, so benchmarks/compare.py can join them.

Run:
    python examples/bench_suite_torch.py [--warmup 5] [--iters 30] [--filter s] [--threads N]
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--filter", default="")
    # Default matches ferro, which threads over available_parallelism() (the
    # process affinity mask where the OS has one); torch otherwise picks its own count.
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    ap.add_argument("--threads", type=int, default=affinity)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    def run(name, group, shape, work, unit, f):
        if args.filter not in name:
            return
        for _ in range(args.warmup):
            f()
        ms = []
        for _ in range(args.iters):
            t = time.perf_counter()
            f()
            ms.append((time.perf_counter() - t) * 1e3)
        ms.sort()
        print(json.dumps({
            "name": name, "group": group, "shape": shape, "work": work, "unit": unit,
            "warmup": args.warmup, "iters": args.iters, "median_ms": statistics.median(ms),
            "min_ms": ms[0], "p90_ms": ms[min(int(len(ms) * 0.9), len(ms) - 1)],
        }), flush=True)

    with torch.no_grad():
        for n in (256, 512, 1024):
            x, y = torch.randn(n, n), torch.randn(n, n)
            run(f"matmul_{n}", "op", f"{n}x{n}@{n}x{n}", 2 * n ** 3, "GFLOP/s", lambda: x @ y)

        n = 1 << 22
        x, y = torch.randn(n), torch.randn(n)
        shape = str(n)
        run("add_4M", "op", shape, 12 * n, "GB/s", lambda: x + y)
        run("mul_4M", "op", shape, 12 * n, "GB/s", lambda: x * y)
        run("relu_4M", "op", shape, 8 * n, "GB/s", lambda: torch.relu(x))
        run("exp_4M", "op", shape, 8 * n, "GB/s", lambda: torch.exp(x))
        run("gelu_tanh_4M", "op", shape, 8 * n, "GB/s", lambda: F.gelu(x, approximate="tanh"))
        run("sum_4M", "op", shape, 4 * n, "GB/s", lambda: x.sum())

        s = torch.randn(1024, 1024)
        run("softmax_1024x1024", "op", "1024x1024 dim=1", 8 * (1 << 20), "GB/s", lambda: torch.softmax(s, 1))

    bn, cin, cout, hw, k = 16, 32, 64, 32, 3
    conv_flops = 2 * bn * cout * hw * hw * cin * k * k
    conv_shape = f"x[{bn},{cin},{hw},{hw}] w[{cout},{cin},{k},{k}] pad=1"
    cx, cw = torch.randn(bn, cin, hw, hw), torch.randn(cout, cin, k, k)
    with torch.no_grad():
        run("conv2d_fwd", "op", conv_shape, conv_flops, "GFLOP/s", lambda: F.conv2d(cx, cw, padding=1))
    cxg, cwg = cx.clone().requires_grad_(), cw.clone().requires_grad_()

    def conv_bwd():
        cxg.grad = cwg.grad = None
        F.conv2d(cxg, cwg, padding=1).sum().backward()
    run("conv2d_fwd_bwd", "op", conv_shape, 3 * conv_flops, "GFLOP/s", conv_bwd)

    def train(model, x, y):
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        def step():
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return step

    batch = 128
    mlp = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    labels = lambda n: torch.tensor([i * 7 % 10 for i in range(n)])
    run("mlp_train_step", "model", "128x784 -> 512 -> 256 -> 10, relu, CE, SGD", batch, "samples/s",
        train(mlp, torch.randn(batch, 784), labels(batch)))

    batch = 32
    cnn = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 8 * 8, 10))
    run("cnn_train_step", "model", "32x3x32x32, conv3-16 pool conv16-32 pool fc2048-10, CE, SGD", batch, "samples/s",
        train(cnn, torch.randn(batch, 3, 32, 32), labels(batch)))


if __name__ == "__main__":
    main()
