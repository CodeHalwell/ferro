"""PyTorch twin of examples/bench_gpu.rs: same cases, shapes, work counts
and JSON lines, on cuda:0 with torch.cuda.synchronize() around every timed
iteration. TF32 is disabled so both sides run true fp32 GEMMs.

    python examples/bench_gpu_torch.py --iters 50
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--filter", default="")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 tensor cores (ferro has no TF32 path yet)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda:0")

    def run(name, shape, work, unit, f):
        if a.filter not in name:
            return
        for _ in range(a.warmup):
            f()
        torch.cuda.synchronize()
        ms = []
        for _ in range(a.iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1e3)
        ms.sort()
        print(json.dumps({"name": name, "group": "gpu", "shape": shape, "work": work, "unit": unit, "warmup": a.warmup,
                          "iters": a.iters, "median_ms": ms[len(ms) // 2], "min_ms": ms[0], "p90_ms": ms[min(len(ms) * 9 // 10, len(ms) - 1)]}))

    for n in (1024, 4096):
        x, y = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
        with torch.no_grad():
            run(f"matmul_{n}", f"{n}x{n}@{n}x{n}", 2 * n ** 3, "GFLOP/s", lambda: x @ y)

    bn, cin, cout, hw, k = 32, 64, 64, 56, 3
    flops = 2 * bn * cout * hw * hw * cin * k * k
    shape = f"x[{bn},{cin},{hw},{hw}] w[{cout},{cin},{k},{k}] pad=1"
    cx, cw = torch.randn(bn, cin, hw, hw, device=dev), torch.randn(cout, cin, k, k, device=dev)
    with torch.no_grad():
        run("conv2d_fwd", shape, flops, "GFLOP/s", lambda: F.conv2d(cx, cw, padding=1))
    cxg, cwg = cx.clone().requires_grad_(), cw.clone().requires_grad_()

    def conv_bwd():
        cxg.grad = cwg.grad = None
        F.conv2d(cxg, cwg, padding=1).sum().backward()
    run("conv2d_fwd_bwd", shape, 3 * flops, "GFLOP/s", conv_bwd)

    px = torch.randn(32, 64, 112, 112, device=dev, requires_grad=True)

    def pool():
        px.grad = None
        F.max_pool2d(px, 2, 2).sum().backward()
    run("max_pool2d_fwd_bwd", "x[32,64,112,112] k=2 s=2", 4 * 32 * 64 * 112 * 112 * 2.5, "GB/s", pool)

    bx = torch.randn(32, 64, 56, 56, device=dev, requires_grad=True)
    bw, bb = torch.ones(64, device=dev, requires_grad=True), torch.zeros(64, device=dev, requires_grad=True)
    rm, rv = torch.zeros(64, device=dev), torch.ones(64, device=dev)

    def bnorm():
        bx.grad = bw.grad = bb.grad = None
        F.batch_norm(bx, rm, rv, bw, bb, training=True, momentum=0.1, eps=1e-5).sum().backward()
    run("batch_norm_fwd_bwd", "x[32,64,56,56] train", 4 * 32 * 64 * 56 * 56 * 6.0, "GB/s", bnorm)

    batch = 256
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 10, bias=False)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(batch, 3, 32, 32, device=dev)
    y = torch.tensor([i * 7 % 10 for i in range(batch)], device=dev)

    def step():
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    run("cnn_train_step", "256x3x32x32, conv3-32 bn pool conv32-64 bn pool fc4096-10, CE, SGD", batch, "samples/s", step)


if __name__ == "__main__":
    main()
