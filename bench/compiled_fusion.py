"""Compiled-fusion benchmark: does a PRECOMPILED fused chain (plan once, replay
many) actually beat eager unfused ferro on a memory-bound pointwise chain?

Chain: out = relu(x) * y + z   (2 loads fused away vs the 3-launch eager path)

We compare, on device-resident tensors, wall time per evaluation of:
  * ferro eager   : (x.relu() * y + z)          3 separate launches, 32n bytes
  * ferro compiled: handle.replay()             1 fused launch,      ~16n bytes
  * torch eager   : torch.relu(x) * y + z        (reference)

Every timed region brackets a real device sync (cuda_synchronize / torch sync).
Warmup then timed loop. MEDIAN is the headline; min is reported only as ratio_best.
"""
import argparse, json, statistics, time, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--sizes", type=str, default="20,22,24,26")
    args = ap.parse_args()

    import ferro
    if not ferro.cuda_is_available():
        print("CUDA not available; this bench needs the GPU.", file=sys.stderr)
        sys.exit(1)
    ferro.cuda_init(0)
    import torch
    assert torch.cuda.is_available()
    dev = "cuda:0"

    def sync():
        ferro.cuda_synchronize()

    exps = [int(s) for s in args.sizes.split(",")]
    rows = []
    for e in exps:
        n = 1 << e
        # ferro device tensors
        x = ferro.Tensor.randn([n], device="cuda:0")
        y = ferro.Tensor.randn([n], device="cuda:0")
        z = ferro.Tensor.randn([n], device="cuda:0")

        # Build a compiled handle from a forward expression. Needs grad tape to
        # capture the op graph; the handle itself is detached at replay.
        xg = x.requires_grad_(True)
        expr = (xg.relu() * y + z)
        b, a = expr.fusion_launches()
        handle = expr.compile_fused()

        def eager():
            r = (x.relu() * y + z)
            sync(); return r
        def replay():
            r = handle.replay()
            sync(); return r

        # torch reference
        tx = torch.randn(n, device=dev); ty = torch.randn(n, device=dev); tz = torch.randn(n, device=dev)
        def teager():
            r = torch.relu(tx) * ty + tz
            torch.cuda.synchronize(); return r

        # correctness: compiled replay vs eager
        re = eager(); rr = replay()
        import numpy as np
        d = float(np.max(np.abs(np.asarray(re.cpu().tolist()) - np.asarray(rr.cpu().tolist()))))

        def bench(fn):
            for _ in range(args.warmup): fn()
            ts = []
            for _ in range(args.iters):
                t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e6)
            return ts
        te = bench(eager); tr = bench(replay); tt = bench(teager)
        med_e, med_r, med_t = statistics.median(te), statistics.median(tr), statistics.median(tt)
        min_e, min_r = min(te), min(tr)
        # bytes: eager unfused chain moves 32n bytes; fused ~16n.
        row = dict(exp=e, n=n, launches_before=b, launches_after=a,
                   med_eager_us=med_e, med_replay_us=med_r, med_torch_us=med_t,
                   ratio_fused_vs_eager=med_e/med_r, ratio_best=min_e/min_r,
                   ratio_fused_vs_torch=med_t/med_r, max_abs_diff=d)
        rows.append(row)
        print(f"2^{e:>2} n={n:>10} launches {b}->{a}  "
              f"eager {med_e:8.1f}us  replay {med_r:8.1f}us  torch {med_t:8.1f}us  "
              f"| fused/eager {med_e/med_r:5.2f}x (best {min_e/min_r:4.2f}x)  "
              f"fused/torch {med_t/med_r:5.2f}x  diff {d:.2e}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print("wrote", args.json)

if __name__ == "__main__":
    main()
