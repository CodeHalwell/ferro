"""Full relu(x)*y+z, compile once and replay; detached inference on both sides.

Wall-clock samples fence the device before and after each call, without host
copies. Warmup and compilation are excluded. Median is the headline; min/min
is secondary. This compares eager PyTorch, not torch.compile. Run twice.
"""
import argparse
import json
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json")
    ap.add_argument("--sizes", default="20,22,24,26")
    args = ap.parse_args()
    assert args.iters > 0 and args.warmup >= 0

    import ferro
    import torch
    assert ferro.cuda_is_available() and torch.cuda.is_available()
    ferro.cuda_init(0)
    torch.manual_seed(18)
    torch.set_grad_enabled(False)
    rows = []
    print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)
    print("All timed paths: no autograd; full relu(x)*y+z; fence-only sync")

    for e in map(int, args.sizes.split(",")):
        n = 1 << e
        tx, ty, tz = [torch.randn(n, device="cuda:0") for _ in range(3)]
        # Import copies the inputs; setup is outside the timed region.
        x, y, z = [ferro.from_dlpack(t) for t in (tx, ty, tz)]
        x.requires_grad_(True)
        expr = x.relu() * y + z
        b, a = expr.fusion_launches()
        handle = expr.compile_fused()
        assert handle.num_steps == 3 and handle.num_operands == 3
        assert (b, a) == (3, 1)
        x.requires_grad_(False)
        del expr
        assert not any(t.requires_grad for t in (x, y, z))
        assert not any(t.requires_grad for t in (tx, ty, tz))

        def eager():
            return x.relu() * y + z

        def replay():
            return handle.replay()

        def teager():
            return torch.relu(tx) * ty + tz

        re, rr, rt = eager(), replay(), teager()
        ferro.cuda_synchronize()
        torch.cuda.synchronize()
        assert re.shape == rr.shape == [n]
        assert not re.requires_grad and not rr.requires_grad
        fe, fr = torch.from_dlpack(re), torch.from_dlpack(rr)
        torch.testing.assert_close(fr, fe, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fr, rt, rtol=1e-5, atol=1e-6)
        d = (fr - fe).abs().max().item()
        dt = (fr - rt).abs().max().item()
        del re, rr, rt, fe, fr

        def bench(fn, sync):
            for _ in range(args.warmup):
                fn()
            sync()
            samples = []
            for _ in range(args.iters):
                sync()
                start = time.perf_counter()
                out = fn()
                sync()
                samples.append((time.perf_counter() - start) * 1e6)
                del out
            return samples

        te = bench(eager, ferro.cuda_synchronize)
        tr = bench(replay, ferro.cuda_synchronize)
        tt = bench(teager, torch.cuda.synchronize)
        me, mr, mt = map(statistics.median, (te, tr, tt))
        row = dict(exp=e, n=n, planned_launches_before=b, planned_launches_after=a,
                   compiled_steps=handle.num_steps, grad_mode="inference_no_grad",
                   med_eager_us=me, med_replay_us=mr, med_torch_us=mt,
                   ratio_fused_vs_eager=me/mr, ratio_best=min(te)/min(tr),
                   ratio_fused_vs_torch=mt/mr, max_abs_diff=d, max_abs_diff_torch=dt,
                   eager_GB_s=32*n/me/1000, replay_GB_s=16*n/mr/1000,
                   samples_eager_us=te, samples_replay_us=tr, samples_torch_us=tt)
        rows.append(row)
        print(f"2^{e}: eager {me:.1f}us replay {mr:.1f}us torch {mt:.1f}us; "
              f"median eager/replay {me/mr:.2f}x torch/replay {mt/mr:.2f}x; "
              f"max_abs_diff={d:.3g}; GB/s eager={row['eager_GB_s']:.1f} "
              f"replay={row['replay_GB_s']:.1f}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                           iters=args.iters, warmup=args.warmup,
                           expression="relu(x)*y+z", rtol=1e-5, atol=1e-6,
                           rows=rows), f, indent=2)


if __name__ == "__main__":
    main()
