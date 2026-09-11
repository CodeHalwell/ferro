"""Independent eager LayerNorm f32 validation; run only after rebuilding bindings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from numerics import compare


def cases(seeds, widths, epsilons, trials):
    for seed in seeds:
        for width in widths:
            for trial in range(trials):
                # Each width/trial is independently reproducible when filtering a failure.
                rng = np.random.default_rng(np.random.SeedSequence([seed, width, trial]))
                shape = [(width,), (5, width), (2, 3, width)][trial % 3]
                for distribution in ("normal", "wide", "constant", "near_constant", "large_offset"):
                    x = rng.normal(size=shape)
                    if distribution == "wide":
                        x *= 10. ** rng.uniform(-4, 4, size=shape)
                    elif distribution == "constant":
                        x = np.broadcast_to(rng.choice([0., .5, -2., 1024.], size=shape[:-1] + (1,)), shape).copy()
                    elif distribution == "near_constant":
                        x = 1. + 1e-4 * x
                    elif distribution == "large_offset":
                        x = 1024. + .25 * x
                    x = x.astype(np.float32)
                    weight = rng.uniform(-2., 2., width).astype(np.float32)
                    bias = rng.normal(0., .7, width).astype(np.float32)
                    upstream = rng.normal(size=shape).astype(np.float32)
                    for eps in epsilons:
                        for affine in ("none", "weight", "bias", "both"):
                            arrays = dict(x=x, upstream=upstream)
                            if affine in ("weight", "both"):
                                arrays["weight"] = weight
                            if affine in ("bias", "both"):
                                arrays["bias"] = bias
                            yield dict(seed=seed, width=width, trial=trial, shape=list(shape),
                                       distribution=distribution, affine=affine, eps=float(np.float32(eps))), arrays


def torch_outputs(torch, arrays, eps, device, dtype, gradients):
    tensors = {k: torch.tensor(v, device=device, dtype=dtype, requires_grad=gradients and k != "upstream")
               for k, v in arrays.items()}
    y = torch.nn.functional.layer_norm(tensors["x"], (arrays["x"].shape[-1],),
                                       tensors.get("weight"), tensors.get("bias"), eps)
    result = {"output": y.detach().cpu().numpy().copy()}
    if gradients:
        (y * tensors["upstream"]).sum().backward()
        result.update({"grad_" + k: t.grad.detach().cpu().numpy().copy()
                       for k, t in tensors.items() if k != "upstream"})
    return result


def ferro_outputs(ferro, torch, arrays, eps, device, gradients):
    # Keep every producer alive until the consumer has completed. No RNG is shared with torch.
    owners = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in arrays.items()}
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    tensors = {k: ferro.from_dlpack(v).requires_grad_(gradients and k != "upstream") for k, v in owners.items()}
    y = tensors["x"].layer_norm(tensors.get("weight"), tensors.get("bias"), eps)
    expected_device = "cuda:0" if device == "cuda" else device
    if y.device != expected_device:
        raise RuntimeError(f"LayerNorm returned {y.device}, expected {expected_device}")
    def export(t):
        if device.startswith("cuda"):
            ferro.cuda_synchronize()
        return torch.from_dlpack(t).cpu().numpy().copy()
    result = {"output": export(y)}
    devices = {"output": y.device}
    if gradients:
        (y * tensors["upstream"]).sum().backward()
        for k, t in tensors.items():
            if k == "upstream":
                continue
            if t.grad is None:
                raise RuntimeError(f"Missing supported gradient: {k}")
            result["grad_" + k] = export(t.grad)
            devices["grad_" + k] = t.grad.device
    return result, devices


def gate_budget(meta, arrays, expected, quantity):
    gradient = quantity != "output"
    base = 3e-4 if gradient else 5e-5
    condition = 1.
    if meta["distribution"] in ("near_constant", "large_offset"):
        x = arrays["x"].astype(np.float64)
        variance = np.var(x, axis=-1)
        sensitivity = np.max(np.abs(x), axis=-1) / np.sqrt(variance + meta["eps"])
        condition = max(1., float(np.max(np.where(variance == 0, 1., sensitivity))))
    # First-order centering sensitivity, not a claim of a proven universal error bound.
    extra = (32 if gradient else 8) * (2. ** -24) * (condition - 1.) * max(1., float(np.max(np.abs(expected))))
    return base + max(0., extra), base, condition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 91])
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 3, 31, 33, 127, 129, 257, 513, 1025])
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-5, 1e-3])
    parser.add_argument("--trials", type=int, default=3, help="three trials cover ranks 1, 2, 3")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.jsonl"))
    args = parser.parse_args()
    if args.trials < 1 or min(args.widths) < 1 or min(args.seeds) < 0 or any(not np.isfinite(e) or e <= 0 for e in args.eps):
        parser.error("positive widths/trials/finite eps and nonnegative seeds required")
    import torch
    import ferro
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a silent CPU substitution")
    # Verify both DLPack directions independently, before any numerical comparison.
    bridge = torch.tensor([[1., -2., 0., .5]], device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    imported = ferro.from_dlpack(bridge)
    if args.device.startswith("cuda"):
        ferro.cuda_synchronize()
    np.testing.assert_array_equal(torch.from_dlpack(imported).cpu().numpy(), bridge.cpu().numpy())
    independent = ferro.Tensor([1., -2., 0., .5], [1, 4]).to(args.device)
    if args.device.startswith("cuda"):
        ferro.cuda_synchronize()
    np.testing.assert_array_equal(torch.from_dlpack(independent).cpu().numpy(), bridge.cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_count = len(args.seeds) * len(args.widths) * args.trials * 5 * len(args.eps) * 4 * 2
    failures, count = 0, 0
    with args.output.open("w", encoding="utf-8") as stream:
        header = dict(type="metadata", python=sys.version, torch=torch.__version__, numpy=np.__version__,
                      ferro_path=ferro.__file__, device=args.device,
                      gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      expected_cases=expected_count, gate="ferro vs torch CPU float64; ULP is diagnostic, not gated")
        stream.write(json.dumps(header) + "\n")
        for meta, arrays in cases(args.seeds, args.widths, args.eps, args.trials):
            digest = hashlib.sha256()
            for k, v in sorted(arrays.items()):
                digest.update(k.encode("ascii")); digest.update(v.tobytes())
            for gradients in (False, True):
                record = dict(type="case", **meta, mode="autograd" if gradients else "inference", input_sha256=digest.hexdigest())
                try:
                    actual, devices = ferro_outputs(ferro, torch, arrays, meta["eps"], args.device, gradients)
                    ref32 = torch_outputs(torch, arrays, meta["eps"], args.device, torch.float32, gradients)
                    ref64 = torch_outputs(torch, arrays, meta["eps"], "cpu", torch.float64, gradients)
                    measurements = {}
                    for quantity, value in actual.items():
                        atol, rtol, condition = gate_budget(meta, arrays, ref64[quantity], quantity)
                        measurements[quantity] = dict(condition=condition, ferro_vs_torch32=compare(value, ref32[quantity], atol, rtol),
                                                     ferro_vs_torch64=compare(value, ref64[quantity], atol, rtol),
                                                     torch32_vs_torch64=compare(ref32[quantity], ref64[quantity], atol, rtol))
                    record.update(devices=devices, measurements=measurements,
                                  passed=all(m["ferro_vs_torch64"]["passed"] for m in measurements.values()))
                except Exception as exc:
                    record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                count += 1
                failures += not record["passed"]
                stream.write(json.dumps(record, default=lambda v: int(v) if isinstance(v, np.integer) else str(v)) + "\n")
                stream.flush()
                if not record["passed"]:
                    print("FAIL", {k: record[k] for k in ("seed", "width", "trial", "distribution", "affine", "eps", "mode")}, record.get("error", "numerical gate"))
        summary = dict(type="summary", cases=count, expected_cases=expected_count, failed_cases=failures,
                       passed=count == expected_count and failures == 0)
        stream.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), "report:", args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
