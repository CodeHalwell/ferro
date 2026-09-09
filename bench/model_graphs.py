"""Real inference graphs, fence-only latency, parity before timing.

Default is a short correctness/capability smoke, NOT a performance result.
Use --benchmark twice on an idle GPU for publishable raw median samples.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


class Blocked(RuntimeError):
    pass


def make_case(name, tokens, width, heads, seed):
    gen = torch.Generator(device="cuda:0").manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(shape, device="cuda:0", generator=gen) * scale

    weights = {
        "w1": rand(width, 4 * width, scale=width ** -0.5),
        "b1": rand(4 * width, scale=0.02),
        "w2": rand(4 * width, width, scale=(4 * width) ** -0.5),
        "b2": rand(width, scale=0.02),
    }
    x = rand(tokens, width)
    if name == "transformer":
        for key in ("q", "k", "v", "o"):
            weights["w" + key] = rand(width, width, scale=width ** -0.5)
            weights["b" + key] = rand(width, scale=0.02)
        for key in ("ln1", "ln2"):
            weights[key + "g"] = torch.ones(width, device="cuda:0") + rand(width, scale=0.02)
            weights[key + "b"] = rand(width, scale=0.02)
    return dict(name=name, x=x, weights=weights, tokens=tokens, width=width, heads=heads)


def torch_forward(case, x):
    w = case["weights"]

    def mlp(y):
        return F.gelu(y @ w["w1"] + w["b1"], approximate="tanh") @ w["w2"] + w["b2"]

    if case["name"] == "mlp":
        return mlp(x)
    if case["name"] == "residual_mlp":
        return x + mlp(x)
    n, d, h = case["tokens"], case["width"], case["heads"]
    z = F.layer_norm(x, (d,), w["ln1g"], w["ln1b"], eps=1e-5)
    q, k, v = [(z @ w["w" + key] + w["b" + key]).reshape(n, h, d // h).transpose(0, 1)
               for key in ("q", "k", "v")]
    scores = (q @ k.transpose(-2, -1)) * ((d // h) ** -0.5)
    attention = torch.softmax(scores, dim=-1) @ v
    attention = attention.transpose(0, 1).contiguous().reshape(n, d)
    y = x + attention @ w["wo"] + w["bo"]
    return y + mlp(F.layer_norm(y, (d,), w["ln2g"], w["ln2b"], eps=1e-5))


def prepare_ferro_eager(case, ferro):
    # DLPack import copies, but every copy is confined to setup/rebind.
    state = {"x": ferro.from_dlpack(case["x"])}
    w = {key: ferro.from_dlpack(value) for key, value in case["weights"].items()}
    assert not any(t.requires_grad for t in [state["x"], *w.values()])

    scale = ferro.Tensor([(case["width"] // case["heads"]) ** -0.5], []).to("cuda")

    def check(y):
        if y.device != "cuda:0" or y.requires_grad:
            raise Blocked("Ferro operation returned host storage or autograd output")
        return y

    def forward(validate=False):
        gate = check if validate else lambda y: y
        x = state["x"]
        if case["name"] == "transformer":
            n, d, h = case["tokens"], case["width"], case["heads"]
            z = gate(x.layer_norm(w["ln1g"], w["ln1b"]))
            q, k, v = [gate(gate(z.matmul(w["w" + key])) + w["b" + key]).reshape([n, h, d // h]).transpose(0, 1).reshape([h, n, d // h]) for key in ("q", "k", "v")]
            kt = k.transpose(1, 2).reshape([h, d // h, n])
            scores = gate(q.bmm(kt)) * scale
            attention = gate(gate(scores.softmax(-1)).bmm(v)).transpose(0, 1).reshape([n, d])
            residual = gate(x + gate(attention.matmul(w["wo"])) + w["bo"])
            x = gate(residual.layer_norm(w["ln2g"], w["ln2b"]))
        y = gate(x.matmul(w["w1"]))
        y = gate(y + w["b1"])
        y = gate(y.gelu())
        y = gate(y.matmul(w["w2"]))
        y = gate(y + w["b2"])
        if case["name"] == "transformer":
            return gate(residual + y)
        return gate(x + y) if case["name"] == "residual_mlp" else y

    def bind(x):
        state["x"].copy_(ferro.from_dlpack(x))

    forward(validate=True)
    return dict(run=forward, bind=bind, sync=ferro.cuda_synchronize,
                export=lambda y: torch.from_dlpack(y), keepalive=(w, state))


def prepare_ferro_compiled(case, ferro):
    """Integration seam: return run/bind/sync/export plus structural metadata.

    Do not wrap compile_fused around the last pointwise tail: that can freeze
    matmul intermediates and does not execute this model from its true leaves.
    """
    path = prepare_ferro_eager(case, ferro)
    root = ferro.capture(path["run"])
    graph = root.compile_fused()
    minimum = {"mlp": 5, "residual_mlp": 6, "transformer": 35}[case["name"]]
    assert graph.num_steps >= minimum, "model work omitted from replay"
    path["run"] = graph.replay
    path["structure"] = dict(operations=graph.num_steps, leaves=graph.num_operands, runs=graph.num_runs)
    return path


def prepare_torch(case, compiled):
    state = {"x": case["x"]}
    fn = lambda x: torch_forward(case, x)
    if compiled:
        # No backend=eager, suppressed exceptions, or graph-break fallback.
        fn = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
    return dict(run=lambda: fn(state["x"]), bind=lambda x: state.update(x=x),
                sync=lambda: torch.cuda.synchronize(0), export=lambda y: y)


def parity(path, expected, rtol, atol):
    out = path["run"]()
    path["sync"]()
    actual = path["export"](out)
    torch.cuda.synchronize(0)
    assert actual.device == expected.device and actual.dtype == expected.dtype
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # Validation only: never called inside a timed closure.
    return (actual - expected).abs().max().item()


def measure(path, warmup, iters):
    for _ in range(warmup):
        out = path["run"]()
    path["sync"]()
    samples = []
    for _ in range(iters):
        path["sync"]()
        start = time.perf_counter_ns()
        out = path["run"]()
        path["sync"]()
        samples.append((time.perf_counter_ns() - start) / 1000)
        del out
    return dict(samples_us=samples, median_us=statistics.median(samples),
                min_us=min(samples), max_us=max(samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true", help="Enable timings; otherwise parity/probe only")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--rtol", type=float, default=2e-4)
    ap.add_argument("--atol", type=float, default=2e-5)
    ap.add_argument("--cases", nargs="+", choices=["mlp", "residual_mlp", "transformer"],
                    default=["mlp", "residual_mlp", "transformer"])
    ap.add_argument("--backends", nargs="+", choices=["torch_eager", "torch_compile", "ferro_eager", "ferro_compiled"],
                    default=["torch_eager", "torch_compile", "ferro_eager", "ferro_compiled"])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if min(args.tokens, args.width, args.heads, args.iters) <= 0 or args.warmup < 0 or args.width % args.heads:
        ap.error("positive dimensions/iters, nonnegative warmup, width divisible by heads required")
    if not torch.cuda.is_available():
        ap.error("CUDA PyTorch required; CPU is not a substitute baseline")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    ferro = None
    ferro_error = None
    if any(b.startswith("ferro") for b in args.backends):
        try:
            import ferro
            if not ferro.cuda_init(0):
                raise Blocked("ferro.cuda_init(0) returned false")
        except Exception as exc:
            ferro_error = str(exc)
    report = dict(platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), dtype="float32", tf32=False,
                  mode="benchmark" if args.benchmark else "smoke_no_performance_claim",
                  inference="torch.inference_mode; Ferro detached leaves", args={**vars(args), "json": str(args.json)},
                  rows=[])
    failed = False
    with torch.inference_mode():
        for name in args.cases:
            start = time.perf_counter_ns()
            case = make_case(name, args.tokens, args.width, args.heads, args.seed)
            expected = torch_forward(case, case["x"])
            changed = case["x"] * 0.73 + 0.19
            expected_changed = torch_forward(case, changed)
            torch.cuda.synchronize(0)
            assert not torch.allclose(expected, expected_changed, rtol=args.rtol, atol=args.atol)
            setup_ms = (time.perf_counter_ns() - start) / 1e6
            for backend in args.backends:
                row = dict(case=name, backend=backend, fixture_and_reference_setup_ms=setup_ms)
                start = time.perf_counter_ns()
                try:
                    if backend.startswith("torch"):
                        path = prepare_torch(case, backend == "torch_compile")
                    else:
                        if ferro_error:
                            raise Blocked(ferro_error)
                        prepare = prepare_ferro_eager if backend == "ferro_eager" else prepare_ferro_compiled
                        path = prepare(case, ferro)
                    out = path["run"]()
                    path["sync"]()
                    del out
                    row["prepare_and_first_call_ms"] = (time.perf_counter_ns() - start) / 1e6
                    row["max_abs_error"] = parity(path, expected, args.rtol, args.atol)
                    path["bind"](changed)
                    row["changed_input_max_abs_error"] = parity(path, expected_changed, args.rtol, args.atol)
                    path["bind"](case["x"])
                    parity(path, expected, args.rtol, args.atol)
                    row["status"] = "passed"
                    if "structure" in path:
                        row["structure"] = path["structure"]
                    if args.benchmark:
                        row.update(measure(path, args.warmup, args.iters))
                except AssertionError as exc:
                    failed = True
                    row.update(status="parity_failed", error=str(exc))
                except Exception as exc:
                    row.update(status="blocked", error_type=type(exc).__name__, error=str(exc),
                               attempt_ms=(time.perf_counter_ns() - start) / 1e6)
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
