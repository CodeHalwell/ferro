"""Run the ferro and PyTorch CPU benchmark suites on the same machine and
print a side-by-side table. Only the stdlib is needed here; the torch side
runs under --python (which must have torch installed).

    python benchmarks/compare.py --python benchmarks/.venv/bin/python \
        [--cpus 0-3] [--warmup 5] [--iters 30] [--filter s] [--transformer] \
        [--json out.json] [--markdown out.md]

--cpus pins BOTH processes to the same CPU set via taskset; ferro threads over
available_parallelism() (which honours the affinity mask) and the torch twin
sets torch.set_num_threads to the same count (passed as --threads to both
torch scripts), so thread budgets always match. --cpus needs Linux taskset.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def pinned(cmd, cpus):
    return (["taskset", "-c", cpus] if cpus else []) + cmd


def json_lines(out):
    return {r["name"]: r for r in (json.loads(l) for l in out.splitlines() if l.startswith("{"))}


def throughput(r):
    scale = {"GFLOP/s": 1e9, "GB/s": 1e9, "samples/s": 1.0, "tokens/s": 1.0}[r["unit"]]
    return r["work"] / (r["median_ms"] / 1e3) / scale


def transformer(cpus, python, warmup, steps, threads):
    flags = ["--warmup", str(warmup), "--steps", str(steps)]
    outs = {
        "ferro": sh(pinned([str(BENCH / "target/release/bench_transformer"), *flags], cpus)),
        "torch": sh(pinned([python, str(ROOT / "examples/bench_torch.py"), *flags, "--threads", str(threads)], cpus)),
    }
    rows = {}
    for side, out in outs.items():
        tps = float(re.search(r"throughput: (\d+) tokens/sec", out).group(1))
        m = re.search(r"mean=([\d.]+) p50=([\d.]+) p90=([\d.]+)", out)
        cfg = re.search(r"config: (.*)", out).group(1)
        rows[side] = {"name": "transformer_train_step", "group": "model", "shape": cfg, "unit": "tokens/s",
                      "work": tps * float(m.group(1)) / 1e3, "median_ms": float(m.group(2)),
                      "p90_ms": float(m.group(3)), "warmup": warmup, "iters": steps}
    return rows


def cpu_model():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def thread_count(cpus):
    probe = "import os;print(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count())"
    return int(sh(pinned([sys.executable, "-c", probe], cpus)))


def machine(python, cpus, threads):
    git = sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()
    dirty = bool(sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--", ".", ":!benchmarks/results"]).strip())
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "commit": git + ("-dirty" if dirty else ""),
        "cpu": cpu_model(),
        "cpus": cpus or f"all ({os.cpu_count()})",
        "threads": threads,
        "rustc": sh(["rustc", "--version"]).strip(),
        "torch": sh([python, "-c", "import torch;print(torch.__version__)"]).strip(),
        "os": platform.platform(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="interpreter with torch installed")
    ap.add_argument("--cpus", default="", help="taskset CPU list applied to both sides, e.g. 0 or 0-3")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--filter", default="")
    ap.add_argument("--backend", default="fast", choices=["fast", "core"])
    ap.add_argument("--transformer", action="store_true", help="also run bench_transformer vs bench_torch.py")
    ap.add_argument("--transformer-steps", type=int, default=20)
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    args = ap.parse_args()

    bins = ["--bin", "bench_suite"] + (["--bin", "bench_transformer"] if args.transformer else [])
    subprocess.run(["cargo", "build", "--release", "--quiet", "--manifest-path", str(BENCH / "Cargo.toml"), *bins], check=True)

    threads = thread_count(args.cpus)
    common = ["--warmup", str(args.warmup), "--iters", str(args.iters), "--filter", args.filter]
    print("running ferro suite...", file=sys.stderr)
    ferro = json_lines(sh(pinned([str(BENCH / "target/release/bench_suite"), *common, "--backend", args.backend], args.cpus)))
    print("running torch suite...", file=sys.stderr)
    torch = json_lines(sh(pinned([args.python, str(ROOT / "examples/bench_suite_torch.py"), *common, "--threads", str(threads)], args.cpus)))
    if args.transformer:
        print("running transformer step...", file=sys.stderr)
        t = transformer(args.cpus, args.python, args.warmup, args.transformer_steps, threads)
        ferro[t["ferro"]["name"]], torch[t["torch"]["name"]] = t["ferro"], t["torch"]

    meta = machine(args.python, args.cpus, threads) | {"ferro_backend": args.backend, "warmup": args.warmup, "iters": args.iters}
    if args.transformer:
        meta["transformer_steps"] = args.transformer_steps
    lines = [
        f"{meta['date']} | ferro {meta['commit']} ({meta['ferro_backend']} backend) | torch {meta['torch']} | "
        f"{meta['cpu']}, {meta['threads']} threads (cpus: {meta['cpus']}) | warmup={args.warmup} iters={args.iters}"
        + (f" (transformer: {args.transformer_steps} steps)" if args.transformer else ""),
        "",
        "| case | shape | ferro ms | torch ms | ferro | torch | unit | ferro / torch speed |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    rows = []
    for name, f in ferro.items():
        t = torch.get(name)
        if t is None:
            continue
        speed = t["median_ms"] / f["median_ms"]
        rows.append({"name": name, "shape": f["shape"], "unit": f["unit"], "ferro": f, "torch": t, "speed": speed})
        lines.append(f"| {name} | {f['shape']} | {f['median_ms']:.3f} | {t['median_ms']:.3f} | "
                     f"{throughput(f):.1f} | {throughput(t):.1f} | {f['unit']} | {speed:.2f}x |")
    lines += ["", "ms columns are medians; speed > 1 means ferro is faster than torch on that case."]
    table = "\n".join(lines)
    print(table)
    if args.markdown:
        Path(args.markdown).write_text(table + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps({"meta": meta, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
