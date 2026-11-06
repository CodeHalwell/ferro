"""Run the ferro and PyTorch GPU suites (examples/bench_gpu.rs and its twin
examples/bench_gpu_torch.py) on cuda:0 and print a side-by-side table.

    python benchmarks/compare_gpu.py --python .venv/bin/python \
        [--warmup 10] [--iters 50] [--filter s] [--tf32] [--json out.json] [--markdown out.md]

The torch side runs under --python, which needs a CUDA build of torch. Both
sides sync the device around every timed iteration and report medians.
TF32 is off on the torch side unless --tf32 (ferro has no TF32 path yet).
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def lines(out):
    return {r["name"]: r for r in (json.loads(l) for l in out.splitlines() if l.startswith("{"))}


def rate(r):
    return r["work"] / (r["median_ms"] / 1e3) / (1e9 if r["unit"] in ("GFLOP/s", "GB/s") else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python3")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--filter", default="")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    a = ap.parse_args()
    common = ["--warmup", str(a.warmup), "--iters", str(a.iters), "--filter", a.filter]
    sh(["cargo", "build", "--release", "--features", "cuda", "--bin", "bench_gpu"], cwd=ROOT / "benchmarks")
    ferro = lines(sh([str(ROOT / "benchmarks/target/release/bench_gpu"), *common]))
    torch = lines(sh([a.python, str(ROOT / "examples/bench_gpu_torch.py"), *common, *(["--tf32"] if a.tf32 else [])]))
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip()
    commit = sh(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).strip()
    tv = sh([a.python, "-c", "import torch; print(torch.__version__, torch.version.cuda)"]).strip()
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "gpu": gpu, "ferro": commit, "torch": tv, "tf32": a.tf32}
    rows = ["| case | shape | ferro ms | torch ms | ferro | torch | unit | ferro / torch speed |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for name, f in ferro.items():
        t = torch.get(name)
        if not t:
            continue
        rows.append(f"| {name} | {f['shape']} | {f['median_ms']:.3f} | {t['median_ms']:.3f} | {rate(f):.1f} | {rate(t):.1f} | {f['unit']} | {t['median_ms'] / f['median_ms']:.2f}x |")
    table = f"{meta['date']} | {gpu} | ferro {commit} | torch {tv} | TF32 {'on' if a.tf32 else 'off'}\n\n" + "\n".join(rows)
    print(table)
    if a.markdown:
        Path(a.markdown).write_text(table + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps({"meta": meta, "ferro": ferro, "torch": torch}, indent=1))


if __name__ == "__main__":
    main()
