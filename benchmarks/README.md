# ferro vs PyTorch throughput benchmark (Tier-1 gate harness)

Trains one pre-norm transformer block (token embedding -> causal multi-head
self-attention -> Gelu MLP at 4x width -> RMSNorm residuals -> LM head) with
AdamW on fixed synthetic data, and reports tokens/sec plus step-time
percentiles. The Rust binary (`examples/bench_transformer.rs`) and the Python
twin (`examples/bench_torch.py`) build the same architecture with the same
parameter count, same shapes, same steps, same optimizer settings
(AdamW, lr 1e-4), so their outputs are directly comparable.

## Running

Rust side (from `benchmarks/`, release build is mandatory):

```
cd benchmarks
cargo run --release --bin bench_transformer -- \
    --batch 8 --seq 128 --d-model 256 --heads 4 --vocab 1024 \
    --warmup 100 --steps 500 --device cpu
```

Add `--features cuda` and `--device cuda` to attempt the CUDA backend; if the
driver or an op transfer is unavailable the harness prints why and falls back
to CPU (the printed `device:` line always says what actually ran).

Python side (CPU-only torch wheel by default; swap the index URL for a CUDA
wheel to compare on GPU):

```
cd benchmarks
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
cd ..
.venv/Scripts/python.exe examples/bench_torch.py   # same CLI flags
```

(on POSIX use `.venv/bin/` instead of `.venv/Scripts/`.)

Timing notes:
- Both sides run N warmup steps before timing; neither timed region contains a
  host sync (`loss.item()` is never called on either side).
- Step-time percentiles are per-step wall clock over the timed region.
- tokens/sec = batch * seq * steps / total_timed_time.

## Results

Machine: Windows 11, RTX 3090 (CUDA 13.1) - CPU runs below used the CPU
backend only. Default config unless noted:
batch=8 seq=128 d_model=256 heads=4 vocab=1024, params = 1,313,536
(identical count on both sides). Each table states its warmup/timed steps.

### Post-kernel wave (softmax/log_softmax/gelu now run as device kernels)

warmup=10 timed=30 on both sides:

| Harness | Device | tok/s | step mean ms | p50 | p90 | p99 |
|---|---|---|---|---|---|---|
| ferro | cuda:0 | 3,554 | 288.16 | 279.56 | 335.28 | 341.81 |
| ferro | cpu | 3,411 | 300.20 | 300.28 | 303.70 | 304.52 |
| PyTorch 2.13.0+cpu | cpu | 79,917 | 12.81 | 13.01 | 13.48 | 13.90 |

CUDA is no longer blocked by i64 transfer and now trains end to end on GPU,
edging out ferro's own CPU path. Moving softmax/log_softmax/gelu onto nvrtc
device kernels removed the per-op host round-trips that previously dominated
the CUDA step time (~2,440-2,630 tok/s before, at ~0% GPU utilisation).
Remaining overhead is still op-graph dispatch rather than kernels; the
remaining host-composed ops in the attention path are the next lever.

### Earlier baseline (pre-kernel wave)

warmup=100 timed=500, CPU backend only:

| Harness | Device | tok/s | step mean ms | p50 | p90 | p99 |
|---|---|---|---|---|---|---|
| ferro (fastcpu kernel installed) | cpu | 2,750 | 372.30 | 363.46 | 413.24 | 479.96 |
| PyTorch 2.13.0+cpu | cpu | 74,647 | 13.72 | 13.85 | 14.52 | 15.99 |
| ferro (cuda attempted) | cpu (fell back: no i64 device transfer yet) | 2,750* | - | - | - | - |

\* cuda attempt ran the CPU fallback path; i64 device residency has since
landed (see PARITY_SESSION_REPORT.md Round 4).

**Measured ratio: ferro CPU sits at ~4% of PyTorch eager CPU throughput
(PyTorch ~20x faster); ferro CUDA reaches ~4.4% of torch CPU.** The CPU gap
is dominated by per-op overhead - elementwise kernels, RMSNorm/embedding
paths, and AdamW are single-threaded host loops re-materializing tensors per
op, while torch uses vectorized multithreaded ATen kernels throughout. Matmul
goes through ferro-fastcpu's blocked AVX2 kernel; softmax/gelu now also have
fastcpu/device paths.

### Known gaps

- No fused SDPA / fused AdamW on the Rust side; every backward materializes
  full-size intermediates.
- Reduction ops still compose on host between device launches.

## Results table template

| Harness | Commit / version | Device | batch | seq | d_model | heads | vocab | tok/s | step mean ms | p50 | p90 | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ferro |  |  |  |  |  |  |  |  |  |  |  |  |
| pytorch |  |  |  |  |  |  |  |  |  |  |  |  |
