# Full transformer inference profile

Measured 2026-09-09/10 on Windows 11, RTX 3090 (WDDM), driver 591.86.
Source baseline: `846948476d9251b1ceb34b7369d384776e49f8a7`.
No production changes, builds, commits or pushes were made for this profile.

## Outcome

Prioritize **layout materialization**, then **resident LayerNorm fusion**.
The evidence is isolated end-to-end component timing plus source inspection,
NOT a CUDA kernel timeline or an additive full-model attribution. Matmul and
attention compute are not the first targets at this shape. Scheduling and
allocation remain unresolved subcomponents of the measured call wall time.

## Workload and method

The harness imports `bench/model_graphs.py` and times its actual complete
pre-norm transformer: two affine LayerNorms, Q/K/V projections, scaled
non-causal multi-head attention, output projection and residual, then GELU
MLP and residual. Tokens=128, width=256, heads=8, seed=18, FP32, TF32 disabled.
Torch is 2.6.0+cu124, inference mode, one CPU thread; Ferro leaves are detached.
Both implementations use identical mathematical inputs and weights.

Each of four fresh processes runs serially: 15 warmups, 60 fence-inclusive
samples per backend and component. Runs 2 and 4 reverse backend order. Raw
samples, medians, min/max, quartiles, errors and compiler metadata are in
`verification/profile/run{1,2,3,4}.json`; console logs are
`verification/profile-run{1,2,3,4}.log`. Outputs are retained until after the
final fence, then dropped outside timing. Intermediate allocation/recycling
inside each call remains timed. DLPack setup and correctness exports are
outside timing. The fence is `ferro.cuda_synchronize` or
`torch.cuda.synchronize`, never a host activation transfer.

Full-model gates pass original, changed, restored input and changed/restored
Q weight against freshly evaluated Torch. Maximum absolute error across those
checks is 1.431e-6 eager and 1.193e-6 compiled (rounded upward), tolerance
rtol=2e-4, atol=2e-5. Component gates also pass; their inputs are actual Torch
intermediates from the same model, imported contiguously before timing.
Component replay compiles the entire component, not a precomputed output tail.

## Full-model baseline

Median wall latency in microseconds (us), including final fence:

| Backend | Run 1 | Run 2 reversed | Run 3 | Run 4 reversed |
|---|---:|---:|---:|---:|
| Torch eager | 274.25 | 276.20 | 278.20 | 278.85 |
| Ferro eager | 1409.25 | 977.30 | 979.40 | 1000.35 |
| Ferro compiled | 1075.00 | 923.85 | 923.45 | 926.10 |
| Torch Inductor fullgraph | blocked | blocked | blocked | blocked |

Inductor was actually attempted in every run and failed with `Cannot find a
working triton installation`; no eager compiler substitution or environment
hack. The complete error is saved in each JSON. No compiled-Torch speed claim.

Run 1 is noisy and is retained, not silently discarded: Ferro eager sample
quartiles 967.6/1450.0 us, compiled 921.0/1363.6 us. Runs 2-4 full-model
compiled medians agree closely, while eager ranges 977.3-1000.35 us. Both
Ferro paths remain materially slower than Torch eager for this model. Do not
use the favorable run-1 eager/compiled ratio as an optimization headline.

Ferro compiled exposes 54 recorded operations, 22 leaves and 44 scheduled
runs. These are **compiler counters, not measured CUDA kernel launches**.

## Isolated components: target selection, not latency shares

The table gives the range of the four run medians, in us. Each row has its own
fence and independently prepared inputs. Do not sum rows, multiply a row by
its occurrence count to claim total attribution, or subtract their sum from
full-model time to invent scheduling overhead. Some Torch and Ferro component
runs show large WDDM/order noise; the raw samples retain it.

| Component | Torch eager | Ferro eager | Ferro compiled |
|---|---:|---:|---:|
| LayerNorm 1 | 19.60-36.90 | 89.60-196.85 | 73.00-116.80 |
| LayerNorm 2 | 19.70-31.30 | 89.20-90.60 | 73.00-116.05 |
| Q layout materialization (one of Q/K/V) | 18.20-37.90 | 164.65-166.45 | 165.40-253.35 |
| K transpose materialization | 18.00-194.30 | 165.60-256.95 | 165.90-249.15 |
| Attention merge materialization | 17.50-41.10 | 165.10-188.60 | 165.10-257.80 |
| QK batched matmul | 32.10-52.70 | 16.10-33.90 | 16.30-34.00 |
| Attention softmax | 16.00-20.25 | 22.90-23.80 | 22.90-38.50 |
| Probabilities-V batched matmul | 43.20-52.50 | 26.70-44.50 | 26.70-41.55 |
| Output projection and residual | 34.00-87.90 | 29.60-55.10 | 28.35-29.70 |
| MLP up, bias, GELU | 40.00-83.30 | 33.95-52.60 | 28.70-41.75 |
| MLP down, bias, residual | 38.10-87.15 | 26.05-51.40 | 33.30-48.40 |

Individual Q/K/V projection timings and all samples are in JSON. Layout Q is
particularly defensible: eager Ferro is consistently about 165 us in every
run, while the Torch materialized equivalent is 18.2-37.9 us. Its compiled
wrapper does not remove the materialization cost. K layout's Torch outlier
means that isolated row alone would not support a universal ratio claim.
LayerNorm 2 provides the cleaner normalization baseline than LayerNorm 1;
the latter has much larger run-to-run noise despite the same shape.

## What is known about transfers, scheduling and allocation

Source evidence at the baseline commit:

- `crates/ferro-core/src/tensor.rs:869-888,898-903`: noncontiguous device
  reshape constructs a host `Vec<i64>` with per-element coordinate arithmetic,
  calls `alloc_i64_from_host`, then `gather_rows_dev`.
- `crates/ferro-cuda/src/lib.rs:1560-1564,1587-1631`: that index path uploads
  i64 data, allocates output storage and launches a gather.
- The model explicitly materializes Q, K, V, transposed K, and attention
  merge. At this shape each has 32768 elements: **source-derived expected**
  index payload is 5 * 128 * 256 * 8 = 1,310,720 bytes per forward. This is
  not a measured transfer counter or PCIe bandwidth measurement. Timing each
  layout includes host index generation, upload, allocation, gather and fence;
  these costs cannot be separately attributed by this harness.
- `crates/ferro-core/src/ops_ext/layer_norm.rs:45-69`: resident normalization
  composes reductions and pointwise operations, with per-call scale/epsilon
  creation in eager mode. Isolated captures report 11 operations and 7
  scheduled runs, not 7 physical kernels. Compiled leaves retain constants.
- `crates/ferro-cuda/src/lib.rs:411-431` exposes Rust allocation stats;
  `crates/ferro-py/src/lib.rs` does not bind them. No measured allocator miss,
  allocation-count, or peak-live-storage claim is possible through this API.

Each sample separately records `call_wall` and `final_fence`. Full-model
compiled call-wall medians are 1047.05, 879.60, 876.25, 888.65 us; final-fence
medians are 23.90, 44.25, 47.30, 37.30 us. Eager call-wall medians are 1383.60,
963.20, 949.05, 967.00 us. Call wall contains Python dispatch, Rust scheduling,
allocation, CUDA submission **and possible blocking driver work**. It is NOT
pure CPU time or exclusive scheduler overhead. Separate medians need not add
to total medians. The empty Ferro fence-control median is 0.4-0.7 us, which
shows the idle fence itself is small, not that pending-work fences are free.

Profiler discovery: neither `nsys` nor `ncu` is on PATH; targeted searches of
NVIDIA Corporation / CUDA Toolkit installation directories found no CLI.
A broader Program Files search encountered a search-tool error, so this is
not a claim of exhaustive absence on the machine. Torch reports only
`ProfilerActivity.CPU`, not CUDA. No big installs were attempted. Consequently
there is no CUDA event timeline, hardware counter capture, exclusive kernel
breakdown or measured transfer trace in this report. Desktop graphics
processes share this WDDM GPU; initial nvidia-smi showed 0% utilization,
34 C, P8. Post-run telemetry is `verification/profile/gpu-after.csv`.

## Follow-on optimization acceptance criteria

1. **Eliminate repeated host layout-index construction/upload.** Prefer a
   stride-aware device copy or layout-aware matmul over dynamic i64 indices;
   immutable per-plan index caching is a narrower alternative but retains
   gather traffic and storage. First instrument index-upload bytes/count,
   gather enqueue count and allocations at the backend seam. Validate all
   five layouts, changed leaves and compiled replay, including bounds and
   noncontiguous semantics. Re-run this identical benchmark in both orders.
   This is the strongest target, but no percentage of full-model latency or
   promised speedup is justified yet.
2. **Dedicated affine row LayerNorm / equivalent fused lowering.** Preserve
   biased variance, eps=1e-5, FP32 numerical behavior and detached capture
   semantics. Use the two component gates plus changed-input/weight full-model
   gates; add appropriate cancellation/constant-row correctness cases before
   production changes. Measure actual enqueue/allocation reduction rather
   than equating scheduled runs with kernels. Baseline LN2 eager 89.2-90.6 us,
   compiled 73.0-116.05 us; Torch eager 19.7-31.3 us.
3. **Instrument before scheduling/allocation redesign.** Expose/read backend
   counters or use a Rust-only profiling binary in a separately authorized
   change. Partition host index generation, HtoD calls, freelist hit/miss,
   scheduler traversal and kernel enqueue time with a real profiler where
   available. The current call/fence split cannot identify these separately.
4. **Defer attention/matmul kernel work for this shape.** Component timings do
   not indicate these are the dominant avoidable regression. This says
   nothing about long context, larger widths, training, FP16 or fused SDPA;
   those require distinct full-model profiles.

## Reproduction

From `C:/Users/DanielHalwell/PythonProjects/ferro` in bash:

```bash
export PATH="$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH"
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --json verification/profile/run1.json > verification/profile-run1.log 2>&1
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --reverse --json verification/profile/run2.json > verification/profile-run2.log 2>&1
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --json verification/profile/run3.json > verification/profile-run3.log 2>&1
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --reverse --json verification/profile/run4.json > verification/profile-run4.log 2>&1
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate,utilization.gpu,memory.used --format=csv > verification/profile/gpu-after.csv
```

All four harness processes returned exit code 0. Each JSON has 47 rows: 4
full-model attempts (one explicit compiler block), 42 component rows, and
one empty-fence control. Existing untracked verification artifacts were not
modified. Production test suites were not rerun because no production source
was changed; correctness was exercised by the actual model/component gates.
