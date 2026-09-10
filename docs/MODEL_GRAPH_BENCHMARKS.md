# Model-graph inference benchmarks

## Verified integration

`bench/model_graphs.py` now captures each full Ferro model with
`ferro.capture(forward).compile_fused()` once and times only `handle.replay()`.
It does not replay a pointwise tail seeded by old matmul outputs. The compiled
schedule evaluates matmuls, branches, reductions and layouts from current leaves.
This is a host-scheduled DAG with fused pointwise runs, NOT a CUDA graph, a
whole-model generated kernel, a training compiler, or fused backward.

## Workloads

- MLP: [128,256] input, 256->1024 projection, bias, tanh-approximate GELU,
  1024->256 projection and bias.
- Residual MLP: the same MLP plus its original input.
- Transformer: batch one, 128 tokens, width 256, eight heads; affine pre-norm,
  independent biased Q/K/V projections, head reshape/transposes, scaled QK^T,
  softmax, attention-times-V, head merge and output projection, residual,
  affine LayerNorm, the same GELU MLP and second residual. Bidirectional
  attention; no dropout, mask, positional encoding, KV cache or FlashAttention.

The compiled graphs contain 5/6/54 operations and 4/5/44 scheduled runs for
MLP/residual/transformer respectively. Runs are not measured kernel launches.
Shared subexpressions execute once; layout operations remain schedule boundaries.

## Method and limits

Windows build 26200; NVIDIA GeForce RTX 3090; PyTorch 2.6.0+cu124 (CUDA 12.4).
Float32, Torch TF32 disabled, one Torch CPU thread. Torch runs inside
`torch.inference_mode`; Ferro inputs, weights and replay outputs do not require
gradients. DLPack setup imports copy identical Torch fixtures into Ferro.
Changed/restored input uses checked copy into captured storage, not Python rebinding.

All available paths pass original, changed (`x*0.73+0.19`) and restored-input
Torch parity before timing (rtol 2e-4, atol 2e-5). Full-model Python regressions
also change a shared projection weight; Rust MLP tests mutate both input and
weight. CPU/CUDA LayerNorm-layout values and gradients match Torch.

Ferro's CUDA LayerNorm composes resident sum_dim, broadcast and sqrt kernels.
Strided reshape materializes using device gather: it constructs/uploads i64
layout indices on each call, but does not download activation data. The harness
explicitly reshapes transposed heads before bmm, ensuring whole contiguous bmm
operands. A counting backend test (`resident_model.rs`) proves zero activation
readbacks through normalization, reshape and replay. Existing device tests cover
matmul/broadcast/softmax dispatch. This is not a profiler trace of the entire
hardware run; unsupported backends retain ordinary host fallbacks. Layout index
construction, upload and allocation costs ARE included in the measured call.
Backward layout operations can still stage through host; no resident training
claim is made here.

Fixture/reference setup and backend preparation including first execution are
separate millisecond fields in JSON. Preparation includes eager capture, validation,
DLPack import and lazy compilation, not just compiler time. Ten warmups precede
40 samples. Every full Python-visible call is bracketed with transfer-free CUDA
fences; median wall latency includes dispatch, allocations and the trailing fence.
No output export, .cpu(), .item(), or parity readback is inside timing.

Two complete runs were executed serially after tests/builds finished. Before the
runs nvidia-smi reported 0% GPU utilization (desktop applications remained open).
Run two reverses backend order. Large run/order sensitivity is visible below:
these short Windows wall-latency results do not establish an MLP speedup.
Transformer replay is slower than Torch eager in both runs; replay's improvement
relative to Ferro eager is inconsistent. Do not headline lucky minima or extend
historical pointwise bandwidth results to these models.

## Actual medians (microseconds)

| Run | Workload | Torch eager | Ferro eager | Ferro compiled replay |
| --- | --- | ---: | ---: | ---: |
| 1 | mlp | 61.10 | 49.55 | 45.80 |
| 1 | residual_mlp | 57.00 | 49.50 | 81.85 |
| 1 | transformer | 521.70 | 1821.65 | 922.60 |
| 2 | mlp | 61.10 | 79.25 | 47.45 |
| 2 | residual_mlp | 121.35 | 50.60 | 47.90 |
| 2 | transformer | 268.05 | 1339.45 | 1311.30 |

Torch compile was attempted with `backend="inductor", fullgraph=True,
dynamic=False`. All three workloads in both runs failed with
`BackendCompilerFailed: Cannot find a working triton installation`.
There are NO Torch compiled timings, no backend=eager substitution and no error
suppression. This is a limitation of this installation, not a universal Windows claim.

Each raw report contains 12 rows: nine passed/timed and three blocked without
latency fields. Counts, 40-sample lengths and medians were programmatically checked.

## Reproduce (Git Bash, repository root)

```bash
export PATH="$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH"
PY=crates/ferro-py/.venv/Scripts/python.exe
"$PY" bench/model_graphs.py --json verification/model-smoke.json
"$PY" bench/model_graphs.py --benchmark --json verification/model-graphs-run1.json
"$PY" bench/model_graphs.py --benchmark --backends ferro_compiled ferro_eager torch_compile torch_eager --json verification/model-graphs-run2.json
```

Evidence: `verification/model-smoke.json`, `verification/model-graphs-run1.json`,
`verification/model-graphs-run2.json`, matching `.log` files and
`verification/model-gpu-before.log`. Default invocation is parity-only; only
`--benchmark` produces steady-state samples. Exit success may include blocked
backends: inspect every status before interpreting results.

Remaining: broader forward-op coverage, efficient cached/generated layout
materialization, backward fusion, and CUDA-graph capture of the whole compiled
model. Explicitly detached computations outside capture remain opaque inputs.
