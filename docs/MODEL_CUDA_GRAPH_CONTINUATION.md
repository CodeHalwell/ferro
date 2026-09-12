# Whole-model static CUDA graph continuation

## Delivered scope

The full MLP, residual MLP and bidirectional transformer from
`bench/model_graphs.py` now run as prepared CUDA graphs, not frozen pointwise
 tails or allocating host-scheduled replay. Core lowering and Python exposure
are implemented. This supersedes the missing-integration status in the
historical `MODEL_CUDA_GRAPHS.md` report.

```python
root = ferro.capture(forward)  # the entire inference expression
compiled = root.compile_fused()
graph = compiled.prepare_static()
graph.replay()                # one CUDA graph launch; returns None
saved = graph.snapshot()      # explicit allocation + device copy
x.copy_(new_x)                # same captured Tensor/storage identity
weight.copy_(new_weight)
graph.replay()
# saved remains unchanged; graph.snapshot() obtains the current result
```

Capture, compile and preparation are setup. Replacing a Python model field with
an unrelated Tensor does not rebind graph leaves; prepare again instead.
Scalar hyperparameters and shapes are static. Rust has the corresponding
`CompiledChain::prepare_static`, `PreparedChain::replay`, `snapshot` and
`replay_count` API. Python StaticGraph is thread-confined; its output does not
escape except through independent snapshots. The pre-existing allocating
`CompiledChain::replay` API retains its semantics.

## Implementation and ownership

- `dispatch.rs`: backend-neutral StaticRun/StaticOp and StaticExecution seam.
  Nonimplementing backends return Unsupported; core remains dependency-free.
- `graph.rs`: lower the existing topological tail schedule. Preserve shared
  branches, actual leaves and semantic nonpointwise metadata. Remove only
  inactive LayerNorm affine placeholders, never computed model tails.
- `tensor.rs`: f32 device Storage owns an Arc of the original DeviceBuffer.
  This pins graph pointers without copying CudaSlice or adding StorageCell
  aliases. Existing checked mutation, autograd and ordinary view gates remain.
- Replay acquires distinct StorageCell write guards in address order. Exclusive
  guards are necessary because existing device mutation uses shared guards.
  Tensor identities and original buffers remain owned for graph lifetime.
- `static_graph/model.rs`: prepared destination commands reuse existing kernels
  for fused pointwise/bias broadcasts, layout materialization, last-axis
  softmax, sum_dim and output-only LayerNorm. Matmul/bmm use a dedicated cuBLAS
  handle, including strided-batched GEMM with batch=1 for matmul. This may choose
  a different cuBLAS algorithm than eager matmul; parity is tolerance-based.
- Allocate all destinations and softmax scratch before capture. Every layout
  boundary currently materializes to a distinct buffer, even a plain reshape.
  No liveness reuse optimization is claimed. Use a private nondefault capture
  stream, with a private, owned 4 MiB cuBLAS workspace and checked setter status.
- The ordinary backend's leaked raw cuBLAS workspace was also replaced with
  owned CudaSlice storage and a checked setter.
- Preserve the primitive's tracked original-buffer replay boundary, exact
  backend-stream rejection, RAII capture/graph ownership and sticky poisoning.
  No global event-tracking disable or legacy step capture is used. Private
  warmup work is fenced during error cleanup before destination storage drops.
  Graph resources are destroyed before releasing legacy-capture exclusion.

Supported preparation is static f32 inference with whole resident leaves on
one validated CUDA backend. Unsupported cases fail explicitly: empty
 destinations, zero-reduction GEMM, expanding pointwise seeds, non-last-axis
softmax, layouts above rank 16, host/mixed-device leaves, gradient-requiring
leaves and operations without recorded forward metadata. Ordinary eager
fallbacks still exist, but are never used by this static path. No training,
KV-cache, FlashAttention, dynamic-shape or arbitrary operator coverage claim.

## Executed verification

All evidence here is under `verification/model-cuda-graphs/continuation/`;
previous evidence and unrelated epsilon edits were preserved. Nothing was
staged, committed or pushed.

- RED/GREEN: core pointwise vertical path, matmul, full attention+MLP, optional
  LayerNorm affine combinations, Python API, malformed operand overflow and
  owned backend workspace. Failed intermediate build logs are retained.
- Required-GPU full-model tests mutate both input and weights, retain snapshots,
  compare eager results, and assert unchanged Ferro allocation requests,
  pointwise host-launch counts and existing upload/download counters during
  replay. These counters are not a CUDA-driver timeline or allocation trace.
- Concurrent whole-tensor update/replay stress passes with consistent outputs;
  graph-owned leaves remain usable after caller/root destruction.
- Injected failure exits after warmup enqueue, captured enqueue, successful
  end/instantiate and upload recover and permit another graph. Existing
  unwind/nested-capture/poison/foreign-stream/output-lease tests also pass.
  These are not exhaustive injected native-driver failures at every API call.
- Final release bindings rebuilt from the verified production source.
- Serialized integration: all 13 commands passed; parsed Rust harness totals
  are 630 core, 20 fastcpu/tokenizer, 73 CUDA and 1 standalone capture-layout
  test. Python discovery has 10 passing tests, including the new full static
  model. General/operator/safetensors/fusion and 200-trial fuzz checks passed.
  Existing compiler warnings and documented platform skips remain.

Reproduce with the runtime DLL directories prepended to PATH:

```bash
export PATH="$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH"
PY=crates/ferro-py/.venv/Scripts/python.exe
python -m unittest discover -s verification/model-cuda-graphs/continuation -p test_verify_results.py -v
"$PY" verification/model-cuda-graphs/continuation/run_checks.py
"$PY" verification/model-cuda-graphs/continuation/benchmark.py --json verification/model-cuda-graphs/continuation/new-run1.json
"$PY" verification/model-cuda-graphs/continuation/benchmark.py --reverse --json verification/model-cuda-graphs/continuation/new-run2.json
python verification/model-cuda-graphs/continuation/verify_results.py \
  --runs verification/model-cuda-graphs/continuation/new-run1.json verification/model-cuda-graphs/continuation/new-run2.json \
  --summary verification/model-cuda-graphs/continuation/new-verified-summary.json \
  --integration-dir verification/model-cuda-graphs/continuation
```

The integration runner sets FERRO_REQUIRE_CUDA=1 and RUST_TEST_THREADS=1 and
uses cargo -j2. Benchmarks must remain serial and run after builds/tests.
Run these commands from the repository root. Choose unused run/summary names
for each new measurement; preserve archived `run3.json`, `run4.json` and
`verified-summary.json`. The verifier requires two explicit input paths,
resolves them relative to the current working directory, records their absolute
paths, and refuses to overwrite an existing summary. Omit `--integration-dir`
for benchmark-only validation; that does not claim integration was verified.
The CPU regression fixtures are synthetic test data, not performance evidence,
and import neither Torch nor Ferro.

Fresh reports declare the original rtol=2e-4 and atol=2e-5 and record four
maximum absolute errors, reference absolute maxima and maximum elementwise
ratios `abs(actual-expected)/(atol+rtol*abs(expected))`. The verifier requires
finite nonnegative values at every gate, ratios <=1, and absolute errors within
`atol+rtol*max(abs(expected))`; the ratio gate preserves the elementwise relative
criterion rather than mistaking a global absolute bound for parity. Timings
must have 40 finite positive samples, exact recomputed median/min/max, and a
complete model/comparator grid in declared order. Failed/unknown statuses and
unrelated compiler/runtime exceptions fail verification. Only the two actual
Inductor modes may be blocked by explicit missing-working-Triton diagnostics
(`Cannot find a working triton installation` or `No module named 'triton'`).
Working compiler rows are accepted and validated, not forced into historical
15-passed/6-blocked counts.

The hardened verifier intentionally rejects archived reports lacking the new
parity metadata: it cannot retrospectively prove the elementwise tolerance
from maximum absolute errors alone. Run two fresh benchmarks after integration;
the historical measurements below remain unchanged, not revalidated evidence.

## Final fenced benchmark evidence

Final validated evidence is `review-run1.json` and `review-run2.json`, with
`review-summary.json`; the second run reverses comparator order. These reports
include synchronous snapshot completion and the required parity metadata.
The hardened verifier validated 15 passing timed rows and 6 explicitly blocked
compiler rows per run, all four finite bounded freshness gates, and timing
statistics recomputed from 40 samples per row. Transformer static+snapshot
medians are 336.60 / 338.50 microseconds. Results remain order-sensitive; no
stable overall speedup is established. Full final timings and commands are in
`verification/pr23-lock-review/APPROVED-RESULTS.md`.

## Historical pre-fence benchmark results

The table and measurements below describe archived `run3.json` and `run4.json`,
not the final synchronous-snapshot implementation. These archives lack the new
parity metadata and cannot pass the current verifier; do not cite this table
as final validated performance evidence. They originally recorded 15 timed
rows and 6 blocked compiler rows, with 40 samples following 10 warmups.
Initial run1/run2 evidence is retained but superseded: their compiler probes
hit a harness module-registration error. After fixing that harness error,
actual Inductor attempts reach the missing-Triton failure below.

Hardware: RTX 3090, driver 591.86, Windows; PyTorch 2.6.0+cu124. Before the first
benchmark nvidia-smi reported 0% GPU utilization, 784 MiB used, 33 C. Both
frameworks run float32 inference in the same interpreter; TF32 is disabled,
Torch uses one CPU thread. Workloads are the existing 128-token, width-256 MLP,
residual MLP and eight-head transformer (34 recorded operations for transformer).

Each comparator passes original, changed-input, changed-input-plus-weight and
restored-state Torch parity before timing (rtol=2e-4, atol=2e-5). Static paths'
maximum observed absolute error is below 1.7e-6. Input/weight copies and parity
readback are excluded from timing. Each Python-visible call includes a trailing
transfer-free CUDA fence. Outputs are released outside timing. Fixture setup,
preparation and first-call times are separate JSON fields; preparation includes
imports, capture and cache warming, not just graph instantiation.

Median call latency in microseconds:

| Run | Model | Torch eager | Ferro eager | Host compiled | Static replay | Static + snapshot |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | MLP | 64.80 | 52.00 | 48.90 | 40.80 | 43.20 |
| 4 | MLP | 85.80 | 51.60 | 49.80 | 40.70 | 43.20 |
| 3 | Residual MLP | 66.40 | 61.30 | 71.25 | 41.90 | 54.50 |
| 4 | Residual MLP | 94.70 | 51.65 | 50.00 | 41.90 | 44.40 |
| 3 | Transformer | 278.95 | 295.10 | 272.90 | 432.70 | 335.70 |
| 4 | Transformer | 280.95 | 279.65 | 298.90 | 339.60 | 371.60 |

Static replay returns no public output alias; static+snapshot includes the
explicit result copy and is the closer output-semantics comparator to eager.
MLP static+snapshot was faster in both archived runs, but transformer static replay
is slower than eager and host-compiled execution. Run/order noise is visible;
no general transformer speedup is claimed. Full per-buffer event boundaries,
materialized reshape buffers and cuBLAS selection are possible optimization
candidates, not profiler-proven explanations.

Both actual Torch compiler modes were attempted for every model in both runs:
Inductor fullgraph/dynamic=False with triton.cudagraphs=False, and Inductor
reduce-overhead. Both fail with `Cannot find a working triton installation`.
There are no compiler timings, eager substitutions or suppressed errors. This
is an environment blocker, not a claim that Windows cannot support Inductor.
