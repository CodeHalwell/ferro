# Fused CUDA LayerNorm wave

## Outcome

Implemented one-enqueue rowwise LayerNorm for whole contiguous CUDA f32 tensors,
including independently optional weight and bias, training, and inference replay.
No CUDA graph architecture changes and no new core dependencies.

Same-session full transformer medians (microseconds, lower is better):

| Path | Baseline normal | Baseline reversed | Fused normal | Fused reversed |
| --- | ---: | ---: | ---: | ---: |
| Ferro eager | 437.30 | 417.95 | 284.65 | 280.30 |
| Ferro compiled replay | 364.50 | 357.85 | 267.75 | 269.75 |
| PyTorch eager | 270.75 | 275.10 | 282.95 | 262.35 |

Paired improvements over the refreshed baseline: Ferro eager 1.49-1.54x
(32.9-34.9% lower latency), compiled replay 1.33-1.36x (24.6-26.5% lower).
These are workload-specific observations, not universal speedups. Ferro versus
PyTorch eager changes ranking with comparator order: no consistent PyTorch win
is claimed. Isolated LayerNorm timings also vary by order; do not add isolated
component timings or subtract them from full-model time.

Hardware: RTX 3090, Windows, torch 2.6.0+cu124, f32, TF32 disabled;
transformer 128 tokens, width 256, 8 heads, 15 warmups and 60 samples.
All timings use stream fences, not readbacks. Both inference paths are detached.
Every full-model run validates original, changed and restored inputs and weights.
Post-change full-model maximum absolute error against torch is 9.536743e-7.
The actual fullgraph Inductor attempt remains blocked by a missing working Triton
installation; raw error rows are retained. No substitute compiler was timed.

## Design

- `Backend::layer_norm_dev` is fallible and defaults to unsupported. It returns
  output, normalized input, and per-row standard deviation buffers.
- `layer_norm.cu` uses one 256-thread block per row, shifted double-precision
  accumulation of the mean and a separate centered variance reduction. It avoids
  the unstable E[x*x]-E[x]^2 formula. Strided thread loops handle awkward and wide
  rows; 64-bit loop indices avoid wrapping at the u32 input-size limit.
- The same enqueue fully writes all three output buffers. Empty outer dimensions
  enqueue nothing. Backend checks sizes, multiplication overflow and affine lengths.
- Core tries the fused seam first, then the existing fallible decomposition, then
  the visible CPU fallback. Storage read locks are deduplicated and address-ordered.
- Backward composes resident tensor operations using private saved normalized data,
  row standard deviations and an unchanged protected weight snapshot. A backend
  declining backward primitives uses the shared host backward formula; autograd
  places gradients on their original devices. Backward itself is not fused.
- Capture records one semantic LayerNorm node. Replay reads current leaves and
  weights; no graph planner or capture architecture was changed.
- Transformer replay topology changed from 54 operations / 22 leaves / 44 runs to
  34 / 18 / 32. The benchmark now checks exact operation and operand counts rather than the
  obsolete minimum of 35 operations. This structural-count guard does not prove
  graph connectivity or run count; input/weight mutation parity is checked separately. Each affine LayerNorm changed from 11 ops to one.
- `layer_norm_counts()` counts successful LayerNorm enqueues plus f32 upload and
  download helper calls. Tests calibrate the counters and assert (1,0,0) per eager
  or replay LayerNorm. Pointwise enqueue counts do not move. These are isolated
  seam proofs, not a claim of a measured zero-transfer whole-model timeline.
  External DLPack copies and in-place copy_into are outside these helper counters.

## Verification

- True GPU RED before production edits: `red.log` reports decomposed pointwise
  counts (0,4,3), expected (0,0,0). Initial test API typo was corrected before this
  behavioral RED. `green1.log` passes after the fused inference implementation.
- Counters: missing API compile RED in `counters-red.log`, then real GPU GREEN in
  `counters-green.log`. Training forward separately failed the structural gate in
  `training-red.log`, then passed in `training-green.log`.
- `regression.log`: actual GPU values, gradients and devices, protected affine
  snapshots, independently optional affine parameters, ranks 1/2/3, empty rows,
  widths 1/3/31/256/513/8193, alias/writer stress, current-input/current-weight
  replay, detached replay inside capture, direct bounds, and large-offset f64
  reference checks including inputs near 1e20. Default missing-GPU runs explicitly
  skip; FERRO_REQUIRE_CUDA=1 turns initialization failure into failure.
- `partial.log`: deliberate fused decline and a fused-only backend declining
  backward primitives preserve values, replay and gradient placement.
- Full integration: 12 commands passed; core 630 passed / 2 existing ignored,
  CUDA 59 passed / 0 ignored. Includes fastcpu/tokenizer, CUDA all-target check,
  Python capture/fusion/general/ops/safetensors and 200-trial torch fuzz checks.
- Standalone capture-layout GPU manifest: 1 passed.
- Both baseline and final Python extensions were installed with maturin release
  builds from their respective source states. No rustc crashes occurred this wave.
  Existing compiler warnings remain. No commits, staging, pushes or branch changes.

The first post benchmark stopped on the obsolete topology guard after some eager
samples; this incomplete attempt is retained as `post-topology-red.log` and is not
used in the comparison. After fixing only the benchmark guard, the all-case smoke
passed, then both complete post runs were collected serially. No builds/tests ran
concurrently with the baseline or post timings.

## Exact commands and evidence

Run from `C:/Users/DanielHalwell/PythonProjects/ferro` under Git Bash:

```bash
export PATH="$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH"
# Before any production edits, and again after implementation:
(cd crates/ferro-py && source .venv/Scripts/activate && CARGO_BUILD_JOBS=2 maturin develop --release)
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --json verification/layernorm-wave/baseline1.json
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --reverse --json verification/layernorm-wave/baseline2.json

FERRO_REQUIRE_CUDA=1 cargo test -j2 -p ferro-cuda --test layer_norm_fused -- --test-threads=1
cargo test -j2 -p ferro-core --test layer_norm_partial
cargo test -j2 -p ferro-core
FERRO_REQUIRE_CUDA=1 cargo test -j2 -p ferro-cuda -- --test-threads=1
export FERRO_REQUIRE_CUDA=1 RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=2
crates/ferro-py/.venv/Scripts/python.exe verification/run_integration.py
cargo test -j2 --manifest-path verification/capture-layout-gpu/Cargo.toml -- --test-threads=1
crates/ferro-py/.venv/Scripts/python.exe bench/model_graphs.py --json verification/layernorm-wave/final-smoke.json

crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --json verification/layernorm-wave/post1.json
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --reverse --json verification/layernorm-wave/post2.json
crates/ferro-py/.venv/Scripts/python.exe verification/layernorm-wave/summarize.py
git diff --check
```

Raw samples, summary JSON, summarizer, RED/GREEN logs, final build log, standalone
manifest log, and copies of all integration logs live in
`verification/layernorm-wave/`. Baseline build output is in the execution transcript;
`final-build.log` records the final release build. Repeated test command logs are
named above; `integration-results.json` records the exact integration commands.

Changed production files: core `dispatch.rs`, `ops_ext/layer_norm.rs`; CUDA `lib.rs`
and new `layer_norm.cu`. New regression files: core `tests/layer_norm_partial.rs`,
CUDA `tests/layer_norm_fused.rs`. Benchmark update: `bench/model_graphs.py`.


## Output-only inference follow-up

The CUDA inference seam now requests only the output buffer (one request rather
than three). The `OUTPUT_ONLY` specialization of the shared CUDA source has no
normalized-input or row-statistic output arguments or stores. The reductions and
numeric algorithm are unchanged. Training still requests and saves all three
buffers, and its protected affine snapshot and backward path are unchanged.
`Backend::layer_norm_output_dev` defaults to the existing fallible three-output
seam, preserving fused-only and declining partial backends. This compatibility
fallback does not promise the CUDA allocation reduction on other backends.

Core chooses the output-only seam when no argument requires gradients, including
inference capture. Capture still records the current input, independently optional
weight/bias, and epsilon in one semantic node, without retaining backward buffers.
Compiled replay also uses the output-only path, even for a graph originally
recorded with gradients.

Strict behavioral GPU RED is retained in
`verification/layernorm-output-only/red.log`: allocator requests were 3, expected
1, before production changes. GREEN and the final CUDA suite prove one output
request for inference and replay, three for training, exactly one successful
nonempty LayerNorm enqueue, no decomposed pointwise enqueue, and zero f32 helper
uploads/downloads. Empty inference returns an empty output without an enqueue.
Existing value, gradient, alias/writer, fresh-input/fresh-weight replay and partial
backend regression tests are preserved.

Final release bindings were rebuilt with maturin. All 12 commands imported from
`verification/run_integration.py` passed, run serially by the scoped
`verification/layernorm-output-only/run_checks.py` to avoid overwriting previous
wave evidence. Core: 630 passed, 2 existing ignored. CUDA: 59 passed, none ignored.
The standalone capture-layout GPU manifest also passed its one test. Required-GPU
mode, CUDA runtime PATH, two build jobs and one Rust test thread were enabled.
Existing compiler warnings remain; no staging, commits, pushes or branch changes.

After builds and tests, two full profile runs ran serially, normal then reversed
comparator order. Raw JSON and logs are `layernorm-output-only/post1` and `post2`;
prior `layernorm-wave/post1` and `post2` were not overwritten. Full-transformer
medians in microseconds:

| Path | Historical normal | Historical reversed | Output-only normal | Output-only reversed |
| --- | ---: | ---: | ---: | ---: |
| Ferro eager | 284.65 | 280.30 | 280.60 | 484.65 |
| Ferro compiled replay | 267.75 | 269.75 | 267.40 | 270.50 |
| PyTorch eager | 282.95 | 262.35 | 267.25 | 506.80 |

**No latency speedup is established.** Compiled replay is effectively unchanged;
reversed-run Ferro eager and PyTorch eager both slow substantially. The latter is
an observed order/run sensitivity, not an attributed root cause. Historical
comparison is not a contemporaneous causal A/B experiment. The structural
allocation/store reduction is proven independently of timings. Component wall
timings remain non-additive. Both full-model runs passed original, changed and
restored input/weight parity; actual Inductor attempts remain blocked by missing
working Triton. Summary and integration command/exit records are retained beside
the raw JSON. The earlier wave's performance claims above describe that earlier
baseline comparison, not this output-only follow-up.

Reproduce from the repo with the CUDA PATH prefix shown above:

```bash
export FERRO_REQUIRE_CUDA=1 RUST_TEST_THREADS=1 CARGO_BUILD_JOBS=2
(cd crates/ferro-py && source .venv/Scripts/activate && maturin develop --release)
crates/ferro-py/.venv/Scripts/python.exe verification/layernorm-output-only/run_checks.py
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --json verification/layernorm-output-only/post1.json
crates/ferro-py/.venv/Scripts/python.exe bench/profile_model_graphs.py --reverse --json verification/layernorm-output-only/post2.json
```
