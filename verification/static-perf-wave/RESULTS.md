# Static replay performance wave (2026-09-12)

## Historical stage outcome

These samples belong to the layout/guard performance stage, before later PR24
core, DLPack and native-ownership changes. They are not final-tree performance
or a benchmark of the review fixes. Archived timing JSON and metadata are kept
unchanged; no historical build provenance has been backfilled.

Transformer static+snapshot improved from fresh pre-edit medians of
344.85 / 337.45 us to 238.60 / 229.75 us (30.81% / 31.92% lower).
Replay without snapshot improved from 333.40 / 343.45 us to
245.25 / 281.95 us (26.44% / 17.91% lower). These are same-session,
normal/reversed-order full-model measurements, not the historical 336/338 us
numbers. Strong order/run sensitivity remains elsewhere; no universal speedup
or stable MLP/residual speedup is claimed.

## Changes and attribution

- `crates/ferro-cuda/src/static_graph/model.rs`: map logical slots to unique
  owned allocations; eliminate interior identity-layout destinations and copy
  commands. Singleton strides are ignored only when their dimension is one.
  All consumers are remapped, including GEMM, pointwise, normalization,
  transposes and softmax scratch. Noncontiguous layouts still materialize.
  Final layouts deliberately still copy: a public output lease must remain
  private and must not become an externally mutable leaf alias.
- `crates/ferro-cuda/src/static_graph.rs`: track original leaves plus exported
  output at replay boundaries, not inaccessible scratch. Preparation fences
  warmup, every replay uses the validated backend stream, the graph is neither
  Send nor Sync, scratch never escapes, and Drop fences before releasing it.
  Context-wide event tracking and original leaf/output dependencies remain.
- Shared primary-context capture/failure tests now use a poison-tolerant mutex
  for their entire lifetime. Two default-parallel runs previously crashed with
  native STATUS_ACCESS_VIOLATION, while serialized execution passed. The common
  fixture lock fixes the changed-module parallel reproduction; the exact native
  failing instruction was not identified. This is test isolation, not a claimed
  general CUDA concurrency fix.
- `crates/ferro-cuda/examples/static_gemm_probe.rs`: reproducible isolated GEMM
  graph probe using the same owned 4 MiB workspace, fence-only timings, ten
  warmups and forty samples, with reversed order in the second round.

The layout-only full-model trial (`layout1.json`) reached transformer
230.15 us replay / 234.80 us snapshot, before the scratch-guard change.
The next trial (`guards1.json`) was 231.10 / 229.55 us. This supports layout
materialization as the substantial contributor, not a separately established
scratch-event latency win. Guard removal is a verified structural reduction;
its independent timing effect is within observed noise.

Plain SGEMM versus batch-one strided SGEMM did not establish a useful advantage:
for 128x256x256, medians were 32.00/31.90 us then 15.10/15.10 us;
128x256x1024 gave 17.50/17.50 then 17.50/17.55 us;
128x1024x256 gave 18.90/18.90 in both orders (plain/batched).
**GEMM production selection is unchanged.** Component wall times are
non-additive and are not a driver timeline or proof of equal native algorithms.

## Historical full-model evidence

All float32 workloads use 128 tokens, width 256, eight transformer heads,
TF32 off, Torch one CPU thread, inference mode, transfer-free fences and
forty timed samples after ten warmups. Each comparator passes original,
changed-input, changed-input-and-weight, restored-state parity first, with
rtol=2e-4 and atol=2e-5 and elementwise tolerance ratios saved in JSON.
Snapshot completes its independent D2D copy before return. Output destruction
is outside timing. Setup and first call are reported separately.

Median us; run 2 reverses comparator order:

| State/run | Model | Torch eager | Ferro eager | Host compiled | Static | Static+snapshot |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 1 | MLP | 95.55 | 72.85 | 69.10 | 59.70 | 62.00 |
| Baseline 2 | MLP | 92.15 | 51.00 | 68.75 | 47.40 | 43.30 |
| Stage final 1 | MLP | 64.55 | 51.20 | 71.90 | 47.20 | 43.20 |
| Stage final 2 | MLP | 94.65 | 72.70 | 69.85 | 47.10 | 61.50 |
| Baseline 1 | Residual | 60.55 | 53.45 | 86.15 | 48.90 | 44.50 |
| Baseline 2 | Residual | 56.85 | 91.10 | 82.55 | 48.80 | 44.40 |
| Stage final 1 | Residual | 67.90 | 90.45 | 85.75 | 41.90 | 70.10 |
| Stage final 2 | Residual | 121.05 | 90.75 | 83.15 | 48.20 | 44.40 |
| Baseline 1 | Transformer | 267.60 | 495.65 | 267.80 | 333.40 | 344.85 |
| Baseline 2 | Transformer | 499.60 | 471.45 | 469.05 | 343.45 | 337.45 |
| Stage final 1 | Transformer | 531.80 | 291.40 | 267.30 | 245.25 | 238.60 |
| Stage final 2 | Transformer | 685.30 | 480.50 | 271.05 | 281.95 | 229.75 |

Disclosed adverse rows: MLP snapshot 43.30 -> 61.50 us in reverse order
(+42.03%); residual snapshot 44.50 -> 70.10 us in normal order (+57.53%).
The opposite order does not reproduce either regression. These rows, and eager
comparators varying widely without source changes, prevent a blanket latency
claim. They are preserved, not replaced by favorable reruns.

Both real Inductor modes were attempted for every model. Each final report has
15 passing timed rows and six missing-working-Triton blockers. No compiler
result was replaced by eager and no missing metadata was retrofitted.
`final-summary.json` validates the explicit final input paths and integration.
`baseline-summary.json` validates only the two pre-edit benchmark reports.
Initial baselines used the already-installed release binding; subsequent
layout, guard and stage-final trials explicitly rebuilt release bindings.
`metadata.json` is a historical, incomplete snapshot: its source allowlist omitted
core lowering and other dependencies, and its binding lookup could silently be
empty on another installation. It does not establish final PR24 tree provenance
or prove which source produced any binary. A current snapshot cannot repair this
retroactively. Fresh summarization now requires an active or explicit native
binding and hashes tracked/non-ignored untracked crate, bench and verification
source dependencies, root Cargo manifests/lockfile and Cargo/toolchain config.
That snapshot is explicitly labeled current-at-summarization, NOT historical
run/build attestation. Third-party installed libraries and build environment are
not a hermetic build manifest.

## Historical stage verification

- Required-GPU RED/GREEN: identity layouts requested three buffers before and
  one after; chained/shared aliases add zero captured nodes relative to direct
  GEMM. Tests cover source mutation, ownership after source drop, persistent
  snapshots, singleton strides, transpose preservation, softmax scratch and
  independent final-layout copies. Replay makes no new allocation requests.
- Required-GPU RED/GREEN: a three-command pointwise graph acquired four usage
  guards before and two after (original leaf and public output).
- Delayed foreign-stream output read passes with the exported-output guard.
  A temporary sensitivity experiment removing that guard failed with overwritten
  2.0 values instead of original 1.0 values. The guard was restored immediately;
  the final changed module passed again under the default parallel harness.
- Full serial integration: all 13 commands passed, including core, CUDA,
  fastcpu/tokenizer, standalone GPU capture-layout, CUDA all-target check,
  Python discovery/general/operator/fusion/safetensors and 200-trial fuzz.
  Parsed Rust pass-line sums: core 640, CUDA 80, fastcpu/tokenizer 20,
  standalone capture-layout 1. These are harness sums, not unique-test claims;
  bounded nested core child-process tests contribute repeated summaries.
- Changed static module: eight tests passed in default-parallel mode after
  fixture locking, both before and after the output-guard sensitivity trial.
  `model_static_graph` and `static_graph_safety`: ten passed in default mode.
- Release bindings rebuilt before full integration and final measurements.
  No core/Python production edits were necessary. Existing compiler warnings
  remain. `git diff --check` passed. No staging, commit, push or branch changes.

## Reproduction

From the repository root in Windows Git Bash, with unused output names.
These commands verify/measure the current source, not historical RED states.
Prerequisites: a working CUDA driver/GPU, installed NVRTC/cuBLAS runtime and a
release binding rebuilt from the intended tree. `run_checks.py` prepends and
validates the documented runtime directories; `--runtime-root` overrides their
parent nvidia directory. Directory existence is not CUDA execution proof.
`--plan` only prints validated paths/commands; real checks retain
`FERRO_REQUIRE_CUDA=1` and fail rather than treat required GPU absence as success.
On POSIX use `$(pwd)/crates/ferro-py/.venv`, its `bin` directory and your system
CUDA loader setup instead of the Windows DLL prefix.

The summarizer is a separate historical comparison utility. Invoke it with
`--input-dir` containing all six stage-named reports, `--output-dir` naming a new
directory, and optionally `--binding path/to/ferro.pyd` (or native `.so`). Without
that flag it resolves the active interpreter installation without loading CUDA.
It refuses missing/ambiguous native bindings and existing output directories.
Do not write new snapshots into archived `metadata.json`.

```bash
export VIRTUAL_ENV="$(pwd -W)/crates/ferro-py/.venv"
export PATH="$VIRTUAL_ENV/Scripts:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin:$LOCALAPPDATA/Temp/cuda-rt/nvidia/cublas/bin:$PATH"
python -m maturin develop --release -j2 --manifest-path crates/ferro-py/Cargo.toml
python verification/static-perf-wave/run_checks.py --output-dir verification/static-perf-wave/new-checks
FERRO_REQUIRE_CUDA=1 cargo test -j2 -p ferro-cuda --lib static_graph
FERRO_REQUIRE_CUDA=1 cargo test -j2 -p ferro-cuda --test model_static_graph --test static_graph_safety
cargo run --release -j2 -p ferro-cuda --example static_gemm_probe
python verification/model-cuda-graphs/continuation/benchmark.py --json verification/static-perf-wave/new1.json
python verification/model-cuda-graphs/continuation/benchmark.py --reverse --json verification/static-perf-wave/new2.json
python verification/model-cuda-graphs/continuation/verify_results.py --runs verification/static-perf-wave/new1.json verification/static-perf-wave/new2.json --summary verification/static-perf-wave/new-summary.json --integration-dir verification/static-perf-wave/new-checks
```

Raw logs include the expected RED failures, first missing-VIRTUAL_ENV build
attempt, two probe compile errors corrected before execution, and native
parallel failures. No fabricated timings or discarded adverse trials.
