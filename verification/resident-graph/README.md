# Resident graph primitives and CPU GNN restart verification

Run: 2026-09-19, Windows, Python 3.11.15, Torch 2.6.0+cu124, local CUDA device 0.
Base: `73b33fc7664d7a503dadd9b240a83f9d719a68dc` (merged PR #26).
Branch: `feat-resident-graph-primitives`. The verification run preceded the
publication commit; source hashes identify the tested implementation.

[Evidence](evidence.json) records the exact commands/exits, environment, package
hashes and GNN results. [Source manifest](sources.json) identifies all 437 selected
source/build inputs, including untracked additions; every hash was unchanged
before/after execution and checked again when publishing this report. Documents
are outside that source manifest and were updated after execution. All Python
facades matched source/wheel/install bytes; the native wheel/install bytes matched.

- Wheel SHA-256: `b4fba8df9adc64072ba5deeac08b7f2ecb277cf695e0c707f1cb5480e18eade9`
- Native SHA-256: `8419b508165f2d3621891f5f2834583dcdab45f564b30016b11b181fe66d2466`
- Local raw logs/wheel: `target/resident-graph/run-01/` (ignored build artifacts).
- Reproduction: `.venv/Scripts/python.exe -B verification/resident-graph/verify.py --output-dir target/resident-graph/run-02`
  Use a new directory each run; the collector refuses to overwrite evidence.
  CUDA is required by this collector. It adds installed Torch runtime DLLs to PATH
  on Windows; install the normal build and test dependencies first.

## Implemented and tested surface

PreparedSegments exposes gather (axis zero), select (arbitrary axis) and
out-of-place scatter_add using one-dimensional immutable integer IDs. Duplicates
accumulate. Convenience Tensor.index_select prepares a plan per call on capable
f32 device backends; tensor IDs still download for validation. Prepared plans
avoid repeated topology uploads. Other dtype/partial-backend fallback paths remain.
This is index-select semantics, not general elementwise torch.gather/scatter.

COO.prepare(device) returns reusable SpMM/SDDMM; both operands have first-order
f32 gradients on CPU/CUDA. CUDA keeps features, outputs and adjoints resident,
including tested transposes, and uses O(edges * features) scratch. COO.batch
returns block-diagonal topology and cumulative row/column offsets, preserving
edge order, duplicates, rectangular graphs and isolated nodes. Unprepared
COO/CSR algebra remains CPU; convert CSR to COO before preparation.

Prepared CUDA sum/softmax prerequisites are included in this branch. Sum/gather
use stable CSR edge order without floating-point atomics. Softmax performs one
explicit four-byte nonfinite-status read per nonempty forward: zero feature
transfers does not mean zero total transfers. Mean/max remain CPU-only. Capture
and higher-order differentiation are outside the prepared graph contract.

The counting-backend seed/reshape regression first failed with two extra uploads
and two downloads. It now asserts zero. Real CUDA tests assert zero feature
uploads/downloads and zero repeated topology uploads for selection/scatter and
SpMM/SDDMM forward+backward. Device materialization is allowed and counted
separately. The initial CUDA test wrongly compared that materialization counter
with upload counters; correcting the assertion exposed no implementation failure.
Operational materialization errors now propagate instead of silently falling back.

## Matching-build results

All 17 collector commands exited zero.

| Suite | Result |
|---|---|
| Core Rust | 776 passed, 2 ignored performance probes |
| CUDA Rust | 121 passed, none ignored; required CUDA graph tests executed |
| Fast CPU + tokenizer | 33 passed, 2 ignored performance probes |
| Python unittest discovery | 114 passed, no skips |
| Rust Python bindings | 15 passed |
| CPU regression/classifier/GNN examples | All three seeds each; fresh-process exact restart |
| Python binding, Torch ops, safetensors, compiled/fusion scripts | All passed |
| Differential fuzz | 200 trials, seed 0; all 19 operations met existing gates |
| CUDA all-target compile check | Passed |

Core totals exclude nine successful filtered child-test runs already represented
by parent tests. Suites/examples overlap; do not add all counts as unique tests.
Ignored probes are named in evidence.json. Existing differential-fuzz reduction
and matmul/BMM gates are percentile-based and permit large maximum ULP outliers;
this run does not resolve the historical CPU numerical or illegal-instruction
investigations. No performance, independent review or hosted-CI claim is made.

## Selected GNN learning/restart proof

The public CPU model performs one-hop neighbor aggregation followed by a learned
tanh readout over 32 disconnected four-node graphs. Fixed disjoint feature draws
use dataset seeds 2026/2027; targets use an independent scalar neighborhood
formula. The learning gate was declared before execution: seeds 7/11/43, 400
steps, held-out MSE <=25% of the training-mean constant baseline. Gradient checks
use independent dense Torch adjacency and atol=1e-5/rtol=1e-4 for input and every
parameter. No private bindings or embedded engine implementations are used.

| Seed | Held-out MSE | Baseline ratio | MSE with messages removed | Restart |
|---|---|---|---|---|
| 7 | 0.000125044 | 0.000452319 | 0.232748 | bit-exact |
| 11 | 0.000056130 | 0.000203036 | 0.232655 | bit-exact |
| 43 | 0.000143261 | 0.000518215 | 0.232685 | bit-exact |

Constant baseline MSE: 0.276451. Maximum gradient discrepancy: 1.1921e-7.
Removing neighbor aggregation keeps the same learned readout and is an ablation,
not a separately trained MLP baseline. The ablation confirms the tested model's
prediction depends on graph messages.

At step 200 gradients are cleared and model/optimizer/config/explicit Generator
state is checkpointed. A fresh process rebuilds the graph plan, restores into
fresh compatible objects and compares the remaining 200 losses, held-out metrics,
next RNG draws and serialized state bytes (excluding publication directory name).
Changing one topology coordinate is rejected with all target state unchanged.
Topology is immutable model config; prepared device allocations are not serialized.

Only this CPU f32 GNN variant is certified. Whole CUDA GNN training/residency,
CUDA checkpoint restore, dynamic topology, loader cursors, sparse graph transformers,
connectome models, general indexed kernels and cross-platform bitwise restart
remain open. The original dirty checkout was preserved; work is isolated in the
sibling `ferro-gnn` worktree.
