# Continuing PyTorch/PyG work: verified candidate 18

This matching-build checkpoint records ongoing CPU implementation. Full
PyTorch/PyG parity remains open. Work continues after the isolated snapshot;
all changes remain local and uncommitted. The prior green checkpoint is kept
in [candidate16](../candidate16/README.md), and the failed intermediate full
comparison remains in [candidate17](../candidate17/README.md).

References are PyTorch 2.13.0+cpu and PyG 2.8.1, used only in development.
Ferro keeps its zero-dependency Rust core and one record_fn/Op autograd engine.
No NVIDIA or Apple GPU is available; Metal is not implemented.

## Matching-build validation

The noneditable ABI3 wheel comes from an isolated copy. All 961 selected
source/fixture inputs and 109 installed files (108 Python plus native) were
rechecked after validation. Source ZIP, wheel and raw logs are retained in
`/workspace/work/completeness-2026-11-03`; [manifest.json](manifest.json)
records their exact hashes and outcomes. The selection includes all 64 portable
aggregation checkpoint fixture files, including hidden generation payloads.

| Check | Result |
|---|---|
| Installed Python discovery | Exit 0: 1,413 discovered, 1,383 passed, 30 GPU cases unavailable. |
| Rust workspace, default test threads | Exit 0: core 1,199 passed / 2 ignored; fastcpu 39 passed / 2 ignored; CUDA 140 runtime harness passes / 5 strict GPU cases filtered; tokenizer 8 harness passes. |
| CUDA doctests | Two intended E0502 borrow errors pass; no E0460 artifact mismatch. |
| CUDA all-target compilation | Exit 0 without a CUDA driver/GPU. |
| Native DLPack | 12 harness passes, one GPU early return, three mandatory GPU cases filtered; explicit Python 3.12 link. |
| Binding, compiled fusion, fuse, operator/Torch and safetensors | All exit 0. |
| Fuzz | 200 trials, seed 0, exit 0. |
| Graph training/restart | Learning/control and exact fresh-process continuation proofs pass. |
| API inventory | Installed public names and hashes recorded; names are not semantic/performance scores. |

Unittest reports 1,407 executions and 25 skip indicators: 24 individual GPU skips
plus one class hiding six cases. Rust totals use final outer summaries: 1,386
runtime passes plus two doctests; 18 parsed nested child summaries are excluded.
Optional GPU/tokenizer bodies can return early, so harness passes do not prove
hardware execution. Workspace cudarc is 0.19.10; this separately locked Python
build uses 0.19.9. Subsequent source aligns the Python lock, requiring next-build
verification. Both retain selected CUDA 12.8 bindings.

The Git-ignored workspace Cargo.lock was resolved during native verification;
its retained bytes/hash are recorded separately as resolved build input, rather
than represented as part of the original 961-input ZIP. Subsequent snapshots
include this lock explicitly. The Python lock was already frozen in this build.

## Additions since candidate16

- Native fused add/sub alpha, exact typed scalar categories/promotion and
  Python named arithmetic. Original F64 alpha precision survives mixed JVP
  directions. U64 floating forward succeeds; only a requested invalid right
  adjoint fails, before ordinary gradient publication. Square now supports
  genuine higher/direct derivatives with Torch's overflow/subnormal order.
- Shape-only view/view_as retain real storage/stride/offset aliases and reject
  layouts requiring copies. Direct layout comparison covers 6,449 Torch cases.
  Existing captured device-leaf snapshot policy remains separate from eager
  and CPU replay alias behavior.
- JVP composes with grad/grad_and_value on the same native recorder. Immutable
  Op rebasing preserves actual root flags, saved versions and opaque callbacks,
  without re-running the user function. Auxiliary no_grad views retain their
  blocked reverse edge and base flags. Installed 70 methods plus direct 18,
  composed 18 and arithmetic/view 16 network configurations pass.
- Typed floor/ceil/trunc/round/frac/rsqrt support real F32/F64 storage, precise
  decimal stages, exact I64 rounding copies and integral rsqrt promotion.
  591 independent layouts match Torch. Rsqrt saves its original output, allows
  unused input-value mutation and uses that output for graph-building VJPs.
- Condition numbers and Cholesky-factor solve/inverse extend real CPU linalg.
  Cholesky adjoints match Torch's full factor-manifold convention, including
  ignored-triangle entries; this differs from entrywise finite differences.
  Singular solves have explicitly undefined nonfinite numerical results.
- AlphaDropout/FeatureAlphaDropout, canonical normalization keywords and general
  p/tuple-axis/F64 normalize. Native p2 preserves NaNs and avoids allocating
  irrelevant huge denominators for empty output. Norm accumulation can select
  a different clamp branch exactly at eps when rounding differs from Torch.
- DenseGAT, public custom Aggregation and dense Data/Batch cat/inc hooks cover
  pair/bipartite node spaces, typed offsets, follow_batch and current-value
  inverse separation with live gradients. Twenty-one old aggregation families
  strictly restore frozen candidate16 checkpoint fixtures. Custom dataset
  persistence, sparse axes, nested fields and negative increments remain open.
- Cat and casts use value-free saved-input semantics while nonlinear producer,
  consumer and saved-output guards remain active. Candidate17's three failed
  methods are repaired and pass in this build.

Earlier typed compute/storage/selection, Parameter interoperability, embedding
options, Module traversal/hooks, 14 optimizer families/schedules, atomic typed
checkpoints, recurrence/attention/spatial layers and graph catalogs remain
covered. Exact prior scopes are preserved in the candidate16 report and active
[compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md).

## Remaining boundaries

Parameters/optimizers remain CPU F32; default Parameter construction owns data.
F64 graph-building VJPs/transform primals, full dtype/operator coverage,
half/complex/FFT, arbitrary advanced indexing/out destinations, detach_ edge
generations, complete transform composition/vmap/compiler/export/distributed
support and complete NVIDIA/Metal execution remain open. Existing linalg/norm
and opaque operator higher-order boundaries are explicit.

Custom Batch scalar zero-offset inverse extraction currently copies the first
field where PyG preserves a view; values/gradients pass, and a source19 alias
repair is separate. Sqrt arithmetic, closeness helpers, standard multi-root
autograd signatures, activation options and graph selectors continue afterward.
No full framework, GPU-residency or performance claim follows from this report.
