# Current capability ledger and next milestone

> Current publication scope and merged-tree evidence: [stabilisation publication](STABILISATION_PUBLICATION.md). Earlier local verification links below are historical provenance, not shipped artifacts. CPU training/restart from PR26 and operators from PR28 are retained.

Original ledger updated 2026-09-19; the scoped 2026-11-03 continuation is recorded
below. This is the current roadmap status index. The
[foundations roadmap](FOUNDATIONS_AND_ARCHITECTURE_ROADMAP.md) retains the
detailed B0-B7 acceptance criteria; [FUTURE.md](FUTURE.md) and
[PARITY_ROADMAP.md](PARITY_ROADMAP.md) retain earlier strategy.

## 2026-11-04 active completeness continuation

The latest verified checkpoint is [candidate32](../verification/completeness-2026-11-03/README.md),
prepared for [PR #33](https://github.com/CodeHalwell/ferro/pull/33).
Python: 1,807 passed and 30 GPU cases unavailable; native: 1,702 outer passes
plus two doctests. CUDA compilation, DLPack and eight standalones pass.

This adds physical-batch Jacobians, shared-output PReLU higher/JVP rules,
ownership-aware accumulation and creation-priority backward scheduling,
unused-gradient/power-zero layouts, and CPU F64 Conv1d/Conv2d. Earlier typed
training, modules and graph coverage remains. Full framework parity is open;
Metal and hardware GPU acceptance remain open. Unfinished follow-ups remain
isolated outside the committed checkpoint. See [handoff](CONTINUATION_HANDOFF.md).

## 2026-11-03 portability and compatibility continuation (historical)

The next implementation stage adds real CPU Bool storage, comparisons, logical
reductions, exact boolean counts, typed selection, gradient-preserving scalar
and bool indexing, safetensors BOOL and DLPack boolean interchange. Boolean
node/subgraph masks retain isolated nodes and survive graph batching.

Training now supports native optimizer parameter groups, owning state dictionaries
and four Python schedulers, with exact fresh-process continuation and atomic
restore validation. New synchronous data pipelines support custom map-style and
iterable datasets, nested typed collation, samplers, subsets and concatenation.
The synchronous native TensorDataset path now yields one batch at a time while
retaining its exact Rust sampler order; native worker mode still holds an epoch.

The graph API adds HeteroData/HeteroConv, RGCN, GINE, APPNP, Cheb and TransformerConv,
plus homogeneous NeighborLoader/LinkNeighborLoader, negative sampling and link
splitting. Supported hetero relations, edge attributes, gradients, shared-module
aliases and checkpoints have real PyG comparisons. Finite fanouts follow the
documented Ferro frontier algorithm. PyG's optional native sampler backend is
unavailable here; all-neighbor cyclic two-layer values/gradients compare against
full PyG models instead. General custom graph Batch index offsets reject.

Spatial additions include Conv1d/3d, ConvTranspose1d/2d and AdaptiveAvgPool1d/2d/3d.
MultiheadAttention supports both batched layouts, unbatched inputs, bool/additive
masks, padding masks and per-head weights. Functional attention supports CPU
masks/custom scale and safely handles fully masked rows. Nonzero attention and
recurrent dropout remain unsupported.

These additions retain the zero-dependency Rust core and existing autograd
mechanism. They are CPU compatibility work. NVIDIA CUDA and Apple Metal are the
GPU targets, with separate backend/residency/performance acceptance gates;
Metal is not yet implemented. See [GPU portability and remaining gaps](GPU_PORTABILITY.md),
the [compatibility matrix](PYTORCH_PYG_COMPATIBILITY.md) and
[matching-build report](../verification/portability-2026-11-03/README.md).
The final matching build passes 388 Python cases, with 30 CUDA cases unavailable,
plus 916 core and 38 fastcpu Rust tests. CUDA host tests/all-target compilation
pass; no GPU bodies were executed. The report identifies 579 unchanged source
inputs and 36 matching installed Python/native files. This adds 101 Python
test methods beyond the prior parity stage.
Full PyTorch/PyG parity, typed arithmetic, AMP, distributed training and broad
compiler/ecosystem coverage remain open. Changes remain local and uncommitted.

## 2026-11-03 PyTorch and PyG API expansion

The Python API now includes PyTorch-style tensor factories, typed metadata and
scalar access, functional shape operations, multi-axis reductions and broadcast
matmul. New `nn.RNN/GRU/LSTM` modules and cells cover stacked, bidirectional,
batch-first and unbatched CPU f32 networks with PyTorch parameter layouts.
Classification losses now support class weights, ignored indices, spatial and
probability targets, label smoothing and reference empty/zero-denominator cases.

The graph API adds PyG-style `GCNConv`, `SAGEConv`, `GINConv`, `GATConv` and
`GATv2Conv`, alongside homogeneous `Data`, `Batch`, `DataLoader`, global pooling,
grouped softmax and topology utilities. Sparse GATv2 uses query-dependent dynamic
attention, with optional shared projections and ordinary Ferro checkpointing.
Existing `graph.GcnConv` and native graph APIs remain available.

Actual PyTorch 2.13.0+cpu and PyG 2.8.1 comparisons cover forward values,
input/state/parameter/edge gradients, empty and strided inputs, state aliases
and checkpoint validation. The reference packages are development dependencies;
Ferro runtime modules do not import them, and core remains dependency-free.
CI installs both references explicitly. The
[compatibility matrix](PYTORCH_PYG_COMPATIBILITY.md) records supported signatures,
examples, deliberate tied-extrema differences and rejected options.

The comparisons exposed two numerical defects, now repaired: large-offset
log-softmax cancellation across core/AVX2/CUDA source, and normalization's
zero/tiny-vector gradients plus overflowed norm handling. New targeted native
and actual-Torch regressions cover both. CUDA tests compile, but real GPU
execution remains outstanding. This is a CPU f32 compatibility expansion,
not full Torch/PyG, heterogeneous-graph, distributed or mixed-precision parity.

See the [matching-build report](../verification/parity-2026-11-03/README.md) for
final validation and the integrated graph training/checkpoint example.
The final installed build passes 287 Python cases, with 30 CUDA cases unavailable;
core has 904 passes and fastcpu 38. CUDA host tests and all-target compilation
pass, with no GPU execution. The new example passes GCN/SAGE/GATv2 learning and
exact epoch-boundary fresh-process continuation at seed 7. All reach 100%
held-out accuracy and 50% with edges removed; GCN/SAGE begin at 100%, so their
large cross-entropy reduction establishes learning. The frozen report identifies
555 source inputs and 27 matching installed Python/native files.
All changes remain local and uncommitted.

## 2026-11-03 sparse attention, streaming state and pipeline continuation

Public `ferro.graph.edge_attention` now composes CPU f32 gather, scaled dot
products, per-destination segment softmax and segment sum. It supports optional
per-edge additive bias, duplicate edges, isolated nodes and eager first-order
gradients. It rejects capture and unsupported dtype/device/layout metadata.
The sparse forward/backward allocation regression uses 4096 nodes and 128 edges,
with total allocation below 8 MiB and no allocation above 256 KiB; a dense node
matrix alone would require 64 MiB. This is no claim of resident CUDA attention.

Two public examples pass fixed learning and exact fresh-process restart gates
across seeds 7, 11 and 43:

- `examples/train_restart_graph_attention.py` learns crossed query/key and
  edge-attribute routing on disjoint feature samples with repeated topology
  templates. At 400 updates, accuracy rises from 50% to 100% and cross-entropy
  falls below 0.0001. Query/attribute/edge removal and uniform attention score
  50%; rewired routing endpoints score 0%. Checkpoint step 197 preserves actual
  topology, attributes, data, cursor and model/optimizer/config/Generator state.
- `examples/train_restart_streaming.py` trains a delayed-sign LSTM across
  three chunks per episode, carrying detached hidden/cell state between updates.
  At 600 updates, final-chunk cross-entropy falls from 0.647-0.687 to
  0.0000224-0.0000565. Final-chunk accuracy was already 100% at initialization; the declared
  loss reduction establishes learning. Erasing state reduces continuation
  accuracy to 50%. Step 293 saves the actual nonzero h/c before the pending third
  chunk; resume does not recompute the prefix with updated weights.

Independent dense PyTorch/manual-LSTM references check all parameter and relevant
input gradients. Separate review uses `torch.nn.LSTM`, tests prefix-free resume,
and checks topology permutations and duplicate-edge invariance. Both fixtures
passed their first declared training configuration without budget/gate changes.
These are synthetic CPU f32 examples, not full graph-transformer certification,
arbitrary stream/data-loader checkpoints or cross-platform bitwise guarantees.

The same review repaired finite-mask leakage in causal attention, incompatible
rank-4 batch/head folding, invalid attention/RoPE configurations, and stale cached
causal masks after backend replacement. Rust data loading now reports unwinding
sample/collation panics as ordered errors, bounds all outstanding prefetched work,
and cancels waiting workers on iterator drop. Running user callbacks finish
independently. Python `Trainer.fit` no longer consumes an extra shuffled epoch
while checking whether a source supports multiple epochs.

See the [matching-build report](../verification/graph-sequence-2026-11-03/README.md)
for validation, exact API boundaries, source identity and independent review.
GPU execution and the historical numerical/native-crash investigations remain
open. All changes remain local and uncommitted.

## 2026-11-03 generative training and optimizer continuation

Two more public CPU f32 fixtures pass held-out learning and exact fresh-process
restart across seeds 7, 11 and 43:

- `examples/train_restart_vae.py` trains a Gaussian VAE with reparameterized
  latent samples, normalized Gaussian NLL and analytic KL. After 600 steps,
  held-out Monte Carlo negative ELBO is 0.5764-0.5797 against a constant baseline
  of 8.7466; decoder-mean MSE is 0.00128-0.00342 against 0.4051. Evaluation uses
  a saved fixed noise bank; restart at step 293 preserves subsequent latent draws.
- `examples/train_restart_wgan.py` learns a Gaussian location/scale transform
  with a tanh critic and differentiable gradient penalty. After 4000 cycles of
  five critic updates and one generator update, empirical held-out W1 is
  0.01315-0.11369, below the fixed 0.25 absolute and 25%-of-initial gates.
  Restart at update 11997 preserves the partially completed critic cycle,
  both named optimizers, data cursor and subsequent noise/interpolation draws.

Independent PyTorch checks cover separate objective terms and all relevant
gradients. WGAN checks inactive parameter/optimizer isolation; VAE checks that
reparameterization and KL paths remain live. The first WGAN configuration failed
seed 7's learning gate, and an independent PyTorch trajectory reproduced that
failure. Revision 2 changed critic initialization, learning rates, penalty weight
and budget before rerunning all seeds; quality gates stayed fixed. Penalty and
gradient-norm measurements are diagnostics, not Lipschitz certification.

Optimizer review repaired per-parameter AdamW step counting and shared mutable
moment buffers in public Rust SGD/Adam/AdamW clones. Frozen or absent gradients
now leave AdamW counters, moments and weights unchanged. Clones retain shared
parameter slots but own optimizer buffers. AdamW snapshot `t` is now f32 `[n]`;
new code accepts the legacy scalar and broadcasts its recorded global count.
Old readers cannot read the new vector form, historical skipped counts cannot
be reconstructed, and the existing large-counter f32 precision limit remains.
Capturable AdamW requires equal counters and all-or-none gradients; this has
host regression coverage, with GPU execution still outstanding.

See the [matching-build report](../verification/adversarial-2026-11-03/README.md)
for exact source/build identity, suites, migration details and failed experiments.
This closes only these synthetic eager CPU f32 fixtures. General generative
model quality, arbitrary data-loader state and real-GPU training remain open.
All changes remain local and uncommitted.

## 2026-11-03 differentiable-derivative training continuation

Two public CPU f32 fixtures now exercise mixed coordinate/parameter derivatives
inside real training objectives, with seeds 7, 11 and 43:

- `examples/train_restart_pinn.py` solves u'' = -pi^2 sin(pi*x), enforcing zero
  endpoint values with u=x(1-x)*MLP(2x-1). Only the strong-form residual enters
  training. At 1500 steps, held-out solution RMSE is 0.000320-0.000587 and
  residual RMSE is 0.0540-0.0851, below the declared 0.01 and 0.1 gates.
- `examples/train_restart_contractive.py` trains a smooth autoencoder with a
  live squared encoder-Jacobian penalty. At 400 steps, reconstruction MSE is
  0.270-0.326% of the constant baseline; the final penalty is 4.64-30.67% of
  its initial value, below the declared 10% and 75% gates respectively. Sorted
  minibatches failed seed 11 in both Ferro and PyTorch. A documented interleaved
  schedule passed with the sample set, model, optimizer, budget and gates retained.

All six runs pass exact fresh-process continuation, including saved data/cursors,
model/AdamW/config/Generator state and subsequent draws. Independent curvature,
Jacobian and finite-difference checks establish that the losses reach parameters
through coordinate derivatives. Detachment faults and optimized-Python acceptance
gates are tested. The [matching-build report](../verification/scientific-2026-11-03/README.md)
records the final suites, source identity and original failed schedule.

Review also repaired three functional-AD contract gaps: derivative capture could
freeze gradients and replay incorrect values; custom backward callbacks could
return non-f32 gradients; and invalid scalar-gradient operands could allocate an
implicit device seed before validation. Functional `grad`/`vjp` now reject capture,
enforce f32 callback gradients, and validate operands before seed allocation.
Ordinary infallible backward asserts gradient dtype before touching its gradient
mutex; it remains nontransactional. The supported higher-order subset is documented
in the Python autograd module. Independent probes passed 188 PyTorch comparisons
through third-order derivatives, with maximum absolute error 2.39e-7.

This closes only these eager CPU f32, synthetic, complete-step fixtures. The PINN
uses sampled coordinates as inputs and hard boundary conditions; the autoencoder penalty can
be reduced through encoder/decoder rescaling and proves no robustness property.
General PDEs, f64 AD, derivative capture and GPU certification remain separate.
The subsequent selected gradient-penalty GAN fixture is recorded above.
Independent review found no blockers; all changes remain local and uncommitted.

## 2026-11-03 CNN, GRU and KAN continuation

Three further public CPU f32 learning/restart fixtures are implemented. All
use seeds 7, 11 and 43, predeclared fixed budgets and gates, independent
PyTorch input/all-parameter gradient checks, and exact fresh-process restart:

| Fixture | Learning result | Restart boundary |
|---|---|---|
| `examples/train_restart_cnn.py` | 100% held-out bar-orientation accuracy from 50% initially; replacing each image with its mean reduces accuracy to 50%. | Step 147 of 300 |
| `examples/train_restart_recurrent.py` | 100% held-out token-order accuracy from 50% initially; erasing history before the last token reduces accuracy to 50%. | Step 137 of 300 |
| `examples/train_restart_kan.py` | Held-out MSE is 0.608-2.231% of a training-fit affine baseline and 0.0419-0.1536% of a constant baseline, below the declared 10% gates. | Step 197 of 400 |

All remaining losses, serialized model/AdamW/config/Generator state, next draws,
dataset identity and nonzero checkpoint cursors match. Maximum gradient errors
are 5.15e-8 (CNN), 1.59e-8 (GRU) and 1.20e-7 (KAN), within atol 1e-5 / rtol 1e-4.
Corruption/configuration checks and acceptance-gate sensitivity under optimized
Python are included. Independent reviews found no blockers in this scoped work.

The fixtures exposed and repaired two API defects. KAN checkpoints now serialize
immutable knots, degree and outside-domain policy, rejecting incompatible bases
before mutation. `KanLayer.basis` remains readable; changing it requires a new
layer. Recurrent unroll validates output layout and allocation capacity before
constructing steps, returning errors for two previously panicking zero-data
inputs. Representable empty and fully masked sequences retain their behavior.

See [architecture continuation evidence](../verification/architectures-2026-11-03/README.md)
for the matching build, complete regression results and source manifest. These
are synthetic CPU fixtures: CNN spatial templates repeat, GRU hidden state is
reset each batch and restart is between complete optimizer steps, and KAN knots
are fixed. General data-loader state, mid-sequence restart, higher-order spline
input derivatives and GPU model certification remain open. Changes are local
and uncommitted; historical numerical/native-crash investigations remain open.

## 2026-11-03 GCN and capture continuation

The selected public CPU GCN learning/restart fixture is implemented and locally
verified. Run `python examples/train_restart_gcn.py`. All three seeds (7, 11, 43)
reach 100% accuracy on disjoint held-out feature samples; removing edges reduces
accuracy to 50%. Topology templates may repeat. The maximum independent dense
PyTorch gradient error is 4.47e-8 against declared atol 1e-5 / rtol 1e-4 gates.
Restoration at step 117 reproduces the remaining losses, serialized
model/AdamW/config/RNG state, next draws, dataset identity and cyclic cursor in
a fresh process. Topology/data persistence belongs to this fixture's explicit
bundle; it does not extend the general checkpoint API to arbitrary data loaders.

Further repairs validate GCN inputs before computation, add recoverable
`EdgeIndex::try_batch` (also used by Python), and avoid allocations for empty
feature axes. The legacy `CapturedChain` now retains exact input allocation
owners and uses the existing static-graph capture/cleanup boundary. Replay
requires `&mut self`; poisoned output access rejects. Input mutation is scoped
to the owning backend stream. Foreign-stream mutation event bookkeeping and
experimental whole-step capture remain open, as does real-GPU execution of the
new ownership regressions.

The release wheel excludes generated bytecode. CI now includes CUDA host
regressions and a wheel-content check; hosted CI has not run for this local tree.
See [continuation evidence](../verification/continuation-2026-11-03/README.md)
for the matching source/build identity, commands and limits. All changes remain
uncommitted. Broader GNN variants, unseen-topology generalization, resident GPU
graph training and other model-family certification remain separate gates.

## 2026-11-03 initial review

Reviewed checkout base: `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`.
The new implementation is an uncommitted working-tree change, not a release.
See [this review's evidence](../verification/review-2026-11-03/README.md) for the
final source manifest, commands, failures and verification limits.

- The recent Rust GNN work already implements `EdgeIndex::batch`, `scatter`,
  `MessagePassing` and `GcnConv`; graph batching is no longer wholly open.
- Python now exposes immutable `ferro.graph.EdgeIndex`, CPU f32 `scatter` and
  `GcnConv`. Focused tests compare outputs and every input/parameter gradient
  against a dense PyTorch oracle, cover batching/permutation and empty graphs,
  and exercise saved Adam/model/config/RNG continuation in fresh objects.
  Topology remains an explicit input saved separately. This is not a fresh-process
  GNN learning/restart certification, resident GPU graph training or higher-order AD.
- Review repairs cover minimum-scatter validation, CUDA gather ownership and
  error propagation, required-CUDA test enforcement, and driver-independent
  native fault-injection error reporting. GPU runtime regressions still require
  hardware; CPU-only harness passes must not be counted as GPU execution.
  Non-whole I64 index views use the existing host path rather than passing
  unvalidated backing entries to CUDA. `Error::GatherUnavailable` is a new
  public enum variant: downstream exhaustive matches need updating, and custom
  backends must return it only when declining gather without attempting work.
- [Rust CUDA research](RUST_CUDA.md) records current NVIDIA and community
  toolchain requirements. Keep cudarc as the baseline and both NVIDIA authoring
  tracks optional. No experimental compiler or Rust GPU kernel was enabled.

The public CPU GCN learning milestone proposed by this initial review is now
recorded in the continuation above. The existing
historical CPU numerical/crash investigations remain unresolved. Earlier
evidence identities and counts below describe their own runs, not this tree.

## Evidence boundary

The original index reconciled source and archived reports. The selected CPU training/restart fixtures now have a fresh matching-build run; see the local result below. Other capability rows retain their separately scoped evidence.
Inspected checkout HEAD: `dc13a7199c23d92be5dd766102af03a6bfcb4bc3`, with
uncommitted implementation, tests and reports. HEAD alone does not identify
the inspected implementation. Do not label working-tree additions released,
merged, or verified at this commit.

"Implemented" means source exists for a bounded surface. "Recorded verification"
means the linked report records execution on its own source/build identity.
"Open" means a declared gate still needs evidence or implementation. No bundle
is globally complete merely because its primitives or tests exist.

Evidence order: current source and matching execution manifests, this ledger,
detailed foundation acceptance criteria, then historical strategy. Preserve
historical failure reports rather than rewriting them into current status.

## Capability matrix

| Area / bundle | Implemented surface | Device, dtype and API boundary | Evidence and remaining gate |
|---|---|---|---|
| Tensor and first-order AD / B0-B1 | Checked shape/layout work, typed selection, storage versions, gradient contracts and Python cotangent entry points | Primary math/AD is f32; typed storage is not full typed arithmetic. Device fallback and layout behavior remain operation-specific. | Core and binding suites are recorded in [combined verification](../verification/post25/combined/README.md). A public per-op dtype/placement matrix and broader resident indexing remain open. |
| Public training / B1 | Python Module registration/containers, Linear, losses, SGD/Adam/AdamW, freezing, Trainer and progress | Python convenience APIs do not imply complete torch compatibility or universal CUDA layer coverage. | [Training API tests](../crates/ferro-py/tests/test_training_api.py) and [architecture fixtures](../verification/python-architecture-api/test_architecture.py). Full model certification remains separate. |
| Training state / B2 | Owning snapshots, immutable generations, named optimizers, ordered parameter groups, ties, persistent buffers, modes/config and one explicit Generator; Python save/load/restore; per-parameter eager AdamW counters and owning optimizer clones | Whole-training restore is paused, compatible, whole-contiguous CPU f32. Optimizer groups and exact Python scalar metadata restore atomically. Scheduler state uses an explicit state_dict sidecar; example bundles persist selected data/cursors/phases. Pending gradients, general data cursors, extra RNG streams and arbitrary loop state remain outside the Python contract. | [Python state report](../verification/post25/python-state/README.md), [checkpoint tests](../crates/ferro-py/tests/test_training_checkpoint.py), [schema tests](../crates/ferro-py/tests/test_checkpoint_schema.py), [optimizer and multi-phase evidence](../verification/adversarial-2026-11-03/README.md), [grouped optimizer/scheduler continuation](../verification/portability-2026-11-03/README.md). Broader state completeness and CUDA transactions remain open. |
| Segments / B3 | CPU sum/mean/max/softmax; prepared CUDA sum/softmax and adjoints; public Python PreparedSegments | First-order f32. CUDA topology uploads once per preparation. Nonempty softmax performs a synchronized four-byte status download; no zero-total-transfer claim. Mean/max remain CPU-only; capture unsupported. | [Native report](../verification/post25/segments/README.md), [Python prepared-plan report](../verification/post25/prepared-python/README.md). General resident indexed adjoints and additional segment kernels remain open. |
| Sparse and graph / B4 | Validated COO/CSR and CPU sparse algebra; checked topology/batching, message passing, GCN and edge-restricted attention; Python wrappers and selected learning/restart fixtures | New graph wrappers are CPU f32 and first-order. Edge attention uses node/edge-sized intermediates. Sparse algebra does not establish resident CUDA SpMM/SDDMM or a complete graph data pipeline. | [Sparse tests](../crates/ferro-core/tests/foundation_sparse.rs), [Rust graph tests](../crates/ferro-core/tests/gnn.rs), [GCN proof](../examples/train_restart_gcn.py), [attention proof](../verification/graph-sequence-2026-11-03/README.md). Resident sparse execution and full connectome/graph-transformer gates remain open. |
| Vision, sequence, basis / B5 | Grouped convolution/window work, RNN/GRU/LSTM cells and explicit masked/reset/truncated unroll, spline basis and checkpoint-compatible KAN; selected CNN/GRU/KAN and streaming LSTM learning proofs | New architecture wrappers are CPU f32 scoped. Explicit unroll is not fused scan or a general bounded-memory guarantee. Streaming h/c persistence belongs to the explicit example bundle. | [Foundation publication](../verification/foundation-wave/pr-readiness/PR_BODY.md), [architecture proofs](../verification/architectures-2026-11-03/README.md), [streaming proof](../verification/graph-sequence-2026-11-03/README.md). GPU implementations, decoder breadth and other model variants remain open. |
| Higher-order AD / B6 | Functional gradients/VJPs and graph-building derivatives for a declared smooth subset; selected strong-form PINN, contractive-AE and gradient-penalty WGAN learning/restart proofs | Eager execution; functional derivative capture rejects. Unsupported derivative paths must reject. No general operator coverage, f64 AD, or all-PINN support claim. | [Implementation](../crates/ferro-core/src/higher_order.rs), [tests](../crates/ferro-core/tests/higher_order.rs), [scientific proofs](../verification/scientific-2026-11-03/README.md), [adversarial proof](../verification/adversarial-2026-11-03/README.md). Other scientific/adversarial variants need separate certification. |
| Compilation and capture | Compiled supported inference DAGs, pointwise fusion and prepared static CUDA model execution | Supported operation/layout subsets only. No automatic whole-training compiler or general torch.compile parity claim. | [Graph implementation](../crates/ferro-core/src/graph.rs), [static CUDA owner](../crates/ferro-cuda/src/static_graph.rs), combined regression report. Mutable state/RNG/restore interactions need explicit gates. |
| CPU matmul / BMM | Packed BLIS-style SGEMM (runtime AVX-512 / AVX2+FMA / scalar kernels) on every fastcpu matmul, BMM and install path; containment retired | f32 only. One ascending-k fused chain per output: bitwise independent of threads/blocking, not bitwise equal to the non-fused reference. Custom registrations can change dispatch. | [CPU GEMM incident report](../verification/cpu-gemm-incidents/README.md): exact mul_add oracles, f64 gamma(k) bound, opt-in FERRO_GEMM_CHECK guard. It rules out the enumerated GEMM numerical mechanisms and ungated SIMD in the scanned configuration; a possible abort and both root causes remain unexplained. |
| Architecture certification / B7 | Selected regression, classifier, GCN, CNN, GRU, KAN, PINN, contractive-AE, VAE, WGAN-GP, sparse attention and streaming LSTM held-out learning and fresh-process restart fixtures | Bounded synthetic CPU f32 proofs within one environment; no all-family or cross-platform certification. | [Architecture evidence](../verification/architectures-2026-11-03/README.md), [scientific evidence](../verification/scientific-2026-11-03/README.md), [generative evidence](../verification/adversarial-2026-11-03/README.md), [graph/sequence evidence](../verification/graph-sequence-2026-11-03/README.md); certify other variants, APIs, dtypes and devices independently. |

## Recorded verification and chronology

1. [Foundation publication](../verification/foundation-wave/pr-readiness/PUBLICATION.md)
   cleared its documented merge blocker through software-path containment,
   explicitly leaving the numerical root cause unresolved. Its Python checkpoint
   exclusion and scalar timing figures describe that earlier scope.
2. [Post25 combined verification](../verification/post25/combined/README.md)
   records a rebuilt noneditable wheel, matching source/wheel/install facade
   hashes, native identity and 438 stable inputs for that run. It reports core
   762 passed / 2 ignored, CUDA 118 passed, Python discovery 86 passed,
   architecture 8 passed, and training checkpoints 13 passed. These are reported
   suite results, not new execution here; overlapping/repeated suites must not
   be summed as unique tests.
3. [Prepared Python topology verification](../verification/post25/prepared-python/README.md)
   separately records 10 public tests including six CUDA tests, six Rust CUDA
   segment tests and 15 binding tests. Its report explicitly says it is not the
   parent combined freeze. Do not merge separate manifests into one tested tree.
4. Current checkpoint schema and alias tests add narrower source contracts.
   Their presence alone does not establish inclusion in an earlier report.

Some linked post25 reports and implementation files are currently untracked.
These links are local evidence references, not a guarantee that a fresh clone
contains them. Before publication, select durable reports/manifests explicitly;
do not publish raw logs, wheels or whole verification directories by default.
Do not rerun archival collectors that overwrite preserved evidence.

## Defects and investigation tracks

| Priority | Issue | Required next evidence |
|---|---|---|
| P0 investigation | Historical 343812-ULP CPU discrepancy remains root-cause unresolved. The suspect kernel is retired and code-level mechanisms are ruled out ([report](../verification/cpu-gemm-incidents/README.md)); a host fault is the leading, unverified hypothesis. | Record CPU model/microcode of the original host and run `FERRO_GEMM_CHECK=1` suites there. Passing repetitions do not resolve it. |
| P0 investigation | Recorded Windows 0xc000001d illegal-instruction exit remains unexplained. Ungated SIMD is ruled out for the crashing build configuration by an asm scan ([report](../verification/cpu-gemm-incidents/README.md)). | Reproduce on the original host with CPU model/microcode recorded; distinguish environment from a ud2 abort. |
| Recorded repair, scoped | F32/I64 host reads after CUDA registry replacement use the allocation-owned stream. The earlier old-allocation download defect is closed within that scope. | [Stabilisation publication](STABILISATION_PUBLICATION.md). Foreign-owner compute, mutation and D2D remain rejected; current GPU execution is still unavailable locally. |
| Recorded repair, scoped | Supported same-device F32 materialization removes the earlier strided backward-seed host round trip. | [Stabilisation publication](STABILISATION_PUBLICATION.md). Legacy backends and CUDA rank greater than 16 retain explicit host fallback; this is not universal strided-adjoint residency. |

These priorities guide follow-up work; they neither reopen a historical merge
decision automatically nor authorize merging. No incident-free claim is made.

## Original CPU public training milestone: reproducible restart

Status: implemented and locally verified for the two CPU f32 fixtures; independent review and broader B7 certification remain open. The work extends existing public APIs and checkpoint
regressions into two small architecture proofs before widening the device scope.
B0/B1/B2 provide the starting surface; this closes only the selected B7 rows.

| Fixture | Learning gate, declared before execution | Numerical and restart gate |
|---|---|---|
| Two-layer nonlinear regression | Fixed disjoint train/held-out samples of a smooth nonlinear function; seeds 7, 11 and 43; held-out MSE at most 25% of a training-mean constant predictor, within a fixed documented step budget. | Compare input and every parameter gradient with a same-weight independent reference before updates; checkpoint midway and compare uninterrupted continuation with restore into fresh compatible objects in a fresh process. |
| Two-layer classification with I64 labels | Fixed balanced synthetic two-class data with disjoint held-out samples; the same seeds; held-out accuracy at least 95%, with class-prior baseline reported. | Preserve I64 labels through minibatching and the public loss; compare gradients to the reference, then compare losses, parameters, optimizer state and next explicit-Generator draws after restart. |

Implementation scope:

1. Add public examples and small automated fixtures for these two models.
   Reuse Module, native optimizers and checkpoint APIs; do not embed engine
   implementations or use private bindings.
2. Use explicit deterministic parameter initialization and one explicit Generator
   for stochastic training draws. Built-in initialization that uses implicit RNG
   must be replaced or controlled within the fixture.
3. Keep input order deterministic from fixed data and the restored step. State
   plainly that this tests continuation at an optimizer-step boundary, not
   checkpoint serialization of DataLoader/prefetch cursors.
4. Compare gradients before optimizing, with predeclared f32 tolerances
   (initial target: atol 1e-5, rtol 1e-4). Require bit-exact resumed trajectory
   within the same binary/environment for these CPU fixtures. Cross-platform
   bitwise reproducibility is not a gate or a claim.
5. Clear gradients before snapshot/restore. Compare saved tensors, moments,
   counters, configuration and RNG state, and retain negative cases for
   incompatible models, aliases and late optimizer validation failure.
6. Run existing regression suites once against the final built extension and
   record source/dirty manifests, extension hash, commands/exits, skips, seeds,
   tolerances and held-out results. Never weaken thresholds silently after a
   failing run; document a revised design before claiming acceptance.

Implemented targets:
`examples/train_restart_regression.py`,
`examples/train_restart_classifier.py`, and
`crates/ferro-py/tests/test_architecture_restart.py`.
Keep the reusable process/restart harness with these tests.

Milestone completion requires both fixtures across all three seeds, successful
fresh-process continuation and negative restore checks, a final matching-build
regression report, and review of the bounded claim. GPU training, general data
resume and all-model-family certification stay open.

## Local result for the selected CPU milestone

Both examples passed all three seeds at 400 steps, with PyTorch input/parameter
gradient checks and bit-exact continuation in fresh processes. Regression
held-out MSE was 0.54-1.21% of the constant baseline; classification accuracy was
100%. Serialized model/optimizer/config/RNG state and next RNG draws matched.
The public CPU integer-label cross-entropy binding was added to support this gate.

[Verification report](../verification/architecture-restart/README.md) records
the matching release wheel, 434 unchanged selected source inputs, 105 passing
Python tests and successful core/CPU/tokenizer, binding, parity, serialization,
fusion and fuzz commands. An initial missing-NVRTC verifier failure is preserved;
the final runtime-path-corrected run passed. No CUDA training, cross-platform
determinism, general data-cursor resume, independent review or hosted CI claim.

## Subsequent sequence

- Complete B3 general resident indexing and B4 sparse/graph batching before
  claiming a resident GNN or sparse graph transformer.
- Extend the selected CNN, GRU, streaming LSTM, sparse attention and KAN proofs to other variants and datasets;
  extend to CUDA only with execution and residency evidence.
- Extend the selected PINN, contractive-AE, VAE and gradient-penalty adversarial
  proofs to other problems and architectures. Retain mixed-derivative, optimizer
  isolation and phase-resume checks when extending the adversarial fixture.
- Expand checkpoint state and architecture variants explicitly, then measure
  representative performance with equal work, grad modes and same-session
  comparisons. Retain the separate CPU incident investigations throughout.

## Planned Rust CUDA kernel-authoring support

Both **cutile-rs** (tile-based) and **cuda-oxide** (explicit SIMT) are roadmap targets, not mutually exclusive alternatives. Neither is claimed implemented or verified. Keep cudarc and existing CUDA/vendor kernels as the baseline, with optional compiler integrations and one tensor/autograd contract. Each track must independently pass platform/toolchain, allocation/stream ownership, forward/backward parity, residency, capture and performance gates before production dispatch.

See [Rust-authored CUDA kernels: support both authoring tracks](FOUNDATIONS_AND_ARCHITECTURE_ROADMAP.md#rust-authored-cuda-kernels-support-both-authoring-tracks) for the scope and acceptance criteria. This future work does not supersede the current foundations milestone or unresolved native investigations.

## Stabilisation follow-up: publication evidence and local history

See the shipped [stabilisation publication](STABILISATION_PUBLICATION.md) for the reconciled 470-input source freeze and matching noneditable release-wheel evidence. The earlier 454-input dirty-working-tree freeze and independent review are historical local provenance; their raw reports are not shipped, and they do not certify the reconciled tree. The later [typed parity follow-up](../verification/pr29-typed-parity/README.md) records 119 Python tests with no skips in its runtime-configured run, superseding the earlier 114-test discovery count for that follow-up only. None of these archived counts verifies subsequent review fixes or establishes current hosted CI status.

- The old-allocation host-read defect above is closed for F32/I64 downloads after CUDA registry replacement. Downloads use the allocation-owned stream; foreign-owner compute, mutation and D2D remain rejected.
- The strided-seed host round-trip above is closed for supported same-device F32 materialization layouts, including backward and first-order identity VJP. Legacy backends without materialization and valid CUDA rank greater than 16 retain explicit host fallback. Operational/ownership errors do not authorize fallback. This is not universal strided-adjoint residency.
- Python tolist preserves I64 integers and F64 precision rather than narrowing through F32. Training/restart acceptance gates and the new registry tests remain effective under optimized Python.
- Historical 454-input local suites (not reconciled publication counts): Python 114 passed; core 784 passed / 2 ignored; CUDA 126 passed; fastcpu 27 passed / 2 ignored; bindings 15 passed. Tokenizer records eight harness passes but only seven exercised bodies: real-GPT2 data was unavailable. Targeted normal and optimized Python tests passed. Performance ignores remain unexecuted, not passes.
- Earlier local independent review closed both introduced materialization compatibility blockers and reconciled that run's source, wheel, installed hashes and raw results. Parent additionally reran the three targeted Python suites under -O successfully. This is not independent review of the later reconciled publication or subsequent review fixes.

Remaining boundaries: Error::MaterializeUnavailable adds a public exhaustive enum variant, so downstream exhaustive matches may need updating. Late CUDA copy/guard-completion error injection remains a coverage gap. Historical numerical/native-crash investigations remain unresolved. No commit, push, merge or broad certification of every pre-existing dirty change follows from this result.

## Maintenance rule

For each status change record the API/device/dtype/layout/derivative subset,
source identity including dirty files, evidence path, executed/skipped counts,
remaining exclusions and reviewer decision. Update this index without erasing
historical reports. New source presence advances "implemented"; only matching
execution evidence advances "recorded verification"; only the declared complete
acceptance surface advances a milestone.
