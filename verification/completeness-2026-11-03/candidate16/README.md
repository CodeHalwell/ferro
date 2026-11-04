# Continuing PyTorch/PyG work: verified candidate 16

This matching-build checkpoint records ongoing implementation. Full PyTorch/PyG
parity is not established. Work continued after this isolated snapshot, and all
changes remain local and uncommitted. The prior green report is preserved in
[candidate14](../candidate14/README.md); failed intermediate comparisons are retained
in [candidate12](../candidate12/README.md) and the task scratch directory.
The later [candidate17](../candidate17/README.md) retains three full-discovery
problems and does not replace this green checkpoint. Implementation continues.

References are PyTorch 2.13.0+cpu and PyTorch Geometric 2.8.1, used only for tests.
Ferro's Rust core has zero external dependencies and one record_fn/Op autograd
mechanism. NVIDIA CUDA and Apple Metal remain intended targets. This Linux
machine has no GPU, and Metal is not implemented.

## Matching-build validation

The noneditable ABI3 wheel was built from an isolated copy. All 872 selected
source/manifest inputs, 101 installed Python files and the native library were
rechecked after validation. The source archive, wheel and raw logs are retained
in `/workspace/work/completeness-2026-11-03`; exact hashes and outcomes are in
[manifest.json](manifest.json).

| Check | Result |
|---|---|
| Installed Python discovery | Exit 0: 1,321 discovered cases, 1,291 passed, 30 GPU cases unavailable. |
| Rust workspace, default test threads | Exit 0: core 1,156 passed / 2 ignored; fastcpu 39 passed / 2 ignored; CUDA 140 runtime harness passes / 5 strict GPU cases filtered; tokenizer 8 harness passes. |
| CUDA compile-fail doctests | Both pass with intended E0502 borrow errors; no E0460 artifact mismatch. |
| CUDA all-target compilation | Exit 0 without a CUDA driver or GPU. |
| Native binding DLPack | 12 harness passes; one GPU body returned early; 3 mandatory GPU cases filtered. Explicit Python 3.12 link. |
| Binding, compiled-fusion and fuse regressions | All exit 0. |
| Operator/Torch and safetensors comparisons | Both exit 0. |
| Fuzz comparison | 200 trials, seed 0, exit 0. |
| Graph training/restart example | Learning, control and exact fresh-process continuation proofs pass. |
| API inventory | Installed hashes/public names recorded; name presence is not a semantic or performance score. |

Unittest reports 1,315 executions and 25 skip indicators: 24 individual GPU
skips and one class skip hiding six more GPU cases. Native totals use each
Cargo block's final outer summary: 1,343 runtime passes plus two doctests;
17 parsed nested child passes are excluded. Optional tokenizer/GPU bodies can return
early, so harness success does not imply hardware execution.

Candidate13's native run exposed a test setup race: a waiting test replaced a
backend before acquiring the shared counter lock, invalidating the active
attention cache. The test locks before registration; unchanged transfer
assertions pass under the default runner and repeated two-thread stress.
Original failed logs and candidate13's serial control remain preserved.

## Additions covered by this checkpoint

- Native typed CPU float32/float64 math, promotion and casts, exact int64/Bool
  storage and selection, DLPack, owning clones/detach aliases, shared gradient
  flags, checked mutations and layout-aware adjoints. Typed fills preserve
  exact integers and reference scalar-versus-Tensor conversion behavior.
- Typed CPU stable sort/argsort, including NaNs, signed zeros, exact int64 keys,
  returned-index version guards and float first-order adjoints. List/tuple view
  overloads and prepended strided expansions preserve real storage aliases.
- Genuine direct forward JVP rules on the existing recorder, structural-zero
  provenance and immutable aliases, no_grad/factory/captured-leaf behavior and
  primal aliases. Opaque operations still use a local reverse-contraction
  fallback. Nested levels, arbitrary reverse queries in JVP, vmap and complete
  transform composition remain open; float64 primals are not supported.
- Canonical custom Module constructors and Parameter tensor inheritance,
  direct expressions/functional arguments, gradient assignment and exact
  parameter/base identity on tested CPU optimizer paths. Default Parameter
  construction remains owning; pretrained embeddings explicitly share storage.
- Embedding/EmbeddingBag CPU float32/float64 functional options, padding,
  frequency scaling, renorm, sum/mean/max bags and live per-sample weights.
  Pretrained whole CPU float32 tables share storage with independent leaf flags;
  checkpoints preserve AD flags separately from optimizer-only masks.
- Fourteen CPU float32 optimizer families including Muon, Adafactor and
  strong-Wolfe L-BFGS; retained Python gradient handles, closure config refresh,
  sequential shared-parameter updates, cyclic/one-cycle/composite schedules and
  exact state/RNG continuation. Active manual gradients aliasing model storage
  reject before updates. Independent strided/offset CPU gradients are materialized for fused kernels
  while original parameter and gradient handles remain unchanged.
- CPU real QR/eigh/SVD, spectral/nuclear norms, rank/pseudoinverse, minimum-norm
  least squares, tensorinv/tensorsolve and solve_ex/inv_ex. Ex status tensors use
  Ferro int64, differing from Torch int32. Singular ex values/adjoints are not
  numerically specified; unsupported drivers/rcond conventions reject.
- GroupNorm and LocalResponseNorm extend typed BatchNorm/InstanceNorm and
  multi-axis LayerNorm/RMSNorm, plus existing losses, projected/packed recurrence,
  Transformers, attention, convolution/pooling/padding and rearrangement layers.
- PyG additions include recurrent/sort/set/deep/degree/attention aggregations,
  Set2Set, GraphMultisetTransformer, MetaLayer, GPSConv, graph BatchNorm and
  DiffGroupNorm, learned SAGE aggregation, dense MinCut/DMoN pooling and
  quantile/median readouts with live q buffers. Earlier heterogeneous data,
  sampling, graph model and exact checkpoint controls remain included.

## Additional candidate 16 coverage

- Module parameter/child registration, optional None slot kinds, private and
  replaceable buffers, prefix/recurse/duplicate traversal and apply ordering.
  Local forward pre/post hooks support input/output replacement, kwargs,
  prepend/removal and always_call exception behavior on native tensor graphs;
  callbacks are caller-owned and explicitly recreated after checkpoints.
- Optimizer lifecycle hooks and ordinary whole CPU F32 leaf Tensor inputs,
  preserving exact original objects/IDs/grad slots and Parameter-only masks.
  Active-AD and duplicate-gradient-slot preflight precedes updates. Independent
  strided/offset gradients keep their public handles through fused updates.
- Atomic checkpoint context/immutable-target preflight, including empty
  restores, before model/scalar/optimizer/RNG commits or Generator locking.
  Late rejected buffers leave every state/identity intact and RNG usable.
- Channel Dropout1d/2d/3d, reviewed CPU LU factor/ex/solve/decomposition,
  matrix_power/multi_dot, HeteroLinear/HeteroDictLinear, AntiSymmetricConv,
  VariancePreservingAggregation, MixHopConv and HANConv. Numerical controls,
  all-input gradients, typed state and exact continuation are scoped by tests.

## Remaining scope and numerical boundaries

This is a tested CPU subset. Parameters and optimizers remain CPU float32.
Default Parameter ownership differs from Torch's shared-data constructor;
explicit pretrained sharing has its own whole-storage contract. Sparse gradients,
all dtypes/operators/layouts, arbitrary advanced indexing/out destinations,
complete catalogs, FFT, compiler/export/quantization, distributed training and
full NVIDIA/Metal execution remain open. Iterator and loader checkpoint coverage
is scoped, not general Python iterator serialization.

Some stable formulas differ from Torch CPU overflow/cancellation. SVD numerical
controls retain exact-input float64 comparisons and directional finite
differences for a close-gap Torch float32 failure. Correlated subnormal values
below matrix precision can be deflated; exact mathematical rank at underflow is
not claimed. Truncated least-squares/pinv adjoints follow actual Torch behavior,
which can differ from finite differences when a nonzero singular value is cut.
No blanket equivalence is inferred from these controls.

Further transform composition, fused scalar arithmetic, shape-only view APIs,
condition numbers, AlphaDropout and graph extensions are subsequent source work.
Their own installed-wheel proof is required before joining this coverage. See the active
[compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md).
