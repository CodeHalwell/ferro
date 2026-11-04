# Continuing PyTorch/PyG work: verified candidate 10

This is a matching-build checkpoint in ongoing implementation. It does not
declare PyTorch/PyG parity. The user requested continued work through the gaps,
and the current working tree has advanced beyond this snapshot. All changes
remain local and uncommitted. The previous green checkpoint is preserved in
[candidate5](../candidate5/README.md).

References are PyTorch 2.13.0+cpu and PyTorch Geometric 2.8.1, used for tests
only. Ferro's Rust core has zero external dependencies and retains its single
record_fn/Op autograd mechanism. NVIDIA CUDA and Apple Metal remain intended
targets; this Linux environment has no GPU, and Metal is not implemented.

## Matching-build validation

Candidate 10 was built as a noneditable ABI3 wheel from an isolated source copy.
All 780 selected source/manifest inputs and 76 installed Python files were
rechecked after validation. The source ZIP, wheel and raw logs are retained in
`/workspace/work/completeness-2026-11-03`. Exact hashes are in
[manifest.json](manifest.json).

| Check | Result |
|---|---|
| Installed Python discovery | Exit 0: 1,006 discovered cases, 976 passed, 30 CUDA cases unavailable. |
| Rust workspace | Exit 0: core 1,059 passed / 2 ignored; fastcpu 39 passed / 2 ignored; CUDA 140 runtime harness passes / 5 strict GPU tests filtered; tokenizer 8 harness passes. |
| CUDA compile-fail doctests | Both pass with the intended E0502 borrow errors. No E0460 artifact mismatch. |
| CUDA all-target compilation | Exit 0 without a CUDA driver or GPU. |
| Native binding DLPack tests | 12 harness passes; one GPU body returned early; 3 mandatory GPU cases filtered. Explicit Python 3.12 link. |
| Binding regression, compiled fusion, fuse regression | All exit 0. |
| Operator/Torch and safetensors comparisons | Both exit 0. |
| Fuzz comparison | 200 trials, seed 0, exit 0. |
| Graph training/restart example | All learning/control/fresh-process proofs pass. |
| API inventory | Installed hashes and public names recorded; this is not a semantic parity score. |

Unittest reports 1,000 executions and 25 skip indicators: 24 individual CUDA
skips and a class skip hiding six more CUDA cases. Rust totals use each Cargo
block's final outer summary; 18 nested child passes are excluded. An independent
audit confirms 1,246 runtime outer passes plus two doctests. The tokenizer's
optional vocabulary body and CUDA-gated bodies may return early; harness passes
do not establish GPU execution or performance.

An initial native run exhausted disk space because earlier debug targets were
retained. The failed logs were preserved, obsolete Cargo caches were cleaned,
and the whole native run was repeated with debug information and incremental
artifacts disabled. Assertions remained enabled. Source and installed hashes
were unchanged. A separate target prevents concurrent development builds from
contaminating compile-fail doctests.

## Additions covered by this checkpoint

- CPU float32/float64 typed computation, promotion, casts, reductions and
  broadcast matrix products; Bool and exact int64 storage/arithmetic; typed
  copies, selection, comparisons and DLPack interchange.
- Real detach aliases, owning clones, contiguous-format handling, typed
  T/mT/t/H/mH views, same-shape reshape aliases, valid empty slice offsets,
  checked in-place bindings and CPU default gradient layouts. Shared dynamic
  requires_grad/base metadata is additional work after this snapshot.
- Functional Jacobian/Hessian/JVP/VJP and selected func transforms on the same
  graph. Missing contributions remain absent internally; Cat/Slice and singleton
  view adjoints preserve real derivative paths. Frozen VJPs preserve selected
  primal operands while retaining detached original saved activation outputs.
  Genuine forward-mode AD and vmap are not established.
- CPU float32/float64 QR, symmetric eigh/eigvalsh and real SVD/svdvals extend
  solves, inverse, triangular solves, Cholesky, determinant/slogdet and stable
  norms. Reduced QR is supported; complete tall QR and mode='r' remain open.
  Repeated-spectrum vector derivatives remain undefined. Factorizations reject
  unsupported higher-order/capture paths explicitly.
- Canonical BatchNorm1d/2d/3d and InstanceNorm1d/2d/3d, multi-axis LayerNorm and
  RMSNorm, typed running buffers/counters, optional affine/state configuration,
  reset behavior and exact fresh-process continuation. Parameters remain CPU
  float32; float64 non-affine modules and float64 functional affine tensors are
  supported. Running-stat writes still require independent whole CPU buffers.
- All reductions for common losses, weighted L1/Huber and binary losses,
  mixed-dtype BCE/logits operand staging, positive-class weights, log-target KL,
  full/non-log Poisson and triplet swap. Native binary adjoints retain original
  operand versions and apply weighting/mean division in the reference order.
- Thirteen optimizer families, strong-Wolfe L-BFGS and expanded schedulers.
  Python optimizer steps retain exact gradient handles, including L-BFGS's last
  closure gradients. Raw Rust steps retain their consuming contract. Composite
  schedules and Module.zero_grad have owning state/continuation proofs.
- Module state_dict/load_state_dict with typed persistent/optional buffers,
  storage-sharing detached state, metadata, legacy BatchNorm counters, strict
  and partial keys, stable parameter/gradient slots and alias-aware staged
  sequential copies. CPU loading preflights all writes; assign=True and
  non-whole destination views remain unsupported. Traversal/dotted getters are
  covered. Standard custom __init__ support is subsequent source work.
- Packed/projected recurrence, persistent dropout, canonical Transformers,
  expanded attention, spatial/padding/pooling and rearrangement modules retain
  actual-reference gradient and saved-training continuation tests.
- Expanded PyG CPU models include GENConv, SignedConv, LGConv, SimpleConv,
  EGConv, hypergraph, dense, conditioned, point, mixture and residual families.
  Graph attention/APPNP/Transformer dropout, heterogeneous batching, GraphSAINT
  samplers, homogeneous/heterogeneous conversion, local dataset persistence,
  aligned transforms, frontier loaders and cluster collation are covered.

## Numerical controls and remaining scope

The independent SVD run retained one failure against Torch float32 in 448
checks. For that close-gap matrix, native float64 agrees with exact-input Torch
float64 within 1.02e-12; native float32 differs by 6.27e-7, while Torch float32
differs by 0.00370535. Directional finite differences support native float64.
The original failing comparison is preserved; no tolerance was relaxed.

Very small BatchNorm epsilon also exposes Torch CPU fused-formula cancellation:
constant nonzero inputs can lose an affine bias. Tests separately compare
controlled reference adjoints and Ferro's centered mathematical formula. Stable
norm/determinant controls retain their earlier documented upstream differences.

Current source continues with shared requires_grad metadata, standard Module
constructors, GroupNorm expansion, recurrent/sort graph aggregators, spectral
norms, pseudoinverse/rank and cyclic schedules. These are not included in the
candidate 10 installed result. Complete dtypes/operators, forward AD/vmap,
compiler/export/quantization, distributed training, data/ecosystem catalogs and
full NVIDIA/Metal execution remain open. Hardware acceptance is separate from
CPU reference and host-side compilation tests.
