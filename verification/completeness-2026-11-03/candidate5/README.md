# Continuing PyTorch/PyG work: verified candidate 5

This is a verified checkpoint in ongoing implementation, not a declaration of
PyTorch/PyG parity. The working tree advanced after this checkpoint. The user
requested continued work through the remaining gaps; implementation continues.
NVIDIA CUDA and Apple Metal remain the intended GPU targets. No GPU hardware
execution was available in this Linux environment, and Metal is not implemented.

The references are PyTorch 2.13.0+cpu and PyTorch Geometric 2.8.1. Neither is a
Ferro runtime dependency. The Rust core retains zero external dependencies and
the existing record_fn Op/autograd mechanism.

## Changes covered by this checkpoint

- Native CPU float64 arithmetic, reductions, vector/batched/broadcast products,
  nonlinear math and softmax/log-softmax, with same-dtype first-order gradients.
  Numeric promotion distinguishes dimensional tensors, scalar tensors and Python
  scalars. Float32/float64 casts are differentiable; integer arithmetic wraps
  exactly. Float64 RNG/dropout and exact CPU float64/int64 DLPack are covered.
- Typed comparisons, triangles and whole-contiguous CPU copies preserve precise
  values. Storage cells retain identity and shared version counters. Raw
  half/bfloat16 copying does not establish half-precision compute or AMP.
- Functional Jacobian/Hessian/JVP/VJP and selected func transforms use the same
  native graph. Higher derivatives cover additional smooth/piecewise operations,
  batched products and softmax. Optional backward contributions support genuine
  missing gradient slots; ordinary backward validates contributions before
  publishing slots. Public functional unused-input results remain zero-filled.
- Packed/projected recurrent modules and persistent dropout streams, including
  complete-step fresh-process optimizer/RNG continuation. Dense Transformer
  encoder/decoder layers, stacks and full Transformer have canonical parameter
  names and reference input/mask/all-parameter gradient tests.
- ConvTranspose3d, max/average/adaptive-max pooling across 1/2/3 dimensions,
  signed crop/pad modes, pixel/channel rearrangements, GLU, Softmax2d and true
  strided Unflatten views. Attention supports masks/layouts, separate key/value
  widths, extra source tokens and persistent same-mask dropout gradient proofs.
- CPU float32/float64 linear solves, inverse, triangular solves, Cholesky,
  determinant/slogdet, stable vector/Frobenius/induced norms, diagonal views and
  trace. Singular determinant gradients use genuine cofactors. Zero-order norms
  and real slogdet sign preserve unused gradient slots. Gradient clipping has
  actual-reference norm, alias, nonfinite and optimizer-trajectory tests.
- RMSprop, Adagrad, Adamax, Adadelta, ASGD, Rprop, RAdam and NAdam join SGD, Adam
  and AdamW. Additional legacy-family options and fixed-step L-BFGS have native
  history/state, grouped strict restores and actual-reference continuations.
  Four more schedulers extend the initial four; their state remains explicit.
- Graph additions include PNA/HGT/GCN2, GraphConv, learned pooling, Set2Set,
  dense_diff_pool, eager custom MessagePassing, spectral, normalization,
  edge-conditioned/gated/residual and explicit-edge point layers. Local dataset
  persistence, aligned transforms, heterogeneous frontier sampling, cluster
  collation, follow-batch/exclusion and standard link-index offsets are covered.

These are scoped eager CPU APIs. Full dtype, device, compilation and higher-order
coverage is not inferred from an API name or an ordinary first-order test.

## Matching-build validation

Candidate 5 was built and installed as a noneditable ABI3 wheel. All 55 installed
Python files matched source; the native extension is identified separately in
[manifest.json](manifest.json). All 652 selected Rust/Python/CUDA source and
manifest inputs remained unchanged through the final validation. A source ZIP,
the wheel and raw logs are retained in
`/workspace/work/completeness-2026-11-03`.

| Check | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 987 passed / 2 ignored; fastcpu 38 passed / 2 ignored; CUDA 141 harness passes including 2 compile-fail doctests, 5 mandatory GPU tests filtered; tokenizer 8 harness passes, 7 bodies exercised. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0, without GPU execution. |
| Installed-wheel Python discovery | Exit 0: 698 discovered cases, 668 passed and 30 CUDA cases unavailable. |
| Native binding DLPack tests | 12 harness passes; one GPU-gated body returned early; 3 mandatory GPU-only tests filtered by the test-name selection. Linked explicitly against Python 3.12. |
| Binding regression, compiled fusion and fuse regression | All exit 0. |
| `examples/ops_vs_torch.py`, safetensors interoperability | Both exit 0. |
| Fuzz comparison | 200 trials, seed 0, exit 0. |
| Graph training/restart example | All three learning/control/fresh-process proofs pass. |
| API inventory | Installed-file hashes and public-name inventory recorded; this is not a semantic parity score. |

Python unittest reports 692 executions and 25 skip indicators: 24 are individual
CUDA skips, and one class skip hides six additional CUDA cases. Thus 668 pass
and 30 remain unavailable. Rust totals use the last outer summary in each Cargo
executable/doc-test section. Nine nested core child summaries are excluded.
An independent audit reproduced the totals. CUDA harness passes do not establish
GPU execution, residency, throughput or numerical accuracy on hardware.

The first frozen candidate passed Rust and all standalone checks but failed one
of its 692 Python executions: a same-shape differentiable VJP unnecessarily
reshaped the cotangent, losing Python identity. The exact-shape reduction path
now returns that tensor directly. Candidate 5 contains only that Rust source
change relative to candidate 4; the wheel was rebuilt and every full command
was rerun. The failed logs and source manifest are retained separately.

Independent reviews also exposed and repaired an enormous empty-RHS solve loop,
typed materialization capacity holes, exact norm-tie gradients, causal attention
hints, malformed dataset index validation and NaN clipping behavior. Regressions
exercise these controls rather than treating API presence as evidence.

## Remaining scope and subsequent work

The next work includes proper shared detach/clone/view semantics and saved-output
versions, strict functional-AD provenance, canonical attention defaults,
Adafactor, QR, dense graph models, heterogeneous batching and additional samplers.
Those later source changes are outside this frozen checkpoint.

Parameters/optimizers and whole-training checkpoints still have their documented
CPU float32 scope. Native optimizer steps consume gradients. L-BFGS here requires
one CPU group and has no strong-Wolfe line search. Stable norm accumulation can
differ from Torch at intermediate overflow. Some extrema/loop conventions and
upstream PyG defects have explicit documented boundaries.

True forward AD/vmap, general advanced indexing/out buffers, complete dtypes and
math/factorization/FFT catalogs, lazy/normalization/embedding/interpolation/loss
catalogs, full compiler/export/profiling/quantization tooling, distributed
training, downloadable graph datasets/explainers and complete GPU coverage
remain open. See the active [compatibility ledger](../../docs/PYTORCH_PYG_COMPATIBILITY.md)
and [GPU target requirements](../../docs/GPU_PORTABILITY.md).
