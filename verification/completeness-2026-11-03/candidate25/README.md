# Continuing PyTorch/PyG work: verified candidate25

This local checkpoint passes every available recorded check. Full PyTorch/PyG
parity remains open. Changes are uncommitted. The previous green
[candidate23](../candidate23/README.md), [candidate22b](../candidate22b/README.md), and
the first failing combined [candidate24](../candidate24/README.md) remain archived.

References are Torch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro retains its zero-dependency Rust core and one record_fn/Op AD engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-source validation

Candidate25 promotes exactly the 1,040 selected source/fixture inputs and wheel
from combined-parity-proof2. All 119 installed files (118 Python plus native)
match that wheel. Both source ZIPs, both isolated trees, main inputs and installed
files were rehashed for this publication. The manifest records the verification
time; the global installation can advance afterward.

Native workspace, CUDA and DLPack checks use combined-parity-proof1. Its entire
Rust/Cargo/CUDA input set and packaged Python source match proof2. The sole
changed input is a standalone test that previously required neg() to be absent;
proof2 tests both named aliases, values and gradients. Proof1's passing full
Python run and failing standalone log remain intact. Proof2's new wheel passes
full discovery and all eight standalones. This is recorded input equivalence,
not a claim that the native checks were rerun against a different source path.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,645 discovered: 1,615 passed and 30 GPU cases unavailable. |
| Rust workspace | Core 1,364 passed / two ignored; fastcpu 39 passed / two ignored; CUDA 140 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Native DLPack | 12 harness passes; one GPU body returns early and three mandatory GPU cases are filtered. |
| Eight standalones | Binding, compiled fusion, fuse, operator/Torch, safetensors, seed0 200-trial fuzz, graph training/restart and API inventory all exit0. |
| Focused integration | All 59 methods covering tangent copies, unary layout/precision, publication, PReLU JVP, normalization and the neural-network catalog pass without overlays. |

Unittest reports 1,639 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals are 1,551 outer runtime
passes plus two doctests, excluding 18 nested child summaries. Optional GPU and
tokenizer bodies can return early; harness passes do not establish hardware
execution. Focused gates overlap full discovery and are not added to its count.

## Added since candidate23

- Ordinary gradient publication retains held slot identity and updates manual
  gradients in place without changing their history, flags or layouts. Fresh
  public slots detach live callback results. Shared gradient storage uses tested
  publication locks; rejected contributions do not publish partial slots.
- Single-direction JVP entry makes genuine differentiable copies when direction
  metadata differs from the primal. Compatible directions preserve identity;
  copies retain primal layout, offset, storage span and working dtype. These
  rules use the existing Op recorder and retain explicit CPU/F32-primal scope.
- Unary operations preserve tested physical layouts and saved-output versions.
  F64 adjoints use typed staged arithmetic; expm1/exp2/reciprocal/tan use their
  original saved outputs. Named neg/negative aliases are available. Widened F32
  coefficients are tested exactly against each side's own coefficient, with a
  working-F32 bound for portable cross-backend comparison.
- PReLU, Bilinear and GaussianNLLLoss have actual-Torch values, gradients and
  state checks. PReLU gains genuine direct JVP support; its higher reverse and
  capture paths remain explicit unsupported boundaries. Bilinear's supported
  higher/JVP paths use native compositions; GaussianNLL active JVP remains open.
- Typed atan2/acos and seven geometric transforms add Center, NormalizeScale,
  LinearTransformation, Cartesian, Distance, Polar and Spherical.
- Graph InstanceNorm, HeteroBatchNorm and HeteroLayerNorm match tested PyG
  gradients, running state, strided inputs and fresh-process continuation. Live-buffer
  rebinding under a structure-locked Trainer remains unsupported explicitly.

The earlier take, cumulative-extrema and Data/HeteroData subgraph additions
remain covered. Product reductions, RGAT/MLP, physical-batch Jacobians, nested
frames, shared multi-output rules, Where-layout and scalar-power follow-ups
remain separate experiments and are excluded from this checkpoint.

## Remaining scope

Parameters/optimizers remain CPU F32. Full dtype/operator/catalog coverage,
F64 graph-building/transform primals, half/complex/FFT, general indexing/out,
vmap/compiler/export/distributed support and full NVIDIA/Metal models remain
open. Higher/capture support is operator-specific. Name presence does not prove
semantic, device or performance parity. See the
[compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and
[Rust CUDA research](../../../docs/RUST_CUDA.md).
