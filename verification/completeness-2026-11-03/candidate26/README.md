# Continuing PyTorch/PyG work: verified candidate26

This local checkpoint passes every available recorded check after recovering
from a build disk-space failure. Full PyTorch/PyG parity remains open. Changes
are uncommitted. The previous green [candidate25](../candidate25/README.md) and
[candidate23](../candidate23/README.md), and the first failing combined
[candidate24](../candidate24/README.md), remain archived with their evidence.

References are Torch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro retains its zero-dependency Rust core and one record_fn/Op AD engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-source validation

Candidate26 promotes exactly the 1,046 selected source/fixture inputs and normal
wheel from candidate26-proof1. All 121 installed files (120 Python plus native)
match that wheel. Both source ZIPs, both isolated trees, main inputs and both
wheel payloads were rehashed for this publication. The manifest records the
verification time; the global installation can advance afterward.

The initial frozen workspace and CUDA all-target commands exited101 when the
build volume ran out of space; the workspace log also records linker SIGBUS.
After idle build-cache cleanup, both commands passed on the same unchanged
frozen sources. Original failed logs and statuses remain recorded separately
from recovered passing commands. No source repair or relaxed test was needed.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,662 discovered: 1,632 passed and 30 GPU cases unavailable. |
| Rust workspace | Core 1,373 passed / two ignored; fastcpu 39 passed / two ignored; CUDA 140 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Native DLPack | 12 harness passes; one GPU body returns early and three mandatory GPU cases are filtered. |
| Eight standalones | Binding, compiled fusion, fuse, operator/Torch, safetensors, seed0 200-trial fuzz, graph training/restart and API inventory all exit0. |
| Focused integration | All 17 new product/RGAT methods and the 371-method graph gate pass against the matching wheel, without overlays. |

Unittest reports 1,656 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals are 1,560 outer runtime
passes plus two doctests, excluding 18 nested child summaries. Optional GPU and
tokenizer bodies can return early; harness passes do not establish hardware
execution. Focused gates overlap full discovery and are not added to its count.

## Added since candidate25

- Product reductions support CPU Bool/I64/F32/F64, all elements or one integer
  axis, optional explicit dtype and axis keepdim. Bool defaults to I64, integer
  products wrap and empty products are one. Floating forward accumulation is
  serial in the selected dtype, so SIMD/parallel reference rounding and extreme
  overflow can differ. Ordinary reverse supports F32/F64; supported F32 higher
  and direct forward AD use genuine native graphs, including F64 directions
  and outputs. Saved input and product-output versions remain guarded.
- Product differentiation preserves the measured invocation policy: ordinary
  backward, autograd.functional, standalone jacrev and returned func.vjp
  pullbacks use ordinary rules; func.grad/grad_and_value and functional-forward
  composition use zero-safe prefix rules. The mode is not captured in a forward
  Op. The existing zero-containing cumulative-product higher-work bound remains.
  Current func.hessian is reverse-over-reverse and can differ at nonfinite
  intermediates from Torch's forward-over-reverse default; that limitation is
  retained explicitly, without hiding failed comparisons.
- RGATConv supports the tested CPU F32 relational attention modes, basis/block
  weights, edge features, head/dimension folding and cardinality modifications.
  Copied-weight actual-PyG controls cover inputs, every parameter, edge fields
  and returned attention; matched dropout masks and fresh-process complete
  model/optimizer/config/alias/RNG continuation also pass. Constructor l2 is
  deterministically initialized to 1/out_channels instead of reproducing PyG's
  uninitialized allocation. Later reset preserves l2 and w like the reference.
  Lazy/sparse/device/capture paths remain explicit unsupported boundaries.

All previously verified candidate25 behavior remains covered, including typed
angle/geometric transforms, graph normalization, gradient publication and the
neural-network catalog. MLP/readout integration, physical-batch Jacobians,
nested frames, shared multi-output rules, Where-layout and original-scalar
power follow-ups remain separate experiments excluded from this checkpoint.

## Remaining scope

Parameters/optimizers remain CPU F32. Full dtype/operator/catalog coverage,
F64 graph-building/transform primals, half/complex/FFT, general indexing/out,
vmap/compiler/export/distributed support and full NVIDIA/Metal models remain
open. Higher/capture support is operator-specific. Name presence does not prove
semantic, device or performance parity. See the
[compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and
[Rust CUDA research](../../../docs/RUST_CUDA.md).
