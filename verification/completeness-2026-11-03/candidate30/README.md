# Continuing PyTorch/PyG work: verified candidate30

The matching installed integration passes every available recorded check. Full
framework parity remains open; changes are local and uncommitted. Previous
[candidate29](../candidate29/README.md) retains its complete proof and failed-attempt
history. Numerical references are Torch 2.13.0+cpu and PyG 2.8.1, development
dependencies only. Ferro retains its zero-dependency Rust core and one
record_fn/Op AD engine. No NVIDIA or Apple GPU is available; Metal is not implemented.

## Matching-source validation

Exactly 1,087 selected inputs and all 124 installed files match the fresh
typed-modules-current29-proof1 wheel. Both source ZIPs, frozen trees, wheel
payloads, main inputs and private/global installed packages were rehashed before
publication. The manifest records exact identities and verification time.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,771 discovered: 1,741 passed, 30 GPU cases unavailable. |
| Rust workspace, reused exact-source evidence | Core 1,425 passed / two ignored; fastcpu 41 passed / two ignored; CUDA 142 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation, reused | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Native DLPack, reused | 12 harness passes; one GPU body returns early and three mandatory cases are filtered. |
| Eight fresh standalones | All eight exit0: binding, compiled fusion, fuse, operator/Torch, safetensors, seed0 200-trial fuzz, graph training/restart and inventory. |
| Focused integration | 16 methods pass: eight normalization and eight Bilinear/typed training controls, including both fresh-process workers. |

Only nine Python module/test paths changed from candidate29. The 569 selected
Rust, CUDA and build inputs are identical to the original
typed-training-current28-proof1 workspace/CUDA proof and the
typed-training-current28-proof2 DLPack proof. The native-equivalence receipt is
explicit: no new workspace, CUDA-check or DLPack command is claimed for this
module integration. The wheel, complete Python discovery, focused methods and
eight standalones ran anew.

Unittest reports 1,765 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals remain 1,616 outer runtime
passes plus two doctests, excluding 18 nested summaries. Optional GPU/tokenizer
bodies can return early; harness passes do not establish hardware execution.
Focused gates overlap full discovery and are not added to its totals.

## Added since candidate29

- CPU F64 affine LayerNorm and RMSNorm use typed parameters and the existing
  functional operators. Multi-axis normalization, bias options, values and all
  input/parameter gradients are compared with actual Torch.
- BatchNorm1d/2d/3d and InstanceNorm1d/2d/3d accept CPU F64 affine parameters and
  typed running buffers. Tests cover train/eval, tracked/untracked state,
  cumulative BatchNorm momentum, batched/unbatched inputs, counters, empties and
  singleton failures. Resets preserve tested parameter/buffer identities, leaf
  flags and held gradients. Typed SGD trajectories and a fresh process verify
  complete model/optimizer/config/RNG state, counters and subsequent losses.
- Bilinear accepts CPU F64 parameters and typed resets, retaining default F32
  construction. Native composition covers matching batch shapes, strided and
  empty inputs, optional bias and all input/parameter gradients. Empty outputs
  now use a real contiguous copy matching the tested reference layout while
  retaining their graph. Existing GaussianNLLLoss is exercised with live double
  inputs and its clamped-variance reverse derivative; its implementation is
  unchanged. Typed SGD and fresh-process state/RNG continuation pass.

These are Python factory/state extensions over unchanged native operators.
Initialization streams are not claimed identical to Torch; numerical references
copy explicit parameters. Corrupt dtype/config/state controls reject before
partial restoration. The first Bilinear scratch comparison's two empty-layout
failures remain recorded; the genuine contiguous output correction was included
before this fresh packaged proof. Earlier candidate29 integration/startup
failures remain in its archive rather than being relabeled as successful runs.

## Remaining scope

The added module support is CPU F32/F64 eager with matching operand dtypes.
Bilinear does not broadcast batch dimensions. Existing F64 higher/forward and
capture boundaries remain; preserved F32 Bilinear higher/JVP controls do not
establish F64 transform coverage. Affine InstanceNorm rejects a zero batch, as
tested against Torch. GroupNorm F64 affine factories, additional F64 optimizers
and convolution work remain isolated. General F64 module or GPU training support
is not established by this checkpoint.

Inherited optimizer limits remain: Adam public F32 step tensors reject counters
above 2^24; native whole-training snapshots retain exact u32 counters without a
higher-counter Torch trajectory claim. Momentum/moment snapshots use contiguous
typed buffers and optimizer loads require exact parameter dtype. SGD's checked
zero_grad(False) boundary for graph-bearing or unsupported-layout manual
gradients remains. Parameter construction owns data; explicit shared storage is
separate. F32 Linear construction keeps its prior zero bias, while explicit
reset initializes bias uniformly. Embedding F64 remains ordinary eager CPU;
RemoveIsolatedNodes has explicit Batch/custom-index boundaries and a same-process
serialized continuation proof.

The known power(x, 0) direct-JVP zero-tangent stride mismatch on transposed inputs
remains. Its repair, generic empty-gradient accumulation/scheduling, unused
noncanonical materialized-gradient layouts, physical-batch Jacobians and shared
multi-output/nested rules remain isolated. Earlier scalar-power ISA and large
virtual empty-reduction layout limits remain explicit. Full framework parity,
half/complex/FFT, general indexing/out, vmap/compiler/export/distributed and
complete GPU models stay open. Name presence does not establish semantics or
performance. See the [compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and
[Rust CUDA research](../../../docs/RUST_CUDA.md).
