# Continuing PyTorch/PyG work: verified candidate28

This integration passes every available recorded check. Full framework parity
remains open; changes are local and uncommitted. Previous green
[candidate27](../candidate27/README.md) retains its complete proof and independent
integrity receipt. This checkpoint adds bounded CPU F64 training and a native
PyG random-walk transform; it does not certify the separate ongoing experiments.

References: Torch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro retains its zero-dependency Rust core and one record_fn/Op AD engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-source validation

Exactly 1,077 selected inputs and all 123 installed files match
candidate28-proof1. Both source ZIPs, frozen trees, wheel payloads, main inputs
and private/global installed packages were rehashed before publication; the
manifest records the verification time and exact promoted wheel.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,728 discovered: 1,698 passed, 30 GPU cases unavailable. |
| Rust workspace | Core 1,417 passed / two ignored; fastcpu 41 passed / two ignored; CUDA 142 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Native DLPack | 12 harness passes; one GPU body returns early and three mandatory cases are filtered. |
| Eight standalones | Binding, compiled fusion, fuse, operator/Torch, safetensors, 200-trial seed0 fuzz, graph training/restart and inventory all exit0. |
| Focused integration | All 25 F64 Parameter/Linear, SGD and RandomWalkPE methods pass. |

Unittest reports 1,722 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals are 1,608 outer runtime
passes plus two doctests, excluding 18 nested summaries. Optional GPU/tokenizer
bodies can return early; harness passes do not establish hardware execution.
Focused gates overlap full discovery and are not added to its totals.

## Added since candidate27

- CPU F64 Parameter and Linear retain the existing native parameter slots and
  AD engine. Tests compare inputs and all parameter gradients, exact typed
  state, aliases, flags, reset behavior and invalid-state rejection.
- Native F64 SGD preserves full-precision options and measured staged updates,
  including momentum, dampening, Nesterov, weight decay, maximize and mixed
  F32/F64 groups. Typed state dictionaries and whole-training snapshots preserve
  exact fresh-process model/optimizer/config/RNG continuation. Existing F32
  update math is retained; other optimizer families remain F32 here.
- AddRandomWalkPE implements the actual PyG dense path for at most 2,000 nodes
  under a bounded work limit. Weighted outgoing degree determines transition
  values without a weight numerator. Zero walk length returns one column.
  Simple live F32 weights have native gradients; unweighted/detached parallel
  edges preserve overwrite values. Live weighted duplicates reject because
  their reference overwrite adjoints differ from native last-writer gradients.
  Typed feature append, shallow Data/Batch ownership and saved-data fresh-process
  training continuation are tested. F64 edge weights reject, matching PyG's
  dense assignment boundary. There are no random draws or runtime PyG calls.

The first focused command named the nonexistent test_sgd_float64 module and
failed before any tests executed. Its log/status remain separate; the corrected
test_sgd_f64 command passes all 25 methods. Earlier isolated SGD proof1 exposed
three problems: a stale F64 Parameter rejection fixture, incorrect F32 bias
reset behavior, and the Parameter capture exception category. Those were fixed
and proved by a fresh SGD proof2 wheel before this integration; original failures
are retained, and no failing run is relabeled green.

## Remaining scope

Parameter construction continues to own its data; explicit sharing is separate.
The default F32 Linear constructor retains its existing zero bias, while explicit
reset initializes bias uniformly. SGD momentum snapshots are contiguous, so
their original physical strides are not part of the state contract. Optimizer
loads require the exact parameter dtype rather than casting state. Step preserves
supported live manual-gradient handles; zero_grad(set_to_none=False) still rejects
graph-bearing or unsupported-layout manual gradients at the checked in-place
boundary. General F64 modules, other F64 optimizers, F64 GPU training and
higher-order optimizer execution remain open.

A known power(x, 0) direct-JVP zero-tangent stride mismatch on transposed inputs
remains in this checkpoint. Its fix, physical-batch Jacobians and shared
multi-output/nested rules are isolated work, not certified here. The reviewed
RemoveIsolatedNodes transform is also not included. Previous scalar-power ISA
and large virtual empty-reduction layout limits remain as recorded in candidate27.
Full dtype/operator/catalog coverage, F64 transform primals, half/complex/FFT,
general indexing/out, vmap/compiler/export/distributed and complete GPU models
remain open. Name presence does not establish semantics or performance.
See the [compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and
[Rust CUDA research](../../../docs/RUST_CUDA.md).
