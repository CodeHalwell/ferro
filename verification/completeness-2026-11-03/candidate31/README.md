# Continuing PyTorch/PyG work: verified candidate31

The corrected matching integration passes every available recorded check. Full
framework parity remains open; changes are local and uncommitted. Previous
[candidate30](../candidate30/README.md) retains its complete proof. The first
adaptive/GroupNorm integration remains separate failed evidence.

References are Torch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro retains its zero-dependency Rust core and one record_fn/Op AD engine.
No NVIDIA or Apple GPU is available; Metal is not implemented.

## Matching-source validation

Exactly 1,092 selected inputs and all 124 installed files match the fresh
adaptive-groupnorm-current30-proof2 wheel. Both source ZIPs, frozen trees, wheel
payloads, main inputs and private/global installed packages were rehashed before
publication. The manifest records identities and verification time.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,787 discovered: 1,757 passed, 30 GPU cases unavailable. |
| Rust workspace | Core 1,433 passed / two ignored; fastcpu 41 passed / two ignored; CUDA 142 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Fresh native DLPack | 12 harness passes; one GPU body returns early and three mandatory cases are filtered. |
| Eight fresh standalones | All eight exit0: binding, compiled fusion, fuse, operator/Torch, safetensors, seed0 200-trial fuzz, graph training/restart and inventory. |
| Focused integration | All 16 methods pass: eight typed RMSprop/Adagrad and eight GroupNorm/LRN controls, including fresh-process continuation. |

Workspace/CUDA checks ran against adaptive-groupnorm-current30-proof1. All 557
selected Rust/CUDA/build inputs outside the Python binding crate match proof2
exactly, so that workspace evidence is reused under the explicit receipt. The
only proof1-to-proof2 source change is the Python binding's training_state.rs
validation dispatch. Its corrected wheel, full Python suite, focused tests,
DLPack unit checks and eight standalones ran anew. No proof2 workspace rerun or
unchanged-binding claim is made.

Unittest reports 1,781 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals are 1,624 outer runtime
passes plus two doctests, excluding 17 nested summaries. Optional GPU/tokenizer
bodies can return early; harness passes do not establish hardware execution.
Focused methods overlap full discovery and are not added to its totals.

## Added since candidate30

- Native CPU F64 RMSprop and Adagrad retain full-precision options and typed
  state, alongside the existing F32 paths. Actual-Torch trajectories cover
  centered/momentum RMSprop, maximize, coupled weight decay, Adagrad rate decay
  and initial accumulators, skipped/frozen slots, shared storage and mixed
  F32/F64 groups. Tests compare gradients and Double Linear training, validate
  state/config atomicity and prove exact fresh-process model/optimizer/RNG
  continuation.
- Adagrad honors constructor defaults for initial accumulators and current
  defaults when an added group first becomes active. Lazy state creation does
  not invoke user state/load hooks, and unused F32 slots do not block a valid
  Double-only initialization. Invalid later group/state checks precede commits.
- GroupNorm accepts CPU F64 affine parameters with typed resets over its existing
  functional composition. Actual-Torch values and all input/parameter gradients,
  strided and empty inputs, state identity and fresh-process training are covered.
  Existing F64 LocalResponseNorm composition is exercised in the same module/SGD
  controls; no new LRN operator is claimed.

The first packaged proof reported 1,781 executions: 1,749 passed, 30 GPU cases
were unavailable and eight unique Adaptive methods failed. One method's eight
subcases produced eight errors, giving 15 error indicators altogether. All eight
GroupNorm methods passed. The common failure was the binding's step-gradient
preflight still applying the old F32-only optimizer validation.

The repair removes the misplaced Adaptive arm from a constructor dispatch and
adds it to validate_gradient_storage. Only training_state.rs changed between
proofs; fixtures were not weakened. The initial full/focused failures, their raw
tracebacks and the corrected fresh wheel's results all remain recorded.

## Remaining scope

New optimizers support ordinary CPU F32/F64 training, finite options and checked
counters through 2^24; attempting a further active step rejects before mutation.
RMSprop's Double path accepts finite alpha above one as Torch does, while its
legacy F32 validation remains unchanged. Unsupported nonfinite options reject
explicitly even where Torch accepts them. Snapshot buffers are contiguous,
state loads require exact dtype, and automatic Torch-style state casting is not
claimed. Graph-bearing or unsupported-layout manual gradients retain the checked
zero_grad(False) boundary; step preserves tested gradient handles.

GroupNorm F64 support uses existing first-order CPU arithmetic. F64 higher/forward
transforms, capture, GPU execution and general module coverage are not established.
The new Adamax/Adadelta/ASGD, convolution/recurrent/pooling and AD scheduling work
remain isolated. Prior Parameter ownership, Linear initialization, Adam public
step-counter and graph-container boundaries remain in the archived reports.

The known power(x, 0) direct-JVP zero-tangent stride mismatch, unused noncanonical
materialized-gradient layout, generic empty-gradient accumulation/scheduling,
physical-batch Jacobians and shared multi-output/nested repairs remain outside
this checkpoint. Full framework parity, half/complex/FFT, general indexing/out,
vmap/compiler/export/distributed and complete GPU models stay open. Name presence
does not establish semantics or performance. See the
[compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md),
[GPU portability](../../../docs/GPU_PORTABILITY.md) and
[Rust CUDA research](../../../docs/RUST_CUDA.md).
