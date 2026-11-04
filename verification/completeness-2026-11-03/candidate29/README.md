# Continuing PyTorch/PyG work: verified candidate29

The corrected integration passes every available recorded check. Full framework
parity remains open; changes are local and uncommitted. Previous green
[candidate28](../candidate28/README.md) retains its complete proof. The failed first
typed-training integration and startup failures remain separate evidence.

References: Torch 2.13.0+cpu and PyG 2.8.1, development dependencies only.
Ferro retains its zero-dependency Rust core and one record_fn/Op AD engine.
No NVIDIA or Apple GPU is available. Metal is not implemented.

## Matching-source validation

Exactly 1,085 selected inputs and all 124 installed files match the fresh
typed-training-current28-proof2 wheel. Both source ZIPs, frozen trees, wheel
payloads, main inputs and private/global installed packages were rehashed before
publication. The manifest records exact identities and verification time.

| Check | Result |
|---|---|
| Full installed Python discovery | 1,755 discovered: 1,725 passed, 30 GPU cases unavailable. |
| Rust workspace | Core 1,425 passed / two ignored; fastcpu 41 passed / two ignored; CUDA 142 harness passes / five mandatory cases filtered; tokenizer eight harness passes. |
| CUDA doctests / all-target compilation | Two intended E0502 borrow errors pass; all targets compile without a driver. |
| Native DLPack | Fresh proof2 run: 12 harness passes; one GPU body returns early and three mandatory cases are filtered. |
| Eight standalones | All eight proof2 runs exit0: binding, compiled fusion, fuse, operator/Torch, safetensors, seed0 200-trial fuzz, graph training/restart and inventory. |
| Focused integration | 27 methods pass: SGD defaults three, Adam/AdamW eight, Embedding/Bag eight and RemoveIsolatedNodes eight. |

Workspace/CUDA results come from typed-training-current28-proof1, reused only
after independently comparing all 569 selected Rust, CUDA and build inputs with
proof2. The only source changes were the Python SGD constructor and its new test.
The equivalence receipt is explicit: there was no new proof2 workspace command.
The wheel, full Python discovery, focused tests, DLPack and standalones ran anew.

Unittest reports 1,749 executions and 25 skip indicators: 24 individual GPU
skips and one class hiding six cases. Native totals are 1,616 outer runtime
passes plus two doctests, excluding 18 nested summaries. Optional GPU/tokenizer
bodies can return early; harness passes do not establish hardware execution.
Focused gates overlap full discovery and are not added to its totals.

## Added since candidate28

- Native CPU F64 Adam/AdamW retain full-precision options and typed moments,
  including AMSGrad, maximize, coupled/decoupled decay, frozen slots and mixed
  F32/F64 groups. Existing F32 math is retained. Tests cover actual-Torch
  trajectories, invalid-state atomicity, shared storage and complete exact
  fresh-process model/optimizer/config/RNG continuation.
- Embedding and EmbeddingBag accept CPU F64 module parameters using the existing
  native lookup/bag operators. Ordinary gradients, padding/frequency/max-norm
  behavior, bag reductions and live sum-mode per-sample weights are compared.
  Explicit pretrained storage sharing, resets, held parameter/gradient handles,
  typed SGD and fresh-process aliased state/RNG continuation are tested.
- SGD now has Torch's default learning rate 0.001 and positional option order.
  Group-specific learning rates work without a redundant top-level rate.
  Disabled foreach/fused/differentiable options are explicit; unsupported active
  execution modes reject before initialization.
- RemoveIsolatedNodes compacts sorted incident nodes across homogeneous and
  heterogeneous relations, preserving edge order, duplicates and self-loops.
  Native F32/F64/I64/Bool gathers preserve tested gradients and exact typed data;
  shallow containers retain untouched field aliases. Sparse remapping handles
  huge zero-width node spaces without a dense node-sized map. Its serialized
  model/data/optimizer continuation test reconstructs in the same process.

The first integration reported 1,746 executions with 1,721 passes, 30 unavailable
GPU cases and one failing method. Its Embedding fresh worker exposed the missing
SGD default learning rate; an exact worker diagnostic retained the TypeError.
The production constructor was corrected and three regression methods added,
then a fresh wheel/full run passed. No fixture workaround replaced that repair.

Eight earlier standalone startup attempts failed because the private environment
had not yet received its dependency path configuration. Their individual import
errors and nonzero statuses remain intact; the installed proof1 retry and fresh
proof2 run both pass. Neither failed attempt is relabeled green.

## Remaining scope

Adam's public F32 step tensors reject counters above 2^24 rather than silently
rounding; native whole-training snapshots retain exact u32 counters. This does
not claim matching Torch update trajectories beyond that public bound. CPU/ISA
rounding and signed-zero differences remain limited by the measured controls.
Momentum/moment snapshots use contiguous typed buffers; optimizer loads require
exact parameter dtype. SGD's checked zero_grad(False) boundary for graph-bearing
or unsupported-layout manual gradients remains. Parameter construction owns its
data; pretrained/from_shared aliasing is explicit. F32 Linear construction keeps
its prior zero bias, while explicit reset initializes bias uniformly.

Embedding F64 support here is ordinary eager CPU differentiation; sparse
gradients, F64 higher/forward transforms, capture and GPU execution are not
established. RemoveIsolatedNodes rejects Batch/sampled containers, custom hooks
and secondary indexed supervision rather than leaving inverse mappings invalid.
Its state test is not a fresh-process restart claim. General F64 module coverage
and other F64 optimizers remain incomplete.

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
