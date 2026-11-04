# Continuing PyTorch/PyG work: verified candidate 21

This matching-build checkpoint passes every available recorded check. It covers
ongoing CPU implementation, not complete PyTorch/PyG parity. Changes remain
local and uncommitted. The prior green [candidate18](../candidate18/README.md) and
failed [candidate20](../candidate20/README.md), [candidate19](../candidate19/README.md)
and [candidate17](../candidate17/README.md) records remain available.

References are PyTorch 2.13.0+cpu and PyG 2.8.1, used only in development.
Ferro retains its zero-dependency Rust core and one record_fn/Op autograd engine.
No NVIDIA or Apple GPU is available; Metal is not implemented.

## Matching-build validation

The noneditable ABI3 wheel comes from an isolated copy of 981 selected
source/fixture inputs. All 112 installed files (111 Python plus native) match
the recorded wheel. Source ZIP, wheel and raw logs remain in
`/workspace/work/completeness-2026-11-03`; [manifest.json](manifest.json)
records their hashes and outcomes. Both Cargo locks are frozen and use cudarc
0.19.10 with CUDA 12.8 bindings. Source archives, isolated inputs and wheel
payloads were rehashed for this publication; installed hashes were recorded
before the subsequent candidate22 installation.

| Check | Result |
|---|---|
| Installed Python discovery | Exit 0: 1,488 discovered, 1,458 passed, 30 GPU cases unavailable. |
| Rust workspace, default test threads | Exit 0: core 1,241 passed / 2 ignored; fastcpu 39 passed / 2 ignored; CUDA 140 runtime harness passes / 5 strict GPU cases filtered; tokenizer 8 harness passes. |
| CUDA doctests | Two intended E0502 borrow errors pass; no E0460 artifact mismatch. |
| CUDA all-target compilation | Exit 0 without a CUDA driver/GPU. |
| Native DLPack | 12 harness passes, one GPU early return, three mandatory GPU cases filtered; explicit Python 3.12 link. |
| Binding, compiled fusion, fuse, operator/Torch and safetensors | All five standalone checks exit 0. |
| Fuzz | 200 trials, seed 0, exit 0. |
| Graph training/restart | Standalone learning/control and exact fresh-process continuation checks exit 0. |
| API inventory | Standalone exit 0; installed names and hashes recorded, without a semantic/performance score. |

Unittest reports 1,482 executions and 25 skip indicators: 24 individual GPU
skips plus one class hiding six cases. Rust totals use final outer summaries:
1,428 runtime passes plus two doctests; 18 nested child summaries are excluded.
One optional tokenizer body returns early. Optional GPU bodies can also return
early, so harness passes do not prove hardware execution. All eight standalone
commands, arguments, exit statuses and logs are retained in the manifest.

## Corrections and additions since candidate18

Candidate19's Softplus stride assertion and two ordinary F64 Sqrt reference
assertions were corrected with explicit numerical controls; their failed
results remain archived. Candidate20 then exposed two stale fixtures: a test
omitted required checkpoint keywords, and the binding example requested an
unsupported in-place ReLU6. Candidate21 changes exactly those two test/example
files, with no production delta from candidate20, and reruns the full matching
validation. It does not replace failures with selective reruns.

- Canonical `autograd.grad` accepts multiple outputs and live seeds through one
  native union traversal. Strict unused-input defaults and materialized-zero
  options are tested; legacy explicit-zero Tensor helpers remain separate.
  Supported forward JVP composition also covers jacrev. Retained tapes and
  unsupported transform modes remain explicit boundaries.
- CPU `equal`, `isclose` and `allclose`, typed rounded division, and checked
  NumPy scalar interoperability extend arithmetic. Sqrt uses its original
  saved output and exact derivative staging, allowing unused input mutation
  while guarding output mutation. Cross/vecdot and empty broadcast linalg
  controls extend the existing CPU linear algebra scope.
- Softplus supports typed beta/threshold options and stable small values;
  Softsign and LogSigmoid retain their staged CPU adjoints and F64 precision.
  Canonical activation keywords reject unsupported in-place mutation. Exact
  saved-value controls and the documented subnormal/ordinary F64 reference
  rounding differences remain visible in the tests.
- Batch/HeteroBatch selectors preserve current field values, typed offsets,
  live gradients and documented inverse aliases. Data/heterogeneous clone
  operations preserve selected-field ownership and tensor history. Custom
  scalar zero-offset extraction now retains the first field's view; negative
  selector and metadata-copy extensions are explicitly documented.

Earlier normalization/dropout, embedding, Module and optimizer hooks, typed
state, recurrence/attention/spatial layers, graph models, sampling and custom
aggregation/batching remain covered. See the archived candidate18 report and
the active [compatibility ledger](../../../docs/PYTORCH_PYG_COMPATIBILITY.md).

## Remaining boundaries

Parameters/optimizers remain CPU F32. F64 graph-building VJPs and transform
primals, complete dtype/operator coverage, half/complex/FFT, arbitrary advanced
indexing/out destinations, general vmap/compiler/export/distributed support,
full transform composition and complete NVIDIA/Metal models remain open.
Higher-order and capture support is operator-specific. Optional PyG native
sampling backends are unavailable; finite sampling is not backend-identical.

Source22 continues separately with multi-column forward transforms, joint
functional pullbacks, activation/layout and cumulative/Where repairs, and graph
field operations. Those later changes are excluded from this report. No full
framework, GPU-residency or performance claim follows from this checkpoint.
