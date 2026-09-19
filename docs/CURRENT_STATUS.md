# Current capability ledger and next milestone

> PR scope: CPU checkpoint/optimizer APIs and the two training/restart proofs are included. CUDA prepared segments and tiled CPU BMM described below are companion development work outside this branch. [Final isolated PR verification](../verification/architecture-restart/PR_VERIFICATION.md) supersedes broad-checkout test counts for this PR.

Updated 2026-09-19. This is the current roadmap status index. The
[foundations roadmap](FOUNDATIONS_AND_ARCHITECTURE_ROADMAP.md) retains the
detailed B0-B7 acceptance criteria; [FUTURE.md](FUTURE.md) and
[PARITY_ROADMAP.md](PARITY_ROADMAP.md) retain earlier strategy.

## Evidence boundary

The original index reconciled source and archived reports. The selected CPU training/restart fixtures now have a fresh matching-build run; see the local result below. Other capability rows retain their separately scoped evidence.
Inspected checkout HEAD: `dc13a7199c23d92be5dd766102af03a6bfcb4bc3`, with
uncommitted implementation, tests and reports. HEAD alone does not identify
the inspected implementation. Do not label working-tree additions released,
merged, or verified at this commit.

"Implemented" means source exists for a bounded surface. "Recorded verification"
means the linked report records execution on its own source/build identity.
"Open" means a declared gate still needs evidence or implementation. No bundle
is globally complete merely because its primitives or tests exist.

Evidence order: current source and matching execution manifests, this ledger,
detailed foundation acceptance criteria, then historical strategy. Preserve
historical failure reports rather than rewriting them into current status.

## Capability matrix

| Area / bundle | Implemented surface | Device, dtype and API boundary | Evidence and remaining gate |
|---|---|---|---|
| Tensor and first-order AD / B0-B1 | Checked shape/layout work, typed selection, storage versions, gradient contracts and Python cotangent entry points | Primary math/AD is f32; typed storage is not full typed arithmetic. Device fallback and layout behavior remain operation-specific. | Core and binding suites are recorded in [combined verification](../verification/post25/combined/README.md). A public per-op dtype/placement matrix and broader resident indexing remain open. |
| Public training / B1 | Python Module registration/containers, Linear, losses, SGD/Adam/AdamW, freezing, Trainer and progress | Python convenience APIs do not imply complete torch compatibility or universal CUDA layer coverage. | [Training API tests](../crates/ferro-py/tests/test_training_api.py) and [architecture fixtures](../verification/python-architecture-api/test_architecture.py). Full model certification remains separate. |
| Training state / B2 | Owning snapshots, immutable generations, named optimizers, ties, persistent buffers, modes/config and one explicit Generator; Python save/load/restore | Whole-training restore is paused, compatible, whole-contiguous CPU f32. Pending gradients, data cursors, optimizer groups, schedulers, extra RNG streams and arbitrary loop state are outside the Python contract. | [Python state report](../verification/post25/python-state/README.md), [checkpoint tests](../crates/ferro-py/tests/test_training_checkpoint.py), [schema tests](../crates/ferro-py/tests/test_checkpoint_schema.py). Broader state completeness and CUDA transactions remain open. |
| Segments / B3 (CUDA work separate) | CPU sum/mean/max/softmax; prepared CUDA sum/softmax and adjoints; public Python PreparedSegments | First-order f32. CUDA topology uploads once per preparation. Nonempty softmax performs a synchronized four-byte status download; no zero-total-transfer claim. Mean/max remain CPU-only; capture unsupported. | [Native report](../verification/post25/segments/README.md), [Python prepared-plan report](../verification/post25/prepared-python/README.md). General resident indexed adjoints and additional segment kernels remain open. |
| Sparse / B4 | Validated COO/CSR and CPU sparse algebra, with Python wrappers | CPU f32 algebra does not establish resident CUDA SpMM/SDDMM or a complete graph data pipeline. | [Sparse tests](../crates/ferro-core/tests/foundation_sparse.rs) and architecture fixtures. Graph batching, resident sparse execution and GNN/connectome/graph-transformer training/restart gates remain open. |
| Vision, sequence, basis / B5 | Grouped convolution/window work, RNN/GRU/LSTM cells and explicit masked/reset/truncated unroll, spline basis and KAN | New architecture wrappers are CPU f32 scoped. Explicit unroll is not fused scan or bounded-memory training. | [Foundation publication](../verification/foundation-wave/pr-readiness/PR_BODY.md), architecture fixtures, recurrent/basis tests. GPU implementations, decoder breadth and full model proofs remain open. |
| Higher-order AD / B6 | Functional gradients/VJPs and graph-building derivatives for a declared smooth subset; Python exposure | Unsupported derivative paths must reject. No general operator coverage, f64 AD, or all-PINN support claim. | [Implementation](../crates/ferro-core/src/higher_order.rs), [tests](../crates/ferro-core/tests/higher_order.rs). Mixed coordinate/parameter gradients and each scientific/adversarial architecture need separate certification. |
| Compilation and capture | Compiled supported inference DAGs, pointwise fusion and prepared static CUDA model execution | Supported operation/layout subsets only. No automatic whole-training compiler or general torch.compile parity claim. | [Graph implementation](../crates/ferro-core/src/graph.rs), [static CUDA owner](../crates/ferro-cuda/src/static_graph.rs), combined regression report. Mutable state/RNG/restore interactions need explicit gates. |
| CPU BMM (separate work) | Independent unpacked tiled arithmetic on production BMM and install-only registry paths; explicit packed nonbatched APIs remain separate | Current source differs from the earlier conservative scalar containment snapshot. Custom registrations can change dispatch. | Combined report records limited same-run parity/timings. Original numerical incident and later illegal-instruction incident remain unexplained. |
| Architecture certification / B7 | Existing training examples, primitive reference/gradient tests and focused stochastic checkpoint continuation | These provide bounded proofs, not all-family held-out learning and fresh-process restart certification. | Next milestone below; certify each family, variant, API, dtype and device independently. |

## Recorded verification and chronology

1. [Foundation publication](../verification/foundation-wave/pr-readiness/PUBLICATION.md)
   cleared its documented merge blocker through software-path containment,
   explicitly leaving the numerical root cause unresolved. Its Python checkpoint
   exclusion and scalar timing figures describe that earlier scope.
2. [Post25 combined verification](../verification/post25/combined/README.md)
   records a rebuilt noneditable wheel, matching source/wheel/install facade
   hashes, native identity and 438 stable inputs for that run. It reports core
   762 passed / 2 ignored, CUDA 118 passed, Python discovery 86 passed,
   architecture 8 passed, and training checkpoints 13 passed. These are reported
   suite results, not new execution here; overlapping/repeated suites must not
   be summed as unique tests.
3. [Prepared Python topology verification](../verification/post25/prepared-python/README.md)
   separately records 10 public tests including six CUDA tests, six Rust CUDA
   segment tests and 15 binding tests. Its report explicitly says it is not the
   parent combined freeze. Do not merge separate manifests into one tested tree.
4. Current checkpoint schema and alias tests add narrower source contracts.
   Their presence alone does not establish inclusion in an earlier report.

Some linked post25 reports and implementation files are currently untracked.
These links are local evidence references, not a guarantee that a fresh clone
contains them. Before publication, select durable reports/manifests explicitly;
do not publish raw logs, wheels or whole verification directories by default.
Do not rerun archival collectors that overwrite preserved evidence.

## Open defects and investigation tracks

| Priority | Issue | Required next evidence |
|---|---|---|
| P0 investigation | Historical 343812-ULP CPU discrepancy remains root-cause unresolved despite containment. | Preserve original failure; isolate cause and independently review any change that restores suspect arithmetic. Passing repetitions do not resolve it. |
| P0 investigation | Recorded Windows 0xc000001d illegal-instruction exit remains unexplained. | Reproduce with binary/source identity and CPU feature/compiler diagnostics; distinguish environment, dispatch and arithmetic causes. |
| P1 correctness | Re-registering CUDA replaces the registry/stream; generic copying of an older allocation can consult the new backend and fail. | The [prepared-plan report](../verification/post25/prepared-python/README.md) links an isolated tensor probe. Define initialization/ownership semantics and add a regression before changing behavior. |
| P1 residency | General backward detaches a strided seed through a host round trip. | Add a counting-backend regression and real CUDA check for arbitrary supported seed layouts; contiguous-seed segment tests do not close this. |

These priorities guide follow-up work; they neither reopen a historical merge
decision automatically nor authorize merging. No incident-free claim is made.

## Next milestone: CPU public training with reproducible restart

Status: implemented and locally verified for the two CPU f32 fixtures; independent review and broader B7 certification remain open. The work extends existing public APIs and checkpoint
regressions into two small architecture proofs before widening the device scope.
B0/B1/B2 provide the starting surface; this closes only the selected B7 rows.

| Fixture | Learning gate, declared before execution | Numerical and restart gate |
|---|---|---|
| Two-layer nonlinear regression | Fixed disjoint train/held-out samples of a smooth nonlinear function; seeds 7, 11 and 43; held-out MSE at most 25% of a training-mean constant predictor, within a fixed documented step budget. | Compare input and every parameter gradient with a same-weight independent reference before updates; checkpoint midway and compare uninterrupted continuation with restore into fresh compatible objects in a fresh process. |
| Two-layer classification with I64 labels | Fixed balanced synthetic two-class data with disjoint held-out samples; the same seeds; held-out accuracy at least 95%, with class-prior baseline reported. | Preserve I64 labels through minibatching and the public loss; compare gradients to the reference, then compare losses, parameters, optimizer state and next explicit-Generator draws after restart. |

Implementation scope:

1. Add public examples and small automated fixtures for these two models.
   Reuse Module, native optimizers and checkpoint APIs; do not embed engine
   implementations or use private bindings.
2. Use explicit deterministic parameter initialization and one explicit Generator
   for stochastic training draws. Built-in initialization that uses implicit RNG
   must be replaced or controlled within the fixture.
3. Keep input order deterministic from fixed data and the restored step. State
   plainly that this tests continuation at an optimizer-step boundary, not
   checkpoint serialization of DataLoader/prefetch cursors.
4. Compare gradients before optimizing, with predeclared f32 tolerances
   (initial target: atol 1e-5, rtol 1e-4). Require bit-exact resumed trajectory
   within the same binary/environment for these CPU fixtures. Cross-platform
   bitwise reproducibility is not a gate or a claim.
5. Clear gradients before snapshot/restore. Compare saved tensors, moments,
   counters, configuration and RNG state, and retain negative cases for
   incompatible models, aliases and late optimizer validation failure.
6. Run existing regression suites once against the final built extension and
   record source/dirty manifests, extension hash, commands/exits, skips, seeds,
   tolerances and held-out results. Never weaken thresholds silently after a
   failing run; document a revised design before claiming acceptance.

Implemented targets:
`examples/train_restart_regression.py`,
`examples/train_restart_classifier.py`, and
`crates/ferro-py/tests/test_architecture_restart.py`.
Keep the reusable process/restart harness with these tests.

Milestone completion requires both fixtures across all three seeds, successful
fresh-process continuation and negative restore checks, a final matching-build
regression report, and review of the bounded claim. GPU training, general data
resume and all-model-family certification stay open.

## Local result for the selected CPU milestone

Both examples passed all three seeds at 400 steps, with PyTorch input/parameter
gradient checks and bit-exact continuation in fresh processes. Regression
held-out MSE was 0.54-1.21% of the constant baseline; classification accuracy was
100%. Serialized model/optimizer/config/RNG state and next RNG draws matched.
The public CPU integer-label cross-entropy binding was added to support this gate.

[Verification report](../verification/architecture-restart/README.md) records
the matching release wheel, 434 unchanged selected source inputs, 105 passing
Python tests and successful core/CPU/tokenizer, binding, parity, serialization,
fusion and fuzz commands. An initial missing-NVRTC verifier failure is preserved;
the final runtime-path-corrected run passed. No CUDA training, cross-platform
determinism, general data-cursor resume, independent review or hosted CI claim.

## Subsequent sequence

- Complete B3 general resident indexing and B4 sparse/graph batching before
  claiming a resident GNN or sparse graph transformer.
- Certify CNN, recurrent and KAN training/restart independently using existing
  CPU primitives; extend to CUDA only with execution and residency evidence.
- Certify PINN/contractive-AE/gradient-penalty models only after required mixed
  derivative depth reaches model parameters.
- Expand checkpoint state and architecture variants explicitly, then measure
  representative performance with equal work, grad modes and same-session
  comparisons. Retain the separate CPU incident investigations throughout.

## Maintenance rule

For each status change record the API/device/dtype/layout/derivative subset,
source identity including dirty files, evidence path, executed/skipped counts,
remaining exclusions and reviewer decision. Update this index without erasing
historical reports. New source presence advances "implemented"; only matching
execution evidence advances "recorded verification"; only the declared complete
acceptance surface advances a milestone.
