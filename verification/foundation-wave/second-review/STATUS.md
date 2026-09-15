> Historical snapshot: current publication status and corrected parent/child test counts are in [PUBLICATION.md](/verification/foundation-wave/pr-readiness/PUBLICATION.md). The documented merge blocker is cleared by independently validated software BMM containment; historical root cause remains unresolved. Omitted logs/collectors are local provenance.

# Second-wave independent integration review

> Historical review evidence, not publication status. Referenced raw logs, earlier status files and result archives remain local-only; current compiled regressions are packaged in the core test suite. The final source-identity/count manifest and scope are under [pr-readiness](../pr-readiness/PUBLICATION.md).

## Outcome

Public `ferro_core::recurrent` and `ferro_core::basis` are registered. Their integration tests now import the public modules, not source paths. `public-red.log` proves both missing exports; `public-green.log` has 16 passing public-API tests.

One reproduced checkpoint blocker was fixed: a custom Module exposing scalar state but inheriting the default restore protocol rejected a complete snapshot, yet accepted one with every scalar removed and wrote its buffers. `custom-red.log` demonstrates the bypass. The default validator now rejects if either incoming or declared scalar state is nonempty. Stateless defaults still work. `second_review_state.rs` covers this failure and exhaustive single-key omission/duplication over a warmed Linear/BatchNorm/Dropout/Adam bundle, asserting all snapshot bits and model/buffer version counters remain unchanged.

## Final execution

- `cargo test -j 2 -p ferro-core --no-fail-fast`: exit 0; **751 passed, 0 failed, 2 ignored**, 175 top-level harnesses (`core-final.log`). Nine nested subprocess summaries are excluded, not counted as additional tests.
- `cargo test -j 2 -p ferro-fastcpu -p ferro-tokenizer --no-fail-fast`: exit 0; **27 passed, 0 failed, 1 ignored**, 10 harnesses (`cpu-tokenizer-final.log`). The historical fastcpu numerical incident remains **OPEN**; passing diagnostics do not resolve its root cause.
- Targeted state run: 29 passed before the additional exhaustive mutation test (`state-green.log`); final adversarial binary: 2 passed (`adversarial.log`). Both are included in the final full suite, not additive totals.
- Numerical rerun: basis 10, recurrent 6, window/convolution 7 passed (`numerical-evidence.log`). Bernstein 4004 values: p50/p95/p99/max 0 ULP. Two-layer KAN MSE initial 0.143284112, final 0.000002490, held-out 0.000002473. Delayed-copy TBPTT SGD loss 0.14829803 -> 0.00000000 at printed precision.
- `git diff --check`: exit 0. Existing unused-variable/import/mut warnings remain; this is not a warning-clean/clippy certification.
- `sources-final-before.json` and `sources-final-after.json` match exactly for core/fastcpu/tokenizer Rust sources and manifests. Python files are intentionally outside this native identity. Core Cargo.toml still has zero external dependencies.
- `results.json` records per-harness counts. Earlier logs/manifests are retained as earlier evidence, not substituted for final source execution.

## Independent source assessment

Read recurrent.rs, basis.rs, window.rs, conv2d.rs, Module defaults and recursive state handling, BatchNorm/Dropout, checkpoint staging/publication, and concrete optimizer prepare/commit paths. Read and exercised module-state contracts at `../module-state/STATUS.md`.

- Recurrent cells implement explicit RNN affine+tanh, LSTM input/forget/candidate/output gates, and reset-after GRU. Weight tensors are shared engine operands. Tests independently check explicit scalar formulas and finite differences, reset/masking/NaN padding, shared-weight unroll and truncation. Row-wise unroll is a reference implementation, not fused scan or bounded-memory training.
- Splines use f64 Cox-de Boor values/analytic derivative, with repeated-knot zero denominators and one-sided endpoint rules. Independent Bernstein and piecewise-polynomial test oracles differ from production recurrence; KAN checks both input/coefficient adjoints and finite differences. Custom basis higher-order input VJP rejects explicitly; coefficient-only graph support is bounded and tested. No all-operator/all-gradient or torch-wide certification.
- Window gather and scatter-add share one checked coordinate/padding mapping; grouped convolution uses CPU GEMM and its transpose adjoints. Direct spatial oracle, nonuniform upstream gradients, strict finite differences, groups/depthwise multiplier, strided views, overlap, empty batches and overflow checks pass. CPU fallback does not imply GPU residency.
- Checkpoint owning snapshots, strict keys/ties/bindings, CPU staging, optimizer buffer-copy validation and alias isolation precede commits. Scalar/RNG decoding is exact. Existing fixtures exercise late optimizer errors, malformed modes/buffers/RNG, retained counters, tie/freeze topology, stale moments, cloned optimizer aliases and token discard. Custom Module/Optimizer implementations remain responsible for pure validators, live stable buffers, and infallible commits; arbitrary trait implementers cannot be made transactional by these defaults.
- Actual disk restart fixture trains Linear -> BatchNorm -> Dropout -> Linear with Adam plus alternating momentum SGD, snapshots at step 3, continues to step 8, then publishes the old snapshot and reconstructs fresh objects. Exact continuation inputs/outputs/losses and all captured end-state bits match (`state-green.log`). This is stochastic CPU optimizer-step-boundary resume, not complete arbitrary training-loop serialization.
- Publication uses immutable generation payloads, flushed files, then one renamed pointer. Pre-publication interruption and corruption fixtures pass. No physical power-loss test was performed; Windows directory-fsync limits remain. No GPU transaction, concurrent-reader isolation, panic/OOM rollback, or device rollback is advertised.

Security heuristic scan (`security-scan.json`) found only the existing `nn::eval` module-mode function name; it is not dynamic evaluation. No embedded-secret or shell-process matches occurred in the scanned source. This does not constitute an exhaustive security audit.

## Boundaries and evidence location

Pending gradients/accumulation phase, data/sampler cursors, schedulers, update phase and prefetch state are excluded. Caller must pause training and rebuild matching architecture/configuration. Legacy optimizer f32 timestep precision limits remain. No GPU execution, binding rebuild, commit or publication was performed. This subagent independently reviewed other workers; the small export/default-validator changes still need parent review (no nested reviewer tool available).

Original KAN evidence is preserved at `../../../evidencefoundation-wave/basis/RESULTS.md` (repository path `evidencefoundation-wave/basis`). It was historically written there, not under verification. The canonical `../basis/README.md` points there and to this integrated public-API evidence. Do not delete or relabel its historical source-inclusion results as public API tests.
