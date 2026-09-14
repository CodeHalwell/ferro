# DRAFT / BLOCKED: frozen Ferro foundations

**Not ready to merge. Never enable auto-merge for this handoff.** The historical fastcpu BMM incident remains **OPEN**: expected `11.969769477844238`, observed `11.641884803771973`, discrepancy **343812 ULP**. Its root cause is unknown. Passing reruns neither explain nor fix it; no naturally recurring production failure was reproduced during final verification. The p=90 omission fingerprint and injected diagnostic schema tests are counterfactual/test-buffer experiments, not a production RED/GREEN fix. Fastcpu changes are test-only diagnostics and independent-oracle coverage, not a kernel fix.

## Frozen scope and review order

1. **Core foundation/state/AD/CPU architecture**: checked shapes/layouts and typed data; partial-backend correctness, loss adjoints and BatchNorm validation; leaf/version/no-grad contracts and bounded smooth higher-order AD. Rust named buffers/scalars, owning snapshots, strict CPU restore, optimizer state, immutable generation publication and stochastic optimizer-step-boundary continuation. CPU segment/sparse graph algebra, recurrent cells and masked/reset/truncated unroll, spline/KAN basis and rectangular grouped convolution/window adjoints. Shared tensor, module and autograd files are intentionally one coherent commit rather than speculative partial patches.
2. **Fastcpu diagnostics**: retain failure evidence and compare with an independent scalar oracle; include the shared test helper and counterfactual scalar reproducer. Assertions and tolerances are not relaxed.
3. **Native mixed Python package**: Module/build registration and containers, Linear/MSE, SGD, Trainer/progress, functional AD and native graph/recurrent/basis/convolution wrappers, with regressions and executable public CPU examples.
4. **Documentation/evidence**: bounded verification manifests, historical review reports and explicit publication/evidence policy. This body and [PUBLICATION.md](verification/foundation-wave/pr-readiness/PUBLICATION.md) supersede stale milestone/status prose in earlier planning documents. This is one PR, not completion of the architecture roadmap.

`Module::validate_scalars` rejects both nonempty supplied state and omission when a custom module declares scalars. Custom stateful modules must provide their own restore protocol; public recurrent/basis exports and adversarial tests are included. AI agents assisted implementation, independent scoped reviews, verification and packaging. Human review is still required; no exhaustive security or review approval is claimed.

## Unsupported / follow-up work

- Full Rust training-state snapshots are **not exposed in Python**. Python model/training convenience APIs do not imply checkpoint parity.
- New graph/sparse/recurrent/basis/window/convolution wrapper computations are **CPU-f32 scoped**, not resident CUDA implementations. Legacy host fallback is not GPU residency.
- Whole-model checkpoint transactions are **CPU-only**, require paused training and matching architecture/configuration, and exclude pending gradients, accumulation/update phase, schedulers, data/sampler/prefetch cursors and arbitrary loop state.
- Custom Module/Optimizer implementations remain responsible for pure validators and correct commits. No device rollback, concurrent-reader isolation, panic/OOM rollback or physical power-loss guarantee.
- Higher-order AD supports a declared operator subset, not universal differentiation or all PINN/AE/GAN families. Recurrent unroll is not fused scan or bounded-memory training. Four decreasing-loss fixtures are not all-model-family/generalization/restart certification.
- No performance benchmark, speedup, warning-clean/clippy, sanitizer or Miri certification. The fastcpu incident must be resolved with causal evidence before a merge-readiness decision; there is no justified ETA.

## Verified evidence

Final frozen-source execution: **41 canonical leaf commands, all exit 0**, without retrying away failures. The unchanged `verification/dlpack-raw-review/run.py all` integration was followed by a fresh noneditable release wheel build/install and a second complete integration against that wheel.

- Each complete integration: core **751 passed / 2 existing timing tests ignored**; fastcpu + tokenizer **27 passed / 1 existing timing test ignored**; CUDA **111 passed / 0 ignored**; capture-layout standalone GPU test **1 passed**; Python discovery **69 passed**, no reported skips.
- Standalone binding Rust unit suite: **15 passed**, default-parallel. Public eager/snapshot DLPack: **5 / 1 passed**. Supplemental CUDA default-parallel: **111 passed / 0 ignored**.
- Public Python contracts **17 passed**; architecture tests **8 passed**; graph, sequence, KAN and convolution native-autograd training examples exit 0.
- Reconciled totals are **1906 Rust test executions and 238 unittest executions**, not unique tests. Nested child summaries, wrapper/copy logs and standalone example assertions are not double-counted.
- `FERRO_REQUIRE_CUDA=1`; NVRTC/cuBLAS runtime bins explicitly on PATH; Cargo `-j2`. Complete integrations use `RUST_TEST_THREADS=1`; supplemental CUDA/binding tests use the default parallel harness.
- All 14 facade files match source/wheel/installed bytes. Wheel SHA256 `bbe16d6b5a354e32356d6fd898fdc94e6e05b4bedc0ea390b795784ddf4f47ae`; native SHA256 `8db7f4c8fa6fb60e9b38a987a756d54a2f5138815b20e0409c1ec873ab431358`. Installed package/native hashes stayed stable. All 418 recorded build/source/test inputs matched before/after verification.

[README](verification/foundation-wave/pr-readiness/README.md), [canonical commands/counts/log hashes](verification/foundation-wave/pr-readiness/leaf-results-counts.json), [summary](verification/foundation-wave/pr-readiness/summary.json) and [source provenance](verification/foundation-wave/pr-readiness/tested-source-provenance.json) retain the details. Packaging rechecked every original non-null allowlist SHA256 and every canonical raw-log SHA256. No production behavior was changed after verification. The staged whitespace gate found and removed one trailing blank line in `ferro/nn/functional.py`; its AST is unchanged and source-loaded CPU contracts/architecture tests were rerun. The 14-file byte identity above describes the pre-packaging tested wheel, not a new wheel containing that EOF-only change. The only runner change adds an optional fresh output directory; it was syntax/help-tested, not represented as a rerun of all GPU gates.

## Test Plan / reproduction

The additional packaged-source smoke passed discovery (69), contracts (17) and architecture tests (8), with all 14 authored facade files staged against the verified native binary. Its expected saved-version panic was caught by the regression test. Run it with `crates/ferro-py/.venv/Scripts/python.exe verification/foundation-wave/pr-readiness/packaging_smoke.py`; it does not rebuild the native wheel.

Literal commands from final verification include:

```text
python verification/foundation-wave/pr-readiness/verify.py
cargo test -j2 -p ferro-core
cargo test -j2 -p ferro-fastcpu -p ferro-tokenizer
cargo test -j2 -p ferro-cuda
crates/ferro-py/.venv/Scripts/python.exe -m unittest discover -s crates/ferro-py/tests -v
crates/ferro-py/.venv/Scripts/python.exe verification/python-api-contract/contract.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/test_architecture.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/training_examples.py
git diff --check
```

For a new full execution, use `python verification/foundation-wave/pr-readiness/verify.py --output-dir verification/foundation-wave/pr-readiness/rerun-01` with the documented Windows Python/CUDA prerequisites. Use a new directory on each attempt; do not delete or overwrite old evidence. Exact original commands/environment and their results are in the canonical manifest.

**Evidence publication policy:** selected source, tests, reports and manifests are committed. Raw logs, prior-wave archives, diagnostic dumps, executables, wheels, caches and credentials are not uploaded; original raw evidence remains local, referenced by SHA256 rather than presented as a downloadable artifact. Historical report references to omitted runners/results/logs are local provenance, not promises that those files exist in this PR. The portable test/reproducer entrypoints and archival-runner requirements are listed in PUBLICATION.md. No broad `git add`, archive deletion, local main checkout, review-bot request or merge is part of this handoff.
