# Frozen foundation bundle: verification passes, merge BLOCKED

> Publication note: this is the preserved pre-packaging verifier report. Statements below about no commits/publication describe that verifier, not the eventual branch state. [PUBLICATION.md](PUBLICATION.md) and [PR_BODY.md](PR_BODY.md) govern the draft scope and evidence policy. Raw logs and omitted historical artifacts referenced below remain local, not downloadable PR artifacts.

One PR scope is prepared on `feat-attention-parity`, based on HEAD `43dc8ea`. No production edits, staging, commits, pushes, PR creation, tolerance changes or performance benchmarks were performed by this verifier.

**DRAFT ONLY.** The historical fastcpu BMM incident remains OPEN: expected `11.969769477844238`, observed `11.641884803771973`, discrepancy `343812 ULP`. It did not recur in this run. Green tests do not establish a root cause or fix. The independent omission-of-p=90-product fingerprint is counterfactual analysis, not a reproduced production defect. See `../fastcpu-second-review/README.md` and `../review-fixes/fastcpu/README.md`. Parent review/commit approval and resolution of this blocker are still required; no honest merge ETA follows from passing tests.

## Current bundled implementation, not the whole roadmap

- Tensor/layout/dtype and typed-data correctness fixes, loss adjoints, BatchNorm validation, leaf/version safety, no-grad and a bounded smooth higher-order AD subset.
- Rust named buffers/scalars, owning snapshots, strict CPU restore and optimizer state, generation-addressed publication and tested stochastic CPU optimizer-step-boundary continuation.
- CPU segment and sparse topology/algebra, recurrent cells and masked/reset/truncated unroll, B-spline/KAN basis, rectangular grouped convolution and unfold/fold adjoints.
- Native mixed Python package with Module/build registration and containers, Linear/MSE, SGD, Trainer/progress, functional AD and graph/recurrent/basis/convolution wrappers. The four CPU training examples execute actual native autograd updates.

**Future/unsupported:** full Rust training-state snapshots are not exposed in Python; new architecture wrapper computations are CPU-f32 scoped, not resident CUDA implementations. Whole-model checkpoint transactions are CPU-only. Pending gradients, accumulation/update phase, data/sampler/prefetch cursors, schedulers and arbitrary training-loop state are excluded. Callers pause training and rebuild matching architecture/configuration. Custom Module/Optimizer implementations remain responsible for pure validators and correct commits. No device rollback, concurrent-reader isolation, panic/OOM rollback or physical power-loss guarantee. Higher-order AD is not universal; recurrent reference unroll is not fused scan/bounded-memory training. The architecture roadmap remains future acceptance criteria, not global completion or all-model-family certification.

## Fresh final execution

`python verification/foundation-wave/pr-readiness/verify.py` exited **0**. All **41 canonical leaf commands exited 0**, without retrying away failures.

| Gate | Actual result |
|---|---|
| Unchanged raw DLPack `all` runner | exit 0, all 13 original integration commands exit 0 |
| Standalone binding Rust unit suite, default parallel | 15 passed |
| Public eager / snapshot DLPack | 5 / 1 passed |
| Fresh noneditable release wheel build/install | both exit 0 |
| Complete integration repeated against installed wheel | all 13 commands exit 0 |
| Core, each integration | 751 passed, 2 ignored timing tests |
| Fastcpu + tokenizer, each integration | 27 passed, 1 ignored timing test |
| CUDA, each integration | 111 passed, 0 ignored |
| Capture-layout standalone GPU crate, each integration | 1 passed |
| Full CUDA default-parallel supplemental run | 111 passed, 0 ignored |
| Full Python discovery, each integration and supplemental | 69 passed, no reported skips; includes the 38 facade regressions |
| Preserved public API contracts | 17 passed |
| Architecture tests | 8 passed |
| CPU graph, sequence, KAN, convolution training examples | exit 0 |
| `git diff --check` | exit 0 |

The ignored tests are `bmm_perf_old_vs_dispatch`, `conv_timing_im2col_vs_naive`, and `bmm_perf_naive_vs_fastcpu`; they remain ignored because this is not performance benchmarking. No test was modified or newly skipped. Required CUDA runtime bins (NVRTC/cuBLAS) were explicitly put on PATH; `FERRO_REQUIRE_CUDA=1`. All Cargo invocations use `-j2`. The two complete integrations set `RUST_TEST_THREADS=1`, while standalone binding and supplemental CUDA use the default parallel harness. Embedded-Python binding tests set the queried base `PYTHONHOME`; wheel builds do not.

Canonical log reconciliation records **1906 Rust test executions and 238 unittest executions**, not unique tests. The final parent summary per Cargo Running/Doc-tests section is counted; nested child summaries are retained separately. Wrapper console logs and copied integration logs are excluded. Standalone parity/fuzz/example assertions are not invented unittest totals. See `leaf-results-counts.json` for every command, exit, count and log SHA256.

Training MSE first -> final (`final/training-examples.log`): graph `2.9866669178009033 -> 5.41345652891323e-05`; sequence `0.14829802513122559 -> 2.220446049250313e-14`; KAN `0.1933799684047699 -> 2.8424021820683265e-06`; convolution `0.972000002861023 -> 1.276771577352065e-09`. These fixtures are not model-family generalization or restart certification.

## Provenance and bounded review

- All 14 facade source files match their wheel entries and installed files byte-for-byte. Native wheel/installed bytes also match. Installed package/native hashes are stable before/after wheel integration and supplemental tests; imports resolve to `.venv/Lib/site-packages/ferro`.
- Wheel SHA256: `bbe16d6b5a354e32356d6fd898fdc94e6e05b4bedc0ea390b795784ddf4f47ae`.
- Native SHA256: `8db7f4c8fa6fb60e9b38a987a756d54a2f5138815b20e0409c1ec873ab431358`.
- `tested-source-provenance.json` identifies 418 current crate/build/config/test inputs and invoked scripts, matching before/after/current. Inclusion is identity, not proof every function executed. The broader conservative before/after manifest also matches. Historical source snapshots and generated result JSON are explicitly listed separately, not silently hidden.
- `collect.py` was added after tests solely for reconciliation and inventory, then executed against the preserved logs; it was not an input to the tested production build. Fresh report JSON/Markdown additions do not falsify production stability. The previous final-review source-alarm exit remains historical evidence and was not rewritten.
- Read the latest second-review STATUS and Python architecture report; sanity-reviewed the actual `nn.rs`/`lib.rs` diff. The default scalar validator rejects both supplied state and omission when a custom module declares state; stateless defaults remain valid. Public recurrent/basis exports are present. This is a focused independent check, not a new exhaustive review of the entire foundation diff.
- Added-line/untracked-source security heuristics found only `Module.eval()` definitions/calls, not dynamic evaluation. No credential, shell execution or pickle match occurred. This is not an exhaustive security audit. Existing compiler warnings and Git CRLF notices remain; no warning-clean/clippy claim.

## Single-PR handoff

`PR_BODY.md` is explicitly DRAFT. `proposed-files.json` enumerates an exact proposed allowlist; `inventory.json` accounts for every Git-visible untracked candidate plus changed tracked files. The proposal includes production/config/tests, the current scope/contract documents, executable acceptance scripts and selected fresh manifests. Archived raw logs, wheels, binaries, caches, unrelated prior-wave scripts/bench probes are excluded from staging, not deleted. Raw proof logs remain local and are referenced by hash; the parent must choose an evidence attachment policy before publication. Git-ignored files are not staging candidates and are not an exhaustive disk/secrets inventory. Self-referential inventory hashes are null deliberately.

No `git add` was run. Parent must inspect exact paths, review sensitive content and decide what to stage; do not broadly add the working tree. The existing roadmap is historical planning and must not be read as this bundle's latest implementation ledger. AI-assisted implementation and verification must be disclosed in the eventual commit/PR. Verification is finished; opening a draft is a parent decision, while merge remains blocked.
