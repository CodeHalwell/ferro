# Foundation publication: DRAFT, merge BLOCKED

The canonical scope and limitations are [PR_BODY.md](PR_BODY.md). The historical **343812 ULP fastcpu BMM incident remains OPEN, root cause unknown**. No natural recurrence, causal production fix or merge-readiness approval is asserted. The p=90 omission probe and deliberately injected output/input corruption tests validate a counterfactual and diagnostic reporting, not a production bug reproduction.

## Review packaging and source identity

Start with core foundation/state/higher-order/CPU architecture code and its integrated tests; shared tensor/module/autograd files stay together so commits do not depend on speculative partial patches. Review fastcpu's test-only diagnostics and scalar oracle next, then the native mixed Python package/tests/public examples, then evidence and contracts. This is a single draft PR against main, never an authorization to merge.

The original `proposed-files.json` contains exactly 100 proposed paths. All 98 non-null SHA256 entries matched before packaging; its two self-referential entries intentionally have null hashes. It and `inventory.json` are retained as the original verifier's proposal, not silently rewritten into a claim that their historical hashes describe post-packaging prose. `publication-files.json` is the exact final publication allowlist, recording original and final working-tree hashes and the Git-clean blob identities; its own hash is null to avoid self-reference. Git normalizes CRLF when appropriate, so raw disk SHA256 and Git blob identity are different checks.

No production behavior changed after the final frozen verification. The staged whitespace check exposed one trailing blank line in `ferro/nn/functional.py`, removed solely to pass `git diff --cached --check`. Its parsed AST is unchanged; public contracts and architecture tests were rerun with this exact source module loaded over the installed native wheel. Historical 14-file source/wheel byte equality remains evidence for the original tested bytes, not a rebuilt post-whitespace wheel. All other crate/config/test source bytes are unchanged. Other packaging changes are limited to publication prose, precedence/local-evidence notices in historical documents, and the optional `--output-dir` argument in `verify.py`. The latter prevents a fresh rerun from colliding with the committed historical `final/` reports; syntax and help are checked separately, not misrepresented as a second fresh execution of its entire GPU orchestration. The extra `omission_probe.rs` is a compact, reviewed, dependency-free forensic reproducer already exercised during the independent investigation. No additional model features or tolerance changes are introduced.

## Evidence policy

The committed reports/manifests state what the local execution observed. All 41 canonical leaf commands exited zero and all 41 referenced raw log SHA256 values were rechecked during packaging. The reported totals (1906 Rust and 238 unittest executions) count executions, not unique tests. Raw logs, binary/wheel artifacts, cache files, diagnostic JSONL and earlier-wave archives remain local, unmodified and unpublished. Local paths/usernames in manifests are provenance, not downloadable artifacts; no credentials or environment dumps are intended for publication. A heuristic content scan found only ordinary module `eval` names, not dynamic execution or credential matches; this is not an exhaustive security audit.

References in historical reports to omitted `forensics.py`, `analyze.py`, older `run.py`/`verify.py`, result JSON, source archives and raw logs denote local-only provenance. Do not execute those historical reproduction blocks expecting omitted files in a fresh clone. `collect.py` is an archival reconciliation/inventory utility requiring the preserved local raw-log tree under `final/`; it is not a standalone fresh-clone test or a publication staging tool. Running it overwrites proposal/report JSON, so do not run it on the publication manifests merely to validate them. The executable current test entrypoints below are packaged with their source dependencies.

## Portable core and focused reproductions

From repository root, using the supported Rust toolchain:

```text
cargo test -j2 -p ferro-core
cargo test -j2 -p ferro-fastcpu -p ferro-tokenizer
cargo test -j2 -p ferro-fastcpu --test bmm_parity -- --nocapture
cargo test -j2 -p ferro-core --test second_review_state
rustc --edition=2021 verification/foundation-wave/fastcpu-second-review/omission_probe.rs -o target/foundation-omission-probe.exe
target/foundation-omission-probe.exe
```

The last two commands reconstruct the seeded scalar counterfactual; they do not invoke fastcpu. Core state, higher-order and architecture tests are part of ordinary Cargo discovery. The fastcpu integration tests include `tests/support/bmm_diagnostics.rs`; no omitted verification helper is required. Their injected diagnostics create clearly labelled local reports and are never evidence of a natural failure.

## Python and full Windows verification

Prerequisites: project venv at `crates/ferro-py/.venv`, a working Rust/MSVC toolchain, maturin and the existing test dependencies (including torch/numpy/safetensors), and the real CUDA setup used by the integration runners. The full verifier explicitly checks `%LOCALAPPDATA%/Temp/cuda-rt/nvidia/cuda_nvrtc/bin` and `cublas/bin`; missing prerequisites are failures, not skip-based success. It builds/installs a noneditable wheel in that project venv. Embedded Python binding tests set the queried base PYTHONHOME; wheel builds do not. CPU wrapper coverage is not CUDA architecture residency.

```text
python verification/foundation-wave/pr-readiness/verify.py --output-dir verification/foundation-wave/pr-readiness/rerun-01
crates/ferro-py/.venv/Scripts/python.exe -m unittest discover -s crates/ferro-py/tests -v
crates/ferro-py/.venv/Scripts/python.exe verification/python-api-contract/contract.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/test_architecture.py
crates/ferro-py/.venv/Scripts/python.exe verification/python-architecture-api/training_examples.py
```

Choose a nonexistent output directory inside the repository on every new full attempt. The original unchanged DLPack and static integration runners are already tracked. Historical original literal commands and exact exits remain in `leaf-results-counts.json`; the packaging-only output option does not retroactively alter those commands or source manifests.

## Packaging smoke

```text
crates/ferro-py/.venv/Scripts/python.exe verification/foundation-wave/pr-readiness/packaging_smoke.py
```

This packages all 14 authored facade files into a temporary directory with a copy of the verified installed native extension, checks exact import locations, and runs discovery (69 passed), contracts (17 passed) and architecture tests (8 passed). Source/native hashes stayed stable. One expected saved-version panic diagnostic was caught by its regression test and the suite passed. This is not a new native build or a replacement for the frozen CUDA validation. The runner intentionally requires the recorded native SHA256 rather than silently accepting a stale extension.

## Acceptance still outstanding

Resolve the fastcpu incident causally and obtain human review before reconsidering merge. Full Python training-state exposure, resident CUDA architecture kernels, complete arbitrary-loop checkpoint state and remaining roadmap model-family proofs are follow-ups, not completed foundations. Custom restore protocols and CPU-only transactional limitations remain as stated in the PR body. Existing compiler warnings remain; no clippy/sanitizer/Miri or performance certification is implied.

AI assistance: implementation, scoped independent review, verification and packaging used AI agents. Publication is for review only; no review-bot requests or merge operations are part of this handoff.
