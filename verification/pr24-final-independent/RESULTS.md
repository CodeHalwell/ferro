# PR24 final independent audit

Verdict: **passed=true**. Blockers: none found in the reviewed quarantine,
empty-DLPack and verification-tooling fixes. This is a pre-push review, not a
merge, performance measurement, or proof of recovery from native CUDA faults.

Base HEAD remained `2adcc52b8493537ec3adff58a173bb7bc99d1297`; index remained
empty. Before/after SHA-256 inventories cover 422 tracked/nonignored untracked
crate, source and configuration files, with no changes during verification.
Independent evidence scripts are excluded from that inventory. Existing archived
evidence was not overwritten. Production, test and benchmark source was not edited.
The release extension was rebuilt and installed into the existing binding venv.
`installed-binding.json` records its final path and SHA-256.

## Audit findings

- Native: CaptureSession, GraphOwner and Flight share the same sticky state.
  Failed inner cleanup cannot be erased by successful outer fences. Published
  graph Drop uses its original adapter and drops native ownership before address
  owners. Remaining raw slots, pending stream, context, full resource tuple and
  exclusion slot survive quarantine. Bind failure stops cleanup; ambiguous exec
  or graph destruction is not retried, and successfully destroyed slots are
  cleared. Unknown/still-active capture cleanup retains ownership/exclusion.
- DLPack: zero logical size bypasses reachable stride arithmetic and pointer
  operations, not descriptor validation. CUDA empties remain CUDA resident;
  nonempty null data is rejected. Capsule rename/deleter transaction remains
  intact and success/rejection cleanup is covered. Incoming nonempty CUDA stream
  handoff remains unchanged and is not claimed fixed by this patch.
- Tooling: output paths are explicit/fresh, archived evidence is not implicitly
  rewritten, current source/binding metadata is explicitly not historical build
  attestation, and CPU fixture tests are not represented as GPU evidence.
- Security: scanned tracked source additions plus all 25 nonignored untracked
  source files (11,140 lines including archive source copies). Three syntactic
  eval/exec matches were reviewed: a constant test-only PyO3 lambda and two
  local-repository AST fixture extractions. No untrusted-input execution, secrets,
  injection or deserialization concern found. Raw matches: security-scan.json.

Quarantine intentionally retains resources for process lifetime, may grow without
bound, and retains legacy exclusion; restart is reclamation. Holding exclusion
while fencing may block preparation. Real-resource injection tests simulate
adapter failures/skipped calls, not driver failure or fatal-context recovery.

## Executed commands and environment

Working directory: repository root. `python` was the active Hermes Python 3.11;
`PY` below denotes `crates/ferro-py/.venv/Scripts/python.exe`.
The evidence driver prepended LOCALAPPDATA/Temp/cuda-rt/nvidia/{cuda_nvrtc,cublas}/bin
and venv Scripts to PATH, set FERRO_REQUIRE_CUDA=1, VIRTUAL_ENV and PYO3_PYTHON,
and removed RUST_TEST_THREADS. Binding Rust tests additionally set PYTHONHOME to
the binding interpreter's queried sys.base_prefix. Maturin did not use PYTHONHOME.
The nested 13-check integration deliberately sets RUST_TEST_THREADS=1; separate
binding Rust and full CUDA verification use the default parallel harness.

```bash
python verification/dlpack-raw-review/run.py --help
python verification/pr24-final-independent/audit_run.py
python verification/pr24-final-independent/security_scan.py
```

The evidence driver executed:

```bash
python verification/dlpack-raw-review/run.py all --output-dir verification/pr24-final-independent/raw-all
"$PY" -m unittest discover -s crates/ferro-py/tests -v
cargo test -j2 -p ferro-cuda
cargo test -j2 -p ferro-cuda --lib static_graph::native -- --nocapture
python -m unittest discover -s verification/pr24-tooling-review -v
python -m unittest discover -s verification -p test_run_integration.py -v
git diff --check
```

Raw `all` successfully executed these five phases; exact absolute commands are
in `raw-all/all-results.json` and each phase log:

```bash
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib -- --nocapture
"$PY" -m maturin develop --release -j2 --manifest-path crates/ferro-py/Cargo.toml
"$PY" crates/ferro-py/tests/test_eager_dlpack.py -v
"$PY" crates/ferro-py/tests/test_static_snapshot_dlpack.py -v
python verification/static-perf-wave/run_checks.py --output-dir verification/pr24-final-independent/raw-all/integration/integration-checks
```

## Actual results

All commands exited zero. Updated CLI orchestration needed no repair.

| Check | Result |
| --- | --- |
| Raw phases | 5/5 passed; release build installed ferro-0.0.1 |
| Standalone binding Rust, default parallel | 15 passed, 0 failed/ignored |
| Eager / snapshot focused Python | 5 / 1 passed |
| Full public Python discovery | 31 passed, no skips; repeated within integration |
| Full CUDA default parallel | 68 library tests; 111 outer-harness passes including integrations/doc; 0 failed/ignored |
| CUDA nested child | 1 additional pass summary; raw pass-line sum 112, not 112 unique tests |
| Native filtered suite | 26 passed, 42 filtered; overlaps full CUDA, not additive |
| Combined integration | 13/13 commands passed |
| Core within integration | 633 outer-harness passes, 2 ignored; 9 nested pass summaries, raw sum 642 |
| CPU/tokenizer | 20 passed, 1 ignored |
| Capture-layout GPU | 1 passed |
| Tooling CPU fixtures | 8 passed |
| Integration-runner CPU fixtures | 3 passed |
| Fuzz | 200 trials, seed 0; all 19 ops within existing G4 ULP gate |
| Diff whitespace check | exit 0; Git line-ending notices only |

No global unique-test total is claimed: focused/repeated suites overlap and nested
child harnesses produce duplicate summaries. Parsed totals are in harness-counts.json
and count-totals.json. Existing compiler unused/dead-code warnings remain.
Integration logs confirm compiled fusion, fuse/general regression, Torch op parity
and safetensors parity success. Exact 13 commands and log paths are in
raw-all/integration/integration-results.json.

Created only this fresh verification directory and its evidence/helpers. No
staging, commit, push, branch change, merge or benchmark was performed.
