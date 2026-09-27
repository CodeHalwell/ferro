# PR29 bounded review-fix publication evidence

## Merge risk remains OPEN

**The earlier fresh default-parallel FastCPU native access violation remains unresolved. This publication is NOT a crash fix, merge-readiness certificate, or a claim of flake-free execution.** The current combined run passed once on a new build target; that does not explain or erase the failed earlier run. Publication is limited to the requested review fixes and test-isolation correction. No merge, draft/ready transition, or auto-merge is authorized by this report.

The earlier `verification/pr29-review-fixes/integration/ferro-fastcpu.log` records exit 101 with native `0xc0000005, STATUS_ACCESS_VIOLATION`, without a final unit-test summary. Eleven completion lines out of twenty announced tests do not identify a faulting test. Subsequent serial and bounded parallel passes on the preserved executable established neither cause nor a reliable reproducer. No faulting instruction, access address, thread, or stack was captured. No causal connection between the test-isolation defect and the native exception was established.

Incident EXE/PDB remain untouched in `target-pr29-review-integration/debug/deps/`, stem `ferro_fastcpu-9a65ade0ef6b3d3f`. Their SHA256 values were verified before and after this run:

```text
e1b69975c025ae812cd5231bc2c3e3af77c08e19446ddecd2a3ee46275eec15c  incident EXE
9d77a3f00c5728c7934d3d1bdb4bcb999ae3c6766bfb0761e9f5a231c4a4d1e2  incident PDB
```

All raw logs, JSON receipts, wheels, binaries, local runners and prior reports referenced with code paths below are **local provenance, not shipped attachments**. This curated report is shipped. None of those excluded artifacts is required to import or compile the changed source/tests.

## Changes and review disposition

Review base: `dbc306e5a82595f50fa755ee3dd6915e745ee499`, PR [29](https://github.com/CodeHalwell/ferro/pull/29). The isolated worktree is `C:/Users/DanielHalwell/PythonProjects/ferro-pr-stabilisation`; the original checkout was not modified.

### Concurrent CUDA installation (4116579104)

The race is valid, although replacement already existed before this PR. Install now holds the typed-selection mutex across both publications. Typed readers take that gate and validate exact Arc identity against the core registry; direct replacement fails closed instead of reporting a fence on an already-stale selection. Synchronization uses a retained, validated snapshot outside the gate. This is a last-selected-stream snapshot fence, not an all-device fence, an old-allocation fence, or a lease preventing subsequent replacement/submission.

Independent review found an arbitrary-destructor reentrancy hazard in the initial correction. Core's additive `dispatch::exchange_backend` now returns the displaced owner after releasing its registry lock. CUDA defers destruction of retired selection/registry owners and reader snapshots until after releasing its gate, including declaration-order cleanup on unwind. Public register_backend retains its original signature and drops exchanged owners after unlock. No unsafe downcast, Backend-trait change, external core dependency or weakening of allocation-owned downloads/resident ownership was introduced.

The original deterministic paused-reader/two-writer/direct-replacement sensitivity run failed against the instrumented old code (2 passed, 3 failed). Three additional reentrant-destructor tests failed by bounded child timeouts before correction (0 passed, 3 failed), then completed successfully. The final independent deadlock review closed that introduced blocker for the exact core/CUDA hashes below. It did NOT certify the historical crash or entire publication.

The three destructor regressions exercise direct registration, replacement during install, and a mismatching reader snapshot forced to be sole owner. Unwind panic injection, destructor callbacks that themselves write/install, and final destruction of the concrete retired CUDA selection remain source-audited rather than explicitly fault-injected. The child-output pipe/deadline harness has not been stress-tested with large diagnostic output.

### Python setup and false CPU-class premise (4116579122)

At the reviewed head, the flagged method was already on PreparedSegmentsCUDA, not PreparedSegmentsCPU. Nothing was moved. Actual discovery is four CPU methods and six CUDA methods; both CUDA-only methods are excluded from CPU discovery. Prior controlled CPU-only discovery ran four real CPU bodies and skipped CUDA class setup in both normal and optimized interpreters, using the installed extension with initialization mocked unavailable, not a CPU-only wheel.

A separate real bug was corrected: `assert fr.cuda_init(0)` suppressed initialization entirely under `-O`. Setup now calls it unconditionally and explicitly rejects false returns. Required-CUDA hard failure and optional skip behavior are preserved for false results and raised exceptions. New stdlib setup tests import real source against dependency stubs; their pre-fix optimized sensitivity run failed. They test initialization, discovery and skip/failure policy, not GPU numerics. Current real installed-wheel CUDA prepared/registry runs passed in both modes.

### Shipped documentation links (4116579135)

[CURRENT_STATUS](../../../docs/CURRENT_STATUS.md) now links shipped [STABILISATION_PUBLICATION](../../../docs/STABILISATION_PUBLICATION.md) and the [typed-parity follow-up](../../pr29-typed-parity/README.md). The historical 454-input/114-test provenance, reconciled 470-input publication, and later 119-test follow-up are distinguished instead of relabeled as current verification. All three links in the edited roadmap/stabilisation passage, including its heading anchor, passed the tracked-target checker.

### FastCPU test-only registry isolation

All four existing registry-using unit tests share a poison-tolerant guard, retained through assertions and local destruction. The guard restores the saved CPU backend Arc and the audited process-start naive_matmul callback before unlocking, also on unwind. MATMUL has no getter; this is baseline restoration, not a general callback snapshot API. A deterministic artificial contamination regression first failed with mutex-only cleanup, then passed with restoration. The prior test-isolation default and serial suites each passed 28 tests with 2 ignored. These are repeated executions, not extra unique coverage or a crash reproducer.

The FastCPU diff is entirely cfg(test)/test code and corrected stale comments. No production arithmetic, benchmark execution or numerical assertion was changed. Other direct-kernel tests remain parallel. The current files match every source hash in the prior test-isolation audit.

## Current frozen rebuilt-artifact run

Entry: `python verification/pr29-review-fixes/final-publication/run.py`, exit 0, one combined execution with no failed-command retries. Runner and raw evidence remain local. It asserted the NEW target `target-pr29-final-publication` did not exist before building, and never rebuilt the incident target. Every command below exited 0.

The input freeze contains **670 files**: tracked existing inputs plus untracked source/config/tests and local verification Python helpers, including the three new regression files and CUDA source inputs. Before/after manifests agree with no additions, removals or byte changes. This is an input identity inventory, not a count of executed files. This curated report was authored after the run and is not part of that freeze. Installed Python/native payloads also remained identical throughout testing.

The release noneditable wheel was freshly built and force-installed without dependencies into this worktree's `.venv`. Imports resolve under `.venv/Lib/site-packages`, not editable sources. Each Python payload agrees between source, wheel and installed package. Native build output, wheel native entry and imported installed native extension agree.

```text
55c89301493a030e617177d795f57d3fc5370a62f897887f2b90c2dc5ca2ab64  ferro-0.0.1-cp311-abi3-win_amd64.whl
a5c3de51d0c0a838a0f666c6b15605f01636a8bb668bc6853f061d6d507ba660  release/_native.dll = wheel ferro/_native.pyd = installed ferro/_native.pyd
```

Environment: Windows x86_64, RTX 3090 ordinal 0; required CUDA via FERRO_REQUIRE_CUDA=1; existing NVRTC/cuBLAS DLL paths prepended from `$LOCALAPPDATA/Temp/cuda-rt/nvidia/{cuda_nvrtc,cublas}/bin`. All Cargo commands use `-j2` (build concurrency, NOT serial libtest). RUST_TEST_THREADS, PYTHONPATH, PYTHONHOME and CUDA_VISIBLE_DEVICES were unset, except native binding tests set PYTHONHOME to the venv interpreter's base prefix for embedded Python. VIRTUAL_ENV and PYO3_PYTHON point to the worktree venv, CARGO_TARGET_DIR to the new target. PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and RUST_BACKTRACE=1 were set.

| Suite | Current result |
| --- | --- |
| Core default targets | 811 passed, 2 ignored |
| CUDA default targets, required runtime | 134 passed, no ignored, including doctest |
| FastCPU default targets | 28 passed, 2 ignored |
| Tokenizer | 8 harness passes; real-GPT2 body unavailable |
| Native binding library | 15 passed |
| Full Python discovery | 124 passed, no skips |
| Prepared segments normal / -O | 10 passed each |
| Setup policy normal / -O | 5 passed each |
| CUDA registry copy normal / -O | 1 passed each |
| CUDA all-target compile, links, diff-check | passed |
| All six standalone correctness workflow invocations | passed |

Native counts use the last parent summary per Cargo Running/Doc-tests section so isolated child summaries are not double-counted. Targeted suites overlap discovery/full CUDA coverage. The current combined run is not another serial diagnostic. Expected/caught panic diagnostics from rejection tests remain in successful logs.

### Literal command inventory

`$ROOT` below abbreviates exactly `C:/Users/DanielHalwell/PythonProjects/ferro-pr-stabilisation`; commands ran with that cwd and the environment above. Absolute native Windows paths in raw argv receipts have equivalent backslash separators.

```text
nvidia-smi
$ROOT\.venv\Scripts\python.exe -m maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked -j2 --out $ROOT\verification\pr29-review-fixes\final-publication\wheels
$ROOT\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps $ROOT\verification\pr29-review-fixes\final-publication\wheels\ferro-0.0.1-cp311-abi3-win_amd64.whl
$ROOT\.venv\Scripts\python.exe -B -m unittest discover -s crates/ferro-py/tests -p test_prepared_segments.py -v
$ROOT\.venv\Scripts\python.exe -B -m unittest discover -s crates/ferro-py/tests -p test_prepared_segments_setup.py -v
$ROOT\.venv\Scripts\python.exe -B -m unittest discover -s crates/ferro-py/tests -p test_cuda_registry_copy.py -v
$ROOT\.venv\Scripts\python.exe -B -O -m unittest discover -s crates/ferro-py/tests -p test_prepared_segments.py -v
$ROOT\.venv\Scripts\python.exe -B -O -m unittest discover -s crates/ferro-py/tests -p test_prepared_segments_setup.py -v
$ROOT\.venv\Scripts\python.exe -B -O -m unittest discover -s crates/ferro-py/tests -p test_cuda_registry_copy.py -v
$ROOT\.venv\Scripts\python.exe -B -m unittest discover -s crates/ferro-py/tests -v
cargo test -j2 -p ferro-core --no-fail-fast
cargo test -j2 -p ferro-cuda --no-fail-fast
cargo test -j2 -p ferro-fastcpu --no-fail-fast
cargo test -j2 -p ferro-tokenizer --no-fail-fast
cargo check -j2 -p ferro-cuda --all-targets
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib
$ROOT\.venv\Scripts\python.exe examples/py_regression.py
$ROOT\.venv\Scripts\python.exe crates/ferro-py/examples/py_compiled_fusion_regression.py
$ROOT\.venv\Scripts\python.exe crates/ferro-py/examples/py_fuse_regression.py
$ROOT\.venv\Scripts\python.exe examples/ops_vs_torch.py
$ROOT\.venv\Scripts\python.exe examples/safetensors_vs_python.py
$ROOT\.venv\Scripts\python.exe examples/fuzz_vs_torch.py --trials 200 --seed 0
$ROOT\.venv\Scripts\python.exe verification/pr29-review-fixes/python-docs/check_links.py
git diff --check
```

### Preserved failures and limitations

Earlier CUDA runs recorded capture invalidation (`cuda/cuda-full.log`), an unconsumed simulated-cleanup injection (`cuda/cuda-isolated-install-tests.log`), and capture invalidation on unchanged base code (`cuda/baseline-parallel-2.log`). Later passes do not establish a common cause or flake freedom. These failures and the native access violation remain preserved, not replaced with green output.

Only one physical GPU was available; successful second-ordinal/cross-ordinal concurrency was not exercised. FERRO_GPT2_DIR is unset; tokenizer's real-vocab test returns after its availability gate. Four performance probes remain ignored (two core, two FastCPU). Windows binding regression skips the resource-based DLPack export leak check. This local environment is not hosted Linux CPU CI: Torch is 2.6.0+cu124 and NumPy 2.4.6.

The 200-trial seed-0 fuzzer passed existing percentile gates for 19 ops, not a uniform worst-case bound. Recorded maxima include GELU 874842661 ULP, BMM 3185852, mean_dim 304425, sum_dim 65536, matmul 14336 and log_softmax 120. Operator parity separately reports ALL OPS MATCH TORCH. Neither result resolves historical numerical incidents. Late CUDA copy/guard-completion error injection remains a coverage gap. Existing compiler warnings and repository-wide formatting differences were not mass-edited.

## Publication scope and identity

Only the following explicit source/test/doc allowlist is intended for the review-fix commit. No raw logs, JSON receipts, binaries, wheels, targets, runners or unrelated working-tree files are staged. All reports under pr29-review-fixes were read before this run; older reports remain historical local provenance. Git-clean blob IDs differ from raw Windows working-tree hashes where line-ending conversion applies. The three independent-review raw hashes were rechecked before testing and the full freeze remained stable.

- `crates/ferro-core/src/dispatch.rs`
- `crates/ferro-cuda/src/lib.rs`
- `crates/ferro-cuda/src/install_tests.rs`
- `crates/ferro-fastcpu/src/containment_tests.rs`
- `crates/ferro-fastcpu/src/elementwise.rs`
- `crates/ferro-fastcpu/src/lib.rs`
- `crates/ferro-fastcpu/src/registry_tests.rs`
- `crates/ferro-fastcpu/tests/bmm_perf.rs`
- `crates/ferro-py/tests/test_prepared_segments.py`
- `crates/ferro-py/tests/test_prepared_segments_setup.py`
- `docs/CURRENT_STATUS.md`
- `verification/pr29-review-fixes/final-publication/REPORT.md`

### Raw source SHA256

```text
60d9ef086c3446c94492eda48d886896272942ccdf6245406b85661f49a4122b  crates/ferro-core/src/dispatch.rs
1db68649ef428182fcf179cd47138fecc76e3f215d5d26d51ac59334cb1fdefa  crates/ferro-cuda/src/lib.rs
e5633c1f66e982e79e7431618ceff46439a45abc79e130f9e002c818144aa8a4  crates/ferro-cuda/src/install_tests.rs
e53869d69cef818ee6b3a5c33b094597d8bfa254cbc1786e38e4b4bb2bafa1c5  crates/ferro-fastcpu/src/containment_tests.rs
b75227ca0410d395b36a5fdfddf9ec5e7887a0d9c7c1287b912d6ded5423b1bc  crates/ferro-fastcpu/src/elementwise.rs
7e3760efc91988c4e87ae88706f10229dcb18921e6cfc605b8d1753ea59eafb3  crates/ferro-fastcpu/src/lib.rs
09e375de99efa51a2d0e22114596fa832cf9001c9daa0f5ea72875ad25d25ee6  crates/ferro-fastcpu/src/registry_tests.rs
ca16cd98c4260ae5614d6971791735c4e29ff2dfb4ebc3ee8ae79826f315e06b  crates/ferro-fastcpu/tests/bmm_perf.rs
35bb9e7459b9b6158a368e0d60b8a8fca8e11f4b77a421bbd487991f3bc88e03  crates/ferro-py/tests/test_prepared_segments.py
00c1b4ef407ce65a5b275628c23ac18accde378a9a966dea7cacde99b450ed89  crates/ferro-py/tests/test_prepared_segments_setup.py
1728c356ff9341c383a82c59aa3f3330f909c773e84c64decb5307420339e703  docs/CURRENT_STATUS.md
```

Hosted CI and reviewer completion are separate from this local result. Thread replies and the post-push PR comment identify the actual committed head; publication receipt records local/remote/API head equality. No pre-push green hosted claim is made. Review threads must not be resolved by claiming the historical native exception fixed.

AI assistance: Hermes Agent assisted implementation review, integration execution, evidence preparation and publication. This report records actual execution, not synthesized results.
