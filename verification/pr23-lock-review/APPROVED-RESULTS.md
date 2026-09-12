# PR23 approved combined review verification

Completed the approved two-block PairGuard edit in crates/ferro-core/src/inplace.rs. Device binary/axpy and same-device copy now acquire deduplicated address-ordered guards, matching the existing optimizer sweep. Backend calls, public autograd/alias gates, self-copy behavior and version bumps are unchanged. Sibling snapshot, registry and benchmark source edits were preserved. No staging, commit or push.

## Executed checks

All logs below are in this directory unless qualified.

- RED: `cargo test -j2 -p ferro-core --lib device_mutation_replay_lock_order -- --nocapture`, exit 101, approved-red.log. With the optimizer fixes already present, copy-desc, binary-desc and axpy each reached the forced AB-BA cycle and were killed/reaped by the bounded harness; optimizer cases passed. Earlier six-case RED remains in red.log.
- GREEN: same command, exit 0, approved-green.log. All nine child scenarios completed.
- Full core: `cargo test -j2 -p ferro-core`, exit 0, approved-core.log.
- Real required CUDA: `cargo test -j2 -p ferro-cuda --test concurrent_copy_replay -- --nocapture --test-threads=1`, exit 0, approved-cuda-green.log. Child completed in 0.97s (parent 1.04s), no timeout.
- Final combined release rebuild: venv Python `-m maturin develop --release -j2 --manifest-path crates/ferro-py/Cargo.toml`, exit 0, approved-release-build.log; release compile 15.86s, editable ferro-0.0.1 installed.
- Venv Python `verification/model-cuda-graphs/continuation/run_checks.py`, exit 0, approved-integration.log: all 13 serialized commands passed. Per-command logs and integration-results.json live in that continuation directory. Python discovery includes 11 passing tests, including direct snapshot DLPack consumption on default/nondefault streams.
- Venv Python `-m unittest discover -s verification/model-cuda-graphs/continuation -p test_verify_results.py -v`: 7 passed, approved-verifier-tests.log.
- Venv Python `-m unittest discover -s verification/layernorm-numerics -p test_validate.py -v`: 3 passed, approved-epsilon-tests.log.

Runtime: FERRO_REQUIRE_CUDA=1, RUST_TEST_THREADS=1, CARGO_BUILD_JOBS=2. PATH prefixed LOCALAPPDATA/Temp/cuda-rt/nvidia/cuda_nvrtc/bin and nvidia/cublas/bin. No cargo/rustc/maturin/test/benchmark processes remained before benchmark collection (process inspection showed only Hermes Python services/kernels and the inspection process).

## Fresh benchmarks

Only after builds/tests completed, ran continuation/benchmark.py with `--json verification/model-cuda-graphs/continuation/review-run1.json`, then `--reverse --json verification/model-cuda-graphs/continuation/review-run2.json`. Both exited 0. Their matching .log files are retained. The new reports include declared tolerances and all four elementwise freshness gate ratios; historical benchmark JSON was not overwritten.

Validated with continuation/verify_results.py `--runs verification/model-cuda-graphs/continuation/review-run1.json verification/model-cuda-graphs/continuation/review-run2.json --summary verification/model-cuda-graphs/continuation/review-summary.json --integration-dir verification/model-cuda-graphs/continuation`, exit 0. Each run has 15 passing and 6 explicitly blocked rows. Both requested Torch compiler modes were actually attempted and blocked by missing Triton; no compiled Torch comparison is claimed.

RTX 3090, Torch 2.6.0+cu124, CUDA 12.4, f32, TF32 disabled; tokens 128, width 256, heads 8; 10 warmups and 40 fenced samples. Parity rtol 0.0002, atol 0.00002.

Median microseconds, normal / reversed comparator order:

| Model | Torch eager | Ferro eager | Ferro compiled | Static private output | Static + synchronous snapshot |
|---|---:|---:|---:|---:|---:|
| MLP | 72.20 / 64.20 | 52.20 / 52.05 | 48.75 / 48.90 | 81.60 / 40.75 | 58.10 / 43.20 |
| Residual MLP | 58.60 / 77.75 | 102.75 / 51.70 | 95.55 / 49.20 | 41.90 / 41.90 | 44.50 / 44.40 |
| Transformer | 284.10 / 661.35 | 294.15 / 294.85 | 280.10 / 280.15 | 344.85 / 332.80 | 336.60 / 338.50 |

Snapshot timings now include its synchronous fence. Static private-output timings exclude independent copy-out. Results show substantial order/run sensitivity, including changing rankings; no stable overall speedup or old speed claim is asserted. Transformer static/snapshot paths are slower than Ferro compiled in both runs.

Verifier Rust pass-line sums are core 640, CPU/tokenizer 20, CUDA 76, capture-layout GPU 1. These are log sums, not deduplicated test counts: child-process harness summaries are included.

Existing unused-variable/dead-code compiler warnings remain; the unused PairGuard warning is gone. Targeted patch tooling also reported pre-existing rustfmt disagreements; no broad reformat was applied. Serialized integration and git diff --check passed. Default-parallel CUDA library registry races documented by the sibling worker were outside this verification's single-threaded scope.
