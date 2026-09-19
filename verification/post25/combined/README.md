# Final post25 combined verification

Final runner exit 0; collector exit 0. No production changes, commits, pushes or PR actions by this verifier. The first launch failed a verifier ROOT prerequisite assertion before any build/test ran; the owned runner path was corrected and its empty output directory removed before the successful fresh run. This is not a hidden product-test retry.

## Build and provenance

- Noneditable release wheel built with maturin `--release -j2`, then force-installed without dependencies into crates/ferro-py/.venv. Build log explicitly compiles core, fastcpu, CUDA and Python crates.
- Wheel: `wheels/ferro-0.0.1-cp311-abi3-win_amd64.whl`.
- Wheel SHA256: `33103ea62bc121bd6ddce0beb9277335e284d1818e9bd8cb4343cabe5527fdbb`.
- Installed native SHA256: `116f707685d04cf09ed3dc9893291598293107cb855e24f9d120ba3f8af138a1`.
- All 15 Python facade source/wheel/installed hashes match; native wheel/installed hashes match. Includes checkpoint and new Adam wrapper facade.
- 438 frozen production/config/test/harness inputs remained unchanged; installed Python/native before/after hashes unchanged. Conservative repository manifest also unchanged during runner. CUDA .cu source included. Exact manifests and packaging checks are JSON alongside this report.
- Supplemental public Python probe followed primary runner; collector rechecked frozen sources and installed hashes unchanged. Probe and collector are additional verification artifacts, not changed production sources.

## Test results

14 outer commands all exit 0; nested integration has 13/13 leaves exit 0. Excluding its outer aggregator yields 26 canonical leaf logs, plus the separately logged public-segment supplement (exit 0). Collector and final outer provenance exit 0. Aggregation selects final Cargo parent summary in each Running/Doc-tests section and preserves nested child summaries separately. 1082 Rust passed executions and 6 ignored executions across repeated suites are NOT unique-test counts.

| Suite | Passed | Ignored |
|---|---:|---:|
| core | 762 | 2 |
| fastcpu + tokenizer | 35 | 2 |
| CUDA serial integration | 118 | 0 |
| CUDA default parallel | 118 | 0 |
| original standalone Rust binding default parallel | 15 | 0 |
| fastcpu release serial | 27 | 2 |
| explicit CUDA segments | 6 | 0 |
| capture/layout GPU | 1 | 0 |
| Python discovery | 86 | 0 |
| contracts | 17 | 0 |
| architecture | 8 | 0 |
| explicit training checkpoints | 13 | 0 |
| explicit public copy | 3 | 0 |
| explicit convolution | 1 | 0 |

All original Python regression, operations parity, compiled fusion, fusion, safetensors and 200-trial seed-0 fuzz commands exited 0, as did CUDA all-target check and diff checks. Fuzz retains its existing accumulation-order percentile exceptions; this is not an all-output exactness claim. Separate CPU probe requires bit-exact parity for every output.

CUDA runtime NVRTC/cuBLAS directories are prepended per process, FERRO_REQUIRE_CUDA=1 throughout. Binding unit tests use the actual interpreter base-prefix PYTHONHOME; maturin does not. Full CUDA and original binding tests use the default parallel harness.

## Runnable public Python segments

`../public_segments.py`, run against the installed wheel, proves public `ferro.graph.segment_sum` GPU forward/backward, GPU softmax and rejection of NaN/+Inf/-Inf. The wrappers already route into core without a CPU guard; no wrapper edit was necessary. The graph facade's old CPU-only docstring is stale, not an execution barrier.

`segments-explicit.log` proves six underlying CUDA tests. Prepared forward/adjoint execution reports no repeated index uploads and no feature transfers, but softmax performs one synchronized **4-byte integer control download** and one control allocation for finite-input validation. This is NOT zero total transfers. Prepared topology creation uploads indices once (64 bytes in fixture). Arbitrary strided backward seeds still have a shared-core detach-copy host-transfer limitation; the resident strided-feature test explicitly materializes a contiguous seed first.

## Fresh serial CPU speed confirmation

After all this runner's builds and GPU suites finished, `../cpu/run.py timing-combined-final` built the live release probe and ran forward then reversed comparator order. Each shape/route has four warmups, 15 raw samples and pre-timing bit-exact all-output parity. Allocation included; destruction outside timing. Same executable for current independent scalar, safe tiled direct, registry and Tensor+to_vec. The scalar reference is not byte-for-byte historical production. Timing artifacts are in `../cpu/timing-combined-final`; all raw samples, medians and ratios also in collected.json.

| Shape [B,M,K,N] | Order | Scalar ms | Direct ms | Registry ms | Tensor+to_vec ms |
|---|---|---:|---:|---:|---:|
| [16,128,128,128] | forward | 14.9582 | 1.2785 | 1.3804 | 1.8285 |
| [16,128,128,128] | reverse | 14.9329 | 1.2967 | 1.3777 | 1.8130 |
| [4,64,256,256] | forward | 6.8113 | 0.5629 | 0.6281 | 0.9246 |
| [4,64,256,256] | reverse | 6.8339 | 0.5617 | 0.6258 | 0.9827 |

Conservative same-run scalar ratio across these two orders: wide first shape direct >=11.51x, registry >=10.83x; second shape direct >=12.10x, registry >=10.84x. No tiny/skinny universal speed claim. Tensor includes output materialization and is not kernel-only. Reviewer CPU tests may have overlapped; no isolated-host claim or end-to-end model-training claim.

## Preserved open incidents

The historical CPU numerical incident remains unexplained. Additionally `../cpu/isolated-green2/command0.log` records Windows `0xc000001d STATUS_ILLEGAL_INSTRUCTION` from `verification/post25/cpu/target/debug/deps/ferro_fastcpu-9a65ade0ef6b3d3f.exe --nocapture` after multiple tests passed. Its cause remains UNKNOWN; passing current tests and serial/repeated historical retries do not diagnose or fix it. Do not claim the branch is incident-free.

Existing unused-variable/dead-code warnings remain. No code or tolerance was edited to make tests pass.
