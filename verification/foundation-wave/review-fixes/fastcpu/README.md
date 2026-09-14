> Historical snapshot: current publication status and corrected parent/child test counts are in [PUBLICATION.md](/verification/foundation-wave/pr-readiness/PUBLICATION.md). The documented merge blocker is cleared by independently validated software BMM containment; historical root cause remains unresolved. Omitted logs/collectors are local provenance.

# Fastcpu intermittent BMM investigation: UNRESOLVED

> Historical investigation report; later bounded diagnostics are in the committed fastcpu tests. The cited analyze/run scripts and raw result/log archives are local-only. [PUBLICATION.md](../../pr-readiness/PUBLICATION.md) identifies the current portable test entrypoints and evidence policy.

No production fix is claimed. The original failure is a real numerical error, not rounding noise; bounded repeated CPU-only attempts did not reproduce it. Existing assertions and tolerances are unchanged. Changes are diagnostic/test coverage only, confined to ferro-fastcpu and this directory. No GPU, bindings, performance measurements, or commits.

## What the historical failure means

Source: `verification/foundation-wave/gpu/final-01/raw-all/integration/integration-checks/cpu-tokenizer.log`. The failing command was `cargo test -j2 -p ferro-fastcpu -p ferro-tokenizer`, with `RUST_TEST_THREADS=1`. The first mismatch was at flat index 231244 in shape `(16,128,128,128)`.

`analyze.py` independently reconstructs the original local LCG in Python, including each f32 rounding operation. It locates the element at batch 14, row 14, column 76; input seeds are 2556 and 3556. The new Rust scalar oracle uses its own fresh Vec, scalar row/column/dot loops, and no dispatch, pool, packing, or matmul backend reference.

- Rust independent oracle and Python sequential f32: **11.969769477844238**, bits **1094681645**. This is the original reference value.
- Python f64 dot: **11.969773428480593**.
- Historical fast BMM output: **11.641884803771973**, bits **1094337833**.
- Historical discrepancy: **343812 ULP**, absolute error versus f64 **0.32788862470862057**.
- A conservative gamma(256) ordinary-f32 dot forward-error bound from the actual sum of absolute products is **0.00208687152820479**. Ordinary summation ordering cannot explain the observed error for these reconstructed inputs. The separate round-once-per-multiply-add f64 simulation is 11.969768524169922, not the erroneous value.

Full failing row/column input bits and source comparisons are in `dot-product-and-source-analysis.json`. This reconstruction establishes the correct value for the specified seeded data; historical input/output memory was not dumped, so it does not prove that the original process's live buffers were intact.

## Ranked hypotheses examined

1. **Packing tails, reused scratch, and batch/thread row decomposition.** Each worker owns its packed A/B Vecs; output chunks are disjoint mutable slices. `g0` is captured by value, each batch segment is bounded by its batch end, and packing fills valid rows/columns plus padding. The first K block overwrites output rather than accumulating stale contents. Tests now cover original shape, MC/MR/NR tails, K=0/255/256/257/300, NaN-poisoned recycled outputs, concurrent independent callers, and explicit BMM worker splits 1/3/7/31/32/33. No mismatch reproduced. This is bounded evidence, not a proof that a latent implementation/compiler defect is impossible.
2. **Process-global registry/reference race.** CpuBackend's default BMM calls the swappable MATMUL pointer; the original test installs fastcpu before comparison, making the old reference share the packed implementation. The two unit tests which install that pointer install the same function, not competing arithmetic. Backend-map installation is a different registry and cannot redirect direct CpuBackend/FastCpuBackend calls here. MATMUL and backend-map accesses use RwLocks. The original failing harness was serial, further weakening the test-registration-race hypothesis. No synchronization-only 'fix' was applied without reproduction.
3. **RNG/data race.** Both input generators use local wrapping-u64 state and owned Vecs, not ferro's Rng or shared global seeds. BMM receives immutable input slices. Independent Rust and Python fixture reconstruction agrees. No shared-RNG source was found on this path.
4. **SIMD, uninitialized storage, or arithmetic-order assumptions.** The unsafe calls only enter runtime-checked AVX2/FMA target-feature wrappers; packing, micro-kernel arithmetic, and writeback use bounds-checked slices and initialized arrays, not raw pointer SIMD loads. Pool allocation is thread-local, fresh Vecs are zero-initialized, recycled debug Vecs are NaN-poisoned, and K=0 uses zeroed output. The new tests poison recycled buffers even in release. The actual historical error is too large for rounding on the reconstructed input.
5. **Unobserved runtime/compiler/hardware corruption.** Still possible, but no evidence identifies one of these as the cause. No historical memory dump, sanitizer trace, alternate-machine reproduction, or hardware diagnosis exists here. Do not relabel this as harmless or a hardware fault.

## Bounded execution evidence

Every Cargo invocation uses `-j 2`. `run-01` preserved the unchanged original unit test; `run-02` added only failure diagnostics to it. Each phase has commands, exits, raw logs, and source hashes.

- **100 exact original-test invocations:** 50 serial-harness and 50 default-harness, all passed. The filtered default-harness invocation runs only one test, so it alone is not a concurrency test.
- **12 full-suite invocations** across those two phases: six serial and six default-parallel, all passed.
- **600 dedicated original-fixture BMM repetitions** across debug/release/oracle/affinity probes, each compared bitwise to the independent scalar oracle. Additional original-fixture runs occur in full suites and are not included in this count.
- Integration oracle tests also exercise three seeds across six tail/K-block shapes, four simultaneous callers with four BMMs each, and the original failing batch slab with explicit matmul thread counts 1/3/7/32.
- An affinity probe attempted masks with 1/3/7 CPUs. **Rust still reported available_parallelism=32 in all three children. These are not evidence of serial BMM dispatch.** Accordingly, `batch_group_splits_match_scalar_oracle` was added to call the actual private BMM worker inline for one thread and explicitly split its row space for 3/7/31/32/33.
- Final source: `final-01/full-serial.log` and `full-parallel.log` each show **16 passed, 1 ignored** (13 unit + 3 integration; the existing ignored test is a benchmark and was not run). The explicit worker-split test also passed in release. `git diff --check` passed.

No failed execution was replaced with a fabricated success. The historical failure remains the only observed RED; no deterministic current RED and hence no production RED/GREEN fix were obtained.

## Provenance and scope

Archived fastcpu source, pool.rs, and dispatch.rs match HEAD when CRLF is normalized. Current production prefixes still match the archived sources; edits in lib.rs and elementwise.rs are under cfg(test). This establishes inherited source identity, **not that the failure was reproduced on a standalone HEAD build**. No reset or checkout was performed.

`run-01` detected another worker changing unrelated core `optim.rs`; `run-02` detected no source changes during execution. Final relevant source hashes stayed unchanged during final verification. Existing unrelated unused-variable warnings in core layer_norm/scatter_add remain. Rustfmt suggests broad reformatting of pre-existing style; no unrelated formatting was applied.

Changed/created code:

- `crates/ferro-fastcpu/src/elementwise.rs`: on mismatch, print independent scalar/f64 dot, seeds, coordinates, input row/column bits, and available parallelism before the original unchanged assertion.
- `crates/ferro-fastcpu/src/lib.rs`: new explicit worker-split scalar-oracle unit test; production unchanged.
- `crates/ferro-fastcpu/tests/bmm_parity.rs`: three independent-oracle tests covering original fixture, tails/recycling, and concurrent callers.
- This directory: runners, Python dot/source analysis, machine-readable counts/results, and logs.

## Reproduction

From repository root, use fresh output names (runners refuse to overwrite existing run directories):

```
python verification/foundation-wave/review-fixes/fastcpu/run.py fresh-run
python verification/foundation-wave/review-fixes/fastcpu/analyze.py
cargo test -j 2 -p ferro-fastcpu --lib tests::batch_group_splits_match_scalar_oracle -- --exact --nocapture
cargo test -j 2 -p ferro-fastcpu --test bmm_parity -- --nocapture
```

Set `FERRO_BMM_REPEATS` to increase original-fixture iterations. Keep the issue open. A future original-test failure now retains enough local input data to distinguish corrupt inputs, the default reference, and fast BMM output instead of losing the evidence on a passing retry.
