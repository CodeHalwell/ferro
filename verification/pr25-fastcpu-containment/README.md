> Historical snapshot: current publication status and corrected parent/child test counts are in [PUBLICATION.md](/verification/foundation-wave/pr-readiness/PUBLICATION.md). The documented merge blocker is cleared by independently validated software BMM containment; historical root cause remains unresolved. Omitted logs/collectors are local provenance.

# PR25 fastcpu BMM: production-path containment, root cause unresolved

## Decision and scope

The historical 343812-ULP error is real. Its cause is still **UNRESOLVED**. The p=90 omission is a counterfactual fingerprint, not a naturally reproduced mechanism. Prior passing repetitions and negative UB audits are not used to declare the incident fixed.

Production `matmul_batch` now computes every output independently using a fresh Vec, bounds-checked direct input slices, and ascending-k f32 multiply then add. It does not call the MATMUL registry, packed arithmetic, AVX target-feature wrappers, batch workers, shared scratch, or host pool. The historical packed BMM and its dispatch helpers are compiled only under `cfg(test)`. There is no environment/debug switch or production opt-in that restores that BMM path. This is replacement, not sampling, a checksum, a widened tolerance, or reexecution of the suspect kernel.

`FastCpuBackend::matmul_batch` already routes here. A second entrypoint audit found that **install-only callers** use CpuBackend's default BMM, which calls the MATMUL callback per slab. `install()` now registers a conservative one-batch callback too. Without this change the containment test deterministically returned the injected wrong output. This also slows ordinary MATMUL-registry operations; that collateral cost is deliberate and disclosed rather than silently leaving a bypass.

Explicit public `matmul` and `matmul_with_threads`, and FastCpuBackend's non-batched `matmul`, remain packed. They are outside this BMM containment claim. A caller manually registering a different MATMUL kernel/backend can override library policy. No other crate, GPU code or binding was changed.

## Numerical contract and limits

For every valid nonempty shape, the sole production branch visits each output and all K products in order; there is no acceptance threshold or unchecked candidate output. K=0 returns positive zeros. NaN/infinity/overflow/subnormal behavior follows ordinary f32 multiply/add, not blanket NaN rejection: an input-generated NaN is valid, while a NaN injected into the bypassed packed kernel cannot enter a returned BMM result. Checked dimension products reject overflow before allocation/indexing for nonempty shapes; short nonempty buffers panic as the existing Vec-returning backend API does. Empty outputs return without dereferencing inputs.

This establishes software-path isolation from the suspect implementation, **not a proof that the historical root cause is removed**. It assumes valid immutable inputs and ordinary Rust/compiler/host arithmetic and memory correctness. Arbitrary host hardware faults, input corruption before entry, or compiler faults affecting the independent implementation are not detectable by path replacement. No evidence diagnoses such a fault here. It would be scientifically wrong to promise universal error immunity or label the original incident root-cause-fixed. Independent reviewer decides whether this bounded containment satisfies the release gate; this report does not mark PR25 ALLFIXED.

## Deterministic validation

- `red.log`: actual packed microkernel injection skips p=90; BMM returns 127 instead of 128. Test fails before rerouting (exit 101).
- `install-red.log`: after direct BMM rerouting, install-only CpuBackend BMM still returns 127 instead of 128. Test fails before correcting the callback (exit 101).
- New cfg(test)-only thread-local faults execute inside real packed arithmetic, not a fabricated backend. RAII clears the fault even on panic, and thread-local state cannot contaminate concurrent tests. Small injected fixtures deliberately run inline, so thread-local injection is not misrepresented as cross-worker fault propagation.
- Positive controls verify the suspect path really returns omitted-product, NaN, infinity, or finite-corrupted output. Direct BMM and FastCpuBackend BMM return correct output while the same fault remains armed.
- The historical row/column extracted from seeds 2556/3556 produces injected bits 1094337833 in the packed path and correct bits 1094681645 in contained BMM.
- Original bitwise assertions and prior bounded failure diagnostics remain unchanged. Existing independent-oracle original-shape, tails, K-blocks, recycling and concurrent-caller coverage remains active; historical packed-worker split diagnostics remain test-only.
- New Tensor API test exercises all four last-two-dimension transpose combinations across empty/batch/K-zero/singleton/tail shapes, exact all-element reference values, CPU placement, gradient checking and shape rejection. CPU here means host Tensor storage; FastCpuBackend does not implement a DeviceBuffer-resident BMM and no GPU/resident-device claim is made.
- Final debug default-parallel suite: **25 passed, 2 ignored**. Final release serial suite: **25 passed, 2 ignored**. These are 18 unit and 7 integration correctness tests per run, not added together as unique tests. Ignored tests are cost/performance probes. Existing core layer_norm/scatter_add unused-variable warnings remain. `git diff --check -- crates/ferro-fastcpu` passes. Rustfmt proposes broad changes to inherited compact style; no repository-wide formatting was applied.

## CPU cost, not a performance win

The separate release cost probe runs single-threaded CPU comparators sequentially: 3 warmups, 9 samples, two orders. No GPU benchmark was run. At shape [16,128,128,128], scalar BMM medians were **13.9863 ms and 13.8991 ms**. The comparator was **explicit one-thread packed matmul separately per slab**, with medians **0.9193 ms and 0.9850 ms**. It is NOT the historical fused batched implementation. Raw samples are in `cpu-cost.log` and `results.json`.

This demonstrates a substantial CPU cost, not an exact historical regression factor or isolated-machine benchmark. Other host Python processes were present; their idle state was not established. No GPU performance claim is inferred from these measurements. The probe links the normal production library, not cfg(test) packed microkernels whose fault hooks would bias timings. The later added historical-dot unit test did not alter production arithmetic or this probe.

## Reproduction and artifacts

Commands from repository root (use fresh log names to retain old evidence):

```sh
CARGO_TARGET_DIR=target-pr25-fastcpu cargo test -j 2 -p ferro-fastcpu
CARGO_TARGET_DIR=target-pr25-fastcpu cargo test -j 2 -p ferro-fastcpu --release -- --test-threads=1
CARGO_TARGET_DIR=target-pr25-fastcpu cargo test -j 2 -p ferro-fastcpu --release --test bmm_containment serial_cpu_cost_probe -- --ignored --exact --nocapture --test-threads=1
python verification/pr25-fastcpu-containment/summarize.py
```

The first two execute current containment checks, not historical RED reproduction. Preserved RED logs are from actual pre-containment/pre-callback-correction states. `summarize.py` validates archived expected statuses and parses all raw CPU samples; it does not rerun tests. `results.json` explicitly labels source hashes **after execution**, not build attestations or retroactive hashes of historical binaries. Compiler used here: stable rustc 1.98.0, x86_64-pc-windows-msvc. No Miri/sanitizer result is claimed.

Files changed/created:

- `crates/ferro-fastcpu/src/lib.rs`: independent production BMM, install-only bypass closure, test-only historical BMM and injection seams.
- `crates/ferro-fastcpu/src/containment_tests.rs`: deterministic injection and edge-contract tests.
- `crates/ferro-fastcpu/tests/bmm_containment.rs`: installed Tensor API checks and ignored serial CPU cost probe.
- This directory: README, summarize.py, results.json, actual RED/GREEN/final/cost/diff logs.

No commit or push. Root-cause investigation remains open separately from the implemented BMM containment.
