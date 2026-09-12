# PR24 empty DLPack import review

## Outcome

Fixed only binding production code in `crates/ferro-py/src/dlpack.rs`.
Empty explicit-stride tensors have no reachable address, so their stride range
must not be computed using `(extent - 1)` on a zero axis. Empty CUDA imports
now construct CUDA storage instead of returning early with CPU storage, without
pointer arithmetic, dereference, or a producer download. Nonempty null data is
still rejected. Dtype, negative rank/extent, nonzero extent-product overflow,
byte-offset alignment and CUDA ordinal validation precede empty construction.
Overflow of nonzero extents is checked independently of zero-axis position.

The importer still calls the producer's `__dlpack__()` with no stream argument.
Its nonempty producer ordering/copy scheme is unchanged; these empty-input tests
are NOT evidence that general incoming CUDA stream handoff has been fixed.
The existing read -> capsule rename -> deleter ownership transaction is unchanged.

## Contract checked

Downloaded upstream header from
https://raw.githubusercontent.com/dmlc/dlpack/main/include/dlpack/dlpack.h
into `dlpack.h`. Lines 259-260 explicitly specify NULL data for size-zero tensors.
Lines 283-285 distinguish legacy NULL contiguous strides from the v1.2 requirement
for explicit strides. This binding uses the legacy `dltensor`/DLManagedTensor ABI;
this patch does not introduce versioned capsules or change that ABI.
No elements means no reachable stride bounds, including negative/extreme strides.
The nonzero-product and alignment checks are Ferro validation constraints, not a
claim that DLPack imposes Ferro's usize storage limits.

## Tests and exact evidence

Commands are preserved in each log and JSON; all runners prefix the NVRTC/cuBLAS
runtime DLL directories and set FERRO_REQUIRE_CUDA=1. Cargo uses -j2.

- `red.log`: original extension, new Python tests FIRST. 4 failures and 8 errors:
  explicit empty strides rejected, contiguous empty returned CPU, leading-zero
  overflow/alignment/negative CUDA ordinal bypassed. Expected production defects.
- `build.log`: successful `maturin develop --release -j2 --manifest-path
  crates/ferro-py/Cargo.toml` and editable extension installation.
- `green.log`: all 3 new DLPack test methods pass. Raw null CUDA descriptors,
  both explicit/NULL strides, extreme empty strides, single consumption/deleter,
  invalid descriptors, Torch CUDA import/export, shaped tolist(), static snapshots,
  retained consumer lifetime, and Torch -> Ferro roundtrip.
- `unit.log`: standalone Rust binding suite, default parallel harness,
  `cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib`:
  **15 passed**, including existing nonempty export/import and recorded-error tests.
  New Rust tests cover driver-free empty stride validation, zero-masked overflow,
  alignment, and the actual capsule destructor's exactly-once behavior after both
  successful import and rejection. The raw Python fixture independently counts
  CUDA import deletion but explicitly owns failure cleanup (no Python destructor).
- `python.log`: `python -m unittest discover -s crates/ferro-py/tests -v`:
  **31 passed**, no CUDA skips. Includes independent eager CUDA softmax across
  empty axes and GEMM with empty output/zero reduction against CUDA Torch values.
  `test_static_coverage.py` now executes CUDA eager for empty cases and asserts
  CUDA residency instead of silently substituting a CPU oracle.
- Scoped `git diff --check` passed.

## Intermediate issues preserved

`build-concurrent-incomplete.log` and `build-concurrent-incomplete-2.log` captured
build attempts during another worker's unfinished native graph edits. Subsequent
build passed; no core/CUDA source was changed by this worker.
`green-test-oracle-error.log` preserves a test mistake exposed after the production
fix: tolist() for (2, 0) is [[], []], not []. Corrected using shaped Torch tolist(),
then reran successfully. Original production RED evidence remains untouched.
`python-before-standalone-eager.log` preserves the earlier 29-test passing run.
Existing Rust unused/dead-code warnings remain. The file patch tool's edition-less
Rust lint rejects existing C-string literals; real Cargo compilation succeeds.

No staging, commits, pushes, benchmarks, or production core/CUDA edits.
