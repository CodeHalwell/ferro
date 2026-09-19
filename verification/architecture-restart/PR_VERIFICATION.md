# Isolated CPU training/restart PR verification

This is the final publication evidence for the selected PR tree based on
dc13a7199c23d92be5dd766102af03a6bfcb4bc3. The earlier run-01/run-02 reports
describe the broader development checkout. They must not be used as the
identity of this isolated PR.

## Scope

The PR includes the Python CPU checkpoint/optimizer prerequisites, persistent
storage-alias validation and tests, the narrow cross-entropy binding, both
public training/restart examples and tests, and roadmap/documentation changes.
CUDA prepared segments and the separate tiled CPU BMM work remain outside
this branch. Historical post25 reports are retained solely as context for
roadmap claims from that broader checkout.

## Final checks

[pr-run-03/results.json](pr-run-03/results.json) records all 13 verifier
commands exiting zero. The release wheel was built and installed into this
worktree's separate environment. Python discovery passed 95 tests. The
difference from the earlier 105-test run is the excluded prepared-segment
test file. Both examples still pass seeds 7, 11 and 43 with identical reported
metrics and bit-exact fresh-process continuation.

[Supplemental results](pr-run-02/supplemental.json) record successful CUDA
all-target compilation and runtime tests. The initial Rust binding build
failed before tests because PyO3 selected a different Python than PYTHONHOME.
With PYO3_PYTHON set explicitly to this worktree's interpreter, the
[final binding run](pr-run-03/binding-final.json) passed all 15 tests.
No source change or tolerance relaxation was needed.

The full core, fastcpu/tokenizer, Python binding/DLPack, operator reference,
safetensors, compiled fusion, fusion and seeded fuzz suites passed.
Existing ignored Rust tests and warnings remain. Fuzz retains its documented
accumulation-order percentile exceptions. No sanitizer or hosted-CI claim.

The first isolated attempt pr-run-01 stopped before compilation because the
new environment had the Python maturin package but not its executable.
The existing executable was copied into that environment before pr-run-02.
The original environment and package were not changed.

[Provenance](pr-run-03/provenance.json), [source manifest](pr-run-03/sources-after.json)
and [stability result](pr-run-03/stable.json) identify 426 selected inputs
unchanged through the verifier, plus matching source/wheel/install Python
bytes and wheel/install native bytes. Documentation was finalized afterward.
The separate environment reuses dependency packages from the original venv
through a .pth entry, while ferro itself is installed locally and hash-checked.
This is isolation from unrelated source changes, not a clean-room dependency
reconstruction.

Final source normalization removed terminal blank lines only; pr-run-03 rebuilt the wheel and repeated all 13 verifier commands plus the binding suite. [Six model results](pr-run-03/metrics.json) are published as structured evidence.

## Reproduction

Choose a fresh output directory and use a Python environment containing
maturin, Torch, NumPy and safetensors:

~~~text
python -B verification/architecture-restart/verify.py --output-dir verification/architecture-restart/fresh-run
cargo check -j2 -p ferro-cuda --all-targets
cargo test -j2 -p ferro-cuda
cargo test -j2 --manifest-path crates/ferro-py/Cargo.toml --lib
~~~

On Windows, prepend the installed Torch lib directory for CUDA DLL discovery.
For Rust binding tests set PYO3_PYTHON to the selected interpreter and PYTHONHOME
to that interpreter's base_prefix. Required runtime CUDA checks used
FERRO_REQUIRE_CUDA=1. Do not set PYTHONHOME for the wheel build.

Raw logs, wheels and executables remain local. Published JSON manifests and
reports preserve identities and command outcomes; references to omitted logs
in those manifests denote local provenance. No exhaustive review or closure
of the older CPU numerical/illegal-instruction investigations is claimed.
AI assisted implementation, verification and PR preparation.
