# Static replay Python acceptance: RED handoff

## Run

From `C:/Users/DanielHalwell/PythonProjects/ferro` in bash:

```bash
python verification/static-coverage-python/run.py green
```

Run this only after the parent rebuilds the bindings. The runner itself never
builds, installs, benchmarks, stages, or commits. It uses the existing
`crates/ferro-py/.venv/Scripts/python.exe`, requires CUDA for both libraries,
and prefixes the local CUDA NVRTC/cuBLAS runtime DLL directories to PATH.
An existing log name is refused, so use `green2` etc. for later attempts.
A method can be selected by adding `StaticCoverageTests.test_softmax_axes`.
Missing GPU availability fails tests rather than skipping with
`FERRO_REQUIRE_CUDA` set. Missing imports also fail, not skip.

## Verified RED

`final-red.log` is the final test-file revision: 10 methods run, 49 failing
cases, process exit 1, environment probe exit 0. Real device was NVIDIA
GeForce RTX 3090; Torch was 2.6.0+cu124. The log records the command,
installed Ferro path, and SHA256 of the test source.

The final errors, counted from traceback sections in `red-summary.json`:

- 18 expanding static seed not implemented.
- 12 empty static destination unsupported.
- 10 softmax last-dimension-only rejection.
- 6 invalid or empty static matmul rejection.
- 2 zero-width softmax divide-by-zero panics during capture.
- 1 zero-column matmul cuBLAS invalid-value failure during capture.

The explicit scalar/same-shape shared-expression controls and invalid-input
rejection method pass. Nonempty last-axis softmax positive/negative subcases
also complete without error. No numeric gate failures appeared.
`softmax-red.log`, `broadcast-red.log`, and `shapes-red.log` retain vertical
exploration evidence; `acceptance-red.log` precedes the added passing controls.

## Coverage and contracts

- Softmax axes 0/1/2/-1/-2/-3 on rank 3; input and scale mutation makes each
  nonempty oracle genuinely change, not just shift softmax by a constant.
- First pointwise seed expands rows, columns, rank, scalar, and multiple axes.
  Subtraction/division preserve operand order; shared leaf and intermediate
  cases exercise a DAG, not only a straight unary chain.
- Empty pointwise ranks 1-4, empty broadcasts, all positive softmax axes on
  selected empty rank 1-3 inputs, empty transpose/reshape, and empty GEMM
  output dimensions.
- Zero GEMM inner dimension produces exact zeros, separately exercised with
  a nonempty mutable bias to verify replay freshness.
- Wrong softmax dimensions, incompatible nonempty/empty GEMM dimensions,
  incompatible broadcast, and unsupported log_softmax remain rejection cases.
- Every successful case checks replay returns None and replay_count == 3,
  detached snapshots, exact output rank/shape, direct nondefault-stream CUDA
  DLPack consumption with no producer fence, preserved old snapshots after
  subsequent replays and graph deletion, and consumer ownership after all
  Ferro snapshots are dropped. Nonempty mutating cases check distinct buffer
  pointers and an actually changed Torch oracle.
- Actual Torch CPU float32 arithmetic is the independent numeric reference;
  nondegenerate cases also compare actual eager Ferro CUDA. Degenerate input
  cases compare eager Ferro CPU, since the old eager CUDA path itself panics
  on zero-width softmax and rejects zero-column GEMM. This is explicit, not a
  fallback on test failure. Static capture/replay always receives CUDA leaves.
- f32 maximum-ULP gates: 32 softmax, 8 composed pointwise, exact empty/GEMM-zero
  cases. Ordering collapses signed zero and subnormals only, checks finite
  values, and never uses a blanket near-zero absolute tolerance. These are
  deterministic acceptance cases, not a distribution-wide numeric proof.

## Parent attention

The final RED still exposes the capture-time empty-softmax panic for `[0]`
axis 0 and `[2,0]` axis 1, and capture-time cuBLAS failure for `[2,3] @ [3,0]`.
Changing only static destination planning will not resolve those public API
paths if capture evaluates the eager operator first. Keep these as acceptance
failures or explicitly coordinate scope; do not relabel them expected rejection.

No production files, existing tests, bindings, dependencies, or git index were
changed by this worker. Final GREEN is intentionally deferred to the parent
following the implementation and binding rebuild.
