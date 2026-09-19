# Resident segment primitives

Implemented in core segment dispatch and CUDA, with no Python, autograd, tensor,
sparse or graph source changes. Existing free `segment::sum` and `softmax` APIs
now dispatch resident f32 inputs to CUDA; CPU behavior remains. Public
`segment::PreparedSegments::new(ids, groups, device)` validates immutable integer
topology and prepares it once. Its `sum` and `softmax` methods reuse that topology
across fresh input tensors; recorded adjoints retain the same plan. Free-function
calls explicitly prepare each invocation. Static capture rejects these operators;
there is no frozen-result replay path or claim of full GNN residency (SpMM remains CPU).

## Contracts and measured counters

- CSR offset + edge-order i64 metadata is uploaded once per preparation, counted
  by `layout_counts`. Duplicates, unsorted edges, isolated/empty groups, zero-width
  features and empty edge lists are supported. Host IDs are never represented as f32.
- One thread owns each (segment, feature) and reduces in original edge order.
  This implementation does not use floating-point atomics; parity is tolerance-
  based rather than a cross-hardware bitwise promise. No performance claim made.
- Sum and softmax each enqueue one forward and one adjoint kernel. Empty output
  work skips enqueues. Each result uses one counted f32 allocation; gradient
  accumulation in the tested step is resident in-place.
- Softmax is max-subtracted. Nonfinite logits remain errors, including all -inf;
  no masking API exists. To retain that eager error contract it explicitly
  allocates and reads one four-byte integer status. This is a synchronizing
  control transfer, not a feature-data download. Do not claim zero total transfers.
- `segment_counts`: successful sum forward/backward, softmax forward/backward,
  status downloads, status allocations. `layer_norm_counts` existing upload/download
  fields count f32 feature transfers; `alloc_stats` counts f32 result allocations.
- Structural test over two fresh-input steps: per step zero topology uploads,
  zero f32 uploads/downloads, four successful operator enqueues, four f32 result
  requests, one status allocation and one four-byte status download.
- Original CudaSlice ownership and stream tracking stay enabled. Foreign backend
  streams/plans and invalid buffer types/sizes reject before kernel enqueue.
- Strided input features materialize through the fallible resident layout seam.
  No fallback suppresses layout or kernel errors. Saved softmax output is a
  detached snapshot and core multi-cell reads use deduplicated address ordering.

## Existing boundaries found

The `record_fn`/`backward_with` interface is infallible. Backward backend errors
therefore panic with the original error instead of silently computing on CPU;
forward methods return `Result` unchanged. A core fake backend tests both paths.
Changing backward into a Result-returning mechanism is outside this task's scope.

Existing `backward_with` calls `cotangent.detach_copy()`: a noncontiguous seed
currently causes a host round trip before reaching any segment adjoint. This was
observed as one upload/one download in the strided test. Tests explicitly prepare
a contiguous resident seed to measure the segment boundary; tensor/autograd files
were not changed. A follow-up general tensor fix is needed for arbitrary strided
public backward seeds. Strided *features* do stay resident in this implementation.

## Verification

Required CUDA uses NVRTC/cuBLAS DLLs from
`$LOCALAPPDATA/Temp/cuda-rt/nvidia/{cuda_nvrtc,cublas}/bin`, `FERRO_REQUIRE_CUDA=1`,
and `cargo -j2`. The CUDA test helper asserts initialization when required.
New forward tests were observed failing on the original CPU-only dispatch before
implementation; strided input failed before resident materialization was added.
Scalar f64 forward/VJP and finite differences (including perturbed CUDA forwards)
cover small unsorted, multi-feature and many-edge single-group cases. Existing CPU
segment gradient-check tests remain part of verification. Logs are adjacent.
No performance benchmark was run while sibling work could be active.

Final recorded runs all returned exit code zero:

- `segments-gpu.log`: six required real-CUDA tests, default parallel harness.
- `core-full-final.log`: full core suite, including three new fake-backend tests
  and six existing CPU segment tests (762 passed across parent suite summaries).
- `cuda-full.log`: full CUDA suite with required runtime (118 passed across
  parent suite summaries), including static graph safety regressions.
- `cuda-check.log`: CUDA crate check.
- `diff-check.log`: owned tracked files pass whitespace/error checks.

`results.json` records commands/status and `source-sha256.json` identifies the seven
owned production/test files. Existing unrelated compiler warnings remain. Other
agents' concurrent repository changes were left untouched. No commit or push.
