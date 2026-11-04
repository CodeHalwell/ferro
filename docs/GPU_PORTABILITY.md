# NVIDIA CUDA and Apple Metal targets

The intended GPU targets are NVIDIA through CUDA and Apple GPUs through Metal.
CUDA already has a backend crate and a tested host-side API; a Metal backend
has not been implemented. The current execution environment is Linux without
an available GPU. Compilation and mock-backend tests do not establish hardware
execution, throughput, residency or numerical parity on either target.

The latest CPU compatibility additions are recorded in the
[compatibility matrix](PYTORCH_PYG_COMPATIBILITY.md) and
[matching-build report](../verification/completeness-2026-11-03/README.md).
This document records the remaining work rather than declaring full parity.

## Shared execution contract

Keep the zero-dependency core and one record_fn autograd mechanism. Device
crates implement dispatch::Backend; model/optimizer code must not depend on a
particular GPU API. Existing CPU-only operators explicitly validate that scope.
Porting a model requires resident implementations for its complete forward and
backward path, not just permitting another device string.

Both backends need validated allocation ownership, typed buffer layouts, storage
version checks, shape/stride materialization, explicit transfers and errors that
preserve the current owner/context guarantees. Device-tagged buffers must be
real device allocations. Unsupported operations must expose their documented
fallback or return an error rather than silently advertise GPU residency.

CPU Bool uses native bool storage; GPU implementations need a defined byte
representation and typed index/mask kernels. DLPack streams and producer
lifetimes need target-specific handling. Native Bool device transfer currently
rejects. General module/optimizer device migration is also a separate task:
stable parameter slots and optimizer moments must move consistently, with alias
and rollback behavior established before enabling Module.to across devices.

## Implementation and acceptance order

1. Validate the existing CUDA backend on an NVIDIA machine: allocation/copy,
   pointwise/broadcast operations, reductions, matrix products and their adjoints.
   Run the required-CUDA tests and exercise failure cleanup/owner replacement.
2. Add an Apple-only Metal backend crate, a real device identity and capability
   queries. Implement the same buffer/dispatch contract with Metal command
   submission and matching CPU-reference tests. Non-Apple builds must continue
   to work without Apple frameworks. Apple hardware is required for acceptance.
3. Port typed indexing and prepared segment operations, then graph scatter,
   sparse products and attention adjoints. The newly added Python graph layers
   and samplers are currently CPU APIs. CUDA segment sum/softmax are existing
   partial foundations; they do not establish complete graph GPU execution.
4. Cover convolution, transposed convolution, normalization, pooling, recurrent
   state and attention kernels. Current spatial compositions establish semantics;
   efficient GPU kernels and layout-specific paths are additional work.
5. Validate optimizer groups, model/state migration, checkpoints and continuous
   training on each backend. Require whole-model forward/backward placement
   checks and compare outputs, losses and gradients with actual references.
6. Add native mixed precision and loss scaling where supported, followed by
   profiling and benchmark gates. Float64 and other dtype availability must be
   declared per backend; matching CPU storage types does not imply every GPU
   has matching arithmetic support.

Numerical comparisons need declared tolerances, device synchronization and
representative shapes. Same-environment bit-exact checkpoint continuation is
a different guarantee from cross-device/cross-platform reproducibility. Neither
is implied by matching a public API name.

## Remaining framework gaps

| Area | Active CPU additions | Still open |
|---|---|---|
| Tensor types/indexing | Bool, exact int64 arithmetic, float64 math/AD, numeric promotion/casts, typed selection/copies and CPU DLPack | Full dtype/operator coverage, float64 parameters/optimizers, half/complex compute, arbitrary advanced indexing and out destinations |
| Training | Fourteen optimizer families including Muon, leaf Tensor inputs, hooks, retained gradients, strong-Wolfe L-BFGS, schedules and atomic typed checkpoints | SparseAdam, remaining options/schedulers, generic iterator state and GPU state migration |
| Data | Lazy map/iterable Python datasets, nested collation and synchronous native minibatches | Python multiprocessing, pinned/asynchronous pipelines and generic iterator checkpointing |
| Vision/sequence | ConvTranspose3d, padding/pooling, packed/projected recurrence, persistent dropout, canonical Transformers, attention options and shuffles | Full normalization/embedding/loss/interpolation/unpooling catalogs, lazy modules, all dtype/layout options and efficient kernels |
| Graph structures/models | HGT/PNA, eager MessagePassing, spectral/conditioned/residual/point layers, learned pooling and graph normalization | Automatic to_hetero, complete operator/aggregation catalogs, dynamic geometry and sparse formats |
| Graph scale/link training | Homogeneous/heterogeneous frontier loaders, link splitting/negative sampling, cluster collation, HeteroBatch, GraphSAINT, typed conversion and standard label-index batching | Temporal/distributed loaders, METIS and arbitrary Batch increment hooks |
| Graph ecosystem | Local dataset persistence, aligned transforms and topology utilities | Downloadable dataset zoo, positional encodings, complete transforms and explainers |
| Differentiation | Direct forward JVP rules with a scoped opaque fallback, functional Jacobian/Hessian/VJP, selected transforms, higher derivatives and absent gradients | F64 forward primals, nested levels, vmap/vectorization, complete transform composition and higher-order coverage |
| Math, compilation and deployment | CPU QR/eigh/SVD/pinv/lstsq/LU, extended solves and matrix chains, typed sorting/norms/math, fusion and scoped capture | Remaining linalg/FFT, general compiler/export, quantization and profiling toolchains |
| Scale and hardware | Existing partial CUDA backend | Metal backend, full GPU model coverage, AMP, distributed training and performance certification |

The full target remains broader than one implementation stage. Each new area
needs source-backed numerical, gradient, state and hardware evidence before its
remaining items can be marked complete.
