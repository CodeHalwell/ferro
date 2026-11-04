# CPU compatibility and GPU portability groundwork, 2026-11-03

This is an uncommitted continuation of the [prior parity expansion](../parity-2026-11-03/README.md),
on base commit `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. The scope is a
substantial additional CPU feature stage, not full PyTorch/PyG parity or GPU
execution. NVIDIA CUDA and Apple Metal are the intended GPU targets; the
[GPU portability ledger](../../docs/GPU_PORTABILITY.md) describes what remains.

## Implemented features and evidence

References are actual PyTorch 2.13.0+cpu and PyG 2.8.1. Ferro runtime code imports
neither reference package nor NumPy. The Rust core remains dependency-free and
new differentiable operations use the existing record_fn mechanism.

| Area | Implemented scope | Added Python methods |
|---|---|---:|
| Native Bool and interchange | True Bool dtype/storage; comparisons/logical operations, exact counts, typed selections/where, full/prefix mask indexing, scalar-index gradients, safetensors BOOL and actual CPU DLPack byte interchange | 16 |
| Functional Bool | Public predicates/reductions/nonzero/where/select, tuple-axis Bool sums and logical reductions, exact typed values and selected-value gradients | 3 |
| Attention | MHA batch/sequence-first and unbatched inputs, bool/additive/padding masks, per-head weights; rank-2 functional SDPA/custom scale, fully masked zeros, empty inputs and capture rejection | 7 |
| Data pipelines | Lazy map/iterable datasets, nested typed collation, custom sampling/collation, subsets/concatenation/splits, native zero-worker batching and exact cross-worker sampler order | 7 |
| Graph masks | Bool subgraphs preserve isolated nodes and edge alignment; Bool graph masks batch/extract exactly; custom link-index batching rejects instead of corrupting offsets | 3 |
| Spatial | Conv1d/3d, ConvTranspose1d/2d, AdaptiveAvgPool1d/2d/3d and functional helpers; groups/dilation/sizing/empty batches/checkpoints | 9 |
| Training | Native SGD/Adam/AdamW groups, owning state dictionaries, atomic loads; StepLR/ExponentialLR/CosineAnnealingLR/ReduceLROnPlateau and exact fresh-process continuation | 10 |
| More graph convolutions | GINE, APPNP, Cheb, TransformerConv; edge/node/parameter gradients, Laplacian modes/lambda, beta-gated skips, batches and checkpoints | 14 |
| Heterogeneous graphs | HeteroData/HeteroConv and bounded RGCNConv, CPU typed stores, relation routing/aggregation/bases, shared-child gradients and checkpoint aliases | 13 |
| Sampling and links | NeighborLoader/LinkNeighborLoader, exact sparse negative sampling, batched negatives and RandomLinkSplit, alignment/leakage/gradient controls | 19 |

These are 101 new Python test methods, with parameterized numerical cases and
negative tests. Existing methods also migrate supported-option expectations.
Spatial tests include 41 reference output/input/all-parameter gradient cases;
an independent reviewer checked six additional configurations (maximum error
1.43e-6). Heterogeneous/RGCN tests exercise per-relation mean normalization and
basis gradients. Extra graph-convolution tests use actual PyG classes and copied
weights, without claiming identical random initialization streams.

## Important correctness and state details

Bool is real native CPU storage, not floating masks under another dtype name.
Typed comparisons preserve I64 values above 2^53 and tiny F64 values. CPU DLPack
uses canonical 8-bit bool metadata/bytes; import treats any nonzero byte as true
without constructing invalid Rust bool representations. Serializer/import bounds,
empty tensors and noncontiguous views have tests. Bool is nondifferentiable;
selection still reaches f32 source values through existing autograd rules.

Scalar tensor indexing formerly detached the selected scalar. Native views now
preserve the autograd path. Numeric wrappers reject unsupported typed arithmetic
before entering float kernels. A review found empty Bool reductions could attempt
an unrepresentable nonempty output allocation; checked typed capacities and
fallible reservation now return errors. Scalar any/all axes0/-1 match Torch.

Masked/custom-scale attention is a CPU eager path. Functional SDPA uses bool True
for allowed positions; MHA uses True for exclusions. Fully masked SDPA rows return
zero values/gradients. MHA's returned-weight path preserves its reference NaN
behavior on fully excluded rows. Empty attention avoids allocating an unused
causal mask, and unused float-mask gradients follow the tested Torch behavior.
Constructor/intermediate allocation guards precede computation. Functional SDPA
rejects explicit masks combined with is_causal in this supported CPU contract.
Nonzero attention/recurrent dropout and broader attention options remain open.

Native optimizer groups preserve legacy single-group snapshots and stage every
group before mutation. CPU state loads validate shapes, parameter partitions,
buffers and scalar configuration before commit. Python scalar metadata preserves
exact learning-rate values beyond native f32 rounding. Tests compare 15-step
Torch group/skip trajectories and scheduler state/rate histories, including cosine
cycles, plateau thresholds and cooldown. Fresh-process continuation compares
remaining losses, complete model/optimizer/RNG state and future draws. Scheduler
state remains an explicit state_dict sidecar. Native steps retain Ferro's existing
gradient-consumption behavior, rather than every torch.optim convention.

Synchronous Python loaders collate one batch at a time, including infinite streams.
TensorDataset zero-worker batching shares the native Rust sampler's exact order
with worker mode. Native workers still materialize an epoch; arbitrary Python
multiprocessing, pinning and generic iterator checkpoints are not implemented.

Neighbor loaders preserve seed-first n_id/e_id/input_id metadata and aligned native
fields/gradients. All-neighbor cyclic two-layer results match a full PyG SAGE
model for seed outputs, original node/edge fields and every parameter gradient.
The installed PyG loader cannot run its optional sampler backend (pyg-lib and
torch-sparse are absent); this is not counted as successful backend validation.
Finite fanouts use Ferro's documented once-discovered frontier algorithm and RNG,
not backend-identical PyG sampling. CPU samplers do not establish GPU graph scale.

Negative sampling operates on sparse occupied/complement ranks without a dense
node-by-node matrix. Link splits control duplicate/reverse-edge leakage and
supervision separation. Independent review caught per-split supplied-label shift
and empty-negative-field differences; both now compare with actual PyG, with the
failed baseline retained. Custom link-label indices reject ordinary Batch
collation because its node-offset convention is unsupported; use LinkNeighborLoader.
Bool node masks define the subgraph node universe when num_nodes is omitted,
including isolated nodes and empty topology.

## Staging corrections and independent review

Independent reviewers covered native Bool/storage/byte interchange, empty output
capacity, graph convolution formulas, sampling/leakage, heterogeneous aliases,
spatial shapes/gradients, grouped atomic checkpointing, data laziness and attention
allocation/capture. No runtime PyTorch delegation or second differentiation engine
was introduced. Source-only overlays were limited to explicitly logged scratch
runners; repository tests and final commands use the installed wheel.

Staging failures are retained. They include a test fixture losing the trailing
shape of an empty nested list, actual scalar Bool axis/allocation defects, an
attention test combining a mask/causal flag rejected by the reference, and earlier
source/wheel differences. The first full frozen run passed the Rust workspace and
all Python unittest cases, but the standalone binding script still asserted three
old limitations. It now checks scalar-index one-hot gradients, valid additive
mask output plus malformed-mask rejection, and typed custom-dataset collation
plus invalid-source rejection. Independent review confirmed these migrations
strengthen supported behavior. Exactly that script differs between the archived
first freeze and final freeze; no production source changed. The full build and
all commands were repeated after the final freeze.

## Final matching-build validation

Linux x86_64, rustc 1.99.1 (58cf8f9c3 2026-10-13), Python 3.12.14, Torch 2.13.0+cpu and
PyG 2.8.1. The final noneditable release wheel was built and installed
after the final freeze. All 36 installed Python/native files match the wheel;
Python files also match the checkout. No generated bytecode is packaged.

| Command / scope | Final result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 916 passed / 2 ignored; fastcpu 38 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies (real GPT2 fixture absent). CUDA 139 unit/integration harness passes plus 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0. |
| Release wheel build and noneditable installation | Both exit 0. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 388 passed; 30 discovered CUDA cases unavailable. |
| `python examples/pyg_parity.py` | Exit 0; all three graph learning and exact fresh-process restart gates retained. |
| `python examples/py_regression.py` | Exit 0; complete binding and DLPack regression script passes. |
| Compiled DAG and fuse regression scripts | Both exit 0. |
| `python examples/ops_vs_torch.py` | Exit 0; operator values and gradients pass. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization passes. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Source/wheel/install identity and `git diff --check` | Passed. |

Python discovers 418 cases. Unittest reports `Ran 412 tests` and 25 skip
records: 24 individual skips plus one class-level setup skip preventing six
additional cases. Thus 388 pass and 30 remain unavailable. Rust totals use the
last outer summary per executable; nested subprocess summaries and targeted
staging reruns are excluded. CUDA harness passes are not GPU execution.

All 579 selected source/config/lock inputs remain unchanged after the final freeze.
Native extension SHA-256:
`aa6a26ab2338c8373c6e4541a6f005b4b89db22d664c54cc5fa95febbbdd8eb6`.
The manifest hashes 85 local logs/results/diagnostics, including staged failures
and archived first-freeze evidence. Raw logs, wheels and diagnostics remain under
`/workspace/work/portability-2026-11-03`; they are not shipped in the repository.


The final independent audit recomputed all 579 source hashes, 85 evidence hashes
and 36 wheel/install identities; verified all 12 command records; independently
recounted Rust/Python outcomes; and reconciled all three graph learning/restart
results with their raw JSON records. It also verified that the archived first
freeze differs only in the reviewed standalone binding test. No mismatch or
remaining blocker was found within the declared CPU scope.

## Remaining scope

All new execution paths are scoped CPU implementations. Real GPU validation is
absent; CUDA compilation/host regressions do not establish GPU body execution.
No Metal backend was added in this stage. The shared tensor/backend architecture
is retained for the intended NVIDIA CUDA and Apple Metal work.

Full typed arithmetic/promotion/complex tensors, arbitrary advanced indexing,
more optimizers/layers, packed/projected/dropout recurrence, HGT/PNA/learned
pooling, automatic hetero tracing, arbitrary MessagePassing, dataset/transform/
explainer ecosystems, general derivative transforms, full linalg/FFT/compiler/
export/quantization/profiling, native AMP and distributed training remain open.
The [compatibility matrix](../../docs/PYTORCH_PYG_COMPATIBILITY.md) and GPU ledger
record specific boundaries rather than marking the full gap list complete.

Performance ignores and the real GPT2 fixture remain unexecuted. Historical
native-crash/numerical investigations remain unresolved. No commit, publication
or deployment was performed; all changes remain local and uncommitted.
