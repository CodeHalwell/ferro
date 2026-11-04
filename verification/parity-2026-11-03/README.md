# PyTorch and PyTorch Geometric API expansion, 2026-11-03

Base commit remains `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. This is an
uncommitted continuation of the [graph/sequence work](../graph-sequence-2026-11-03/README.md).
The [manifest](manifest.json) identifies the frozen source, rebuilt noneditable
wheel, installed native extension and local evidence. Prior reports retain
separate source identities.

## New public API

The [compatibility matrix](../../docs/PYTORCH_PYG_COMPATIBILITY.md) gives imports,
supported options and exact boundaries. The additions include:

- Tensor construction and metadata: lowercase tensor/as_tensor, constants,
  ranges, eye, random and like factories; typed dtype/size/numel/item/truth
  behavior; exact i64/f64 storage. f32 remains the arithmetic/autograd dtype.
- Functional shapes, partitioning and reductions: inferred reshape, tuple-axis
  sum/mean, view-preserving unbind, empty split/chunk behavior, and vector,
  matrix and broadcast-batch matmul with correct gradient reduction.
- PyTorch-layout RNN, GRU and LSTM networks and cells, including multiple layers,
  bidirectionality, batch-first, unbatched input and explicit states. Eighty-four
  numerical recurrent cases compare outputs and all input/state/parameter
  gradients against actual torch modules; checkpoint continuation is also tested.
- Weighted cross-entropy and NLL with reductions, ignored indices, spatial and
  unbatched input, label smoothing and CE probability targets. Tests cover
  extreme logits, supported target/weight gradients, empty/all-ignored targets,
  zero denominators and loss configuration/buffer checkpoints.
- PyG-shaped GCNConv, SAGEConv, GINConv, GATConv and GATv2Conv. Weighted edges,
  bipartite options, trainable GIN epsilon, multihead attention, edge projections,
  self-loop fills, residuals and GATv2 shared projections have scoped reference
  tests. GATv2 computes dynamic attention after source/target feature addition;
  a ranking-reversal regression distinguishes it from static GAT scoring.
- Homogeneous Data, Batch and synchronous DataLoader; global sum/mean/max pools,
  grouped softmax/degree, self-loop/coalesce/undirected/subgraph utilities.
  Typed labels and aligned fields survive batching and extraction, with gradients
  preserved. Existing native graph APIs and legacy GcnConv remain available.

There are 80 additional Python test methods, with many parameterized numerical
and gradient cases. References are actual PyTorch 2.13.0+cpu and PyG 2.8.1.
Pinned test dependencies and CI installation make reference coverage explicit.
No Torch, PyG or NumPy runtime dependency was added; core has zero dependencies
and gradients still use the existing record_fn engine.

## Numerical repairs

Log-softmax formerly evaluated x - (max + log_sum_exp_of_shifted_values), losing
its normalization term when the common logit offset was large. It now evaluates
(x - max) - log_sum. Core, AVX2 and CUDA source use the same stable ordering.
The initial regression failed at offset 1e8; final tests cover positive/negative
1e8 and 3e38, several axes, values and seeded gradients. Weighted/smoothed CE
comparisons also pass at these extreme finite offsets.

L2 normalize formerly differentiated through sqrt(0) and divisions whose squared
or cubed denominators underflowed. A direct backward rule now projects the
cotangent and divides once by max(norm, eps), gating the projection on the clamp
branch before division. Snapshots remain detached and the ordinary record_fn
version checks remain active. Infinite overflowed norms are not capped at f32
max. Finite input tests include exact zero, tiny nonzero vectors/epsilon, the
clamp equality boundary, huge values, strided/empty tensors and stale parents.
NaN epsilon rejects explicitly; capture rejects before computation. Numerical
parity is scoped to finite inputs and finite positive epsilon. Higher-order
normalize and actual GPU execution are not established.

Independent review also caught unbind copying instead of retaining views, and
empty partition count differences. Functional wrappers now use native narrow
views and match the tested empty split/chunk contracts. The first development
functional test used an unsupported legacy constructor keyword; its correction
and all earlier staging logs are retained separately from final evidence.

The first complete suite also found two outdated test expectations. The mock
backend in gpu_sticky still used the old log-softmax arithmetic and disagreed
with the corrected reference by rounding; its formula now matches while exact
assertions remain intact. The standalone binding regression still expected
classification reduction="none" to reject. It now checks per-sample shape and
sum consistency, retains other unsupported reductions and adds invalid-reduction
checks for every loss. An independent review confirmed both test migrations.
The original frozen build/logs are preserved under `frozen-attempt1`; a new
source freeze and full rebuild/suite run follow these test-only corrections.

## Integrated graph training and fresh-process restart

Run `python examples/pyg_parity.py`; it emits one JSON result for each of GCN,
GraphSAGE and GATv2. Runtime uses only Ferro and the standard library. The model
is one graph layer, tanh, global mean pool and a two-class linear readout,
trained through CrossEntropyLoss and AdamW. DataLoader shuffles variable-size
homogeneous graphs; an explicit Generator supplies input jitter.

Each split has 32 graphs with 4-6 nodes. Opposite-label pairs have identical
node features and node count; topology connects the marked node to a positive
or negative neighbor. Training and held-out f32 feature values are disjoint,
while topology templates repeat. The feature-only baseline and edge-removal
control are both 50% by construction, checked against the actual paired data.

Before training, gates were fixed at seed 7, 60 epochs of four batches, held-out
accuracy >=95%, CE <=25% of its initial value, and exactly 50% edge-erased
accuracy with >=40 percentage points of degradation. All three models pass the
first declared configuration, without tuning budgets or weakening gates.

| Model | Initial accuracy | Final accuracy | Initial CE | Final CE |
|---|---:|---:|---:|---:|
| gcn | 100% | 100% | 0.66317326 | 0.00394267 |
| sage | 100% | 100% | 0.63039416 | 0.00183574 |
| gatv2 | 12.5% | 100% | 0.71134937 | 0.00509483 |

GCN and SAGE already classify all held-out examples correctly at initialization;
their declared CE reduction establishes learning, not an accuracy increase.
GATv2 improves from 12.5% to 100%. All three edge-erased accuracies are 50%.

At epoch 13 (update 52 of 240), the example saves actual graph tensors/labels,
sample identities, dataset digest and loader epoch alongside the native
model/AdamW/config/Generator checkpoint. A fresh worker restores with deliberately
wrong initial optimizer settings and RNG seed, then must reproduce all 188
remaining losses, shuffled batch identities, final serialized state, metrics
and future RNG draws exactly. Saved data are decoded rather than regenerated;
data hash and epoch/step consistency validate before model construction.
Three integration tests also cover data/cursor corruption and preservation of
graph identity through collation/extraction. Independent review found no blocker.

This is a synthetic, same-environment CPU f32 proof at completed epoch boundaries.
It does not checkpoint arbitrary loader iterators, prove unseen-topology
performance or establish broad graph-learning quality.

## Final matching-build validation

Linux x86_64, rustc 1.99.1 (58cf8f9c3 2026-10-13), Python 3.12.14, torch 2.13.0+cpu and
PyG 2.8.1. After the final source freeze, maturin built a noneditable
release wheel and pip installed it with `--force-reinstall --no-deps`. All
27 installed Python/native files match the wheel; Python files also match the
checkout. Generated bytecode is excluded from the wheel.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 904 passed / 2 ignored; fastcpu 38 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies because the real GPT2 fixture is absent. CUDA 139 unit/integration harness passes plus 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0. |
| Release wheel build and noneditable installation | Exit 0. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 287 passed; 30 discovered CUDA cases unavailable. |
| `python examples/pyg_parity.py` | Exit 0; GCN, SAGE and GATv2 learning/restart gates pass. |
| `python examples/py_regression.py` | Exit 0; binding/DLPack regressions pass. |
| Compiled DAG and fuse regression scripts | Both exit 0. |
| `python examples/ops_vs_torch.py` | Exit 0; operator/reference checks pass. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization checks pass. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Compatibility documentation snippet | Passes against the frozen installed wheel. |
| Source/wheel/install identity and `git diff --check` | Passed. |

Python discovers 317 cases. Unittest reports `Ran 311 tests` and
25 skip records: 24 individual skips plus one class-level setup skip
preventing six more cases. Thus 287 passed and 30 are unavailable. Rust counts
use the last outer summary per executable, excluding nested subprocess summaries
and staging reruns. CUDA harness passes do not establish GPU body execution.

All 555 selected source/config/lock inputs remained unchanged after the final
freeze. Native extension SHA-256:
`100fbc844dc8f1ab73aacb49ad3f6583c76b9971448e8806662853b07cc3832d`.
The manifest hashes 80 local logs/result/diagnostic files. Raw logs, wheels,
staging failures and generated diagnostics remain under
`/workspace/work/parity-2026-11-03`; they are not shipped in the repository.


## Boundaries and review

Full compatibility remains open. The new graph layers/utilities and recurrent
modules are CPU f32 eager APIs. Nonzero recurrent/attention dropout, packed
sequences, LSTM projections, lazy graph channels, cached GCN normalization,
heterogeneous graphs, arbitrary MessagePassing and SparseTensor adjacency reject
or remain unavailable. Typed storage does not add typed arithmetic, bool masks,
automatic promotion, AMP or distributed training.

Two reference corner cases are deliberately documented: weighted duplicate GCN
self-loop insertion rejects rather than imitating PyG's indexed-write gradient;
zero-valued extrema ties use actual input winners instead of PyG CPU scatter's
initial zero. No-batch maximum tie routing also differs. Rectangular loop
attribute reduction supports the full target count and is tested against an
explicit PyG scatter reference where PyG's automatic min-size path rejects.
See the compatibility matrix for the full list.

Independent reviews covered graph helpers/layers and GATv2 ordering/aliasing,
classification derivatives, recurrent state/configuration, factories, functional
wrappers, normalize snapshots and the integrated training proof. A separate
final audit recomputed all 555 source hashes, 80 log hashes and 27 installed
wheel-file identities, reconciled all 12 command records and independently
recounted Rust/Python results and all three graph proofs. No mismatch or blocker
remained. Earlier staging
failures, stale-wheel checks and source overlays remain visible in local logs;
only the frozen installed build supplies final validation counts. There were no
source overlays in repository tests or the final suite.

No commit, PR publication or deployment was performed. CUDA tests compile, but
real GPU validation, performance ignores, the real GPT2 fixture and historical
native-crash/numerical investigations remain outstanding.
