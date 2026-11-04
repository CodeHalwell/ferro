# PyTorch and PyTorch Geometric compatibility

This ledger describes the local 2026-11-03 API and portability expansions. Numerical references
are PyTorch 2.13.0+cpu and PyTorch Geometric 2.8.1. Ferro remains an independent
Rust engine: these packages are test dependencies, not runtime dependencies.
The Rust core still has zero external dependencies.

## Active completeness work

The latest green matching checkpoint is [candidate31](../verification/completeness-2026-11-03/README.md):
1,757 Python cases pass and 30 GPU cases are unavailable out of 1,787 discovered.
Its 1,092 selected source/fixture inputs, 123 installed Python files and native
library are hash-identified. Corrected proof2 wheel/full Python/16 focused methods,
eight standalones and DLPack ran anew. Workspace/CUDA evidence is reused from
proof1 under exact557-input equality excluding the changed binding crate:
1,624 outer runtime passes plus two doctests. Optional/filtered GPU bodies and
nested summaries are not hardware evidence. Full parity remains open.

Previous [candidate30](../verification/completeness-2026-11-03/candidate30/README.md),
[candidate29](../verification/completeness-2026-11-03/candidate29/README.md) and
earlier checkpoints remain archived. Candidate31 adds CPU F64 RMSprop/Adagrad
and affine GroupNorm, with actual-Torch state/gradient/training and fresh-process
continuation controls. The first proof's eight Adaptive failures (15 indicators)
were fixed in production step validation; only training_state.rs changed before
the corrected proof. Its failures and all passing GroupNorm controls remain
recorded. The following rows describe candidate31; new adaptive/AD/spatial
experiments and known layout gaps remain outside it.

| Area | Installed candidate31 coverage |
|---|---|
| Typed compute | CPU float32/float64 arithmetic, nonlinear math, softmax, reductions and broadcast matrix products; fused alpha arithmetic and exact scalar conversions; typed rounding/frac/rsqrt and rounded division; equal/isclose/allclose; checked NumPy scalar interoperability; promotion/casts; exact int64/Bool selection/copies and DLPack. CPU F64 Parameter/Linear, Bilinear, LayerNorm/RMSNorm/BatchNorm/InstanceNorm/GroupNorm affine factories, Embedding/Bag and SGD/Adam/AdamW/RMSprop/Adagrad are supported; other factories and optimizers retain documented dtype limits. |
| Views/ordering | Tested CPU binary/copy physical layout planning; real detach/clone, typed T/mT/t/H/mH, shared requires_grad metadata, shape-only view/view_as and variadic permute/expand; CPU typed stable sort/argsort and take with tested saved-index/input versions, negative/repeated indices and supported live adjoints. Cast/Cat adjoints need no input values; saved nonlinear outputs and sort indices remain guarded. |
| Linear algebra | Solves/inverse/triangular/Cholesky/determinants/norms, QR/eigh/SVD, spectral/nuclear norms, rank/pinv, minimum-norm lstsq, tensorinv/tensorsolve, solve_ex/inv_ex, LU factor/ex/solve/decomposition, matrix_power, multi_dot, cond, Cholesky-factor solve/inverse and cross/vecdot. Ex info is I64 versus Torch int32; singular ex values/adjoints are unspecified. |
| Differentiation | Native multicolumn jacfwd with one primal evaluation; joint functional VJP traversal; genuine optional Where adjoints; direct JVP on the existing Op recorder, supported JVP/grad/jacrev composition and activation backward graphs. Single-direction entry makes real metadata/dtype copies when required and preserves compatible direction identity. Ordinary gradient publication preserves held/manual slot identity and history while detaching fresh slots. Forward transforms accept F64 outputs/directions from F32 primals. Current public Hessian and reverse transforms retain their earlier limits; nested private frames and batched higher-zero semantics, F64 primals, general reverse queries during JVP and vmap remain open. |
| Normalization/losses | CPU F32/F64 affine BatchNorm/InstanceNorm and multi-axis LayerNorm/RMSNorm with typed resets/running state and fresh-process continuation; CPU F64 affine GroupNorm and existing LRN training controls, general p/tuple-axis/F64 normalize, expanded loss options, channel Dropout1d/2d/3d and AlphaDropout/FeatureAlphaDropout. Typed Softplus beta/threshold, Softsign and LogSigmoid have staged adjoint controls. Unsupported activation in-place mutation rejects. GroupNorm F64 retains first-order CPU scope; norm higher-order boundaries and reference rounding differences remain explicit. |
| Additional neural-network catalog | PReLU with native branch-wise first reverse and direct JVP, Bilinear CPU F32/F64 factories, matching-batch ordinary gradients, typed SGD and fresh-process state; supported F32 higher/JVP compositions, and GaussianNLLLoss with cloned clamped variance and genuine reverse paths. PReLU JVP logical-axis ordering preserves tested structural-zero reverse layouts. PReLU higher reverse/capture and GaussianNLL active JVP remain explicit unsupported boundaries; Bilinear F64 higher/forward/capture and batch broadcasting remain unsupported; module parameter defaults stay F32. |
| Unary/angle repairs | Original scalar exponent precision and special square/cube/reciprocal staging; ordered Clamp NaN/signed-zero and genuine Where adjoints/layouts. Empty reductions preserve layouts up to 2^20 output elements; larger virtual zeros can differ in strides. A known power(x, 0) direct-JVP zero-tangent stride mismatch on transposed inputs remains; its fix is isolated. Tested raw physical layouts and typed F64 staged adjoints; saved-output rules for expm1/exp2/reciprocal/tan; named neg/negative aliases; typed atan2/acos. Portable F32 coefficients widened to F64 retain F32 working precision, with exact per-side staging controls. |
| Embedding | CPU F32/F64 functional lookup/bags, padding/frequency/renorm, sum/mean/max bags and live sum-mode per-sample weights. Embedding/EmbeddingBag factories accept CPU F32/F64 parameters, with typed resets, explicit pretrained sharing and fresh-process state; F64 support is ordinary eager differentiation, not higher/forward/capture/GPU coverage. Explicit pretrained whole storage sharing preserves independent leaf flags; sparse gradients reject. |
| Modules/state | Canonical custom constructors, Parameter as a native Tensor subtype, gradient assignment, Parameter containers, tensor state_dict/load_state_dict, typed optional buffers, metadata and stable tested slots. CPU F32/F64 Parameter construction still owns data; explicit shared storage is separate. Linear accepts CPU F64 and retains F32 defaults (constructor zero bias, explicit reset uniform bias). Explicit parameter/child/buffer registration, optional slot kinds, traversal/apply and local forward hooks are included; callbacks are caller-owned across checkpoints. |
| Recurrence/spatial | Packed/projected RNN/GRU/LSTM, persistent dropout, convolution/transposed convolution, pooling/padding/rearrangements. Values, input/all-parameter gradients and exact restart controls have documented dtype/layout scopes. |
| Attention/Transformers | Canonical MHA/SDPA options, masks/causal hints, separate widths, extra tokens/dropout, canonical encoder/decoder layers/stacks and full Transformer. |
| Optimizers/schedulers | Fourteen CPU F32 families including Muon, Adafactor and strong-Wolfe L-BFGS, plus native CPU F64 SGD/Adam/AdamW/RMSprop/Adagrad with full-precision options, typed moments and measured staged updates; lazy Adagrad added groups honor current defaults without state/load hook side effects; closure config refresh, sequential shared-storage leaves, retained gradient handles, cyclic/one-cycle/restart/composite schedules, exact typed-buffer/RNG continuation. Manual gradients aliasing model storage reject. Ordinary whole CPU F32 leaf Tensor inputs, F64 SGD leaves and strided manual-gradient identity are covered. Momentum/moment snapshots are contiguous and state loads require exact dtype. Adam public F32 step tensors reject counters above2^24 while native snapshots preserve exact u32 counts; higher-counter Torch trajectories are not claimed. SGD zero_grad(False) rejects graph-bearing or unsupported-layout manual gradients; default slot clearing is supported. RMSprop/Adagrad require finite options, exact-dtype loads and counters through2^24; Double RMSprop accepts alpha>1 while legacy F32 validation remains. SGD default lr=.001 and positional options match Torch; active unsupported execution flags reject. Lifecycle hooks preserve ordering/replacement/error boundaries; restore preflights active AD and immutable targets before any commit. |
| Graph models | Previous catalogs plus recurrent/set/deep/degree/quantile aggregation, Set2Set/GMT, MetaLayer/GPSConv, graph BatchNorm/DiffGroupNorm, dense spectral pooling, HeteroLinear/HeteroDictLinear, AntiSymmetricConv, VPA, MixHop, HAN, DenseGAT and public Aggregation customization. RGATConv adds tested CPU F32 relational attention modes, basis/block weights, edge features and cardinality modifications with actual-PyG gradients and fresh-process state. Its constructor safely initializes l2 instead of reading PyG uninitialized memory; later reset preserves l2/w. PyG MLP and MLP-aware DeepSets/readout routes cover tested batch-aware norms, COO/CSR, gradients, dropout and fresh-process state; reference global-index rejection behavior is retained. Twenty-one older aggregation checkpoint families restore strictly. |
| Graph normalization/geometry | Graph InstanceNorm, HeteroBatchNorm and HeteroLayerNorm with tested actual-PyG state/gradients and fresh-process continuation; live-buffer rebinding under a structure-locked Trainer remains unsupported. Center, NormalizeScale, LinearTransformation, Cartesian, Distance, Polar and Spherical preserve tested typed fields and gradients. AddRandomWalkPE follows the bounded dense PyG path (N<=2000), denominator-only weighted degree, zero-length one-column behavior, typed append and saved-data fresh-process continuation. Simple live F32 weights are differentiable; live weighted duplicate pairs and F64 edge weights reject. |
| Graph data | Data/Batch/HeteroData/HeteroBatch, dense custom cat/inc hooks, pair/bipartite offsets, follow_batch, live inverse separation, batch selectors and selected-field clone ownership, local datasets, frontier/link loaders, GraphSAINT, cluster collation and transforms. Custom dataset persistence remains unsupported. Optional PyG native sampling backends are unavailable; backend-identical finite sampling is not claimed. |
| Graph field operations | Eager Data/HeteroData node/edge subgraphs preserve selected order/repetitions and aligned typed gradients; batched/sampled/custom multi-space and additional indexed fields reject. RemoveIsolatedNodes retains sorted incident nodes across homogeneous/heterogeneous relations, typed gathered gradients, edge order and untouched aliases; batches/custom hooks/secondary indices reject, and its serialized continuation proof is same-process only. Tensor-field apply/detach/contiguous/cpu and supported to retain tested ownership and live gradients; Data/Batch/Hetero variants preserve private inverse metadata. Callbacks are sequential; detach_ remains unsupported. |
| Product reductions | CPU Bool/I64/F32/F64, all elements or one axis, explicit dtype and axis keepdim; Bool defaults I64, integers wrap, empty result one. Serial selected-dtype accumulation can differ from SIMD reference rounding. Ordinary F32/F64 reverse and supported F32 higher/JVP graphs retain saved input/output versions and measured ordinary-versus-functional invocation policies. Current reverse-over-reverse func.hessian can differ at nonfinite intermediates from Torch forward-over-reverse; capture/device paths reject. |
| Cumulative operations | Typed CPU cumsum/cumprod over F32/F64/I64/Bool, mixed precision, integer wrap, supported higher/JVP and linalg.vander; typed cummax/cummin values/int64 indices with tested tie/NaN/signed-zero and indexed adjoint behavior. Zero-containing higher cumprod has an explicit 1,048,576 element-work bound; resident GPU kernels and full transform/capture scope remain open. |

Run `python examples/parity_inventory.py --output /tmp/ferro-api-inventory.json`
after installing a wheel to find public-name omissions. Name presence does not
certify signatures, semantics, derivatives, devices or performance.

Whole-training checkpoints preserve independent parameter AD/optimizer flags,
CPU F32/F64/I64/Bool buffers and optional None slots. Loading requires supported
whole CPU destinations and assign=False. Higher-order/capture scope is
operator-specific; F64 graph-building VJPs are not established. Full dtype and
operator coverage, arbitrary advanced indexing/out destinations, half/complex
compute, FFT, vmap, compiler/export/profiling/quantization, distributed training,
ecosystem catalogs and complete GPU models remain open.

Stable formulas can differ from Torch CPU overflow/cancellation or close-gap
F32 factorization results. Retained failed comparisons, exact-input F64 controls
and finite differences define the evidence. Correlated subnormal SVD values can
be deflated below matrix precision; exact underflow rank is not guaranteed.
Truncated pinv/lstsq adjoints follow actual Torch behavior and may disagree with
finite differences across a nonzero spectral cutoff. See the matching report.

Physical-batch Jacobians, shared multi-output/nested rules, other F64 optimizers,
power-zero direct-JVP layout and generic empty accumulation/scheduling repairs
remain outside this checkpoint. Unused noncanonical empty materialized gradients
can differ in layout. The supported F64 modules/optimizers do not imply general
F64 module, higher-order optimizer or GPU training support.

## Earlier portability snapshot

The tables and restrictions below describe the earlier portability publication.
Some restrictions were lifted by the active work above; its final matching-build
report remains separate historical evidence.

### Supported additions at that publication

| Area | Public API | Supported scope |
|---|---|---|
| Construction | `ferro.tensor`, `as_tensor`, `zeros`, `ones`, `full`, `arange`, `eye` | Rectangular Python data, scalars/empty shapes, f32/f64/i64 and native bool storage; exact typed scalar access. `tensor(existing)` copies/detaches; unchanged `as_tensor(existing)` retains identity/history. Bool random/arange are unsupported. |
| Booleans/indexing | Comparison and logical functions/operators, `any`, `all`, `nonzero`, `masked_select`, `where` | Native CPU bool storage, broadcast predicates, exact bool counts, typed selections, single full/prefix bool-mask indexing and live f32 selected-value gradients. Mixed numeric promotion and general advanced-index tuples remain unsupported. Bool safetensors and CPU DLPack round trips are tested. |
| Random construction | `rand`, `randn`, `rand_like`, `randn_like` | Native f32 RNG, explicit Ferro Generator where accepted. Torch random sequences are not reproduced. |
| Metadata | `dtype`, `ndim`, `dim()`, `size()`, `numel()`, `item()`, `len(tensor)`, `bool(tensor)` | Dtype names are strings, not `torch.dtype` objects. Truth values require one element; scalar length and ambiguous truth values reject. |
| Shapes | `cat`/`concat`, `reshape`, `flatten`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `split`, `chunk`, `unbind` | Functional wrappers, negative dimensions, inferred reshape dimension, tuples of split outputs, strided views and gradients. `unbind` preserves views. |
| Reductions/products | `sum`, `mean`, `mm`, `bmm`, `matmul` | f32, tuple-axis reductions, vector/matrix products and broadcast batch matmul with gradients; scalar/empty dimensions tested. |
| Like construction | `zeros_like`, `ones_like`, `full_like` | Inherit shape/dtype/device; independent leaves with optional requires-grad. |
| Classification | `nn.functional.cross_entropy`, `nll_loss`, `nn.CrossEntropyLoss`, `NLLLoss` | CPU f32, class weights, none/sum/mean, ignore-index, spatial/unbatched input, CE label smoothing and probability targets. Supported target/weight gradients and empty/all-ignored behavior match reference cases. |
| Recurrence | `nn.RNN`, `GRU`, `LSTM`, `RNNCell`, `GRUCell`, `LSTMCell` | CPU f32, Torch parameter names/layout, stacked/bidirectional networks, batch-first/unbatched input, supplied/default states, tanh/relu RNN, input/state/all-parameter gradients. |
| Spatial layers | `nn.Conv1d`, `Conv3d`, `ConvTranspose1d/2d`, `AdaptiveAvgPool1d/2d/3d` and matching functional helpers | CPU f32 native compositions, grouped/strided/dilated, supported same/valid padding, unbatched/empty batches, transposed output sizing, exact adaptive bins and checkpoint-compatible parameters. Compatibility implementations are not fused GPU kernels. Existing Conv2d remains available. |
| Attention | `nn.MultiheadAttention`, functional scaled-dot-product attention | Batch/sequence-first and unbatched MHA, bool/additive masks, padding masks and per-head/averaged weights. Functional attention adds rank-2 input and finite custom scale; fully masked rows return zero on the SDPA path. Masks/custom scale are CPU eager only; nonzero dropout rejects. |
| Training | `optim.SGD`, `Adam`, `AdamW`, `optim.lr_scheduler` | Ordered parameter groups, owning state dictionaries and atomic CPU loads. StepLR, ExponentialLR, CosineAnnealingLR and ReduceLROnPlateau have actual-Torch trajectory tests. Scheduler state is saved explicitly as a sidecar. |
| Data pipelines | `data.Dataset`, `IterableDataset`, `Subset`, `ConcatDataset`, `random_split`, `default_collate`, `DataLoader` | Lazy synchronous Python datasets/streams, nested typed collation, custom samplers/batches/collation; synchronous native TensorDataset batches retain the Rust shuffle order. Native worker mode still materializes an epoch. No arbitrary Python multiprocessing or generic loader checkpointing. |
| Graph convolution | `graph.GCNConv`, `SAGEConv`, `GINConv` | Eager CPU f32; weighted GCN, SAGE root/project/normalize and mean/sum/max aggregation, GIN fixed/trainable epsilon. SAGE/GIN support documented bipartite inputs. |
| Graph attention | `graph.GATConv`, `GATv2Conv` | Sparse additive multihead attention, concat/mean, bipartite features, edge projection, self-loop fills, bias/residual, returned weights, and GATv2 shared weights/dynamic scoring. |
| More graph layers | `graph.GINEConv`, `APPNP`, `ChebConv`, `TransformerConv`, `RGCNConv` | Sparse CPU f32, supported edge attributes/weights and gradients; Chebyshev Laplacian modes/per-graph lambda, Transformer key/value edge projection and beta gating, relation-specific RGCN sums/means and optional bases. Unsupported caches, lazy widths, block decomposition and dropout reject. |
| Heterogeneous graphs | `graph.HeteroData`, `HeteroConv` | Typed CPU node/relation stores, dictionary routing, registered relation modules, sum/mean/min/max/cat/None aggregation, shared-module identities and checkpoints. Automatic to_hetero tracing, HGT and heterogeneous sampling are not implemented. |
| Sampling/link prediction | `graph.NeighborLoader`, `LinkNeighborLoader`, `negative_sampling`, `batched_negative_sampling`, `RandomLinkSplit` | Synchronous homogeneous CPU frontier sampling, seed-first IDs and aligned differentiable fields, sparse exact negative sampling, directed/undirected split controls. Finite fanouts follow Ferro's documented frontier algorithm and RNG; backend-identical PyG sampling is not claimed. |
| Graph reductions | `graph.degree`, `softmax`, `global_add_pool`, `global_mean_pool`, `global_max_pool` | CPU grouped reductions, optional graph count, isolated/empty groups, differentiable f32 values. |
| Topology utilities | `graph.add_self_loops`, `remove_self_loops`, `coalesce`, `to_undirected`, `subgraph` | CPU i64 `[2,E]` topology or Ferro EdgeIndex; aligned attributes and reduction/relabeling options. Subgraph accepts a bool node mask and can return an aligned bool edge mask. |
| Graph containers | `graph.Data`, `Batch.from_data_list`, `Batch.to_data_list`, `DataLoader` | Homogeneous graphs, exact i64 labels/offsets and bool mask fields, gradient-preserving collation and extraction, explicit field roles, synchronous seeded batching. |

All graph names above are available directly from `ferro.graph`, not from a
`torch_geometric` package shim. Existing `graph.GcnConv` is preserved; the new
`GCNConv` has PyG's parameter layout and options. Graph/recurrent modules use
Ferro's normal Module registration, optimizers and checkpoint format.
Numerical tests copy reference weights; matching initialization draws is not
claimed. Arbitrary Torch state dictionaries need explicit tensor conversion.

```python
import ferro as fr
from ferro import graph, nn

x = fr.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
edge_index = fr.tensor([[0, 1, 2], [1, 2, 0]], dtype=fr.int64)
layer = graph.GATv2Conv(2, 4, heads=2)
nodes = layer(x, edge_index)
pooled = graph.global_mean_pool(nodes, fr.tensor([0, 0, 0]))
pooled.sum().backward()

sequence = fr.randn(5, 2, 3)
output, (hidden, cell) = nn.LSTM(3, 4, num_layers=2, bidirectional=True)(sequence)
```

The original `Tensor(data, shape)` constructor and native method signatures are
retained. Use the new lowercase factories and functional wrappers for the
listed PyTorch-style conventions; adding `ferro.matmul` does not change every
legacy method/operator into PyTorch's overload set.

Run `python examples/pyg_parity.py` for a complete GCN/SAGE/GATv2 graph classifier
using `DataLoader`, pooling, cross-entropy and AdamW. It checks loss reduction on
held-out feature samples, an edge-removal control, and exact fresh-process
checkpoint continuation. The fixture saves graph data and loader epoch alongside
the training checkpoint; this does not add general iterator checkpoint support.

## Explicit boundaries

- Arithmetic and autograd remain f32. f64/i64 storage and shape operations do
  not imply typed arithmetic, mixed-precision training or automatic promotion.
  General advanced indexing, `out=`, distributed training,
  sparse tensor formats, AMP and general compilation parity remain open.
- New graph layers/utilities, recurrent wrappers and expanded classification
  losses are CPU eager APIs. Graph operations use ordinary first-order AD;
  their CUDA residency, capture and arbitrary higher derivatives are not
  established. Existing native CUDA operators have a separate capability scope.
- Recurrent nonzero dropout, packed sequences, LSTM projections and explicit
  dtype overrides reject. Graph attention nonzero dropout, lazy channels,
  SparseTensor adjacency and arbitrary MessagePassing options reject. Cached
  GCN normalization is unsupported. Supported attention fill reductions are
  add/sum/mean/min/max, not product.
- GCN insertion rejects weighted duplicate self-loops: PyG's indexed overwrite
  gradient for this case differs from Ferro's last-writer derivative. Ordinary
  duplicate edges, unique weighted loops and unweighted duplicate loops are
  tested. Disable loop insertion only when that is the intended model.
- Extrema ties have deliberate boundaries. PyG CPU scatter includes an initial
  zero in zero-extremum gradient ties; Ferro assigns gradients to actual input
  winners. No-batch max pooling uses native shared-tie gradients, while Torch's
  dimension max chooses its first winner. Do not expect gradient parity at
  these tied extrema. Non-tied cases and ordinary nonzero scatter ties are tested.
- Rectangular self-loop attribute reductions use the full target-node count;
  this handles a case where PyG 2.8.1's automatic min-size reduction rejects
  valid target IDs. The extension is tested against an explicit PyG scatter
  reference. GAT/GATv2 `return_attention_weights=False` returns a Tensor;
  TransformerConv returns its attention tuple for either boolean flag, matching
  the installed 2.8.1 runtime rather than assuming one shared convention.
- `Data` supports declared node/edge/graph tensor fields; `y` defaults to a
  graph field. Ambiguous custom fields require `field_roles`. Arbitrary
  metadata/custom index increments, CSR grouped softmax, asynchronous graph
  loader workers and generic iterator
  checkpoints are unsupported. Loader shuffle uses Python seed plus epoch,
  not Torch's random sequence.
- Grouped graph softmax currently supports `dim=0`; global pools accept `[N]`
  or `[N,F]`. Topology attribute reductions require f32, although `Data` can
  retain i64 and bool fields without conversion.
- Native optimizer steps consume gradients. Group membership uses ordered
  Parameter slots and add_param_group; cross-group duplicate slots reject.
  SGD/Adam/AdamW remain the available families, without arbitrary Torch options
  or direct Torch pickle/state loading. Scheduler state is an explicit owning
  state_dict sidecar, not automatically captured by the training checkpoint.
- Native TensorDataset workers retain the whole-epoch path; arbitrary Python
  callbacks use num_workers=0. Generic datasets/streams are lazy; their shuffle
  and split RNG streams are not Torch-identical. Pinned memory and multiprocessing
  Python dataset workers are not implemented. Split/link-label index fields
  reject graph Batch collation because custom node offsets are unsupported;
  LinkNeighborLoader is the supported link-supervision minibatch route.
- Native Bool storage and predicates are CPU-only and nondifferentiable.
  Selected f32 values retain gradients; typed storage/selections do not add
  bool/integer/f64 arithmetic. Whole-training model restore still targets its
  documented CPU f32 parameter/buffer contract. Saving arbitrary Bool data with
  safetensors is supported separately. Non-F32/Bool DLPack export rejects.
- Finite neighbor fanouts use once-discovered frontiers, with explicit seed IDs
  and topology metadata. All-neighbor cyclic two-layer values/gradients compare
  against full PyG models. PyG's optional pyg-lib/torch-sparse sampling backend
  is unavailable in the validation environment; backend-identical finite
  sampling is not claimed. Heterogeneous/temporal/distributed sampling remains open.
- Index-target classification rejects trainable class weights, as Torch does.
  Probability-target CE supports target and class-weight gradients. Mean over
  all-ignored or empty targets returns NaN with zero/empty gradients; a zero
  selected-weight denominator has different NaN/Inf gradient behavior and is
  tested separately. Deprecated reduction arguments are unsupported.

## Numerical repairs and validation

Log-softmax now subtracts the row maximum before subtracting the log of the
exponential sum. Large equal finite logits retain `-log(class_count)` instead
of cancelling to zero. The repair covers core, AVX2 and CUDA kernel source.

L2 normalization records a direct first-order derivative. Zero rows no longer
differentiate through `sqrt(0)`, tiny positive epsilon avoids squared-denominator
underflow, and overflowed finite-input norms yield zero outputs/gradients like
the reference. NaN epsilon rejects explicitly; numerical parity tests cover
finite inputs with finite positive epsilon, including the clamp boundary.

Tests compare values and seeded input, state, parameter and supported attribute
gradients against real PyTorch/PyG. They also cover invalid configurations,
noncontiguous/empty inputs, aliases and checkpoint compatibility. Install
`crates/ferro-py` with its `test` extra for the pinned reference packages; CPU
CI installs the CPU Torch wheel and PyG explicitly. Neither reference package
is imported by Ferro runtime modules.

See the [portability matching-build report](../verification/portability-2026-11-03/README.md)
for current validation and source identity, and the
[earlier parity report](../verification/parity-2026-11-03/README.md) for the graph
training example. [GPU targets and remaining gaps](GPU_PORTABILITY.md) records
the NVIDIA CUDA and Apple Metal work still required. These additions expand a
tested subset; they do not establish full PyTorch or PyG compatibility.
