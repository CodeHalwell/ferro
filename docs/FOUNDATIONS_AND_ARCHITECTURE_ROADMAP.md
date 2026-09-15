# Ferro foundations and architecture roadmap

> Historical baseline roadmap, not the final implementation ledger. Its missing/in-progress milestones describe the original audit. For the frozen draft scope, verified results and OPEN fastcpu blocker, see [publication status](../verification/foundation-wave/pr-readiness/PUBLICATION.md) and [PR body](../verification/foundation-wave/pr-readiness/PR_BODY.md). Remaining architecture gates are future acceptance criteria.

## Goal and evidence boundary

Build a trustworthy general-purpose differentiable engine, then prove architecture breadth with public training programs. The scope is not LLM-only: dense models, CNNs, RNNs, GNNs, connectome-constrained networks, graph transformers, KANs, PINNs, autoencoder families, GANs, and explicit extension seams for other models.

This is a planning and source-audit consolidation, not an implementation report. No builds, tests, GPU execution or commits were performed for this document. Source audits identify the baseline as branch `feat-attention-parity`, commit `43dc8ea`; the supplied concatenated label `feat-attention-parity43dc8ea` must not be treated as a separately verified branch name. Concurrent correctness workers may change the working tree. Their first foundation bundle remains **in progress**, not verified complete here.

Paths below are repository-relative. Source symbols are preferred over line numbers because concurrent edits move lines. Existing test source demonstrates intended coverage, not that a test passed on the current tree. Absence means not found in the inspected core/binding surface, not a proof about every possible external package.

### Status vocabulary

- **Present (source)**: implementation exists; runtime support still depends on backend, dtype, layout and API.
- **Partial**: a useful subset exists, or functionality is composable but not an end-to-end supported model.
- **Missing**: required public primitive/contract was not found in the inspected implementation.
- **Source-suspected defect**: source reasoning identifies a likely failure; reproduce before claiming a runtime bug fixed.
- **In progress**: explicitly assigned work, without a completion claim.
- **Verified (bounded)**: an identified execution artifact supports only its stated fixture, build and environment.
- **Deferred/unknown**: not accepted into a current bundle, or insufficient evidence to characterize support.

### Reading order and precedence

1. `CLAUDE.md`: zero-dependency core, single `Tensor::record_fn` autograd path, storage/version invariants, gradient-device contract and counting-backend tests.
2. Current implementations and executable test evidence, scoped to exact source/build identity.
3. This roadmap for priorities and architecture acceptance criteria.
4. `docs/PARITY_ROADMAP.md`, `docs/FUTURE.md`, `docs/CAPABILITY.md`, `docs/ARCHITECTURE.md` for earlier plans and design context, not blanket status authority.

Older documents mix historical gaps with later status updates. For example, PARITY_ROADMAP still lists a missing Python GPU bridge below an update saying it landed; FUTURE lists non-scalar backward as future while Rust `backward_with` exists. ARCHITECTURE describes backends as future and an older storage representation. This document does not inherit their timelines, universal parity claims or performance targets as achievements. Do not rewrite those documents as part of this document-only task.

The input audits were read from local delegation transcripts `deleg_433132be/task-0.log` (tensor/autograd), `deleg_433132be/task-1.log` (training), `deleg_99a5a7ac/task-0.log` (graph/sequence/KAN) and `deleg_53333092/task-0.log` (PINN/AE/GAN), under the Hermes `cache/delegation/live` directory. These are provenance, not portable proof artifacts. Durable source anchors and proposed regression gates are recorded below; final audit clarifications were also supplied by the coordinating agent.

## Non-negotiable distinctions

- A static execution graph or CUDA graph is not graph-neural-network topology support.
- A CUDA input accepted through CPU fallback is not a resident CUDA implementation. Uploading a host-computed output does not change this.
- Storage dtype support is not compute dtype or autograd support. Half checkpoint IO does not establish mixed-precision training.
- Retaining and replaying a first-order tape is not differentiating the backward computation. A VJP API alone is not `create_graph`.
- Dense masked connectivity can prove a connectome model mathematically, but does not prove sparse memory or compute scaling.
- An example that decreases one loss does not establish correct gradients, generalization, resumability or support for every variant in a family.
- A passing CUDA test command may have skipped all hardware tests. Record actual executed/skipped cases.

## Current foundation ledger

| Area | Status and evidence | Required next contract |
|---|---|---|
| Storage, aliases, mutation | Present (source): `crates/ferro-core/src/tensor.rs` (`StorageCell`, `detach_copy`, `accumulate_grad`), `inplace.rs`, `params.rs`; tests `version_counters.rs`, `device.rs`. Arc sharing and version checks are valuable existing infrastructure. Device detached snapshots can share buffers. | Preserve immutable cell variant/buffer identity and checked mutation seams; explicitly distinguish alias, detached handle and owning snapshot. Test stale saved inputs, tied leaves and asynchronous snapshot lifetime. |
| Shapes and typed views | Source-suspected defects/in progress: `shape.rs` (`numel`, `default_strides`), tensor constructors/materialization, I64 device view handling. | Checked products/strides/offset bounds before allocation or indexing; scalar, empty, zero-stride, transposed and oversized-shape cases; exact I64 preservation. |
| Dtypes | Partial: `dtype.rs`, `half.rs`, tensor typed constructors/casts, `safetensors.rs`; primary math and AD are F32. | Publish per-op storage/compute/accumulation/output/gradient dtype matrix. Reject unsupported math before lossy host conversion. Selection and serialization must preserve exact typed values. |
| First-order autograd | Present (source): `autograd.rs` (`build_topo`, `backward`, `backward_with`), `Tensor::record_fn`; tests `autograd.rs`, `backward_with.rs`, `version_counters.rs`. | Keep shape/arity/device checks, repeated-backward accumulation, branched DAGs, deep teardown, no-grad and mutation behavior. Add functional gradients without polluted persistent grad slots. |
| Layout adjoints | Partial/source-suspected residency issue: `tensor.rs`, `Tensor::reshape` backward reconstructs via host values on relevant paths. | Correct view adjoints without avoidable device download/upload; count transfers for noncontiguous and broadcast cases. |
| Python indexing and VJP | Missing differentiable basic indexing: `crates/ferro-py/src/lib.rs`, `PyTensor::__getitem__` explicitly makes detached F32 host copies. `PyTensor::backward` accepts no cotangent, although Rust has `backward_with`. | Recorded dtype-preserving indexing, scalar/empty/negative-step semantics and duplicate-index accumulation; expose validated non-scalar cotangents without pretending higher-order support. |
| Partial backend behavior | Source-suspected defects/in progress: `tensor.rs` reduction dispatch and `unbroadcast`; unsupported optional reduction kernels have used `expect`. | Supported device dispatch, explicit observable host fallback, or descriptive unsupported error; never backend-capability panics on valid input. |
| Placement | Partial/inconsistent: `tensor.rs::raw_binary`, `ops_ext` host loops, `dispatch.rs::Backend`. Some ops return CPU, others re-upload host results. | One documented fallback policy plus a strict-residency mode or test equivalent; distinguish transfer correctness from transfer-free execution. |
| Dense modules and initialization | Present (source): `nn.rs` (`Module`, `Linear`, initialization, containers), `modules.rs`; not a complete Python module ecosystem. | Named parameters AND buffers, nested mode/placement propagation, initialization validation, stable parameter identity, freeze semantics and shared-parameter update-once rules. |
| Losses and normalization | Partial; `ops_ext/bce_with_logits_loss.rs`, `huber_loss.rs`, `smooth_l1_loss.rs` have source-suspected custom-backward placement failures. `batch_norm.rs` singleton-eval/zero-channel validation needs correction. | CPU and real-CUDA values/input/target gradients where defined; stable extremes and valid train/eval behavior; never infer backward correctness from forward success. |
| Optimizers | Present (source): `optim.rs`, `optim_ext.rs`, `params.rs`; SGD/Adam/AdamW and related helpers exist. | Parameter-group/config serialization, freeze filtering, tied-parameter deduplication, missing-grad semantics, accumulation/clipping tests and independent multiple-optimizer state. |
| Data | Partial: `data.rs` dataset/sampler/collation/loader, `ops_ext/index_select.rs`, `cat.rs`. Typed selection/collation is in progress. | I64 labels from dataset through shuffled/remainder minibatches into CE; reproducible but distinct epochs; checkpointable epoch/cursor and worker behavior. |
| Training checkpoint | Partial/source-suspected defects: `checkpoint.rs` (`from_module`, `load_into_module`, `save_to_dir`, `load_from_dir`), `nn.rs::Module`, `rng.rs`, `modules.rs`, optimizer snapshots. | Owning snapshots; complete buffers/RNG/config/data/phase state; prevalidated transactional restore preserving placement; crash-consistent generation publication. |
| Python training API | Partial: `crates/ferro-py/src/lib.rs` exposes tensors and execution handles, not the complete Rust module/optimizer/loss/data surface. | Public ergonomic modules, losses, optimizers, state dict/checkpoint and data paths; no examples depending on private binding internals. |
| Higher-order AD | Missing: `autograd.rs::backward_with` explicitly detaches the cotangent and describes `create_graph` as later. `docs/CAPABILITY.md`, section 1.2 explains the detached-backward limitation. | Genuine graph-building gradients for a declared smooth subset; unsupported derivative paths reject loudly; mixed coordinate/parameter derivatives must reach model parameters. |

### Immediate assigned work: do not duplicate or mark complete

The coordinating task identifies `deleg4555f45a` as the first foundation bundle:

1. Tensor shape bounds, I64 views and partial-backend reduction behavior.
2. BCE-with-logits, Huber and Smooth-L1 custom backward corrections, plus BatchNorm eval validation.
3. Typed `index_select`/`cat` and I64 data-to-cross-entropy regressions.

These are **in progress** at this document's evidence boundary. Code appearing in a concurrent working tree is not a completion signal. Require regression reproduction, final diff, source identity, independent review and real test logs before changing status. They do not cover Python indexing, complete dtype policy, higher-order AD, graph primitives or checkpoint completeness.

## Checkpoint and training-state design requirements

`Checkpoint::from_module` currently collects `p.tensor()` handles, not an independently owned point-in-time copy. `load_into_module` validates and sets parameters incrementally, allowing earlier mutation before a later error; replacing tensors can also change placement/identity. `Module` exposes named parameters but no general named-buffer contract. The current container models one optimizer namespace and seed/offset fields, not arbitrary multi-optimizer training state.

`save_to_dir` publishes a tensor file and JSON sidecar by separate renames. Its comment that the pair remains atomic is not established: an interruption between replacements can pair new tensors with old metadata. Temporary files plus rename per file do not provide a transaction over the pair. Define generation-addressed immutable files and a single committed manifest/pointer, or another demonstrably crash-consistent format; test replacement semantics on Windows as well as the supported CI filesystem. State what durability guarantees require flush/fsync and what concurrent readers may observe.

Required state schema and tests:

- Model parameters, named buffers (BatchNorm running statistics, masks, codebooks), training/eval modes and tied-parameter identity metadata.
- Owning snapshots immune to subsequent optimizer steps; device-buffer copying must not silently alias mutable training storage.
- Optimizer type, hyperparameters, parameter groups/order/identity, all moments/counters and schedulers; multiple optimizers with separate steps.
- Sequential `Rng` state as well as counter-based dropout seed/offset; module-local dropout streams, sampling streams and reproducible restoration policy across devices.
- Data sampler seed/state, epoch, sample/batch cursor, remainder/drop-last configuration and prefetch policy. Restore logical consumption, not merely global step.
- Adversarial update phase and discriminator/generator counters; VQ EMA count/sum/codebook state; gradient accumulation phase and pending gradients if mid-accumulation checkpoints are supported.
- Validate names, duplicates, versions, dtype, shape, optimizer configuration and destination placement before changing any live object. Failed restore must leave every parameter/buffer/optimizer unchanged.
- Test uninterrupted versus interrupted continuation, including losses, weights, next sample IDs, next stochastic draws and optimizer state; test checkpoint creation followed by more training before save.

`requires_grad_(false)` creates a fresh leaf identity; calling it on a retrieved parameter tensor is not sufficient to freeze the `Param` slot. Define a module/parameter freeze API and optimizer filtering explicitly. Deduplicate tied parameters by stable identity for optimizer steps while allowing gradients from every use to accumulate. These are prerequisites for tied AEs, alternating GAN updates and constrained models.

## Architecture coverage and minimum proofs

All rows below describe source-supported starting points and future gates, not completed training certifications. Every model gate includes the common gradient, numerical, placement and checkpoint protocol later in this document. CPU and CUDA are separate statuses; variants inherit only primitives actually covered.

| Family | Present/partial starting point and source anchors | Missing/shared dependencies | Minimum architecture proof |
|---|---|---|---|
| Dense regression/classification | `nn.rs`, `optim.rs`, `data.rs`, loss ops; Rust building blocks exist. | Python training exposure, typed labels, state completeness. | Public minibatched nonlinear regression and I64-label MLP classification; held-out metric, input/parameter gradients and restart trajectory. |
| CNN | `ops_ext/conv2d.rs` im2col+GEMM, `max_pool2d.rs`, `avg_pool2d.rs`, functional `batch_norm.rs`; `modules.rs::Conv2D`. `crates/ferro-core/examples/train_classifier_cnn.rs` is an existing synthetic CPU-oriented starting fixture, not a new execution result. | Conv groups/dilation/depthwise, additional dimensions/transpose/interpolation as demanded; resident CUDA forward/backward; reconcile module rank-2 BatchNorm with rank-2/rank-4 functional API. | Conv -> BatchNorm -> activation -> pool -> classifier; gradients for input/filter/bias/affine parameters, train/eval statistics, singleton eval, held-out images and resumed training. Add grouped/dilated fixtures separately. |
| RNN/GRU/LSTM | Dense matmul, activations, shared `Param`, indexing/cat can express explicit unroll. No dedicated scan/recurrent module found. | Differentiable slicing, recurrent state API, sequence masking/lengths, optional scan/checkpoint policy; no claim that `cumsum` is general scan. | Short sequence prediction with variable lengths; parameter gradients against an independent unroll, early-timestep input gradients, state reset and truncated-BPTT detach semantics; checkpoint hidden state when streaming. |
| GNN | `ops_ext/gather.rs`, `scatter_add.rs`, `index_select.rs` provide first-order dense indexed pieces, mainly host paths beyond restricted row gathering. | Resident indexed adjoints, segment sum/mean/max/softmax, edge/index validation, sparse topology/algebra and graph batching. | GCN/message-passing model on tiny known graphs; dense-reference values/gradients, node permutation equivariance, isolated nodes, duplicate edges and batched-versus-separate agreement; save topology metadata. |
| Connectome-constrained | Dense masked learned weights and recurrent dynamics are conceptually composable. | Fixed/learnable topology contract, sign/strength constraints and projections, sparse representation, data provenance, recurrent state. | Fixed wiring with trainable edge strengths; forbidden edges remain exactly absent/zero after backward, optimizer and reload; directed/sign-constrained tests, topology identity and no unsupported claim of biological realism. |
| Graph transformer | Dense `bmm`, softmax and composed attention exist; graph execution facilities do not supply sparse graph attention. | Edge-wise Q/K scoring, edge features/bias, stable segment softmax, gather/scatter/SDDMM/SpMM, ragged batching and masking. | Tiny edge-restricted attention versus dense masked reference; gradients to Q/K/V and edge attributes, disconnected graphs, empty neighborhoods, permutation equivariance and restart. Assert no dense N-by-N allocation for the sparse path. |
| KAN | Broadcast arithmetic, powers/trigonometric ops, parameters and sums can express basis models. `ops_ext/sin.rs` illustrates host fallback. | Edge-wise basis API/contraction, knot lookup, spline recurrence and boundary rules, learnable-knot policy and resident implementations. | True edge-wise map `y_j = sum_i sum_k c[j,i,k] B_k(x_i)`; independent edge coefficients and input/coefficient gradients on nonlinear regression. Polynomial/Fourier first, splines separately; a shared activation MLP is not a KAN. |
| PINN | Smooth dense networks and first-order VJPs exist. | Genuine higher-order/functional AD, mixed derivatives, stable coordinate scaling and preferably real F64 compute/AD for demanding residuals. | Analytic derivative fixtures then a small ODE/PDE with boundary/initial conditions; differentiate residual loss into parameters, compare residual and parameter gradients to reference, held-out residual grid and resumed optimizer state. |
| Basic/tied/denoising/sparse AE | Encoder/decoder dense ops and reconstruction losses are composable, not end-to-end demonstrated here. | Shared identity/update-once; corruption RNG; sparsity penalties/reductions; decoder shape utilities for convolutional forms. | Reconstruct held-out structured data; encoder/decoder and tied-weight gradient parity, no identity-only bypass, controlled corruption, separate reconstruction/regularizer metrics and restart. |
| Gaussian VAE, beta-VAE, conditional VAE | Reparameterization and analytic Gaussian KL are composable from exp/arithmetic/reductions. `ops_ext/kl_div_loss.rs` is discrete probability KL, not Gaussian KL. | Sampling/RNG ownership, conditioning API, reduction/scaling convention and complete stochastic resume. | Fixed-epsilon reference `z = mu + exp(logvar/2) * epsilon`; nonzero correct mu/logvar/decoder gradients, reconstruction and KL tracked separately, held-out ELBO trend and next-sample reproducibility. |
| VQ-AE/VQ-VAE | Distances, argmin, embedding/indexing and detach permit a prototype. | Explicit straight-through estimator, commitment/codebook losses, typed lookup residency, EMA state and dead-code policy. | Nearest-code assignments versus exhaustive reference; prescribed STE gradients, codebook/encoder update isolation, EMA resume equivalence and utilization metrics. STE/EMA is not a higher-order-AD requirement. |
| Contractive AE | Reconstruction is composable. | Differentiating a Jacobian-norm penalty requires higher-order AD. | Analytic small-network Jacobian penalty and parameter gradient versus reference, then reconstruction-plus-penalty training and restart. |
| Basic/conditional GAN | Dense generator/discriminator and BCE-with-logits or other explicit objectives are composable. | Loss-backward fix, freeze/identity semantics, separate optimizers, update phase and RNG checkpoints. | Low-dimensional known distribution; alternate D/G updates with exact parameter-isolation checks, discriminator input gradients reaching G, distribution metrics rather than loss-decrease alone, resume both optimizers and phase. |
| WGAN-GP and derivative-regularized GANs | Base critic/generator can compose. | Gradient penalty must differentiate input gradients back to critic parameters; true higher-order AD. Spectral normalization has its own state/linear-algebra requirements. | Analytic critic penalty and mixed-gradient check, then tiny distribution experiment with separately logged transport objective/penalty; full phase/RNG restart. |

### Autoencoder breadth is a dependency map, not a turnkey promise

The family cannot be closed by certifying one vanilla AE. Convolutional AEs depend on CNN/decoder shape work; recurrent/sequence AEs on masking/state/unroll; graph AEs on graph batching and indexed/sparse algebra; masked AEs on differentiable gather/scatter and reconstruction masks. Denoising and sparse AEs need corruption/regularization contracts. Conditional/hierarchical VAEs need explicit conditioning and latent-state structure. Adversarial AEs inherit GAN phase/isolation requirements. Contractive AEs inherit higher-order requirements. Flow-based posteriors, discrete latent estimators, diffusion/score AEs and other research variants remain separately scoped extensions, not blanket support claims.

### PINN derivative-order warning

Even a first-derivative strong-form residual such as `r = du_theta(x)/dx - f(x)` needs a mixed coordinate/parameter derivative when optimizing `r^2` with respect to theta. Existing first-order `backward_with` does not establish this. Second-coordinate-derivative PDEs require correspondingly deeper differentiable derivative composition. Test the final parameter gradient, not just whether a displayed coordinate derivative looks correct. Numerical/weak-form residual approximations can be separate supported formulations, but must not be presented as solving the missing strong-form AD contract.

## Dependency DAG and priority policy

```text
B0 correctness + typed data + first-order safety
 |--> B1 public tensor/training API + placement + parameter identity
 |      |--> B2 complete resumable state
 |      |--> B3 resident indexing + segment algebra
 |      |      `--> B4 sparse algebra + graph batching --> GNN/connectome/graph transformer
 |      |--> B5 vision + recurrent/basis primitives --> CNN/RNN/KAN/structured AE
 |      `--> B6 functional VJP + true higher-order subset --> PINN/contractive AE/WGAN-GP
 `------------------> numerical/error-contract gates for every bundle
B1 + B2 --> basic AE/Gaussian VAE/VQ-AE/basic GAN proofs
B2 + each relevant primitive bundle --> B7 architecture certification + API documentation
All correctness/residency/model gates --> measured optimization experiments (deferred)
```

B3 does not require sparse storage to prove segment semantics; B4 builds on that proof. Basic models need not wait for higher-order AD. Higher-order CPU correctness need not wait for sparse CUDA kernels. State design and functional-AD design can proceed concurrently after contracts are reviewed, but shared core files require explicit ownership. Prioritize silent wrong values/gradients/state and public-API breaks over new model classes, then architecture-enabling primitives, then optimization. Do not let attention-specific work consume the foundational acceptance budget.

## Reviewable PR bundles

These are coherent workstreams with approximately seven or eight acceptance tasks each, not instructions to submit enormous single diffs. Split a bundle into stacked PRs when necessary; retain the bundle gate and dependency order. Every implementation task starts with a failing regression, proceeds to minimal change, then full applicable regression and independent review. No dates or throughput estimates are assigned.

### B0 - Immediate correctness and typed data (priority P0, in progress)

Likely files: `shape.rs`, `tensor.rs`, the three loss files, `ops_ext/batch_norm.rs`, `index_select.rs`, `cat.rs`, `data.rs`, and matching `crates/ferro-core/tests` fixtures.

1. Check shape products, stride arithmetic and address bounds before use.
2. Preserve exact I64 device-view ordering/materialization, including empty/noncontiguous cases.
3. Remove unsupported-reduction panics from valid partial-backend paths and broadcast adjoints.
4. Reproduce/correct custom loss backward placement for BCE-with-logits, Huber and Smooth-L1.
5. Separate BatchNorm training sample requirements from eval validation; reject invalid zero-channel input safely.
6. Make typed selection/cat preserve supported data dtypes without F32 round trips.
7. Exercise I64 dataset -> shuffle -> remainder collation -> CE -> backward/step.
8. Independent review plus CPU, counting-backend and actual CUDA regression evidence; hand off remaining defects explicitly.

### B1 - Public contracts, placement and training surface (P0/P1)

Likely files: `tensor.rs`, `autograd.rs`, `params.rs`, `nn.rs`, `modules.rs`, `optim.rs`, `optim_ext.rs`, `dispatch.rs`, `crates/ferro-py/src/lib.rs`, binding tests/examples.

1. Publish and enforce a minimal strict compute/storage dtype matrix; test large integers and F64 values that cannot survive F32 conversion.
2. Replace Python basic indexing detachment with recorded selection and explicit scalar/empty/negative-step semantics.
3. Bind validated `backward(cotangent)`/functional first-order VJP behavior without claiming `create_graph`.
4. Remove reshape-adjoint host round trips where a device path is claimed.
5. Unify observable fallback/error/placement policy and add capability/transfer accounting.
6. Define named buffers, train/eval/placement propagation, parameter freezing and tied-identity deduplication.
7. Expose public modules, optimizers, losses and data composition in Python with regression/classification parity examples.
8. Review public error/compatibility behavior, stale saved-input safety and independent CPU/CUDA coverage.

### B2 - Checkpoint and reproducible training transactions (P0/P1)

Likely files: `checkpoint.rs`, `safetensors.rs`, `params.rs`, `nn.rs`, `modules.rs`, `rng.rs`, `data.rs`, optimizer state implementations and `tests/checkpoint.rs`.

1. Specify versioned schema, ownership, alias metadata and compatibility errors.
2. Snapshot owning parameter/buffer/optimizer values, including tied parameters once.
3. Capture full optimizer groups/configs, schedulers and multiple optimizer namespaces.
4. Capture sequential and counter RNG streams, buffers/modes, EMA state and update phase.
5. Capture epoch/cursor and define prefetch/accumulation resume boundaries.
6. Prevalidate complete restore and preserve destination placement/identity or explicitly invalidate compiled handles.
7. Publish crash-consistent generations; fault-inject every write/publication boundary and concurrent-reader case.
8. Prove deterministic and stochastic interrupted/uninterrupted continuation on CPU and supported CUDA paths.

### B3 - Indexed and segment execution (P1)

Likely files: `ops_ext/gather.rs`, `scatter.rs`, `scatter_add.rs`, `index_select.rs`, new segment op files/tests, `dispatch.rs`, `crates/ferro-cuda/src`, Python bindings.

1. Specify I64 index bounds, rank/broadcast rules and topology validation lifetime.
2. Provide general resident gather/index-select and correct duplicate-index adjoints, not only embedding-row special cases.
3. Provide resident scatter-add with explicit deterministic/nondeterministic accumulation policy.
4. Add segment sum/mean with unsorted IDs, explicit segment count and empty semantics.
5. Add segment max with empty/tie/NaN and gradient policy.
6. Add stable segment softmax and backward, including empty neighborhoods and extreme logits.
7. Bind primitives and test dense reference, permutation, repeated indices, negative/error and empty cases.
8. Count transfers/allocations in forward and backward; avoid repeatedly downloading static topology for validation.

### B4 - Sparse and graph training (P1, after B3)

Likely new modules: `crates/ferro-core/src/sparse.rs`, graph-data module and sparse op files (proposed, not present); dispatch/backend implementations, Python graph API and tests.

1. Define validated COO/CSR shape/index/value contracts, duplicate coalescing and immutable topology ownership.
2. Implement CPU conversion/coalescing with exact index checks; gradients apply to values, not integer topology.
3. Implement SpMM and value/dense-input adjoints against dense reference.
4. Implement SDDMM and adjoints for sampled edge scores; define transpose/layout behavior.
5. Add resident CUDA sparse/indexed paths with declared dtype, determinism and memory behavior.
6. Add disjoint graph batching, offsets, node/edge features, graph IDs, pooling and no-cross-graph-leak tests.
7. Certify tiny GNN, fixed-connectome and sparse graph-transformer training/restart fixtures.
8. Independent topology/permutation/isolated-node/duplicate-edge review; prove no hidden dense adjacency allocation.

### B5 - Vision, sequence and basis breadth (P1/P2, independently splittable)

Likely files: convolution/pooling/normalization ops, `modules.rs`, new recurrent/basis modules, dispatch/CUDA kernels, architecture examples/tests.

1. Unify functional/module normalization shape and train/eval/buffer semantics.
2. Extend convolution groups/dilation/depthwise with input/filter/bias adjoints and checked output shapes.
3. Add resident CUDA conv/pool/norm paths; choose cuDNN versus native lowering by correctness/maintenance evidence, not presumed speed.
4. Scope decoder operations (transpose convolution/interpolation) and conv dimensional variants to explicit workloads.
5. Implement recurrent cells and explicit unroll with shared weights, lengths/masks, hidden state and truncated-BPTT contract.
6. Introduce scan only with declared state/shape/autograd semantics and an equivalent-unroll oracle.
7. Implement edge-wise polynomial/Fourier KAN reference, then separately validated splines/knots and residency.
8. Certify CNN, recurrent predictor, KAN and structured-AE examples with all gradient/state gates.

### B6 - Differentiable derivatives and scientific training (P1, not optional for PINNs)

Likely files: `autograd.rs`, `tensor.rs::accumulate_grad`, `ops.rs`, selected `ops_ext`, `testkit.rs`, Python bindings, new derivative and scientific-model tests.

1. Specify functional gradients, cotangent handling, unused inputs, accumulation and graph lifetime independently of persistent `.grad` mutation.
2. Add genuine `create_graph` mode while preserving the single recording seam and first-order fast path.
3. Implement a declared smooth subset: arithmetic, matmul, sum/broadcast/layout and chosen smooth activations; retain original-input dependence safely.
4. Make gradient accumulation differentiable in this mode; prevent cycles and reject unsupported derivatives rather than detach silently.
5. Add analytic second/mixed derivatives and HVP/finite-difference-of-gradient reference tests; check parameter gradients through coordinate derivatives.
6. Plan real F64 compute/AD as a distinct extension, not a storage cast workaround; document conditioning limitations meanwhile.
7. Certify strong-form PINN, contractive AE and gradient-penalty critic fixtures, including required derivative depth.
8. Review mutation/lifetime/error behavior independently; add JVP/jacobian utilities only with their own coverage, not by implied support.

### B7 - Architecture certification and extensibility (P1/P2, incremental)

Likely new files: `crates/ferro-core/tests/architecture_*.rs`, `crates/ferro-py/tests/test_architecture_*.py`, public examples and `verification/foundations/` artifacts (all proposed).

1. Establish shared public-API training fixture runner, deterministic data, reference parameter import and evidence manifest.
2. Certify dense/CNN/recurrent/KAN fixtures as their primitive bundles land.
3. Certify graph/connectome/graph-transformer fixtures with topology and residency assertions.
4. Certify basic/tied/denoising/sparse AE, Gaussian/conditional VAE and VQ/EMA fixtures individually.
5. Certify GAN alternating-update isolation, multi-optimizer phase resume and distribution metrics.
6. Certify PINN/contractive/WGAN-GP only after genuine derivative-order gates pass.
7. Publish operator extension procedure: validation, recorded forward/VJP, derivative-order declaration, backend fallback, metadata/capture support and reference tests; keep external dependencies out of core.
8. Update capability matrix per fixture/API/device/dtype/order, obtain independent spec and code-quality review, and leave unsupported variants explicitly open.

## Acceptance gates and runnable entry points

### Existing repository commands (run by the implementation owner, not executed here)

From repository root, after confirming the selected tree and build ownership:

```bash
cargo test -p ferro-core
cargo test -p ferro-fastcpu
cargo check -p ferro-cuda
cargo test -p ferro-cuda
cargo test -p ferro-core --test backward_with
cargo test -p ferro-core --test version_counters
cargo test -p ferro-core --test device
cargo test -p ferro-core --test checkpoint
cargo run -p ferro-core --example train_classifier_cnn
```

Python is a standalone maturin crate; use the selected project environment and record the actual imported extension path/hash. On this Windows host the available interpreter command is `python`, not `python3`. Build and GPU jobs must be serialized under the coordinating owner's lock:

```bash
cd crates/ferro-py
python -m maturin develop --release
cd ../..
python examples/ops_vs_torch.py
python examples/py_regression.py
python examples/safetensors_vs_python.py
python -m unittest discover -s crates/ferro-py/tests -p 'test_*.py'
```

These commands are entry points, not promised passing results or a complete future architecture suite. Confirm environment dependencies (maturin, torch, safetensors, CUDA runtime) before running. Existing `CLAUDE.md` shows example paths relative to its binding instructions that should be resolved against actual repository layout rather than copied blindly.

Create the proposed architecture fixtures before claiming these future gate commands exist:

```bash
cargo test -p ferro-core --test architecture_dense
cargo test -p ferro-core --test architecture_cnn
cargo test -p ferro-core --test architecture_recurrent
cargo test -p ferro-core --test architecture_graph
cargo test -p ferro-core --test architecture_connectome
cargo test -p ferro-core --test architecture_graph_transformer
cargo test -p ferro-core --test architecture_kan
cargo test -p ferro-core --test architecture_pinn
cargo test -p ferro-core --test architecture_autoencoders
cargo test -p ferro-core --test architecture_gan
python -m unittest discover -s crates/ferro-py/tests -p 'test_architecture_*.py'
```

The autoencoder and GAN targets must enumerate individual variants, not report one generic family pass. Add real-CUDA counterparts in `crates/ferro-cuda/tests` for every claimed resident model, alongside CPU counting-backend fixtures. Missing targets are work, not evidence. No CUDA acceptance may be inferred from a skipped test or CPU reference run.

### Common numerical and structural protocol

- For each differentiable primitive, compare forward values and each relevant input/parameter gradient to an independent reference and finite differences via `ferro_core::testkit::grad_check`. Use O(1) smooth inputs away from ties/kinks for grad checks; add separate defined kink/tie cases.
- Test scalar, empty, noncontiguous, broadcast, duplicate-index and invalid-shape/dtype/device cases. Fuzz dimensions and layouts; large I64 values must compare exactly without float conversion.
- Predeclare tolerances by operation, dtype, accumulation length and oracle. Report absolute/relative error and ULP where meaningful; do not transplant attention tolerances to all operators or loosen thresholds after seeing a failure without explanation.
- Cover overflow/underflow, NaN/Inf, extreme logits, zero/one-sized axes, singular residuals, reduction cancellation and masked/empty neighborhoods with explicit behavior. Stable logsumexp/logits losses are correctness features, not only optimizations.
- Assert selected backend/kernel dispatch, gradient placement, upload/download counts, synchronizations and allocation behavior. Count forward and backward separately; exclude explicitly documented setup/readback only. Serialize process-global counting backends with poison-tolerant mutexes.
- Compare CPU reference, optimized CPU and actual CUDA independently. Capture GPU identity, driver/runtime, dtype/TF32 settings, source and build hashes, exact commands/exits, skip counts and raw results.
- For every architecture, check gradients before optimizer steps; then show a predeclared held-out learning criterion over fixed seeds against a meaningful baseline. Save/reload in a fresh training object and compare continuation, not only inference output. Stochastic models additionally compare RNG/EMA/phase state.
- Keep diagnostic scalar readbacks outside strict resident-step assertions. A performance claim requires synchronized timing and same-session A/B with equal work/grad modes, warmups, repeated samples and disclosed environment noise.
- Independent reviewers inspect spec compliance and code quality separately from the author. Require negative fixtures that actually fail if a claimed safety/residency/parity property is removed.

## Numerical, systems and exotic-extension backlog

These domains remain visible, but are not silently included in the first bundle:

| Domain | Next decision/gate |
|---|---|
| Full dtype semantics | F64 math/AD, integer arithmetic, Bool/index policy and explicit promotion; F16/BF16 compute, accumulation, autocast and loss scaling each need separate coverage. AMP/storage files alone do not certify training. |
| General tensor algebra | N-D broadcast matmul, einsum/tensordot, diagonal/triangular operations, sorting/search and richer views; prioritize a workload and gradient contract, not API-count parity. |
| Scientific linear algebra | Solve/decompositions/eigen/SVD, conditioning, complex values and differentiable solvers remain separately audited work. PINN support does not imply differentiable ODE/PDE solver support. |
| Higher-order beyond initial subset | JVP, jacrev/jacfwd, batched transforms, custom higher-order VJPs, implicit differentiation, differentiable optimizers/meta-learning and neural ODE adjoints each require explicit designs. |
| Probability/generative long tail | General distributions, flows/log-determinants, discrete estimators, diffusion objectives and adversarial normalization/regularizers; compose only after estimator and RNG contracts are declared. |
| Exotic graph/neural models | Hypergraphs, heterogeneous/temporal graphs, equivariant/geometric nets, spiking/surrogate-gradient models, reservoir/liquid models and neural operators are unverified extension directions, not current capabilities. |
| Data scale | Ragged/packed data, transforms, async/pinned staging, multiprocessing/distributed samplers, streaming/error recovery and deterministic prefetch; preserve core dependency boundary. |
| Distributed training | Inspect `ddp.rs` and backend integrations independently; no inference here of production NCCL/multi-node correctness. Multi-device reductions, failure handling, state sharding and distributed resume need separate gates. |
| Compilation/capture | `capture.rs`, `graph.rs`, fusion/static graph and CUDA capture work remain bounded execution facilities. Unknown ops must reject or explicitly fall back; mutable buffers, RNG and restore invalidation need tests. No automatic whole-training compiler claim. |
| Interop/deployment | DLPack ownership/streams/producer behavior, non-CUDA backends, export, quantization, edge/wasm and serving are independent audits. Existing interop support is not universal zero-copy support. |
| Observability and safety | Anomaly detection/hooks, source op diagnostics, memory/lifetime stress, reproducibility manifests and fuzzing for hostile shapes/files; retain zero-dependency core and explicit unsupported errors. |

## Attention evidence: preserve, do not extrapolate

`verification/attention-wave/baseline/REPORT.md` records a rebuilt measurement at source `43dc8ead5e4f2460efc7ba5bca14c2e06c3a5603` with native binding SHA256 `b157d8c18467c3a3c8ce2e936b3ab6299c3f30c9c9fd08e0eb3f6428bc41a788`. It is bounded verified evidence from the archived run, not a new run by this document's author.

The report covers composed rank-3 attention and selected full-model inference/replay fixtures, F32 with TF32 disabled, changed inputs/weights and explicit tolerances. It does not establish Rust legacy SDPA parity, fused attention, training gradients, all mask semantics or a full integration-suite pass. Fully masked rows/infinities/extreme logits were not covered. Real torch.compile attempts were blocked by missing Triton. The display-attached RTX 3090/WDDM runs have substantial timing/order noise and ranking reversals; no universal speedup or stable bottleneck fraction follows.

Keep the raw baseline immutable. A later scale/mask/softmax experiment may be justified after foundational gates, but must retain composed fallback, define masks, prove structural execution and rerun same-session A/B. Dense attention optimization is deferred behind general correctness, training state and architecture-enabling primitives, not abandoned or declared complete.

## Update and completion protocol

For each bundle/architecture row, maintain a small evidence record: owner, status, exact commit plus dirty-source manifest if relevant, touched paths/symbols, implemented API/device/dtype/layout/derivative subset, negative regression, commands/exits and executed/skipped counts, reference/tolerances, raw artifact path, reviewer decisions and remaining exclusions.

1. Reconcile concurrent worker reports against final diffs before editing status.
2. Change source-suspected -> reproduced only with a failing fixture; in-progress -> verified only with reviewed implementation and passing relevant gates.
3. Keep source presence, CPU verification, CUDA correctness, CUDA residency, Python exposure, model training and restart as distinct columns/claims.
4. Record blockers honestly (missing hardware/dependency, unimplemented fixture, invalidated build) and rerun after source changes. Do not carry old hashes forward as current proof.
5. Update older contradictory documentation in a separately owned documentation change after implementation evidence exists.
6. Close a bundle only for its declared acceptance surface. Keep deferred domains and unsupported architecture variants open; never mark the general engine or all model families globally 'done'.
