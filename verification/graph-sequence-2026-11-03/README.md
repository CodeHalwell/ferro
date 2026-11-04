# Sparse attention, streaming state and pipeline continuation, 2026-11-03

Base commit remains `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. This is an
uncommitted continuation of the [generative training work](../adversarial-2026-11-03/README.md).
The [manifest](manifest.json) records 538 selected source/config/lock inputs,
rebuilt wheel, installed extension and local logs. Prior reports retain their
own source identities and results.

## Public sparse edge attention

Rust `gnn::edge_attention` and Python `ferro.graph.edge_attention` implement
single-head scaled dot-product attention restricted to directed edges. Queries
and keys have shape [N,D], with D>0; values have shape [N,Dv], and optional
additive bias has shape [E]. Destination queries attend source keys/values.
Duplicate edges remain independent softmax entries; missing destinations return
zeros. Empty topology, zero nodes and zero value width have defined empty/zero
outputs and correctly shaped adjoints.

The implementation composes existing index selection, multiplication, reduction,
per-destination segment softmax and segment sum. It introduces no dependency or
second autograd mechanism. CPU f32, eager first-order gradients are supported;
capture and unsupported higher derivatives reject. Shape, dtype, device and
derived allocation capacity are checked before computation. Nonfinite computed
logits reject through the segment-softmax contract.

Eight Rust tests compare a separate f64 dense reference, strict finite differences
for every operand, duplicate bias gradients, node/edge permutations, disjoint
batches, empties and invalid requests. Six public Python tests compare
PyTorch forward/input/edge-bias gradients, strided inputs and the same structural boundaries.
Independent review found no blockers.

A thread-local allocation regression measures the sparse forward and backward
with 4096 nodes, 128 edges, query/key width four and value width three. It requires
maximum allocation <=256 KiB and aggregate allocated bytes <8 MiB. A single
dense 4096-by-4096 f32 matrix would require 64 MiB. Inputs are constructed before
measurement; forward, backward and their intermediates are included. This proves
the scoped implementation avoids hidden dense adjacency, not CUDA residency or
a performance ranking.

## Learned routing through query/key scores and edge attributes

Run `python examples/train_restart_graph_attention.py`. Each synthetic graph has
one signed query and four candidate neighbors crossed by key sign and edge-
attribute sign. The label comes from the value whose key matches the query and
whose edge attribute is positive. Paired examples have balanced value patterns,
so query-local information and uniform aggregation cannot solve the routing.

The 32 training and 32 held-out graphs use disjoint continuous feature/attribute
samples, 5-8 nodes with rotated indices and isolated decoys, and 4 or 8 incoming
edges. Some duplicate complete neighborhoods. Topology templates can repeat;
unseen-topology generalization is not claimed. Six parameter tensors implement
three 3-to-2 Q/K/V projections, a 1-to-1 edge-attribute projection and a two-class
readout. One explicit Generator supplies Gaussian feature/attribute jitter 0.002.

Before training, the design declared seeds 7, 11 and 43, 400 AdamW steps, lr 0.03,
zero weight decay, batch size eight and checkpoint step 197. Gates are held-out
accuracy >=95%, CE <=0.05 and <=25% of its initial value. Removing query scores,
edge attributes, all edges or learned weighting, or rewiring routing endpoints,
must yield <=60% accuracy and a drop of at least 35 percentage points. All three
runs pass on the first declared configuration, without tuning or gate changes:

| Seed | Initial accuracy | Final accuracy | Final held-out CE |
|---|---:|---:|---:|
| 7 | 50% | 100% | 0.0000962913 |
| 11 | 50% | 100% | 0.0000591204 |
| 43 | 50% | 100% | 0.0000275970 |

Query removal, edge-attribute removal, edge removal and uniform attention each
score 50%; swapping attribute-routing endpoints scores 0%. The rewiring preserves
node degrees and feature/attribute multisets while reversing the selected label.
Initial CE is approximately 0.693. Before updates, independent dense PyTorch
attention and cross-entropy check inputs, edge attributes and all parameters at
atol 2e-5 / rtol 2e-4; maximum absolute error is 2.24e-8. Q/K/edge-score gradients
are explicitly nonzero, and detached-score faults fail.

All remaining losses, model/AdamW/configuration/Generator state, next draws,
metrics and cyclic cursor match after a fresh-process restart at step 197
(next graph eight). The worker reads actual saved node counts, topology,
attributes, features and labels. Nine tests include learning, disjoint batching,
duplicate-edge invariance, isolated-node output, corruption/configuration,
no-regeneration checks, gradient faults and optimized-Python acceptance gates.
The full seed-7 CLI also passes under `python -O`.

## Streaming LSTM with saved hidden and cell state

Run `python examples/train_restart_streaming.py`. A CPU f32 LSTM of hidden width
eight and a two-class readout learn the sign of an initial cue after eight blank
tokens. There are 32 balanced training and 32 disjoint held-out sequences, each
split into three chunks of three tokens. Hidden/cell state resets at episode
start, then is detached and carried across chunk optimizer boundaries. Six
parameter tensors train with one explicit Generator supplying input noise 0.002.

The unchanged predeclared design uses seeds 7, 11 and 43, 600 AdamW chunk updates,
lr 0.02, weight decay 0.0001, batch size eight and checkpoint step 293. Each
continuation chunk must score >=95%; final-chunk CE must be <=0.05 and <=10% of
initial CE. Resetting state each chunk must remove at least 40 accuracy points.
At the checkpoint itself, saved state must classify the pending blank chunk at
>=95%, while erased state scores 50%. All runs pass on the first configuration:

| Seed | Initial final-chunk CE | Final final-chunk CE | Carried-state accuracy | Erased-state accuracy |
|---|---:|---:|---:|---:|
| 7 | 0.68289 | 0.0000565052 | 100% | 50% |
| 11 | 0.68661 | 0.0000374317 | 100% | 50% |
| 43 | 0.64706 | 0.0000224113 | 100% | 50% |

Initial final-chunk accuracy was already 100% for these seeds. Learning is established by
the predeclared CE reduction, not an accuracy improvement. The fixed blank
suffix and balanced classes make state erasure a meaningful control.

Independent manual PyTorch unrolls compare logits, input/initial-h/c gradients
and every parameter gradient for a single chunk, a live two-chunk bridge,
detached carry and native truncate=3; maximum absolute error is 2.98e-8. Detached
and truncated history gradients are exactly zero while shared parameter
gradients remain live. A separate reviewer used `torch.nn.LSTM`, agreeing
within 1.94e-8, and independently checked prefix-free continuation.

Checkpoint 293 is episode 97, pending chunk index two, with next sequence batch
starting at sample eight. It saves actual nonzero h/c as typed F32 [8,8]
safetensors alongside the model/optimizer/configuration/Generator checkpoint,
data and sequence identifiers. This state was computed before the preceding
optimizer update; recomputing the prefix with updated weights is not equivalent.
The worker validates shape, type, finite values, hashes, cursor and reset/detach
policy before constructing/restoring the model and optimizer, then resumes without
regenerating data or the prefix.

All remaining losses, h/c, complete model/optimizer/RNG state, metrics and
subsequent draws match exactly in a fresh process. Nine tests cover learning,
typed/corrupt/nonfinite carry, configuration and cursor binding, prefix-free
restart, zero-state and gradient-boundary faults, and optimized-Python gates.
This is restart between complete chunk updates, not mid-backward serialization,
arbitrary streaming data-loader state or a general bounded-memory guarantee.

## Attention and training-pipeline repairs

Review found and reproduced these production defects:

- Causal SDPA used a finite -1e9 mask. For q=[[[1e5]]], k=[[[0],[1e4]]],
  v=[[[3],[7]]], it returned 5 with value gradients [0.5,0.5], although only the
  first key is allowed. The repaired -infinity mask returns 3 and [1,0], matching
  PyTorch. Nonsquare causal alignment remains upper-left. Python multihead
  returned weights use the same exclusion and place their mask on the score
  device. This covers finite computed logits, not arbitrary nonfinite inputs.
- Rank-4 functional attention flattened mismatched batch/head axes into equal
  products: q [2,2,1,1] and k/v [1,4,1,1] could be paired silently. Matching axes
  now validate before folding. Broadcasting/GQA is not added to this functional
  wrapper; the separate SelfAttention grouped-query path remains explicit.
- Zero query/key width produced NaNs. It now rejects before computation rather
  than adopting PyTorch's empty-reduction behavior. Core attention also checks
  f32, same-device inputs and derived buffer capacity before allocation. Empty
  batch/query/key lengths and zero value width retain their valid shape/gradient
  behavior. Constructors reject nonpositive dimensions and invalid head counts;
  enabled RoPE requires a finite positive base and even head width.
- The causal mask cache was keyed only by device ordinal and shape. Replacing a
  registered backend could reuse a mask owned by its predecessor and reject
  valid fresh inputs as foreign allocations. Cache reuse now also compares
  backend owner identity; displaced entries release after the cache lock.
  The regression uses an owner-checking mock backend, not GPU execution.
  This covers sequential registry replacement, not concurrent registry changes
  during an attention call.
- Python `Trainer.fit` called `iter(batches)` merely to detect one-shot sources.
  For an eager shuffled loader, two requested epochs advanced the loader three
  times and discarded epoch zero. Non-consuming iterator-protocol validation
  preserves seeds 7/8 instead of training seeds 8/9. Three tests cover exact
  epoch order, iterator call counts, reiterable sources and non-consuming
  one-shot rejection.
- A Rust worker panic could leave a missing batch ticket; channel closure then
  silently ended the iterator. The reproducer returned only two successful
  batches from six, without an error. Unwinding sample-loading/collation panics
  now become ordered errors in both inline and threaded modes, preserving later
  batches. Unexpected worker-channel closure yields one error and then ends.
- A bounded result channel did not bound the reorder buffer. With batch zero
  gated, workers=2/prefetch=1 fetched all 32 items before the first yield. A
  shared ticket window now bounds loading plus queued and reordered batches;
  the same probe observes two outstanding items. Capacity is capped by the
  epoch batch count, including oversized prefetch settings. Iterator drop
  cancels new claims, wakes waiters and drops the receiver without joining
  running user callbacks. Those callbacks finish independently. Thread-spawn
  failures produce a terminal iterator error.

The loader changes retain the existing stateless Rust RandomSampler contract;
they do not add epoch/cursor checkpointing or asynchronous callback cancellation.
The bound applies to Rust prefetch work. Python's TensorDataset DataLoader still
materializes a whole epoch before yielding, and sampler/index lists remain
epoch-sized. Thread-spawn failure and unexpected channel closure handling were
reviewed but not directly fault-injected by the new tests.
Dataset length and sampler-order callbacks are outside the panic conversion.
No claim covers panic-abort builds or hung user callbacks. Four new Rust loader
tests and thirteen existing tests pass; independent review found no blockers.

## Final matching-build validation

Linux x86_64, Rust 1.99.1, Python 3.12.14 and torch 2.13.0+cpu. The final
noneditable wheel was built after the source freeze and installed with
`--force-reinstall --no-deps`. All 20 Python/native files match the wheel and
installed package; Python sources match the checkout. Generated bytecode is
excluded from the wheel.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 897 passed / 2 ignored; fastcpu 37 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies because the real-GPT2 fixture is absent. CUDA 137 unit/integration harness passes plus 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0. |
| `maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked --out <work>/dist-frozen` and noneditable installation | Exit 0. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 207 passed; 30 discovered CUDA cases unavailable. |
| Both public graph-attention/streaming CLIs above | Exit 0; all six learning/restart proofs pass against the frozen build. |
| `python examples/py_regression.py` | Exit 0; binding/DLPack regressions pass. |
| `python crates/ferro-py/examples/py_compiled_fusion_regression.py` and `py_fuse_regression.py` | Exit 0; compiled DAG and fusion checks pass. |
| `python examples/ops_vs_torch.py` | Exit 0; operator/reference checks pass. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization checks pass. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Source/wheel/install identity and `git diff --check` | Passed. |

Python discovers 237 cases. Unittest reports `Ran 231 tests` and 25 skip records:
24 individual skips plus one class-level setup skip preventing six more cases.
Thus 207 pass and 30 are unavailable. Rust counts use the last outer summary of
each executable, excluding nested subprocess summaries and targeted reruns.
CUDA harness passes do not establish GPU body execution. Performance ignores
and the missing real-GPT2 fixture remain unexecuted.

All 538 selected source inputs remained unchanged after final execution. Native
extension SHA-256:
`18fe4b24e95a606ca2d351657246bb6a593ae4474171849629e84e7515a7e8a4`.
Raw logs, failure reproductions, wheels and generated diagnostics remain under
`/workspace/work/graph-sequence-2026-11-03`; the manifest hashes the local logs.
Raw logs and wheels are not shipped. Independent implementation/model reviews
found no remaining blockers in this scoped work.

## Remaining boundaries

All new model proofs are synthetic CPU f32 runs within one binary/environment.
Sparse edge attention is not a complete graph transformer, resident CUDA sparse
execution or universal topology generalization. Streaming state is persisted by
the example's explicit bundle, not by a broadened generic checkpoint API.

No GPU is exposed. CUDA ownership/stream/capture and model gates still require
real hardware. Foreign-stream mutation bookkeeping, experimental whole-step
capture and historical CPU numerical/native-crash investigations remain open.
The [Rust CUDA research](../../docs/RUST_CUDA.md) remains applicable; cudarc is
still the baseline. No compiler/dependency migration, commit, push, merge or
deployment was performed.
