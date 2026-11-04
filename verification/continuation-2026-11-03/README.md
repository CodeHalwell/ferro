# GCN learning and capture ownership continuation, 2026-11-03

Base commit remains `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. This is an
uncommitted continuation of the [initial review](../review-2026-11-03/README.md).
The [manifest](manifest.json) identifies the 499 source/config/lock inputs used
for this run, the rebuilt wheel and installed extension, and local log hashes.
Earlier reports retain their own source identities and are not overwritten.

## GCN learning and fresh-process restart

The public fixture is `examples/train_restart_gcn.py`, with reusable logic in
`examples/gcn_restart.py` and tests in `crates/ferro-py/tests/test_gcn_restart.py`.
It uses two GCN layers, eight hidden channels, tanh, AdamW, and 240 CPU f32 steps.
Weights are explicitly initialized for seeds 7, 11 and 43; one explicit Generator
supplies stochastic training perturbations.

Training and held-out splits each contain 32 balanced small directed graphs.
Paired labels share topology and identical query-node features. Neighbors carry
the useful signal; an isolated decoy cancels the global feature sum. Feature
samples are disjoint across splits; topology templates may repeat. This proves
learning on the declared synthetic fixture, not unseen-topology or real-dataset
generalization.

Gates were declared before successful training: at least 95% held-out accuracy,
at least 40 percentage points lost after removing edges, input/all-parameter
gradient agreement with independent dense PyTorch algebra at atol 1e-5 / rtol
1e-4, and bit-exact restart. Before any successful training run, the checkpoint
was moved from step 120 to 117 so the saved batch cursor would be nonzero.
No learning or numerical threshold was weakened.

| Seed | Initial held-out accuracy | Final held-out accuracy | Edges removed | Max gradient absolute error | Fresh-process restart |
|---|---:|---:|---:|---:|---|
| 7 | 50% | 100% | 50% | 2.98e-8 | Bit-exact |
| 11 | 56.25% | 100% | 50% | 2.24e-8 | Bit-exact |
| 43 | 84.375% | 100% | 50% | 4.47e-8 | Bit-exact |

Both class-prior and node-local baselines are 50%. The independently launched
worker loads the saved graph/features/labels and validates their digest, run
configuration, checkpoint publication and cursor before constructing the model.
It does not regenerate topology. The restored run matches every remaining loss,
serialized model/AdamW/config/RNG state, subsequent random draws, held-out metrics,
dataset identity and cursor. Seven tests include corruption, rewritten digests,
configuration rejection without live-state mutation, and acceptance-gate fault
injection under optimized Python. Separate agent reviews found no blockers.

The bundle is an explicit example-level persistence format. It does not change
the general checkpoint API or certify arbitrary data-loader resume. Restart is
bit-exact within this binary/environment; cross-platform identity is not a gate.

## Additional repairs

- Core GCN validates CPU f32, capture state, node counts and all parameter ranks
  before matrix multiplication or self-loop construction. Reproductions covered
  a scalar replacement weight and a tiny feature matrix paired with huge graph
  metadata. These now return errors rather than panic or allocate first.
- `EdgeIndex::try_batch` checks aggregate node/edge counts and allocation limits;
  Python batching maps its errors. Existing Rust `batch` remains a convenience
  wrapper that can panic on failure, preserving its return type.
- Zero-width mean/max/min reductions avoid unnecessary per-group count/mask
  allocations and preserve empty adjoints. A reproduced capacity failure now
  passes without allocating for nonexistent feature coordinates.
- CUDA buffers share exact allocation owners internally. The final owner alone
  retires a slice to its allocator. `CapturedChain` retains its inputs and uses
  existing static-graph private-stream preparation, RAII cleanup and failure
  quarantine instead of disabling context tracking.
- Pointwise capture validates broadcast ranks, products, references and address
  bounds before compilation/allocation. Mutable aliases relinquish ownership
  on success, error and unwind, including their event/stream metadata.
- Wheels exclude local Python bytecode caches. CI gains CUDA host regression
  execution and a wheel-content check after intentionally generating caches.
  The workflow was reviewed locally; hosted CI was not run.

Capture migration: bind `let mut captured` because replay requires `&mut self`.
An output borrow therefore prevents replay from overwriting the same allocation.
The handle is neither Send nor Sync. `output()` panics on a poisoned graph;
the copy method returns an error. Zero-element captured chains reject explicitly.

## Final matching-build validation

Linux x86_64, Rust 1.99.1, Python 3.12.14 and torch 2.13.0+cpu. No GPU is exposed.
The source freeze was checked again after all final commands. The wheel excludes
generated caches and its Python/native files match the installed package.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 863 passed / 2 ignored; fastcpu 37 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies (real-GPT2 fixture absent). CUDA 136 unit/integration harness passes and 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0, including updated capture callers and GPU regressions. |
| `maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked --out <work>/dist` plus noneditable installation | Exit 0 after the final production edits. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 133 collected, 108 passed / 25 skipped, including all graph/restart tests. |
| `python examples/train_restart_gcn.py` | Exit 0; all three seeds passed against the frozen build. |
| `python examples/py_regression.py` | Exit 0; all binding/DLPack checks passed. |
| `python examples/ops_vs_torch.py` | Exit 0; all operator/reference checks passed. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization checks passed. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Wheel inspection and `git diff --check` | Passed. |

Counts use final outer harness summaries and do not add subprocess or targeted
reruns. CUDA harness successes are not evidence that GPU bodies executed.
Raw logs remain in `/workspace/work/continuation-2026-11-03`; the manifest records
their hashes. Raw logs/wheels are not shipped. Initial API-mismatch and graph
capacity-failure reproductions are preserved there.

## Still open

Real-GPU retention/pool-pressure, broadcast, stream/capture and cleanup regressions
must execute before claiming runtime correctness or performance. In-place mutation
remains scoped to the owning backend stream: the existing alias helper does not
provide original-allocation write-event bookkeeping for concurrent foreign-stream
readers. Experimental whole-training-step capture remains separate and unresolved.
The shared-owner chain change received independent review under this explicit scope.

Historical CPU numerical/native-crash investigations remain unresolved. General
resident GNN training, other graph architectures and other model-family learning
proofs remain open. No experimental Rust CUDA compiler, dependency migration,
commit, push or merge was performed.
