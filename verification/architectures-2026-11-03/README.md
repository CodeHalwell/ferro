# CNN, GRU and KAN continuation, 2026-11-03

Base commit remains `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. This is an
uncommitted continuation of the [GCN/capture work](../continuation-2026-11-03/README.md).
The [manifest](manifest.json) identifies 509 unchanged source/config/lock inputs,
the rebuilt release wheel, installed extension and local execution logs.
Earlier reports retain their own source identities and results.

## Three public learning and restart proofs

Run each CLI with the built Python extension and PyTorch installed:

```sh
python examples/train_restart_cnn.py
python examples/train_restart_recurrent.py
python examples/train_restart_kan.py
```

Each uses the public Module, optimizer, checkpoint and explicit Generator APIs.
Every parameter is explicitly initialized for seeds 7, 11 and 43. Budgets,
datasets and acceptance thresholds were declared before successful training;
no learning or numerical threshold was weakened after execution.

| Fixture | Model and data | Predeclared gate | Checkpoint |
|---|---|---|---|
| CNN | Conv2d(1,8,3), tanh, average pooling and linear head; 64 train / 64 held-out 8x8 images containing horizontal or vertical bars. Label pairs have identical intensity histograms and balanced location distributions. | At least 95% held-out accuracy after 300 AdamW steps; mean-image ablation gives 50% accuracy and loses at least 40 percentage points. | Step 147; next-image cursor 24 |
| GRU | Eight hidden channels and linear readout; 32 train / 32 held-out length-five sequences. Each label pair swaps its first two tokens and shares its later tokens, unordered multiset and final zero token. | At least 95% held-out accuracy after 300 AdamW steps; erasing history before the final token loses at least 40 percentage points. | Step 137; next-sequence cursor 8 |
| KAN | Two fixed cubic spline banks, 2-to-3-to-1; 64 train / 64 held-out points on disjoint offset grids for sin(2*x0+0.8*x1)+0.3*x0*x1. | After 400 AdamW steps, held-out MSE is at most 10% of both training-fit constant and affine baselines. | Step 197; next-sample cursor 16 |

CNN and GRU accuracy starts at 50% for every seed and reaches 100%. Their
respective spatial/history ablations give 50%. Both input/reference and every
parameter-tensor gradient are checked before training at atol 1e-5 / rtol 1e-4.
CNN uses independent PyTorch convolution/pooling operations; GRU uses explicit
reset-after gate algebra. Maximum absolute gradient errors are 5.146e-8 and
1.584e-8 respectively.

| KAN seed | Initial held-out MSE | Final held-out MSE | Fraction of affine-baseline MSE | Fraction of constant-baseline MSE |
|---|---:|---:|---:|---:|
| 7 | 0.723518 | 0.000240675 | 0.608% | 0.0419% |
| 11 | 0.524349 | 0.000882992 | 2.231% | 0.1536% |
| 43 | 0.509890 | 0.000292219 | 0.738% | 0.0508% |

The independent KAN oracle evaluates Cox-de Boor basis algebra in PyTorch and
checks input plus both coefficient-tensor gradients. Samples must be more than
1e-3 from input and hidden knots; observed minimum margin exceeds 0.0028. The
maximum gradient error is 1.193e-7. A separate least-squares reference validates
the training-fit affine baseline; the held-out affine MSE is 0.03958569136 and
constant MSE is 0.57493172728.

All nine runs reproduce every remaining loss, serialized model/AdamW/config/RNG
state, next Generator draws, held-out metrics, dataset identity and cursor in a
fresh process. The worker loads saved samples rather than regenerating them.
Dataset and checkpoint-publication digests are checked before model construction,
and the expected dataset digest is also bound to immutable model configuration.
This is an explicit example-level data bundle, not a general data-loader format.

Seven tests per family include learning/restart, data dependence, saved-data
loading, corruption, configuration rejection without state mutation and
fault-injected acceptance gates under optimized Python. The actual seed-7 KAN
proof also passed under `python -O`. Independent reviewers inspected all nine
new example/test files and found no blockers in the bounded claims.

## API repairs found during the work

`KanLayer` previously exposed a native basis object as public built-in module
configuration. `snapshot_training()` rejected it with a TypeError. The layer now
records immutable knots, degree and outside-domain policy through the existing
checkpoint schema. The native basis is frozen and exposes its outside policy.
The constructor and readable `layer.basis` remain available; assigning a new
basis is no longer supported. Construct a new compatible layer instead.
Roundtrip tests cover direct and inherited layers. Same-shaped knot, degree and
outside-policy mismatches reject before model/optimizer/RNG mutation. The generic
serializer was not broadened to accept arbitrary objects.

Recurrent `unroll` previously panicked for valid zero-data inputs whose requested
output layout or output-handle allocation was unrepresentable. It now checks
the output layout, f32 byte capacity and fallible handle reservation before
constructing steps or invoking the cell. Three regressions cover the failures
and valid empty/fully masked behavior. Separate review found no blocker and
confirmed existing active/reset/truncated recurrent tests still pass.

CI already discovers these new Python tests. Its test-step label now describes
the public API, capture and training/restart scope. Hosted CI was not run.

## Final matching-build validation

Environment: Linux x86_64, Rust 1.99.1, Python 3.12.14, torch 2.13.0+cpu. No GPU
is exposed. The final noneditable wheel was built after the source freeze and
installed with `--force-reinstall --no-deps`. All 20 Python/native wheel files
match the installed package, and Python sources match the checkout. Generating
local bytecode before building did not put any bytecode into the wheel.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 866 passed / 2 ignored; fastcpu 37 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies because the real-GPT2 fixture is absent. CUDA 136 unit/integration harness passes plus 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies may return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0. |
| `maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked --out <work>/dist-frozen` and noneditable installation | Exit 0. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 132 passed; 30 discovered CUDA cases unexecuted. See the accounting note below. |
| All three public architecture CLIs above | Exit 0; all seeds pass against the frozen build. |
| `python examples/py_regression.py` | Exit 0; binding/DLPack regressions pass. |
| `python crates/ferro-py/examples/py_compiled_fusion_regression.py` and `py_fuse_regression.py` | Exit 0; compiled DAG and fusion checks pass. |
| `python examples/ops_vs_torch.py` | Exit 0; operator/reference checks pass. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization checks pass. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Source/wheel/install identity and `git diff --check` | Passed. |

Python accounting distinguishes test cases from skip records: discovery finds
162 cases. Unittest reports `Ran 156 tests` and 25 skip records. Those records
are 24 individual skips plus one class-level setup skip, which prevents six
additional cases from running. Thus 132 cases pass and 30 cases are unavailable;
subtracting all 25 skip records from 156 would undercount passes. Rust counts use
the final outer summary of each executable and do not add nested subprocess or
targeted reruns. CUDA harness passes are not real GPU execution.

The source freeze was checked after final execution. Native extension SHA-256:
`76099d5addb5ffadd5238edae1187282a38c25ac42fe6de22f2f31be846cc1de`.
Raw logs, initial failure reproductions, wheels and test-generated diagnostics
remain in `/workspace/work/architectures-2026-11-03`; they are not shipped here.

## Remaining boundaries

These are small synthetic CPU f32, first-order fixtures. CNN spatial templates
may repeat; the held-out amplitudes differ. GRU hidden state starts at zero for
each batch, and restart occurs between complete optimizer steps, not inside a
sequence. KAN knots are fixed; input higher derivatives are not certified.
Bitwise restart is scoped to this binary/environment, not cross-platform runs.

Real-GPU capture/ownership/stream and model execution gates remain unexecuted.
The earlier foreign-stream mutation bookkeeping and experimental whole-step
capture limitations remain open. Historical CPU numerical/native-crash
investigations are unchanged. [Rust CUDA research](../../docs/RUST_CUDA.md)
remains the current toolchain recommendation; no experimental compiler or
dependency migration was introduced. No commit, push, merge or deployment was
performed, and this review does not certify every pre-existing working-tree change.
