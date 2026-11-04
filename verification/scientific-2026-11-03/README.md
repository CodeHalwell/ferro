# Differentiable-derivative training continuation, 2026-11-03

Base commit remains `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. This is an
uncommitted continuation of the [CNN/GRU/KAN work](../architectures-2026-11-03/README.md).
The [manifest](manifest.json) identifies 517 unchanged source/config/lock inputs,
the rebuilt release wheel, installed extension and local execution logs. Earlier
reports retain their separate source identities and results.

## Strong-form PINN

Run `python examples/train_restart_pinn.py`. The public CPU f32 model solves
u'' = -pi^2 sin(pi*x) on [0,1], with u(0)=u(1)=0 enforced by
u=x(1-x)*MLP(2x-1). The MLP has one tanh hidden layer of width 12. Training uses
only the second-coordinate-derivative residual. Analytic solution values are
used for evaluation, not as supervised training targets.

The design was declared before successful training: seeds 7, 11 and 43; 1500
AdamW steps at lr 0.01; 32 saved base collocation points, batches of eight and
explicit-Generator jitter bounded by +/-0.003; 64 held-out points and endpoints.
The nearest held-out/base distance is 0.00390625, greater than the jitter bound.
The gates are solution RMSE <=0.01, residual RMSE <=0.1 and endpoint error <=1e-6.
No PINN design, budget or threshold was changed after training began.

| Seed | Initial solution RMSE | Final solution RMSE | Final residual RMSE | Endpoint error |
|---|---:|---:|---:|---:|
| 7 | 0.740533 | 0.000325938 | 0.0540125 | 0 |
| 11 | 0.695992 | 0.000319634 | 0.0604210 | 0 |
| 43 | 0.655463 | 0.000586597 | 0.0851276 | 0 |

Public `grad(..., create_graph=True)` is applied twice; the residual loss then
backpropagates into all four parameter tensors. Before updates, independent
PyTorch output/first/second-coordinate derivatives and residual parameter
gradients are checked at atol 5e-5 / rtol 2e-4. Maximum errors are 2.385e-7 for
coordinate derivatives and 3.815e-6 for parameter gradients. Every parameter
tensor has nonzero mixed gradients. Float64 reference finite differences also
check curvature and selected entries of each parameter gradient at declared
2e-3 tolerances; these are reference calculations, not native f64 AD support.
An independent review additionally used closed-form tanh curvature.

Eight tests cover learning/restart, saved collocation data, physics/configuration
mismatch, corrupted bundles, derivative detachment and optimized acceptance
gates. Both a detached curvature and a detached leaf with requires_grad re-enabled
fail the oracle. The actual seed-7 CLI also passed under `python -O`.

## Contractive autoencoder

Run `python examples/train_restart_contractive.py`. A Linear(2,3)-tanh encoder
and Linear(3,2) decoder train on 64 curve samples [t,sin(1.5*t)], with 64 distinct
half-offset held-out samples. The objective is reconstruction MSE plus 0.05 times
the mean squared Frobenius norm of the encoder Jacobian. Public graph-building
VJPs select each Jacobian row; the penalty remains connected to encoder parameters.
The batches are independent samples, so summing row-selecting cotangents does not
introduce cross-sample terms.

The predeclared gates are reconstruction MSE <=10% of a training-mean constant
baseline and final Jacobian penalty <=75% of initialization. Seeds are 7, 11 and
43, with 400 AdamW steps, lr 0.025, weight decay 0.001, batch size eight and one
explicit Generator supplying Gaussian input perturbations of scale 0.005.

The original sorted cyclic schedule passed seed 7 but failed seed 11's
reconstruction gate: MSE/baseline was about 0.477. Execution stopped there;
seed 43 was not run under that schedule. Independent PyTorch training with the
same noise reproduced the failure (Ferro MSE 0.193062037 versus PyTorch
0.193062633), identifying a training/schedule issue rather than a derivative
discrepancy. These logs are preserved.

After that failure, a deterministic interleaved schedule was declared as version
`interleaved-cyclic-contractive-v2`. Each batch spans the curve. This is a changed
experiment: only training order changed; the sample set, held-out set, model,
optimizer, 400-step budget and gates were retained. Ordered data and schedule
identity are bound to the checkpoint. All three revised runs pass:

| Seed | Held-out MSE | MSE / constant baseline | Final / initial Jacobian penalty |
|---|---:|---:|---:|
| 7 | 0.001319403 | 0.325694% | 4.63795% |
| 11 | 0.001094501 | 0.270177% | 30.6667% |
| 43 | 0.001209346 | 0.298526% | 21.7376% |

Before updates, reconstruction, penalty and total losses and their input/all-four-
parameter gradients are checked separately against an analytic PyTorch Jacobian
at atol 1e-5 / rtol 1e-4. Maximum error is 1.193e-7. Penalty gradients into both
encoder tensors are explicitly nonzero; its decoder gradients are correctly zero.
A scalar-preserving detached-penalty fault fails. Separate review used a full
batch Jacobian rather than the analytic-Jacobian oracle. Eight tests include the
learning run, saved data, unchanged sample sets after interleaving, corruption,
configuration/penalty mismatches, detachment and optimized-Python gates.

The penalty does not establish robustness or latent identifiability: decoder
rescaling can compensate for a smaller encoder Jacobian.

## Restart and core repairs

All six final runs reproduce every remaining loss, serialized model/AdamW/config/
Generator state, next draws, held-out metrics, dataset identity and cyclic cursor
in a fresh process. PINN resumes at step 749 (next base point 8); the autoencoder
at step 197 (next sample 40). Workers read the saved base data and validate
publication/dataset digests rather than regenerating it. PINN jitter is recreated
from its restored Generator and immutable physics configuration. This is explicit
example-level data persistence, not arbitrary data-loader or mid-step resume.

Review found and fixed three functional-AD contract gaps:

- Inference capture could freeze a functional derivative. For grad(x*x)*x,
  capture at x=2 followed by replay at x=3 returned 12 instead of 18. Both
  `grad_wrt` and `vjp_wrt` now reject capture early, for both create_graph modes.
  Derivative capture has not been implemented; evaluate derivatives eagerly.
  Tests cover persistent gradients, recording modes, nested capture recovery
  and subsequent valid inference replay.
- A public custom backward callback could return a same-shaped I64 or F64
  tensor as a gradient. Functional VJP now returns DtypeMismatch before
  accumulation/return. Ordinary infallible backward asserts f32 before device
  transfer or gradient-mutex access. Its broader backward operation remains
  nontransactional; this does not promise rollback of earlier contributions.
- Scalar functional gradients previously allocated their implicit seed before
  validating operand dtype/device. A counting backend observed one unnecessary
  allocation. Shared metadata validation now rejects first; the regression
  observes zero calls for invalid requests and an injected backend failure for
  valid metadata as a positive control.

Four Rust regressions and two Python capture tests exercise these repairs.
Independent review found no blockers. A separate review-stage numerical probe
passed 188 PyTorch comparisons across four seeded shared-weight networks, with
maximum error 2.385e-7 through third-order derivatives. It covered strided inputs,
broadcast reductions, live/output-dependent cotangents, mixed symmetry and
functional gradient-state preservation. The final frozen suites below verify
the rebuilt implementation; review-stage probes are not added to suite totals.

The Python autograd module now documents functional state, live cotangents,
unused inputs, the smooth derivative subset and eager-only execution. The status
ledger also reconciles two previously repaired CUDA issues with their existing
publication scope; it does not claim fresh GPU execution.

## Final matching-build validation

Linux x86_64, Rust 1.99.1, Python 3.12.14 and torch 2.13.0+cpu. No GPU is exposed.
The final noneditable wheel was built after the source freeze and installed with
`--force-reinstall --no-deps`. All 20 Python/native wheel files match the installed
package, and Python sources match the checkout. Bytecode generated before building
remains excluded from the wheel.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 870 passed / 2 ignored; fastcpu 37 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies because the real-GPT2 fixture is absent. CUDA 136 unit/integration harness passes plus 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies may return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0. |
| `maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked --out <work>/dist-frozen` and noneditable installation | Exit 0. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 150 passed; 30 discovered CUDA cases unavailable. |
| Both public scientific CLIs above | Exit 0; all seeds pass against the frozen build. |
| `python examples/py_regression.py` | Exit 0; binding/DLPack regressions pass. |
| `python crates/ferro-py/examples/py_compiled_fusion_regression.py` and `py_fuse_regression.py` | Exit 0; compiled DAG and fusion checks pass. |
| `python examples/ops_vs_torch.py` | Exit 0; operator/reference checks pass. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization checks pass. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Source/wheel/install identity and `git diff --check` | Passed. |

Python discovers 180 cases. Unittest reports `Ran 174 tests` and 25 skip records:
24 individual skips plus one class-level setup skip preventing six more cases.
Thus 150 pass and 30 are unavailable. Rust counts use the last outer summary of
each executable, excluding nested subprocess counts and targeted reruns. CUDA
harness passes do not establish execution of GPU bodies.

All 517 selected source inputs remained unchanged after final execution. Native
extension SHA-256:
`c41f3ed4fac67127d62f659015ec06e43e0eca6f500bb4883cda633c51b89ad3`.
Raw logs, failure reproductions, wheels and generated diagnostics remain in
`/workspace/work/scientific-2026-11-03`; the manifest hashes the local logs. Raw
logs and wheels are not shipped with this report.

## Remaining boundaries

These are eager CPU f32, synthetic examples within one binary/environment.
The PINN differentiates the model with respect to sampled input coordinates and
trains parameters; it does not optimize collocation locations. Hard boundary
conditions are built into its model form. General PDE conditioning, f64 AD,
arbitrary higher-order operators, derivative capture and gradient-penalty GAN
training require separate work. Cross-platform bitwise restart is not claimed.

Real-GPU ownership/stream/capture and model gates remain unexecuted. Historical
CPU numerical/native-crash investigations and the earlier foreign-stream mutation
and experimental whole-step capture limitations remain open. The existing
[Rust CUDA research](../../docs/RUST_CUDA.md) remains applicable; no compiler or
dependency migration was introduced. No commit, push, merge or deployment was
performed, and this review does not certify every pre-existing working-tree change.
