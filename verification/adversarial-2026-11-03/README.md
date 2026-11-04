# Generative training and optimizer continuation, 2026-11-03

Base commit remains `5f1ebe9b54742a690cd7cbdd6b17568329b1c701`. This is an
uncommitted continuation of the [scientific training work](../scientific-2026-11-03/README.md).
The [manifest](manifest.json) identifies 526 selected source/config/lock inputs,
rebuilt wheel, installed extension and local logs. Earlier reports retain their
own source identities and results.

## Gaussian VAE

Run `python examples/train_restart_vae.py`. The public CPU f32 VAE has a
2-to-8 tanh encoder, separate one-dimensional mean/log-variance heads, and a
1-to-8 tanh-to-2 decoder. Reparameterization uses
`z = mu + exp(0.5 * logvar) * epsilon`. The objective is Gaussian reconstruction
NLL with observation sigma 0.2, including its normalization constant, plus
analytic standard-normal KL with beta 1. NLL sums observation features, KL sums
latent coordinates, and both average samples.
The ten parameter tensors train on 64 curve samples `[t, sin(1.5*t)]`, with
64 disjoint half-offset held-out samples.

The unchanged design declared seeds 7, 11 and 43, 600 AdamW steps at lr 0.015,
weight decay 0.0001, batch size eight and deterministic interleaved training
order. One checkpointed Generator supplies training latent noise. Held-out
Monte Carlo evaluation uses four fixed noise draws per sample, persisted with
the dataset; evaluation neither regenerates noise nor advances the training RNG.

Gates: held-out Monte Carlo negative ELBO <=50% of initialization and <=50% of
the constant Gaussian baseline; decoder-mean reconstruction MSE <=25% of the
constant predictor; KL strictly between 0.1 and 10. All pass:

| Seed | Initial MC negative ELBO | Final MC negative ELBO | Final KL | Decoder-mean MSE |
|---|---:|---:|---:|---:|
| 7 | 14.877728 | 0.576381 | 1.368161 | 0.001278593 |
| 11 | 9.327246 | 0.579742 | 1.511537 | 0.003415372 |
| 43 | 14.616918 | 0.579327 | 1.340876 | 0.003263830 |

The constant Gaussian negative-ELBO baseline is 8.746631; its MSE is 0.405105.
Decoder-mean MSE is a separate diagnostic, not the stochastic likelihood or ELBO.
NLL can be negative for this continuous density. Independent fixed-epsilon
PyTorch comparisons check NLL, KL and total losses, input/epsilon gradients and
all ten parameter gradients at atol 1e-5 / rtol 1e-4; maximum absolute error is
3.815e-6. Separate review used `torch.distributions.Normal` log probabilities
and KL and agreed within 1.91e-6. Scalar-preserving detachment of latent samples,
scale or KL fails the gradient oracle. Nine tests cover learning, objective
reductions/normalization, saved evaluation noise, configuration/corruption,
restart and optimized-Python gates.

## WGAN with a differentiable gradient penalty

Run `python examples/train_restart_wgan.py`. This is a bounded Gaussian
location/scale problem, targeting N(1, 0.6^2). The generator is an affine
`exp(log_scale)*z + shift` transform with two parameters. The critic is a
1-to-16 tanh-to-1 network with three parameter tensors and no output bias.
Training uses 512 saved real quantiles in interleaved order, distinct 512 held-out
real quantiles, and 512 saved evaluation latent quantiles. An explicit Generator
supplies training Gaussian latent draws and uniform interpolation draws.

Critic loss is mean fake score minus mean real score plus lambda times
`mean((sqrt(D'(interpolation)^2 + 1e-12) - 1)^2)`. Public
`grad(..., create_graph=True)` keeps the penalty connected to critic parameters.
Generator loss is minus mean critic score. Native Adam uses betas (0, 0.9),
batch size 32 and five critic updates per generator update.

The original configuration failed seed 7 after its full 1500-cycle budget.
It used positive critic initialization, critic/generator lr 0.002 and lambda 10.
Generated mean reached about 2.2410 and held-out W1 about 1.243, failing the
unchanged quality gates, although restart and derivative/isolation checks passed.
An independent PyTorch trajectory with identical draws reproduced the mean to
2.4e-7. This was a failed experiment, not evidence of an engine discrepancy.
No success or execution is claimed for its other seeds. Its source and logs are
preserved locally and hashed by the manifest.

Revision 2 was declared before its training runs. It changed critic initialization
to centered mixed-sign normals (weight scales 0.5 and 0.3, bias scale 0.1),
critic lr to 0.005, generator lr to 0.001, lambda to 0.1, and budget to
4000 cycles / 24000 updates. This is a weaker penalty and longer experiment.
The target, data, architecture size, 5:1 update ratio, seeds and quality gates
were retained. No best checkpoint was selected.

Gates: empirical held-out W1, absolute mean error and absolute standard-deviation
error each <=0.25; W1 also <=25% of its initial value. All revised runs pass:

| Seed | Initial W1 | Final W1 | Mean error | Std error | Held-out penalty |
|---|---:|---:|---:|---:|---:|
| 7 | 1.755147 | 0.048002 | 0.046011 | 0.020549 | 0.025925 |
| 11 | 1.755410 | 0.013147 | 0.007640 | 0.011500 | 0.010517 |
| 43 | 1.733592 | 0.113694 | 0.113116 | 0.012711 | 0.167834 |

The known target-transform baseline W1 is 0.004127 due to distinct finite
quantile grids. Penalty and input-gradient norms are diagnostics only. Final
mean norms are 1.1280, 0.9336 and 1.3974; these do not establish a Lipschitz bound.
This fixture proves neither general GAN convergence nor high-dimensional quality.

Before updates, an independent PyTorch oracle checks critic transport, penalty,
total loss, generator loss, critic input derivatives and all active parameter gradients, including nonzero
penalty mixed gradients. Maximum absolute error is 1.193e-7 at atol 5e-5 /
rtol 3e-4. A scalar-preserving detached penalty fails. Both optimizers clear
gradients at phase boundaries. Critic updates detach fake samples. Generator
updates freeze critic parameter slots and explicitly read detached critic weights,
preserving input gradients into the generator. Parameter freezing alone masks
optimizer gradients; it does not disable tensor differentiation.

Isolation checks after warmup compare inactive parameters and optimizer buffers,
counters and configuration, including signed-zero-sensitive scalar bytes. They
check gradients are cleared and trainable flags restored. Nine tests include
all-seed learning, data/phase persistence, isolation, corruption, configuration,
fault injection and optimized-Python gates. The complete seed-7 CLI also passes
under `python -O`.

## Restart and optimizer repairs

All six runs reproduce every remaining objective term/loss, held-out metric,
model/configuration/optimizer/Generator state and subsequent draws in a fresh
process. VAE resumes at step 293 with cursor 40. WGAN resumes at update 11997:
cycle 1999, three critic updates into the cycle, 9998 total critic updates,
1999 generator updates and next real-sample cursor 448. Actual data and evaluation
noise are read from saved bundles. This is example-level persistence between
complete updates, not arbitrary data-loader or partial-backward restart.

The optimizer review exposed and repaired two production defects:

- AdamW used one global bias-correction count. A parameter receiving its first
  gradient after another parameter had stepped therefore used the wrong
  correction (a focused case updated 1.0 to 0.9418873 instead of 0.9). Eager
  AdamW now owns per-parameter counters; missing/frozen gradients skip weights,
  moments, decay and counters. A step with no gradients changes nothing. An
  independent 11-step PyTorch oracle covers delayed gradients, freeze/unfreeze,
  empty steps, nondefault betas and nonzero decay, comparing all state after
  each step. It failed the old wheel and passes the rebuilt implementation.
- Derived Rust optimizer `Clone` copied counters but shared populated moment
  buffers. Stepping a clone could alter the original's moments without advancing
  its counters. SGD, Adam and AdamW clones now own state buffers while intentionally
  retaining shared parameter slots. Device cloning may transfer through host;
  no performance or residency claim is made. Private restore staging retains
  shallow temporary handles, replaces staged state, validates, then commits.

AdamW snapshot key `t` now stores f32 `[number_of_parameters]`, including `[0]`
for an empty optimizer. Restore accepts this vector or the old scalar `[]`.
The old recorded global count is broadcast; historical skipped counts cannot be
recovered. Old readers do not accept the new vector form. Counts must be finite,
nonnegative integers in u32 range. The existing f32 serialization precision
limit remains: this is not an exact round-trip promise for arbitrary u32 counts.

Capturable AdamW retains a shared device step/correction triplet. Activation and
restore require synchronized parameter counters. Each update requires gradients
for every parameter or none; partial sets reject before device-counter advancement,
and an empty set is a no-op. Repeated activation preserves the current authoritative
device counter. This does not make already-captured graphs dynamically honor
freeze changes. CPU mock tests cover preflight rejection and positive controls;
the new GPU regression compiles but its GPU body was not exercised.

Additional named-optimizer checkpoint tests preserve unequal AdamW counters and
frozen flags, restore by namespace despite reordered bindings, reject corrupt
later optimizer state atomically, and read legacy scalar counts. Independent
review found no production blockers.

The first full Rust run found an older transaction test that intentionally relied
on public optimizer clones sharing moments. Its expectation conflicted with the
new owning Clone contract. The revised tests verify owning-clone restore and
retain rejection coverage using genuinely aliased optimizer state. That failure,
the first successful Python checks, and the original freeze are preserved under
`validation-attempt1`; final validation uses a new freeze.

## Final validation

Linux x86_64, Rust 1.99.1, Python 3.12.14 and torch 2.13.0+cpu. The final
noneditable wheel was built after the source freeze and installed with
`--force-reinstall --no-deps`. All 20 Python/native files match the wheel and
installed package; Python sources match the checkout. Generated bytecode is
excluded from the wheel.

| Command / scope | Result |
|---|---|
| `cargo test --workspace -- --skip required_cuda_ --nocapture` | Exit 0: core 880 passed / 2 ignored; fastcpu 37 passed / 2 ignored; tokenizer 8 harness passes, 7 exercised bodies because the real-GPT2 fixture is absent. CUDA 137 unit/integration harness passes plus 2 compile-fail doctests; 5 mandatory GPU tests filtered. Other GPU-gated bodies return early. |
| `cargo check -p ferro-cuda --all-targets` | Exit 0. |
| `maturin build --manifest-path crates/ferro-py/Cargo.toml --release --locked --out <work>/dist-frozen` and noneditable installation | Exit 0. |
| `python -m unittest discover -s crates/ferro-py/tests -v` | Exit 0: 171 passed; 30 discovered CUDA cases unavailable. |
| Both public VAE/WGAN CLIs above | Exit 0; all six learning/restart proofs pass against the frozen build. |
| `python examples/py_regression.py` | Exit 0; binding/DLPack regressions pass. |
| `python crates/ferro-py/examples/py_compiled_fusion_regression.py` and `py_fuse_regression.py` | Exit 0; compiled DAG and fusion checks pass. |
| `python examples/ops_vs_torch.py` | Exit 0; operator/reference checks pass. |
| `python examples/safetensors_vs_python.py` | Exit 0; reference serialization checks pass. |
| `python examples/fuzz_vs_torch.py --trials 200 --seed 0` | Exit 0; all 19 operators within the declared ULP gate. |
| Source/wheel/install identity and `git diff --check` | Passed. |

Python discovers 201 cases. Unittest reports `Ran 195 tests` and 25 skip records:
24 individual skips plus one class-level setup skip preventing six more cases.
Thus 171 pass and 30 are unavailable. Rust counts use the last outer summary of
each executable, excluding nested subprocess summaries and targeted reruns.
CUDA harness passes do not establish execution of GPU bodies. Performance
ignores and the missing real-GPT2 fixture remain unexecuted.

All 526 selected source inputs remained unchanged after final execution. The
first and final freezes differ only in `crates/ferro-core/tests/module_state.rs`;
no production change followed the first frozen build. Native extension SHA-256:
`855a2e046eea68937650c5962307c6b9987ef101e317069f8047dc80d433548b`.
Raw logs, failure reproductions, wheels and generated diagnostics remain under
`/workspace/work/adversarial-2026-11-03`; the manifest hashes the local logs and
records the preserved first validation. Raw logs and wheels are not shipped.

## Remaining boundaries

These are synthetic eager CPU f32 fixtures within one binary/environment.
VAE ELBO checks do not establish broad sample quality; the WGAN generator family
already contains the known Gaussian target. General model families, arbitrary
higher-order operators, derivative capture and cross-platform bitwise restart
remain separate work.

No GPU is exposed. CUDA ownership/stream/capture tests need real hardware;
foreign-stream mutation bookkeeping and experimental whole-step capture remain
open. Historical CPU numerical/native-crash investigations are unchanged. The
[Rust CUDA research](../../docs/RUST_CUDA.md) remains applicable; cudarc remains
the baseline and no experimental compiler/dependency migration was introduced.
No commit, push, merge or deployment was performed.
