# Python API design: modular models without routine constructor boilerplate

> Historical contract/design document with later progress notes. The baseline RED and missing-facade statements below are preserved history, not current status: the frozen native package now passes all 17 contracts. Current supported scope, higher-order limitations, Python checkpoint gap and OPEN fastcpu blocker are authoritative in [PUBLICATION.md](../verification/foundation-wave/pr-readiness/PUBLICATION.md) and [PR body](../verification/foundation-wave/pr-readiness/PR_BODY.md).

## Status and evidence boundary

This is a TARGET contract, not an implemented facade or a claim of architecture support. The opt-in executable RED contracts are `verification/python-api-contract/contract.py`. They use real public imports and computations, no mocks, skips or expected-failure decorators. The filename deliberately avoids default `test_*.py` discovery; invoke it explicitly. No binding, core or existing integration files are changed by this proposal.

Inspected baseline: `crates/ferro-py/src/lib.rs` exports Tensor, Generator, capture, StaticGraph, FusedChain, DLPack, cat/where, safetensors and CUDA utilities. `crates/ferro-py/pyproject.toml` currently uses maturin module-name `ferro`; the installed wheel has a generated `ferro/__init__.py` wrapping its extension. Neither that generated wrapper nor existing Rust nn/optim implementations constitutes the proposed Python ecosystem. Current Python Tensor construction is `Tensor(flat_values, shape)`, with `tolist()`, `item()`, `requires_grad_()`, `grad`, `backward()`, `copy_()` and `to(device)`. Preserve those public spellings.

Sources and priority: `CLAUDE.md`, actual core/binding source, then `docs/FOUNDATIONS_AND_ARCHITECTURE_ROADMAP.md` (especially B1/B2/B6). Concurrent core workers own foundational state, AD, optimizers and data; this document specifies integration seams without declaring their work complete.

## One composable public package

Use a maturin mixed Rust/Python package with a private native extension (proposed `ferro._native`) and authored Python facade. Preserve `from ferro import Tensor` and existing imports when relocating the extension; do not merely add runtime sys.modules aliases. A dedicated implementation worker must verify packaging in a clean environment and run the old binding regressions.

| Public namespace | Responsibility |
|---|---|
| `ferro` | Existing Tensor/Generator and IO/interop exports; grad contexts as explicit new APIs |
| `ferro.nn` | Module, Parameter, Linear, registered containers and subsequent layers |
| `ferro.nn.functional` | Stateless differentiable operations; no separate autograd engine |
| `ferro.nn.losses` | MSELoss first, then validated existing loss primitives |
| `ferro.optim` | SGD first, then Adam/AdamW, groups, state and freezing |
| `ferro.data` | Dataset/loader/sampler/collation, ordinary iterable interoperability |
| `ferro.checkpoint` | Versioned complete training state, distinct from tensor-only safetensors |
| `ferro.training` | Optional Trainer convenience; never a required base class or callback DSL |

Layers, losses and custom models call the same native Tensor operations. Python must not reconstruct tensors through lists/NumPy, detach live intermediates, or force CPU placement to make a missing binding appear supported. Explicit user logging with `item()` or `tolist()` is allowed and visibly synchronizes. Counting-backend/device tests, not these CPU tests, must prove transfer-free execution.

## Module lifecycle and configuration

```python
from ferro import nn

class MLP(nn.Module):
    in_features: int
    hidden_features: int = 32
    out_features: int = 1

    def build(self):
        self.hidden = nn.Linear(self.in_features, self.hidden_features)
        self.output = nn.Linear(self.hidden_features, self.out_features)

    def forward(self, x):
        return self.output(self.hidden(x).relu())

model = MLP(in_features=6, out_features=2)
```

No routine `__init__`, `super()`, decorator, dataclass or input-shape inference is required. Configuration fields are explicit annotations, not guessed from class-body integer values. Unannotated constants remain constants. Exclude ClassVar annotations and private names; merge inherited config in deterministic base-to-derived order. Defaults must be immutable values; reject mutable defaults with an actionable error (a future explicit default factory can relax this). Reject live class-body Parameter, Tensor and Module instances, including annotated defaults, to avoid accidental cross-instance training state. Deferred declarative references may be explored later, but are not the baseline.

Implementation outline: `Module.__init_subclass__` derives an inspectable keyword-only constructor signature from declared fields, without executing generated source or evaluating arbitrary annotation expressions. The common constructor binds/validates arguments before allocating registered state, initializes registries and training=True, assigns per-instance config, then invokes the most-derived `build(self)` exactly once. Missing/unknown arguments raise TypeError before build. Type annotations define constructor fields; do not imply a universal runtime type validator. Layers validate dimensional constraints explicitly.

`build` is an overridable lifecycle hook, not a user-requested rebuild operation. Calling forward, train, eval or to never builds again. A derived build replaces its parent's hook; deliberate composition may call `super().build()` but no superclass call is routine or required. Failed construction propagates its exception and never exposes a usable half-built instance. Structural configuration is read-only after construction: changing dimensions requires a new model. Advanced users compose arbitrary Python in build/forward. A handwritten initializer, if supported later, must not silently bypass registration: provide a documented explicit escape hatch or reject it clearly in v1.

## Registration, identity and placement

- Attribute assignment automatically registers Parameter and Module instances. Ordinary Tensor attributes are transient state unless explicitly registered with `register_buffer(name, tensor, persistent=True)`.
- `Parameter(tensor)` owns a stable Python parameter object and native parameter slot; `.tensor()` gives the differentiable leaf used in forward. This explicit accessor avoids assuming subclassability of the current PyO3 Tensor. Native arithmetic must remain native; no Python value-copy adapter.
- `parameters()` and `named_parameters()` recurse in assignment order, deduplicating by stable parameter identity. First encountered dotted path is canonical. `named_buffers()` returns registered buffers separately. Aliases must remain recoverable for state serialization even though optimizer enumeration is deduplicated.
- Assigning/replacing/deleting registered attributes updates registries. ModuleList and ModuleDict explicitly intercept container mutations; plain Python containers are not recursively guessed or silently registered. Reject cycles in module ownership. Shared acyclic children and explicitly tied parameters are legal.
- Optimizers keep references to parameter slots, not copied tensors or a one-time stale leaf. Deduplicate again defensively at optimizer construction. Contributions from all tied uses accumulate; each parameter is updated once per step. Replacing a parameter is an intentional structural edit: existing optimizers retain their old parameter set until explicitly rebuilt/updated.
- `train(mode=True)` and `eval()` recursively set module behavior and return self. They never change requires_grad, disable recording, clear recurrent state or step optimizers. Buffers follow placement but are not optimized.
- `to(device)` returns self, preserves parameter object/slot identity and alias relationships, and moves registered buffers and nested parameters. A transfer replaces the slot's underlying storage only via a sanctioned core seam; NEVER mutate a StorageCell variant/buffer identity. Existing optimizers resolve the live slot. Validate the whole migration first; do not leave partial placement on failure. Outstanding tapes/captured executions must be rejected or invalidated explicitly when migration makes them stale.
- CPU contracts prove CPU no-op placement and subsequent updates only. Real cross-device identity, buffer movement, optimizer-state migration and capture invalidation need dedicated native/device tests before support claims.

## Three training levels, same model and optimizer machinery

### 1. Standard supervised fit

```python
from ferro.nn.losses import MSELoss
from ferro.optim import SGD
from ferro.training import Trainer

trainer = Trainer(model, optimizer=SGD(model.parameters(), lr=0.01),
                  loss_fn=MSELoss(reduction="mean"))
reports = trainer.fit(batches, epochs=10)
```

Each standard batch is exactly `(inputs, targets)`; users with other signatures choose an objective or custom step. Per batch: zero_grad, forward, scalar loss, backward, step. No mandatory Dataset or callback DSL. Data transfer is explicit before iteration or inside a user adapter, never inferred from model placement.

### 2. Custom objective, trainer-owned updates

```python
def objective(model, batch):
    reconstruction, mu, logvar = model(batch["inputs"], batch["epsilon"])
    return reconstruction_loss(reconstruction, batch["target"]) + gaussian_kl(mu, logvar)

trainer = Trainer(model, optimizer=optimizer, objective=objective)
trainer.fit(batches, epochs=10)
```

`objective(model, batch) -> scalar Tensor`; Trainer performs exactly one zero_grad/backward/step per batch. The objective owns model calls and loss construction, not optimizer updates. Fixed epsilon, KL scaling and RNG streams remain explicit; this syntax does not certify a VAE implementation. Optional `metrics(model, batch, loss) -> dict[str, object]` runs after loss construction and before backward, once per batch, to add reporting without owning updates or recomputing forward. Reserved `loss` may not be overwritten.

### 3. Full custom step, or no Trainer at all

```python
def train_step(model, batch):
    optimizer.zero_grad()
    loss = custom_loss(model, batch)
    loss.backward()
    optimizer.step()
    return {"loss": loss}

Trainer(model, train_step=train_step).fit(batches, epochs=10)

# Equally supported: a completely ordinary Python loop.
for batch in batches:
    optimizer.zero_grad()
    loss = custom_loss(model, batch)
    loss.backward()
    optimizer.step()
```

`train_step` is the concrete public keyword for the full custom-step level. Trainer calls it once and performs ZERO zero_grad/backward/step operations of its own. GAN phase scheduling, multiple optimizers, accumulation, clipping, closures, freezing, projection, explicit detach/truncated BPTT and RNG management belong to this user code. It can return an empty dict when metrics are unnecessary. Arbitrary batches pass through unchanged by identity. External secondary models remain the callback owner's responsibility for modes and restoration.

### Ownership validation and reporting

`Trainer(model, *, optimizer=None, loss_fn=None, objective=None, train_step=None, eval_step=None, metrics=None)` is the minimal constructor. loss_fn/objective/train_step are mutually exclusive. Managed loss/objective requires an optimizer. Custom train_step rejects a trainer optimizer and trainer metrics callback to avoid ambiguous ownership (return metrics directly). An eval-only Trainer may omit all training strategies; fit then raises before consuming data. Invalid combinations fail at construction, before consuming batches or mutating models.

`fit(batches, *, epochs=1)` returns a flat list of per-batch report dictionaries in epoch/batch order. Managed strategies include the exact scalar Tensor under `loss`, plus optional metrics. Custom steps return dictionaries with arbitrary named metrics and no mandatory loss. No implicit detach, item, averaging, device conversion or NumPy conversion occurs. These live results may retain memory; users doing long runs should choose an explicit scalar-logging callback and a future streaming/non-retaining report policy rather than assuming free history. Validate scalar objective/loss before backward; callback return schema errors are descriptive. Non-reiterable iterators cannot silently provide empty later epochs: reject multi-epoch one-shot iterators or require an explicit iterable factory in a later extension.

Trainer scopes recursive training=True during fit and restores EVERY previously registered module's original mode in finally, including heterogeneous child modes, on success or failure. Exceptions propagate; batches are not retried, optimizer updates already performed are not rolled back, and recurrent state is not reset. Mid-loop structural mutation is unsupported in the first version and should fail clearly rather than corrupt the restoration traversal.

## Built-in progress and scalar metrics (implemented Python facade)

`ferro.Progress` (also `ferro.training.Progress`) is shared by ordinary loops and
Trainer. It prints ordinary newline-delimited text to stdout or `stream=StringIO()`;
there are no TTY assumptions, progress bars, background threads, or extra dependencies.

```python
import ferro as fr
from ferro.training import Trainer

# Opt in explicitly to scalar Tensor.item() logging at the selected cadence.
progress = fr.Progress(log_every=10, tensor_metrics=True)
trainer = Trainer(model, optimizer=optimizer, loss_fn=loss_fn, progress=progress,
                  progress_weight=lambda batch: batch[0].shape[0])
trainer.fit(batches, epochs=3)
trainer.evaluate(validation_batches)

# Manual loop: no Trainer and no formatting strings required.
progress = fr.Progress(log_every=10)
for _ in range(3):
    with progress.epoch():
        for inputs, targets in progress.batches(batches):
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            progress.update()  # record one successful optimizer update
            if progress.should_log:
                progress.log(loss=loss.item(), weight=inputs.shape[0])
```

- `epoch_count` counts train epoch entries (including an interrupted epoch), starting
  at one. `epoch(phase="validation")` resets phase-local metrics/batches without
  incrementing the train epoch. Evaluation before training is epoch zero.
- `batch` counts yielded/started batches in the current phase; `global_batch`
  counts them across phases and calls. Neither claims that a yielded batch succeeded.
  Sized iterables display `batch=2/5`; generators display `batch=2`, with no guessed
  total and no materialization. An epoch is `complete` only after source exhaustion;
  early break prints `stopped`, and exceptions propagate without a success line.
- `updates` counts optimizer operations, not batches or backward calls. Managed
  Trainer training records one after each successful step. Custom train steps and
  manual loops call `progress.update()` themselves (or `update(count)` for multiple
  updates); Trainer never guesses, performs, or double-counts their updates.
- Entering an epoch clears metrics and `batch`, not global counters. `reset()` clears
  all counters/metrics between epochs and is rejected while an epoch is active.
- `log(loss=..., accuracy=..., weight=...)` or `log({"nll": ..., "kl": ...})`
  aggregates separately weighted means for each named metric. `weight` defaults to
  one; pass actual sample/token counts for unequal or short batches. Values must be
  explicit scalar batch means, not sums. Multiple `log` calls may supply different
  weights for different metrics. `metrics` exposes a fresh dictionary of means.
  `accuracy` is an explicitly supplied fraction formatted as a percentage;
  other names use four decimals. Classification, target formats, and accuracy
  calculations are never inferred. Missing metrics do not add to their denominator.
- `log_every` controls printing after completed batches, plus an epoch summary.
  `should_log` is a cadence hint inside the loop. Metrics aggregate only values
  actually submitted: sparse logging produces a sampled mean, NOT a full-epoch
  mean. For exact epoch means submit every batch with its correct weight.
- Trainer defaults to `progress=None` (unchanged silent behavior); `False` also
  disables reporting. `True` creates a default reporter for scalar-valued reports
  and counters. Native Tensor-valued reports are not converted by default.
  `Progress(tensor_metrics=True)` explicitly authorizes Trainer to call `item()` on
  scalar tensors only at `should_log` batches; this can synchronize CUDA. It never
  extracts an unlogged final batch merely to fill the summary, so an epoch shorter
  than `log_every` has no sampled tensor metric. Python scalar metrics are aggregated
  every batch. `enabled=False` suppresses output and automatic tensor extraction,
  but keeps counters and supplied scalar metrics.
- `progress_metrics(batch, report) -> dict[str, scalar]` optionally replaces the
  Trainer's metric selection, once per successful batch. `progress_weight(batch)`
  supplies its weight. These explicit adapters may choose extraction cadence and
  custom metrics; user callbacks still run in silent mode. Original report Tensor
  identity, returned report history, and objective/custom-step ownership are unchanged.

Executable coverage: `crates/ferro-py/tests/test_progress.py` uses real CPU tensors
and StringIO to test weighted short batches, generator totals, phases, cadence,
silent reporting, explicit tensor sampling, reset, early break/failure, managed
updates, custom update ownership, and original report identity. No CUDA residency
or synchronization-cost claim is inferred from these CPU tests.

## Evaluation mode is independent from gradient recording

`evaluate(batches, *, grad_enabled=False)` scopes recursive eval mode and an explicit grad-recording context, invokes `eval_step(model, batch)` or the configured managed loss/objective, and returns per-batch metric dictionaries without backward or optimizer steps. It preserves Tensor result identity and returns live graphs when enabled. Both modes and the previous grad-recording context restore in finally, even on errors. `ferro.no_grad()` and `ferro.enable_grad()` must be nested, exception-safe native recording controls, not output detachment after graph construction.

```python
# eval() alone must still permit differentiation with respect to coordinates.
model.eval()
x = coordinates.requires_grad_(True)
value = model(x)

# Trainer evaluation can likewise preserve input-gradient computation.
results = Trainer(model, eval_step=residual_metrics).evaluate(
    coordinate_batches, grad_enabled=True)
```

PINN residual optimization, contractive AE penalties and WGAN-GP require genuinely differentiable gradients and mixed input/parameter derivatives. Enabling recording is necessary but NOT sufficient: current first-order backward is not create_graph. Do not ship a fake higher-order facade that detaches a VJP or uses finite differences without an explicitly separate approximation API. This suite asserts first-order evaluation/input gradients only; B6 analytic mixed-derivative contracts remain mandatory dependencies.

## Architecture-specific extensibility and state

- CNN: same module tree with functional convolution/norm, registered running statistics and train/eval behavior; not implicit training-flag inference from gradient mode.
- RNN/GRU/LSTM: explicit `(output, state)` or clearly owned transient state, explicit reset and detach boundaries, lengths/masks carried in batches. Trainer never infers sequence boundaries.
- GNN/connectome/graph transformer: topology/indices/edge features are explicit batch or buffer objects; graph batching and sparse primitives need their own native proofs. No forced `(x, y)` schema in objectives/custom steps and no hidden dense adjacency conversion.
- KAN: independent edge-wise basis coefficients in Parameters and knots/grids in explicit config or buffers; not a disguised shared-activation MLP.
- AE/VAE/VQ: arbitrary structured forward results, explicit stochastic streams and loss terms; tied weights register once, EMA/codebooks are buffers or explicitly named training state.
- GAN: independent optimizers/phases, deliberate parameter freezing without blocking discriminator-input gradients, explicit gradient penalties only where higher-order AD is supported.
- PINN: arbitrary coordinate batches, boundary/interior losses and grad-enabled evaluation; higher-order derivative capability errors remain visible.

`state_dict`/checkpoint design must cover named parameters, persistent buffers, ties, heterogeneous modes and owning snapshots. Complete training checkpoints additionally need optimizer groups/state, RNG, sampler/cursor, accumulation and multioptimizer phase, and explicitly opted-in recurrent state. Restore must validate everything before mutation, preserve parameter-slot identities and target placement, and use crash-consistent publication. Tensor-only safetensors is not this contract. B2 owns implementation and transactional tests; this initial suite does not pretend to certify them.

## Acceptance and implementation handoff

Run CPU-only against the existing project environment from repository root:

```text
crates/ferro-py/.venv/Scripts/python.exe verification/python-api-contract/contract.py
```

Observed RED run against the existing `crates/ferro-py/.venv/Scripts/python.exe`: CPU `Tensor([1.0], [1]).tolist()` returned `[1.0]`; the contract runner reported `Ran 17 tests`, `FAILED (failures=17)`, exit 1, with no skips or test errors. Failures identify missing `ferro.nn`, `ferro.optim`, `ferro.training`, and `ferro.no_grad`. These first missing seams prevent deeper assertions from executing; the later behavioral assertions remain targets, not individually reproduced runtime defects. No bindings were rebuilt.

The suite deliberately imports missing public modules inside test bodies so unavailable facade modules are clear assertion failures rather than collection errors. It is not included in the existing `crates/ferro-py/tests` suite. Implement one vertical slice at a time: packaging + Module/Parameter/Linear; registration/identity and SGD; grad contexts and MSE; the three Trainer ownership levels; evaluation/restoration. Keep the remaining contracts RED until their actual native seams exist.

These contracts cover constructor/build behavior, per-instance parameter independence, nesting/ties/buffers, registered containers, CPU identity, gradients in eval, grad-context restoration, supervised/explicit parity, objective update ownership, custom multioptimizer execution, mode restoration and explicit recurrent resets. They do not certify all layer families, complete checkpoint transactions, real CUDA residency, cross-device migration, typed loaders or higher-order differentiation. Those need the roadmap's independent gates, not passing mocks or broad claims derived from one scalar model.
