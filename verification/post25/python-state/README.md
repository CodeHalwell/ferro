# Python CPU training state

## Outcome

Native Adam and AdamW are exposed alongside SGD, with `Parameter.trainable`
freezing over the same shared Rust Param slots. `ferro.checkpoint` provides
owning training snapshots, save/load, inspection copies and restore into an
existing compatible Python Module. No model code is serialized or constructed.

```python
import ferro as fr

optimizer = fr.optim.AdamW(model.parameters(), lr=0.001)
generator = fr.Generator(42)
# Train using this explicit generator; stop at a quiescent boundary.
optimizer.zero_grad()
fr.checkpoint.save_training('run/checkpoint', model,
    optimizers={'main': optimizer}, generator=generator, step=100)

# Fresh compatible model/optimizer/generator objects are supported.
step = fr.checkpoint.load_training('run/checkpoint', model,
    optimizers={'main': optimizer}, generator=generator)

# Alternatively snapshot now, continue training, and publish the owning copy later.
snapshot = fr.checkpoint.snapshot_training(model,
    optimizers={'main': optimizer}, generator=generator, step=step)
snapshot.save('run/another-checkpoint')
```

## Supported state and boundaries

- Named parameters, tied slots, freeze flags, persistent buffers, private/shared
  child modules, heterogeneous training modes and immutable configuration.
- Named independent native SGD/Adam/AdamW optimizers, configuration, moments,
  counters, parameter order/binding and one explicit ferro Generator state.
- Whole-contiguous CPU f32 restore targets. Typed checkpoint I/O and owning
  inspection preserve exact dtype bits; this does not enable typed training
  restore. F64 rejection and bit-exact F64 I/O have independent tests.
- Pending gradients are rejected, including during restore. Clear them first.
  Data cursors, accumulation, implicit/global RNG, extra RNG streams, optimizer
  groups and unregistered recurrent state are not captured. Unsupported cursor
  arguments are rejected, not silently serialized. Persistent=False buffers
  intentionally remain target-local.
- The Rust StateModule adapter delegates to core `from_training_state` and
  `load_training_state_into`. All model, optimizer, RNG, tie, mode and config
  validation precedes the native model/optimizer/RNG CPU commit.
- Python mode dictionaries are updated afterward without user setter hooks.
  This is NOT a cross-language transaction against asynchronous exceptions,
  MemoryError, process interruption or concurrent readers. Pause training;
  arbitrary hostile Module subclasses are outside this protocol.
- Core immutable generation publication is reused unchanged: delayed snapshots
  own data, old payloads stay immutable, and one pointer publishes the payload.
  Windows directory power-loss durability remains filesystem/OS dependent.

## Verification

Rebuilt the noneditable wheel with release `-j 2`, installed it into the existing
ferro-py venv, and executed the real native implementation. No GPU or benchmark
was run. No core files were edited by this worker.

```
crates/ferro-py/.venv/Scripts/python.exe -m maturin build --release -j 2 --manifest-path crates/ferro-py/Cargo.toml --out verification/post25/python-state/wheels
crates/ferro-py/.venv/Scripts/python.exe -m pip install --force-reinstall --no-deps verification/post25/python-state/wheels/ferro-0.0.1-cp311-abi3-win_amd64.whl
crates/ferro-py/.venv/Scripts/python.exe verification/post25/python-state/verify.py
cargo test -p ferro-core --test checkpoint --test checkpoint_dtype --test module_state
```

`results.json`: 79 Python tests passed across eight suites, including 13 new
checkpoint/optimizer tests and all 17 frozen API contracts. Trainer, Progress,
architecture and copy/no-grad guards passed. Core targeted suites passed
5 checkpoint, 8 dtype and 16 module-state tests. CPU training examples also ran.

Coverage includes bit-exact stochastic continuation into fresh objects with
Adam, AdamW and momentum/Nesterov SGD together; tied weights; multiple
optimizers; delayed publication; heterogeneous/private children; buffers;
freezing; exact large steps; owning inspection; Torch CPU optimizer parity;
changed parameter order/ties/config; corrupt model/optimizer/RNG/mode/config;
and no live-state mutation after rejected restores.

Initial public RED tests failed against the prior installed wheel for missing
Adam, checkpoint and trainable APIs (six failures before the first wheel build).
A transient concurrent core segment edit initially blocked cargo check; a
subsequent check and release build passed. Rust compiler warnings from other
crates remain. Copy/no-grad tests intentionally catch stale-version panics.

`verify.py` records stable hashes for owned sources and verifies installed
facade bytes match source. It records the loaded native binary hash. Concurrent
other-worker core/CUDA changes are not frozen by this report; it is not a claim
that the installed wheel contains later edits to those other workers' files.
No commits or pushes were made.
