"""Quiescent CPU f32 training checkpoints, using core immutable generations.

Includes parameters/ties/freezing, persistent buffers, heterogeneous modes,
immutable configuration, named native optimizers and one explicit Generator.
No code is serialized or constructed. Restore into an existing compatible model.
Persistent parameter/buffer StorageCell ties must match, even across distinct
Python wrappers. Only identical whole-contiguous CPU f32 aliases are supported;
distinct aliased views are rejected. External/nonpersistent alias relations and
shared device allocations behind distinct StorageCells are outside this contract.
Older whole-training snapshots without storage schema fail closed on restore.
Pending gradients, data cursors, implicit/global RNG, optimizer groups and
unregistered recurrent/training state are not captured. Clear gradients first.
Native model/optimizer/RNG validation and commits form one CPU transaction.
Python mode dictionaries are updated afterward without user setter hooks; this
is not concurrent-reader isolation or rollback against asynchronous exceptions,
MemoryError, process interruption or malicious Module subclasses. Pause training.
"""
import json
import os
from ._native import _TrainingCheckpoint, Generator
from .nn import Module, Parameter
from .optim import SGD, Adam, AdamW

FORMAT_VERSION = 1


def _value(value):
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        return ['float', value.hex()]
    if type(value) is bytes:
        return ['bytes', value.hex()]
    if type(value) in (tuple, frozenset):
        values = [_value(v) for v in value]
        return [type(value).__name__, sorted(values, key=repr) if type(value) is frozenset else values]
    raise TypeError('Checkpoint configuration must contain immutable scalar/tuple/frozenset values')


def _state(model, optimizers, generator):
    if not isinstance(model, Module) or not isinstance(generator, Generator):
        raise TypeError('Checkpoint requires a Module and explicit ferro.Generator')
    if type(optimizers) is not dict or any(type(k) is not str or not k or
            not all(c.isascii() and (c.isalnum() or c == '_') for c in k) for k in optimizers):
        raise TypeError('optimizers must be a dict of nonempty ASCII identifier names')
    if any(type(o) not in (SGD, Adam, AdamW) for o in optimizers.values()):
        raise TypeError('Only native SGD, Adam and AdamW checkpoints are supported')
    if len({id(o._native) for o in optimizers.values()}) != len(optimizers):
        raise ValueError('Each optimizer must have one unique namespace')
    params, buffers, modes, schema, modules = [], [], [], [], {}
    canonical = {}
    def visit(module, path):
        identity = id(module)
        first = canonical.setdefault(identity, path)
        config = {}
        for name in module._config:
            config[name] = _value(getattr(module, name))
        # Built-in layers use handwritten initializers with scalar attributes.
        initializer = next(base for base in type(module).__mro__ if '__init__' in base.__dict__)
        if initializer.__dict__.get('_builtin_init', False):
            for name, value in module.__dict__.items():
                if not name.startswith('_') and name != 'training' and name not in module._registered and name not in module._buffers:
                    config[name] = _value(value)
        schema.append([path, first, type(module).__module__, type(module).__qualname__, config,
                       list(module._buffers.items())])
        if type(module.training) is not bool or 'training' not in module.__dict__:
            raise TypeError('Module training mode must be an ordinary boolean attribute')
        if first == path:
            modes.append(('mode.' + path, int(module.training)))
            modules['mode.' + path] = module
        for name, value in module._registered.items():
            if not name or '.' in name:
                raise ValueError('Checkpoint registration names must be nonempty and contain no dots')
            key = path + '.' + name if path else name
            if isinstance(value, Module):
                visit(value, key)
            elif isinstance(value, Parameter):
                params.append((key, value))
        for name, persistent in module._buffers.items():
            if persistent:
                key = path + '.' + name if path else name
                buffers.append((key, getattr(module, name)))
    visit(model, '')
    encoded = json.dumps([FORMAT_VERSION, schema], ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')
    scalars = [('schema.length', len(encoded))] + [('schema.' + str(i), b) for i, b in enumerate(encoded)] + modes
    opts = [(name, opt._native) for name, opt in optimizers.items()]
    return params, buffers, scalars, opts, generator, modules


class TrainingCheckpoint:
    """Owning snapshot. Delayed save never observes later training mutations."""
    def __init__(self, native):
        if not isinstance(native, _TrainingCheckpoint):
            raise TypeError('Use snapshot_training() or load()')
        self._native = native

    @property
    def step(self):
        return self._native.step

    def tensors(self):
        """Return owning native tensor inspection copies, with exact dtype."""
        return dict(self._native.tensors())

    def save(self, path):
        self._native.save(os.fsdecode(os.fspath(path)))

    def restore(self, model, *, optimizers, generator):
        params, buffers, scalars, opts, generator, modules = _state(model, optimizers, generator)
        restored = self._native.restore(params, buffers, scalars, opts, generator)
        for name, value in restored:
            if name.startswith('mode.'):
                modules[name].__dict__['training'] = bool(value)
        return self.step


def snapshot_training(model, *, optimizers, generator, step=0):
    """Capture an owning snapshot at a zero-gradient, quiescent boundary."""
    if type(step) is not int or not 0 <= step < 2**64:
        raise ValueError('step must be an unsigned 64-bit integer')
    params, buffers, scalars, opts, generator, _ = _state(model, optimizers, generator)
    return TrainingCheckpoint(_TrainingCheckpoint.snapshot(params, buffers, scalars, opts, generator, step))


def save_training(path, model, *, optimizers, generator, step=0):
    snapshot_training(model, optimizers=optimizers, generator=generator, step=step).save(path)


def load(path):
    """Read a published generation without mutating any live state."""
    return TrainingCheckpoint(_TrainingCheckpoint.load(os.fsdecode(os.fspath(path))))


def load_training(path, model, *, optimizers, generator):
    return load(path).restore(model, optimizers=optimizers, generator=generator)
