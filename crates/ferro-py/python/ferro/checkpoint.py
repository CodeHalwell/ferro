"""Quiescent CPU training checkpoints, using core immutable generations.

Includes parameters/ties/freezing, persistent buffers, heterogeneous modes,
immutable configuration, named native optimizers and one explicit Generator.
No code is serialized or constructed. Restore into an existing compatible model.
Persistent parameter/buffer StorageCell ties must match, even across distinct
Python wrappers. Parameters may be CPU float32 or float64 (SGD only for float64); persistent buffers may
also use float64, int64 or bool. Persistent None buffer slots are schema-bound.
Only identical whole-contiguous CPU aliases of these types are supported;
distinct aliased views are rejected. External/nonpersistent alias relations and
shared device allocations behind distinct StorageCells are outside this contract.
Older whole-training snapshots without storage schema fail closed on restore.
Ordered optimizer groups, their exact Python scalar settings and native moments
are included. Scheduler state_dict values remain explicit caller-owned state.
Pending gradients, data cursors, implicit/global RNG and unregistered recurrent/
training state are not captured. Clear gradients first.
Native model/optimizer/RNG validation and commits form one CPU transaction.
Python mode dictionaries are updated afterward without user setter hooks; this
is not concurrent-reader isolation or rollback against asynchronous exceptions,
MemoryError, process interruption or malicious Module subclasses. Pause training.
"""
import json
import os
from ._native import _TrainingCheckpoint, Generator
from .nn import Module, Parameter
from .optim import SGD, Adam, AdamW, RMSprop, Adagrad, Adamax, Adadelta, ASGD, Rprop, RAdam, NAdam, LBFGS, Adafactor, Muon

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


def _state(model, optimizers, generator, optimizer_metadata=True):
    from .nn.layers import Dropout
    from .nn.containers import ParameterList, ParameterDict
    if not isinstance(model, Module) or not isinstance(generator, Generator):
        raise TypeError('Checkpoint requires a Module and explicit ferro.Generator')
    if type(optimizers) is not dict or any(type(k) is not str or not k or
            not all(c.isascii() and (c.isalnum() or c == '_') for c in k) for k in optimizers):
        raise TypeError('optimizers must be a dict of nonempty ASCII identifier names')
    if any(type(o) not in (SGD, Adam, AdamW, RMSprop, Adagrad, Adamax, Adadelta, ASGD, Rprop, RAdam, NAdam, LBFGS, Adafactor, Muon) for o in optimizers.values()):
        raise TypeError('Checkpoint requires a supported native optimizer')
    if len({id(o._native) for o in optimizers.values()}) != len(optimizers):
        raise ValueError('Each optimizer must have one unique namespace')
    params, buffers, modes, schema, modules = [], [], [], [], {}
    canonical = {}
    def visit(module, path):
        if isinstance(module, Dropout):
            Dropout._validate_rng_buffer(module.rng_state)
        identity = id(module)
        first = canonical.setdefault(identity, path)
        config = {}
        for name in module._config:
            config[name] = _value(getattr(module, name))
        # Explicit constructors keep immutable public attributes as schema.
        initializer = next(base for base in type(module).__mro__ if '__init__' in base.__dict__)
        if initializer is not Module:
            for name, value in module.__dict__.items():
                if not name.startswith('_') and name != 'training' and name not in module._registered and name not in module._buffers:
                    config[name] = _value(value)
        if isinstance(module, (ParameterList, ParameterDict)):
            entries = tuple(str(index) for index in range(len(module))) if isinstance(module, ParameterList) else tuple(module.keys())
            config['container.entries'] = _value(entries)
            config['container.kinds'] = _value(tuple(module._entry_kinds.items()))
            config['container.registration_order'] = _value(tuple(module._registered))
            names = tuple(dict.fromkeys(entries + tuple(module._entry_kinds)))
            config['container.values'] = _value(tuple((key, getattr(module, key)) for key in names if not isinstance(getattr(module, key), (Module, Parameter))))
        else:
            kinds = getattr(module, '_registered_kinds', {})
            optional = tuple((name, kinds[name]) for name, value in module._registered.items() if value is None and name in kinds)
            if optional:
                config['registration.none'] = _value(optional)
        buffer_schema = [(name, persistent, 'none') if persistent and getattr(module, name) is None else (name, persistent)
                         for name, persistent in module._buffers.items()]
        schema.append([path, first, type(module).__module__, type(module).__qualname__, config, buffer_schema])
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
            if persistent and getattr(module, name) is not None:
                key = path + '.' + name if path else name
                buffers.append((key, getattr(module, name)))
    visit(model, '')
    encoded = json.dumps([FORMAT_VERSION, schema], ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')
    scalars = [('schema.length', len(encoded))] + [('schema.' + str(i), b) for i, b in enumerate(encoded)] + modes
    if optimizer_metadata:
        for name, optimizer in optimizers.items():
            scalars.extend((f'optim_meta.{name}.{key}', value) for key, value in optimizer._checkpoint_metadata().items())
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
        from .nn.layers import Dropout
        tensors = self.tensors()
        has_metadata = any(key.startswith('scalars.optim_meta.') for key in tensors)
        params, buffers, scalars, opts, generator, modules = _state(model, optimizers, generator, has_metadata)
        for name, module in modules.items():
            if isinstance(module, Dropout):
                path = name[len('mode.'):]
                key = 'buffers.' + (path + '.' if path else '') + 'rng_state'
                Dropout._validate_rng_buffer(tensors.get(key))
        staged = {}
        if has_metadata:
            for name, optimizer in optimizers.items():
                prefix = f'scalars.optim_meta.{name}.'
                metadata = {}
                for key, tensor in tensors.items():
                    if not key.startswith(prefix):
                        continue
                    if tensor.dtype != 'float32' or tensor.shape != [4]:
                        raise ValueError('invalid optimizer metadata scalar')
                    words = tensor.tolist()
                    if any(not 0 <= value <= 65535 or value != int(value) for value in words):
                        raise ValueError('invalid optimizer metadata scalar word')
                    metadata[key[len(prefix):]] = sum(int(value) << (16 * i) for i, value in enumerate(words))
                configuration = tensors.get(f'optim.{name}.config')
                if configuration is None:
                    raise ValueError('optimizer configuration is missing')
                staged[name] = optimizer._prepare_checkpoint_metadata(metadata, configuration)
            # Accepted legacy optimizer metadata omits only newly introduced
            # neutral options. Preserve its native scalar key schema while the
            # validated Python staging above upgrades the public dictionaries.
            scalars = [(name, value) for name, value in scalars
                       if not name.startswith('optim_meta.') or 'scalars.' + name in tensors]
        restored = self._native.restore(params, buffers, scalars, opts, generator)
        for name, optimizer in optimizers.items():
            optimizer._commit_checkpoint_metadata(staged.get(name))
        for name, value in restored:
            if name.startswith('mode.'):
                modules[name].__dict__['training'] = bool(value)
        return self.step


def snapshot_training(model, *, optimizers, generator, step=0):
    """Capture an owning snapshot at a zero-gradient, quiescent boundary."""
    if type(step) is not int or not 0 <= step < 2**64:
        raise ValueError('step must be an unsigned 64-bit integer')
    params, buffers, scalars, opts, generator, _ = _state(model, optimizers, generator)
    for optimizer in optimizers.values():
        optimizer._sync()
    return TrainingCheckpoint(_TrainingCheckpoint.snapshot(params, buffers, scalars, opts, generator, step))


def save_training(path, model, *, optimizers, generator, step=0):
    snapshot_training(model, optimizers=optimizers, generator=generator, step=step).save(path)


def load(path):
    """Read a published generation without mutating any live state."""
    return TrainingCheckpoint(_TrainingCheckpoint.load(os.fsdecode(os.fspath(path))))


def load_training(path, model, *, optimizers, generator):
    return load(path).restore(model, optimizers=optimizers, generator=generator)
