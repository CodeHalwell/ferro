"""Explicit configuration, automatic registration, and stable parameter identity."""
import inspect
from typing import ClassVar, get_origin
from .._native import Tensor, Parameter


def _immutable(value):
    return isinstance(value, (str, bytes, int, float, bool, type(None))) or (
        isinstance(value, (tuple, frozenset)) and all(_immutable(x) for x in value))


class Module:
    _config = {}
    __signature__ = inspect.Signature()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__init__' in cls.__dict__ and not cls.__dict__.get('_builtin_init', False):
            raise TypeError('Define build(), not __init__(), for Module construction')
        fields = {}
        for base in reversed(cls.__mro__[1:]):
            fields.update(getattr(base, '_config', {}))
        for name, value in cls.__dict__.items():
            if isinstance(value, (Tensor, Parameter, Module)):
                raise TypeError(f'{name}: live class state is shared; create it in build()')
        for name, annotation in cls.__dict__.get('__annotations__', {}).items():
            if name.startswith('_'):
                continue
            if get_origin(annotation) is ClassVar or annotation is ClassVar or (
                isinstance(annotation, str) and annotation.replace('typing.', '').startswith('ClassVar')):
                fields.pop(name, None)
                continue
            default = cls.__dict__.get(name, fields.get(name, inspect.Parameter.empty))
            if default is not inspect.Parameter.empty and not _immutable(default):
                raise TypeError(f'{name}: mutable configuration default; create state in build()')
            fields[name] = default
        cls._config = fields
        cls.__signature__ = inspect.Signature([
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
            for name, default in fields.items()])

    def __init__(self, **kwargs):
        bound = self.__signature__.bind(**kwargs)
        bound.apply_defaults()
        self._initialize()
        for name, value in bound.arguments.items():
            object.__setattr__(self, name, value)
        self.build()

    def _initialize(self):
        object.__setattr__(self, '_registered', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_structure_locked', False)
        self.training = True

    def build(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if name in self._config and '_registered' in self.__dict__:
            raise AttributeError(f'{name} is read-only configuration; construct a new model')
        registry = self.__dict__.get('_registered')
        if registry is not None and name not in ('_registered', '_buffers', '_structure_locked', '_config'):
            structural = name in registry or name in self._buffers or isinstance(value, (Module, Parameter))
            if structural and self._structure_locked:
                raise RuntimeError('Module structure cannot change during Trainer execution')
            if isinstance(value, Module) and any(child is self for child in value.modules()):
                raise ValueError('Module ownership cycle')
            if name in self._buffers:
                if not isinstance(value, Tensor):
                    raise TypeError('Registered buffer must be a Tensor')
            if isinstance(value, (Module, Parameter)):
                registry[name] = value
            else:
                registry.pop(name, None)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._config:
            raise AttributeError('Configuration is read-only')
        if (name in self._registered or name in self._buffers) and self._structure_locked:
            raise RuntimeError('Module structure cannot change during Trainer execution')
        object.__delattr__(self, name)
        self._registered.pop(name, None)
        self._buffers.pop(name, None)

    def register_buffer(self, name, tensor, persistent=True):
        if not isinstance(name, str) or not name or '.' in name or name.startswith('_'):
            raise ValueError('Buffer name must be a nonempty public attribute name without dots')
        if hasattr(self, name) or self._structure_locked:
            raise ValueError('Buffer name already exists or module structure is locked')
        if not isinstance(tensor, Tensor):
            raise TypeError('Buffer must be a Tensor')
        setattr(self, name, tensor)
        self._buffers[name] = bool(persistent)

    def modules(self):
        seen = set()
        def visit(module):
            if id(module) in seen:
                return
            seen.add(id(module))
            yield module
            for child in module._registered.values():
                if isinstance(child, Module):
                    yield from visit(child)
        return visit(self)

    def _named(self, buffers=False, remove_duplicate=True):
        seen = set()
        def visit(module, prefix):
            entries = ((name, getattr(module, name)) for name in module._buffers) if buffers else module._registered.items()
            for name, value in entries:
                if isinstance(value, Module):
                    yield from visit(value, prefix + name + '.')
                elif isinstance(value, (Tensor if buffers else Parameter)):
                    if not remove_duplicate or id(value) not in seen:
                        seen.add(id(value))
                        yield prefix + name, value
            if buffers:
                for name, child in module._registered.items():
                    if isinstance(child, Module):
                        yield from visit(child, prefix + name + '.')
        return visit(self, '')

    def named_parameters(self, *, remove_duplicate=True):
        return self._named(remove_duplicate=remove_duplicate)

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())

    def named_buffers(self, *, remove_duplicate=True):
        return self._named(buffers=True, remove_duplicate=remove_duplicate)

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError('mode must be bool')
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        tensors = [p.tensor() for p in self.parameters()] + [b for _, b in self.named_buffers()]
        if device != 'cpu' and not (isinstance(device, str) and (device == 'cuda' or device.startswith('cuda:'))):
            raise ValueError('Expected cpu or cuda device')
        normalized = 'cuda:0' if device == 'cuda' else device
        if any(t.device != normalized for t in tensors):
            raise NotImplementedError('Cross-device Module migration requires validated parameter-slot and optimizer-state migration; no state changed')
        return self


class ModuleList(Module):
    _builtin_init = True

    def __init__(self, modules=()):
        self._initialize()
        for module in modules:
            self.append(module)

    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError('ModuleList accepts Modules only')
        setattr(self, str(len(self)), module)
        return self

    def __len__(self):
        return len(self._registered)

    def __iter__(self):
        return iter(self._registered.values())

    def __getitem__(self, index):
        return list(self._registered.values())[index]

    def __setitem__(self, index, module):
        if not isinstance(module, Module):
            raise TypeError('ModuleList accepts Modules only')
        key = list(self._registered)[index]
        setattr(self, key, module)

    def __delitem__(self, index):
        keys = list(self._registered)
        key = keys[index]
        delattr(self, key)
        values = list(self._registered.values())
        for key in list(self._registered):
            delattr(self, key)
        for module in values:
            self.append(module)


class ModuleDict(Module):
    _builtin_init = True

    def __init__(self, modules=()):
        self._initialize()
        for key, value in dict(modules).items():
            self[key] = value

    def __setitem__(self, key, value):
        if not isinstance(key, str) or not key or '.' in key or key.startswith('_'):
            raise ValueError('Invalid ModuleDict key')
        if key not in self._registered and hasattr(self, key):
            raise ValueError('ModuleDict key conflicts with an attribute')
        if not isinstance(value, Module):
            raise TypeError('ModuleDict accepts Modules only')
        setattr(self, key, value)

    def __getitem__(self, key):
        return self._registered[key]

    def __delitem__(self, key):
        if key not in self._registered:
            raise KeyError(key)
        delattr(self, key)

    def __iter__(self):
        return iter(self._registered)

    def __len__(self):
        return len(self._registered)

    def items(self):
        return self._registered.items()
