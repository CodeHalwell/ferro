"""Annotation/build and standard constructors with automatic state registration."""
import inspect
from typing import ClassVar, get_origin
from .._native import Tensor, Parameter


def _immutable(value):
    return isinstance(value, (str, bytes, int, float, bool, type(None))) or (
        isinstance(value, (tuple, frozenset)) and all(_immutable(x) for x in value))


def _registration_name(name):
    if not isinstance(name, str):
        raise TypeError('State names must be strings')
    if not name or '.' in name:
        raise ValueError('State names must be nonempty and contain no dots')


class Module:
    _version = 1
    _config = {}
    __signature__ = inspect.Signature()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        initializer = next(base for base in cls.__mro__ if '__init__' in base.__dict__)
        custom_init = initializer is not Module and not initializer.__dict__.get('_builtin_init', False)
        fields = {}
        if not custom_init:
            for base in reversed(cls.__mro__[1:]):
                fields.update(getattr(base, '_config', {}))
        for name, value in cls.__dict__.items():
            if isinstance(value, (Tensor, Parameter, Module)):
                raise TypeError(f'{name}: live class state is shared; create it in the constructor or build()')
        for name, annotation in cls.__dict__.get('__annotations__', {}).items():
            if name.startswith('_') or custom_init:
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
        if initializer is not Module:
            signature = inspect.signature(cls.__init__)
            cls.__signature__ = signature.replace(parameters=list(signature.parameters.values())[1:])

    def __init__(self, **kwargs):
        if type(self).__init__ is not Module.__init__:
            if kwargs:
                raise TypeError('Module initialization expects no arguments in a custom constructor')
            self._initialize()
            return
        bound = self.__signature__.bind(**kwargs)
        bound.apply_defaults()
        self._initialize()
        for name, value in bound.arguments.items():
            object.__setattr__(self, name, value)
        self.build()

    def _initialize(self):
        object.__setattr__(self, '_registered', {})
        object.__setattr__(self, '_registered_kinds', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_structure_locked', False)
        self.training = True

    def build(self):
        pass

    def __call__(self, *args, **kwargs):
        forward = self.forward
        if not self.__dict__.get('_forward_pre_hooks') and not self.__dict__.get('_forward_hooks'):
            return forward(*args, **kwargs)
        from ._module_hooks import call
        return call(self, args, kwargs, forward)

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        """Register a local input callback; its handle removes the registration."""
        from ._module_hooks import register_pre
        return register_pre(self, hook, prepend, with_kwargs)

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        """Register a local output callback, optionally also on forward errors.

        Hooks can replace outputs and remain caller-owned across checkpoints.
        """
        from ._module_hooks import register_post
        return register_post(self, hook, prepend, with_kwargs, always_call)

    def __setattr__(self, name, value):
        if name in self._config and '_registered' in self.__dict__:
            raise AttributeError(f'{name} is read-only configuration; construct a new model')
        registry = self.__dict__.get('_registered')
        if registry is None and isinstance(value, (Module, Parameter)):
            raise AttributeError('Call super().__init__() before assigning parameters or child modules')
        if registry is not None and name not in ('_registered', '_registered_kinds', '_buffers', '_structure_locked', '_config'):
            structural = name in registry or name in self._buffers or isinstance(value, (Module, Parameter))
            if structural and self._structure_locked:
                raise RuntimeError('Module structure cannot change during Trainer execution')
            kinds = self._registered_kinds
            previous = kinds.get(name)
            if isinstance(value, Parameter):
                kind = 'parameter'
            elif previous == 'parameter':
                if value is not None:
                    raise TypeError('Registered parameters can only be replaced by Parameters or None')
                kind = previous
            elif isinstance(value, Module):
                kind = 'module'
            elif previous == 'module':
                if value is not None:
                    raise TypeError('Registered children can only be replaced by Modules or None')
                kind = previous
            else:
                kind = None
            if kind is not None:
                _registration_name(name)
            if isinstance(value, Module) and any(child is self for child in value.modules()):
                raise ValueError('Module ownership cycle')
            if kind is not None:
                self._buffers.pop(name, None)
            if name in self._buffers:
                if value is not None and not isinstance(value, Tensor):
                    raise TypeError('Registered buffer must be a Tensor or None')
            if kind is not None:
                if previous == 'module' and kind == 'parameter':
                    registry.pop(name)
                    kinds.pop(name)
                registry[name] = value
                kinds[name] = kind
            else:
                registry.pop(name, None)
                kinds.pop(name, None)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._config:
            raise AttributeError('Configuration is read-only')
        if (name in self._registered or name in self._buffers) and self._structure_locked:
            raise RuntimeError('Module structure cannot change during Trainer execution')
        object.__delattr__(self, name)
        self._registered.pop(name, None)
        self._registered_kinds.pop(name, None)
        self._buffers.pop(name, None)

    def register_parameter(self, name, param):
        _registration_name(name)
        if '_registered' not in self.__dict__:
            raise AttributeError('Call super().__init__() before registering parameters')
        if param is not None and not isinstance(param, Parameter):
            raise TypeError('Registered parameters must be Parameters or None')
        if hasattr(self, name) and self._registered_kinds.get(name) != 'parameter':
            raise ValueError('Parameter name already exists as another attribute')
        if self._structure_locked:
            raise RuntimeError('Module structure cannot change during Trainer execution')
        setattr(self, name, param)
        self._registered[name] = param
        self._registered_kinds[name] = 'parameter'

    def add_module(self, name, module):
        _registration_name(name)
        if '_registered' not in self.__dict__:
            raise AttributeError('Call super().__init__() before registering children')
        if module is not None and not isinstance(module, Module):
            raise TypeError('Registered children must be Modules or None')
        if hasattr(self, name) and self._registered_kinds.get(name) != 'module':
            raise ValueError('Module name already exists as another attribute')
        if self._structure_locked:
            raise RuntimeError('Module structure cannot change during Trainer execution')
        setattr(self, name, module)
        self._registered[name] = module
        self._registered_kinds[name] = 'module'

    def register_module(self, name, module):
        self.add_module(name, module)

    def register_buffer(self, name, tensor, persistent=True):
        """Register explicit state; only persistent buffers enter checkpoints.

        Training checkpoints validate persistent parameter/buffer StorageCell ties,
        not Python wrapper identity. Matching whole-contiguous CPU F32/F64/I64/Bool aliases are
        supported; distinct aliased views are rejected. Nonpersistent buffers and
        unregistered tensor attributes (including their alias relations) are excluded.
        None keeps an optional registration slot but has no tensor state.
        """
        _registration_name(name)
        if '_registered' not in self.__dict__:
            raise AttributeError('Call super().__init__() before registering buffers')
        if (hasattr(self, name) and name not in self._buffers) or self._structure_locked:
            raise ValueError('Buffer name already exists or module structure is locked')
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError('Buffer must be a Tensor or None')
        object.__setattr__(self, name, tensor)
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

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        memo = set() if memo is None else memo
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, child in self._registered.items():
                if isinstance(child, Module):
                    path = prefix + ('.' if prefix else '') + name
                    yield from child.named_modules(memo, path, remove_duplicate)

    def named_children(self):
        seen = set()
        for name, child in self._registered.items():
            if isinstance(child, Module) and child not in seen:
                seen.add(child)
                yield name, child

    def children(self):
        return (child for _, child in self.named_children())

    def get_submodule(self, target):
        if not isinstance(target, str):
            raise TypeError('module path must be a string')
        value = self
        for name in target.split('.') if target else ():
            value = getattr(value, name)
            if not isinstance(value, Module):
                raise AttributeError(name + ' is not a Module')
        return value

    def get_parameter(self, target):
        path, _, name = target.rpartition('.')
        value = getattr(self.get_submodule(path), name)
        if not isinstance(value, Parameter):
            raise AttributeError(name + ' is not a Parameter')
        return value

    def get_buffer(self, target):
        path, _, name = target.rpartition('.')
        owner = self.get_submodule(path)
        if name not in owner._buffers:
            raise AttributeError(name + ' is not a registered buffer')
        return getattr(owner, name)

    def _named(self, buffers=False, prefix='', recurse=True, remove_duplicate=True):
        if not isinstance(prefix, str) or type(recurse) is not bool or type(remove_duplicate) is not bool:
            raise TypeError('Traversal expects a string prefix and bool flags')
        seen = set()
        def visit(module, prefix):
            entries = ((name, getattr(module, name)) for name in module._buffers) if buffers else module._registered.items()
            for name, value in entries:
                if isinstance(value, (Tensor if buffers else Parameter)):
                    if not remove_duplicate or id(value) not in seen:
                        seen.add(id(value))
                        yield prefix + name, value
            if recurse:
                for name, child in module._registered.items():
                    if isinstance(child, Module):
                        yield from visit(child, prefix + name + '.')
        return visit(self, prefix + ('.' if prefix else ''))

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self._named(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def parameters(self, recurse=True):
        return (parameter for _, parameter in self.named_parameters(recurse=recurse))

    def requires_grad_(self, requires_grad=True):
        if type(requires_grad) is not bool:
            raise TypeError('requires_grad must be bool')
        from .. import _native
        if _native._is_capturing():
            raise RuntimeError('requires_grad_ requires eager execution')
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        from ._module_state import state_dict
        return state_dict(self, destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        from ._module_state import load_state_dict
        return load_state_dict(self, state_dict, strict, assign)

    def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
        return self._named(buffers=True, prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def buffers(self, recurse=True):
        return (buffer for _, buffer in self.named_buffers(recurse=recurse))

    def apply(self, fn):
        if not callable(fn):
            raise TypeError('Module.apply expects a callable')
        for child in self.children():
            child.apply(fn)
        fn(self)
        return self

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise TypeError('mode must be bool')
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self, set_to_none=True):
        """Clear parameter gradients, including frozen slots; leave buffers alone."""
        from ..optim import _zero_parameter_gradients
        _zero_parameter_gradients(self.parameters(), set_to_none)

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
