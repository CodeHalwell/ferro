"""Parameter containers with stable registered slots and ordered iteration.

Tensor entries use Ferro's owning CPU float32 Parameter constructor. Unlike
Torch, this does not make the source tensor's storage mutable parameter state.
"""
import operator
from collections import OrderedDict
from collections.abc import Mapping

from .._native import Tensor, Parameter
from .module import Module


def _parameter(value):
    return Parameter(value) if isinstance(value, Tensor) and not isinstance(value, Parameter) else value


def _unlocked(container):
    if container._structure_locked:
        raise RuntimeError('Parameter container structure cannot change during Trainer execution')


def _assign(container, key, value):
    value = _parameter(value)
    previous = getattr(container, key, None)
    kind = container._entry_kinds.get(key)
    if kind is None:
        kind = 'parameter' if isinstance(previous, Parameter) else 'module' if isinstance(previous, Module) else None
    if kind == 'parameter' and value is not None and not isinstance(value, Parameter):
        raise TypeError('a registered parameter can only be replaced by a Parameter, Tensor or None')
    if kind == 'module' and value is not None and not isinstance(value, (Module, Parameter)):
        raise TypeError('a registered module can only be replaced by a Module, Parameter or None')
    registered = dict(container._registered)
    Module.__setattr__(container, key, value)
    if isinstance(value, Parameter):
        container._entry_kinds[key] = 'parameter'
        if kind == 'module':
            registered_value = container._registered.pop(key)
            container._registered[key] = registered_value
    elif isinstance(value, Module):
        container._entry_kinds[key] = 'module'
    elif value is None and kind is not None:
        container._entry_kinds[key] = kind
        registered[key] = None
        container._registered.clear()
        container._registered.update(registered)


class ParameterList(Module):
    _builtin_init = True
    _parameter_container = True

    def __init__(self, values=None):
        self._initialize()
        self._size = 0
        self._entry_kinds = {}
        if values is not None:
            self.extend(values)

    def __setattr__(self, key, value):
        if '_entry_kinds' in self.__dict__ and (key in self._entry_kinds or key in self._registered):
            _unlocked(self)
            _assign(self, key, value)
        else:
            Module.__setattr__(self, key, value)

    def __delattr__(self, key):
        Module.__delattr__(self, key)
        self._entry_kinds.pop(key, None)

    def __len__(self):
        return self._size

    def __iter__(self):
        return (getattr(self, str(index)) for index in range(len(self)))

    def _index(self, index):
        index = operator.index(index)
        if not -len(self) <= index < len(self):
            raise IndexError('ParameterList index out of range')
        return str(index % len(self))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ParameterList(self[item] for item in range(*index.indices(len(self))))
        return getattr(self, self._index(index))

    def __setitem__(self, index, value):
        _unlocked(self)
        key = self._index(index)
        _assign(self, key, value)

    def append(self, value):
        _unlocked(self)
        _assign(self, str(len(self)), value)
        self._size += 1
        return self

    def extend(self, values):
        _unlocked(self)
        if isinstance(values, Tensor):
            raise TypeError('ParameterList.extend requires an iterable of entries, not a Tensor')
        if values is self:
            values = list(values)
        for value in values:
            self.append(value)
        return self

    def __iadd__(self, values):
        return self.extend(values)

    def forward(self, *args, **kwargs):
        raise RuntimeError('ParameterList is a container and cannot be called')


class ParameterDict(Module):
    _builtin_init = True
    _parameter_container = True

    def __init__(self, parameters=None):
        self._initialize()
        self._keys = {}
        self._entry_kinds = {}
        if parameters is not None:
            self.update(parameters)

    def __setattr__(self, key, value):
        if '_entry_kinds' in self.__dict__ and (key in self._entry_kinds or key in self._registered):
            _unlocked(self)
            _assign(self, key, value)
        else:
            Module.__setattr__(self, key, value)

    def __delattr__(self, key):
        Module.__delattr__(self, key)
        self._entry_kinds.pop(key, None)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, key):
        return key in self._keys

    def __reversed__(self):
        return reversed(self._keys)

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        _unlocked(self)
        if not isinstance(key, str):
            raise TypeError('ParameterDict keys must be strings')
        if not key or '.' in key:
            raise ValueError('ParameterDict keys must be nonempty strings without dots')
        if key not in self._keys and hasattr(self, key):
            raise ValueError('ParameterDict key conflicts with an attribute')
        _assign(self, key, value)
        self._keys[key] = None

    def __delitem__(self, key):
        _unlocked(self)
        if key not in self._keys:
            raise KeyError(key)
        delattr(self, key)
        del self._keys[key]
        self._entry_kinds.pop(key, None)

    def keys(self):
        return self._keys.keys()

    def values(self):
        return (self[key] for key in self)

    def items(self):
        return ((key, self[key]) for key in self)

    def update(self, parameters):
        _unlocked(self)
        if isinstance(parameters, (OrderedDict, ParameterDict)):
            entries = parameters.items()
        elif isinstance(parameters, Mapping):
            entries = sorted(parameters.items())
        else:
            entries = parameters
        if entries is self:
            entries = list(self.items())
        for key, value in entries:
            self[key] = value

    def clear(self):
        _unlocked(self)
        for key in list(self):
            del self[key]

    def pop(self, key):
        value = self[key]
        del self[key]
        return value

    def popitem(self):
        if not self._keys:
            raise KeyError('popitem from an empty ParameterDict')
        key = next(reversed(self._keys))
        return key, self.pop(key)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def copy(self):
        return ParameterDict(self.items())

    def fromkeys(self, keys, default=None):
        return ParameterDict((key, default) for key in keys)

    def forward(self, *args, **kwargs):
        raise RuntimeError('ParameterDict is a container and cannot be called')
