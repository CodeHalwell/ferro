"""Tensor state dictionaries with stable parameter and buffer destinations."""
from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from itertools import product

from .. import _native
from .._native import Tensor, Parameter, no_grad


class _IncompatibleKeys(namedtuple('_IncompatibleKeys', 'missing_keys unexpected_keys')):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()


def _modules(module, prefix=''):
    from .module import Module
    yield prefix, module
    for name, child in module._registered.items():
        if isinstance(child, Module):
            yield from _modules(child, prefix + name + '.')


def _entries(module, prefix=''):
    from .module import Module
    for name, value in module._registered.items():
        if isinstance(value, Parameter):
            yield prefix + name, value
    for name, persistent in module._buffers.items():
        value = getattr(module, name)
        if persistent and value is not None:
            yield prefix + name, value
    for name, value in module._registered.items():
        if isinstance(value, Module):
            yield from _entries(value, prefix + name + '.')


def state_dict(module, destination=None, prefix='', keep_vars=False):
    """Return shallow Tensor aliases; configuration belongs to training checkpoints."""
    if not isinstance(prefix, str) or type(keep_vars) is not bool:
        raise TypeError('state_dict expects a string prefix and bool keep_vars')
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    metadata = getattr(destination, '_metadata', None)
    if metadata is not None:
        for path, value in _modules(module, prefix):
            metadata[path[:-1]] = {'version': value._version}
    for name, value in _entries(module, prefix):
        destination[name] = value if keep_vars else value.detach()
    return destination


def load_state_dict(module, values, strict=True, assign=False):
    """Preflight CPU tensor state before copying into stable existing slots.

    Assigning replacement parameters and non-whole destination views are
    unsupported. Missing/unexpected keys are returned when strict is false;
    incompatible values reject before any destination is changed.
    """
    if _native._is_capturing():
        raise NotImplementedError('load_state_dict requires eager execution')
    if _native._forward_ad_is_active():
        raise NotImplementedError('load_state_dict does not support active forward AD')
    if not isinstance(values, Mapping):
        raise TypeError('load_state_dict expects a mapping')
    if type(strict) is not bool or type(assign) is not bool:
        raise TypeError('strict and assign must be bool')
    if assign:
        raise NotImplementedError('assign=True requires parameter-slot replacement; no state changed')
    metadata = getattr(values, '_metadata', {})
    values = OrderedDict(values)
    from .layers import _BatchNorm
    from ._instance_norm import _InstanceNorm
    for prefix, child in _modules(module):
        version = metadata.get(prefix[:-1], {}).get('version')
        if isinstance(child, _BatchNorm) and not isinstance(child, _InstanceNorm) and (version is None or version < 2):
            name = prefix + 'num_batches_tracked'
            if child.track_running_stats and child.num_batches_tracked is not None and name not in values:
                values[name] = child.num_batches_tracked
    targets = OrderedDict(_entries(module))
    missing = [name for name in targets if name not in values]
    unexpected = [name for name in values if name not in targets]
    errors = []
    if strict:
        if missing:
            errors.append('Missing key(s): ' + ', '.join(repr(name) for name in missing))
        if unexpected:
            errors.append('Unexpected key(s): ' + ', '.join(repr(name) for name in unexpected))
    staged, shadows = [], []
    with no_grad():
        for name, target in targets.items():
            if name not in values:
                continue
            value = values[name]
            if not isinstance(value, Tensor):
                errors.append(name + ': expected a Tensor')
                continue
            if target.device != 'cpu' or value.device != 'cpu':
                errors.append(name + ': loading currently requires CPU tensors')
                continue
            if target.shape == [] and value.shape == [1]:
                value = value.reshape([])
            if value.shape != target.shape:
                errors.append(name + ': size mismatch, expected ' + str(target.shape) + ', got ' + str(value.shape))
                continue
            try:
                target.copy_(target)
                for owner, shadow in shadows:
                    if value._shares_storage(owner):
                        if value.numel() == 0:
                            value = shadow.narrow(0, 0, 0).reshape(value.shape)
                        elif value.is_contiguous():
                            value = shadow.narrow(0, value.storage_offset(), value.numel()).reshape(value.shape)
                        else:
                            offset, strides, shape = value.storage_offset(), value.stride(), value.shape
                            indices = [offset + sum(index * stride for index, stride in zip(position, strides))
                                       for position in product(*(range(size) for size in shape))]
                            value = shadow.index_select(0, indices).reshape(shape)
                        break
                payload = value.detach().to(dtype=target.dtype).clone()
                staged.append((target, payload))
                for index, (owner, _) in enumerate(shadows):
                    if target._shares_storage(owner):
                        shadows[index] = (owner, payload.reshape([payload.numel()]))
                        break
                else:
                    shadows.append((target, payload.reshape([payload.numel()])))
            except (ValueError, RuntimeError, NotImplementedError) as error:
                errors.append(name + ': ' + str(error))
        if errors:
            raise RuntimeError('Error(s) in loading state_dict for ' + type(module).__name__ + ':\n\t' + '\n\t'.join(errors))
        for target, value in staged:
            target.copy_(value)
    return _IncompatibleKeys(missing, unexpected)
