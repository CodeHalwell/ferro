"""Local forward hooks; registrations are caller-owned and not serialized."""
from collections import OrderedDict
import warnings
from .._hooks import RemovableHandle


def _registries(module, pre):
    if '_registered' not in module.__dict__:
        raise AttributeError('Call super().__init__() before registering hooks')
    name = '_forward_pre_hooks' if pre else '_forward_hooks'
    kwargs = name + '_with_kwargs'
    names = (name, kwargs) if pre else (name, kwargs, '_forward_hooks_always_called')
    all_names = ('_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_forward_hooks', '_forward_hooks_with_kwargs', '_forward_hooks_always_called')
    for key in all_names:
        if key not in module.__dict__:
            object.__setattr__(module, key, OrderedDict())
    return tuple(module.__dict__[key] for key in names)


def register_pre(module, hook, prepend=False, with_kwargs=False):
    registry, kwargs = _registries(module, True)
    handle = RemovableHandle(registry, extra_dict=kwargs)
    registry[handle.id] = hook
    if with_kwargs: kwargs[handle.id] = True
    if prepend: registry.move_to_end(handle.id, last=False)
    return handle


def register_post(module, hook, prepend=False, with_kwargs=False, always_call=False):
    registry, kwargs, always = _registries(module, False)
    handle = RemovableHandle(registry, extra_dict=(kwargs, always))
    registry[handle.id] = hook
    if with_kwargs: kwargs[handle.id] = True
    if always_call: always[handle.id] = True
    if prepend: registry.move_to_end(handle.id, last=False)
    return handle


def call(module, args, kwargs, forward):
    pre = module.__dict__.get('_forward_pre_hooks', {})
    post = module.__dict__.get('_forward_hooks', {})
    pre_kwargs = module.__dict__.get('_forward_pre_hooks_with_kwargs', {})
    post_kwargs = module.__dict__.get('_forward_hooks_with_kwargs', {})
    always = module.__dict__.get('_forward_hooks_always_called', {})
    result, called = None, set()
    try:
        for key, hook in tuple(pre.items()):
            if key in pre_kwargs:
                value = hook(module, args, kwargs)
                if value is not None:
                    if not isinstance(value, tuple) or len(value) != 2:
                        raise RuntimeError('forward pre-hook must return None or a tuple of (new_args, new_kwargs)')
                    args, kwargs = value
            else:
                value = hook(module, args)
                if value is not None:
                    args = value if isinstance(value, tuple) else (value,)
        result = forward(*args, **kwargs)
        post = module.__dict__.get('_forward_hooks', {})
        post_kwargs = module.__dict__.get('_forward_hooks_with_kwargs', {})
        always = module.__dict__.get('_forward_hooks_always_called', {})
        for key, hook in tuple(post.items()):
            if key in always: called.add(key)
            value = hook(module, args, kwargs, result) if key in post_kwargs else hook(module, args, result)
            if value is not None: result = value
        return result
    except Exception:
        post = module.__dict__.get('_forward_hooks', {})
        post_kwargs = module.__dict__.get('_forward_hooks_with_kwargs', {})
        always = module.__dict__.get('_forward_hooks_always_called', {})
        for key, hook in post.items():
            if key in always and key not in called:
                try:
                    value = hook(module, args, kwargs, result) if key in post_kwargs else hook(module, args, result)
                    if value is not None: result = value
                except Exception as error:
                    warnings.warn('module forward hook with ``always_call=True`` raised an exception '
                                  'that was silenced as another error was raised in forward: ' + str(error), stacklevel=2)
        raise
