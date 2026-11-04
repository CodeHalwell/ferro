"""Optimizer callbacks; registrations are caller-owned and never checkpointed.

State-dict hooks operate on Ferro's owning snapshots. Native training checkpoint
I/O bypasses these callbacks. Callback side effects are not rolled back: a post
hook can raise after a successful update or restore has already committed.
"""
from collections import OrderedDict
from functools import wraps
from ._hooks import RemovableHandle
from ._native import _is_capturing


def _access(optimizer):
    if getattr(optimizer, '_busy', False):
        raise RuntimeError('LBFGS optimizer access is unavailable during its closure')


def _wrap_step(function):
    @wraps(function)
    def call(*args, **kwargs):
        optimizer = args[0]
        _access(optimizer)
        if _is_capturing():
            raise NotImplementedError('Python optimizer steps require eager execution')
        for hook in optimizer._hook_registry('step_pre').values():
            changed = hook(optimizer, args, kwargs)
            if changed is not None:
                if not isinstance(changed, tuple) or len(changed) != 2:
                    raise RuntimeError('optimizer step pre-hook must return None or (new_args, new_kwargs)')
                args, kwargs = changed
        result = function(*args, **kwargs)
        for hook in optimizer._hook_registry('step_post').values():
            hook(optimizer, args, kwargs)
        return result
    return call


def _wrap_state(function):
    @wraps(function)
    def call(optimizer):
        _access(optimizer)
        for hook in optimizer._hook_registry('state_dict_pre').values():
            hook(optimizer)
        result = function(optimizer)
        for hook in optimizer._hook_registry('state_dict_post').values():
            changed = hook(optimizer, result)
            if changed is not None:
                result = changed
        return result
    return call


def _wrap_load(function):
    @wraps(function)
    def call(optimizer, state_dict):
        _access(optimizer)
        pre = optimizer._hook_registry('load_state_dict_pre')
        post = optimizer._hook_registry('load_state_dict_post')
        if pre or post:
            state_dict = state_dict.copy()
        for hook in pre.values():
            changed = hook(optimizer, state_dict)
            if changed is not None:
                state_dict = changed
        result = function(optimizer, state_dict)
        for hook in post.values():
            hook(optimizer)
        return result
    return call


class OptimizerHooks:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, wrapper in (('step', _wrap_step), ('state_dict', _wrap_state), ('load_state_dict', _wrap_load)):
            if name in cls.__dict__:
                setattr(cls, name, wrapper(cls.__dict__[name]))

    def _hook_registry(self, name):
        key = '_optimizer_' + name + '_hooks'
        if key not in self.__dict__:
            self.__dict__[key] = OrderedDict()
        return self.__dict__[key]

    def _register_hook(self, name, hook, prepend=False):
        registry = self._hook_registry(name)
        handle = RemovableHandle(registry)
        registry[handle.id] = hook
        if prepend:
            registry.move_to_end(handle.id, last=False)
        return handle

    def register_step_pre_hook(self, hook):
        return self._register_hook('step_pre', hook)

    def register_step_post_hook(self, hook):
        return self._register_hook('step_post', hook)

    def register_state_dict_pre_hook(self, hook, prepend=False):
        return self._register_hook('state_dict_pre', hook, prepend)

    def register_state_dict_post_hook(self, hook, prepend=False):
        return self._register_hook('state_dict_post', hook, prepend)

    def register_load_state_dict_pre_hook(self, hook, prepend=False):
        return self._register_hook('load_state_dict_pre', hook, prepend)

    def register_load_state_dict_post_hook(self, hook, prepend=False):
        return self._register_hook('load_state_dict_post', hook, prepend)
