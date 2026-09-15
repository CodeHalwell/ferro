"""Synchronous thread-local contexts for the native autograd recorder."""
from contextlib import contextmanager


@contextmanager
def _grad_mode(enabled):
    from . import _native
    previous = _native.set_grad_enabled(enabled)
    try:
        yield
    finally:
        _native.set_grad_enabled(previous)


def no_grad():
    return _grad_mode(False)


def enable_grad():
    return _grad_mode(True)
