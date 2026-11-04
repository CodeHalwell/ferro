"""Native arithmetic methods and exact scalar-alpha CPU primitives."""
from . import _native


def add(input, other, *, alpha=1, out=None):
    """Fused CPU add with typed promotion and Torch scalar conversion rules."""
    if out is not None:
        raise NotImplementedError('arithmetic out destinations are unsupported')
    return _native._arithmetic('add', input, other, alpha=alpha)


def sub(input, other, *, alpha=1, out=None):
    """Fused CPU subtraction; bool operands are unsupported as in Torch."""
    if out is not None:
        raise NotImplementedError('arithmetic out destinations are unsupported')
    return _native._arithmetic('sub', input, other, alpha=alpha)


subtract = sub


def mul(input, other, *, out=None):
    if out is not None:
        raise NotImplementedError('arithmetic out destinations are unsupported')
    return _native._arithmetic('mul', input, other)


multiply = mul


def true_divide(input, other, *, out=None):
    if out is not None:
        raise NotImplementedError('arithmetic out destinations are unsupported')
    return _native._arithmetic('true_divide', input, other)


def div(input, other, *, rounding_mode=None, out=None):
    """Divide, optionally rounding toward zero or negative infinity."""
    if out is not None:
        raise NotImplementedError('arithmetic out destinations are unsupported')
    return _native._division('div', input, other, rounding_mode=rounding_mode)


divide = div


def floor_divide(input, other, *, out=None):
    """Floor division; Torch's distinct floor-divide derivative is unsupported."""
    if out is not None:
        raise NotImplementedError('arithmetic out destinations are unsupported')
    return _native._division('floor_divide', input, other)
