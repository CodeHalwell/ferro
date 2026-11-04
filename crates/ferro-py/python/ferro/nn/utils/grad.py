"""CPU gradient clipping using native tensors and stable real norms.

Stable norm accumulation can remain finite where Torch's intermediate powers
overflow. This is an eager implementation; foreach and sparse gradients reject.
"""
import math

from ... import _native
from ..._native import Tensor, Parameter, no_grad


def _list(values):
    return [values] if isinstance(values, (Tensor, Parameter)) else list(values)


def _validate(tensors, foreach):
    if _native._is_capturing():
        raise NotImplementedError('gradient utilities require eager execution')
    if foreach not in (None, False) or foreach is not None and type(foreach) is not bool:
        raise NotImplementedError('foreach gradient utilities are unsupported')
    for value in tensors:
        if not isinstance(value, Tensor):
            raise TypeError('gradient values must be Tensors')
        if value.device != 'cpu' or value.dtype not in ('float32', 'float64'):
            raise NotImplementedError('gradient utilities support dense CPU float32/float64 only')


def _gradients(parameters, foreach):
    tensors = []
    for parameter in _list(parameters):
        if isinstance(parameter, Parameter):
            parameter = parameter.tensor()
        if not isinstance(parameter, Tensor):
            raise TypeError('parameters must contain Tensors or Parameters')
        tensors.append(parameter)
    _validate(tensors, foreach)
    gradients = [value.grad for value in tensors]
    gradients = [value for value in gradients if value is not None]
    _validate(gradients, foreach)
    return gradients


def _scalar(value, dtype):
    return Tensor.from_f64([value], []) if dtype == 'float64' else Tensor([value], [])


def _number(value, name):
    if type(value) not in (int, float):
        raise TypeError(name + ' must be an int or float')
    return float(value)


def get_total_norm(tensors, norm_type=2., error_if_nonfinite=False, foreach=None):
    """Return the detached norm of per-tensor norms; an empty list has norm zero.

    ord=0 counts nonzero per-tensor norms, matching the hierarchical Torch API.
    """
    from ...linalg import vector_norm
    tensors = _list(tensors)
    _validate(tensors, foreach)
    order = _number(norm_type, 'norm_type')
    if type(error_if_nonfinite) is not bool:
        raise TypeError('error_if_nonfinite must be bool')
    dtype = 'float64' if any(value.dtype == 'float64' for value in tensors) else 'float32'
    with no_grad():
        norms = []
        for value in tensors:
            if order == 0:
                norms.append(float(sum(item != 0 for item in value.reshape([value.numel()]).tolist())))
            elif math.isnan(order):
                norms.append(float('nan'))
            else:
                norms.append(vector_norm(value, ord=order).item())
        if not norms:
            norm = _scalar(0., dtype)
        elif order == 0:
            norm = _scalar(float(sum(value != 0 for value in norms)), dtype)
        elif math.isnan(order):
            norm = _scalar(float('nan'), dtype)
        else:
            values = Tensor.from_f64(norms, [len(norms)]) if dtype == 'float64' else Tensor(norms, [len(norms)])
            norm = vector_norm(values, ord=order)
    if error_if_nonfinite and not math.isfinite(norm.item()):
        raise RuntimeError('the total gradient norm is non-finite; gradients were not modified')
    return norm


def _clip(gradients, max_norm, total_norm):
    if not isinstance(total_norm, Tensor) or total_norm.ndim != 0:
        raise TypeError('total_norm must be a scalar Tensor')
    _validate([total_norm], None)
    limit = _number(max_norm, 'max_norm')
    with no_grad():
        coefficient = (total_norm + 1e-6)
        coefficient = limit / coefficient
        coefficient = coefficient.clamp(-float('inf'), 1.)
        for value in gradients:
            value.copy_(value * coefficient.to(dtype=value.dtype))


def clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=None):
    """Scale existing gradient slots in place using a precomputed scalar norm."""
    gradients = _gradients(parameters, foreach)
    _clip(gradients, max_norm, total_norm)


def clip_grad_norm_(parameters, max_norm, norm_type=2., error_if_nonfinite=False, foreach=None):
    """Clip gradients in place and return their original detached total norm."""
    gradients = _gradients(parameters, foreach)
    _number(max_norm, 'max_norm')
    norm = get_total_norm(gradients, norm_type, error_if_nonfinite, foreach)
    _clip(gradients, max_norm, norm)
    return norm


def clip_grad_value_(parameters, clip_value, foreach=None):
    """Clamp existing gradients without changing parameters or missing slots."""
    gradients = _gradients(parameters, foreach)
    limit = _number(clip_value, 'clip_value')
    with no_grad():
        for value in gradients:
            if math.isnan(limit):
                replacement = _scalar(limit, value.dtype).expand(*value.shape)
            elif limit < 0:
                replacement = value.clamp(limit, limit)
            else:
                replacement = value.clamp(-limit, limit)
            value.copy_(replacement)
