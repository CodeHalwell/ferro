"""Joint functional derivatives on the existing native autograd graph.

The native call is one union traversal. Tensor.grad_wrt and autograd.vjp retain
legacy permissive zero behavior. Graph release and batched cotangents are not
implemented; create_graph currently requires float32 native derivative paths.
"""
import warnings
from .._native import Tensor, _is_capturing, _forward_ad_is_active
from .._tensor_api import full


def _tensors(value, name):
    if isinstance(value, Tensor):
        return (value,)
    try:
        result = tuple(value)
    except TypeError:
        raise TypeError(name + ' must be a Tensor or iterable of Tensors') from None
    if any(not isinstance(item, Tensor) for item in result):
        raise TypeError(name + ' must contain only Tensors')
    return result


def grad(outputs, inputs, grad_outputs=None, retain_graph=None,
         create_graph=False, only_inputs=True, allow_unused=None,
         is_grads_batched=False, materialize_grads=False):
    """Return one derivative per input using a single jointly seeded traversal.

    Unused inputs raise by default. allow_unused=True returns None, and
    materialize_grads=True instead creates independent zeros requiring grad
    exactly when create_graph is true. Live cotangents retain identity on
    passthrough paths, even without create_graph. Existing .grad slots never
    change. Explicit retain_graph=False, batched cotangents, GradientEdge
    objects, capture and active forward AD are unsupported; default/True retain
    the native graph. Float64 supports first order; higher order is float32.
    """
    for name, value in (('create_graph', create_graph), ('only_inputs', only_inputs),
                        ('is_grads_batched', is_grads_batched),
                        ('materialize_grads', materialize_grads)):
        if type(value) is not bool:
            raise TypeError(name + ' must be bool')
    for name, value in (('retain_graph', retain_graph), ('allow_unused', allow_unused)):
        if value is not None and type(value) is not bool:
            raise TypeError(name + ' must be bool or None')
    if materialize_grads and allow_unused is False:
        raise ValueError('materialize_grads=True requires allow_unused=True or None')
    if retain_graph is False:
        raise NotImplementedError('retain_graph=False requires graph release, which is not implemented')
    if is_grads_batched:
        raise NotImplementedError('batched cotangents are not implemented')
    if _is_capturing():
        raise ValueError('functional differentiation during capture is unsupported')
    if _forward_ad_is_active():
        raise ValueError('reverse queries inside an active forward AD level are not supported')
    if not only_inputs:
        warnings.warn('only_inputs is deprecated and ignored', FutureWarning, stacklevel=2)
    output_values = _tensors(outputs, 'outputs')
    keys = tuple(inputs) if isinstance(inputs, dict) else None
    input_values = _tensors(tuple(inputs.values()) if keys is not None else inputs, 'inputs')
    if not input_values:
        raise ValueError('inputs cannot be empty')
    for name, values in (('outputs', output_values), ('inputs', input_values)):
        for index, value in enumerate(values):
            if not value.requires_grad:
                raise RuntimeError(f'{name}[{index}] does not require grad')
            if value.dtype not in ('float32', 'float64'):
                raise ValueError(name + ' must have floating dtype')
            if create_graph and value.dtype != 'float32':
                raise ValueError('create_graph currently requires float32 outputs and inputs')
    if grad_outputs is None:
        seeds = (None,) * len(output_values)
    elif isinstance(grad_outputs, Tensor):
        seeds = (grad_outputs,)
    else:
        try:
            seeds = tuple(grad_outputs)
        except TypeError:
            raise TypeError('grad_outputs must be a Tensor, iterable, or None') from None
    if len(seeds) != len(output_values):
        raise ValueError('one cotangent is required per output')
    for index, (output, seed) in enumerate(zip(output_values, seeds)):
        if seed is None:
            if output.numel() != 1:
                raise RuntimeError('implicit cotangents require one-element outputs')
        elif not isinstance(seed, Tensor):
            raise TypeError(f'grad_outputs[{index}] must be a Tensor or None')
        elif seed.shape != output.shape:
            raise ValueError('cotangent shape must match its output')
        elif seed.device != output.device:
            raise ValueError('cotangent device must match its output')
        elif seed.dtype not in ('float32', 'float64', 'int64', 'bool'):
            raise ValueError('unsupported cotangent dtype')
    for output in output_values:
        if any(value.device != output.device for value in input_values):
            raise ValueError('outputs and selected inputs must share a device')
    # Cast in the caller's grad mode, not the mode used by derivative recording.
    seeds = tuple(full(output.shape, 1., dtype=output.dtype, device=output.device)
                  if seed is None else seed if seed.dtype == output.dtype else seed.to(dtype=output.dtype)
                  for output, seed in zip(output_values, seeds))
    result = Tensor._vjp_many_wrt_optional(list(output_values), list(input_values), list(seeds), create_graph)
    allow_unused = materialize_grads if allow_unused is None else allow_unused
    if not allow_unused and any(value is None for value in result):
        raise RuntimeError('a differentiated input was not used; set allow_unused=True to receive None')
    # Native Tensor handles are stable. Reuse Python wrappers for passthrough
    # cotangents and repeated requested inputs, just as the original handles.
    identities = {value._id(): value for value in input_values}
    identities.update({seed._id(): seed for seed in seeds})
    values = []
    for value, original in zip(result, input_values):
        if value is None and materialize_grads:
            value = (original._zeros_like_layout().requires_grad_(create_graph)
                     if original.device == 'cpu' else
                     full(original.shape, 0., dtype=original.dtype, device=original.device,
                          requires_grad=create_graph))
        elif value is not None:
            value = identities.setdefault(value._id(), value)
        values.append(value)
    return dict(zip(keys, values)) if keys is not None else tuple(values)


def vjp(output, inputs, seed, *, create_graph=False):
    """Use a same-shape, same-dtype cotangent, held fixed for this VJP.

    Later derivatives can reach ``seed`` only when create_graph is true.
    """
    return tuple(output.vjp_wrt(list(inputs), seed, create_graph=create_graph))


from . import functional
