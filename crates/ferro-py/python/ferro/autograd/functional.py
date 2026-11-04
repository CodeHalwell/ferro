"""CPU functional derivatives composed from native reverse-mode VJPs.

Jacobian rows use individual reverse sweeps. JVP contracts these rows with a
direction; this is not a forward-mode or vectorized implementation. Float64
and mixed float32/float64 arguments support first order only. Jacobian blocks
use each input's dtype; mixed JVP accumulation therefore includes rounding at
the input precision before conversion to the output dtype. Float32 higher derivatives retain native operator
support limits and reject unsupported derivative paths. Strict JVP uses two
reverse sweeps to validate derivative connectivity and requires these float32
higher-order rules even when its returned result is detached.
Directions may use float32, float64, int64 or bool; each is converted to its
paired primal/output dtype, preserving live floating-point cast derivatives.
"""
from .._native import Tensor, stack, enable_grad, _is_capturing, _forward_ad_is_active, _forward_ad_vjp_many
from .._tensor_api import full, _shape


def _tuple(value, name):
    single = isinstance(value, Tensor)
    values = (value,) if single else value
    if not isinstance(values, tuple) or not values or any(not isinstance(item, Tensor) for item in values):
        raise TypeError(name + ' must be a Tensor or a nonempty tuple of Tensors')
    return values, single


def _options(create_graph, strict, vectorize=False, strategy='reverse-mode'):
    if any(type(value) is not bool for value in (create_graph, strict, vectorize)):
        raise TypeError('create_graph, strict and vectorize must be bool')
    if vectorize or strategy != 'reverse-mode':
        raise NotImplementedError('functional derivatives currently use unvectorized reverse-mode sweeps')
    if _is_capturing():
        raise ValueError('functional differentiation during capture is unsupported')


def _check(values, dtype=None):
    for value in values:
        if value.device != 'cpu':
            raise NotImplementedError('functional derivatives currently support CPU tensors only')
        if value.dtype not in ('float32', 'float64') or dtype is not None and value.dtype != dtype:
            raise ValueError('functional derivative tensors require float32 or float64 dtype, matching each direction')
    return values[0].dtype if dtype is None else dtype


def _prepare(inputs, create_graph):
    values, single = _tuple(inputs, 'inputs')
    dtype = _check(values)
    if create_graph and any(value.dtype != 'float32' for value in values):
        raise NotImplementedError('create_graph currently supports float32 only')
    # A separate view per argument splits duplicate Tensor argument identities.
    prepared = tuple(value.reshape(value.shape) if create_graph and value.requires_grad
                     else value.detach().requires_grad_(True) for value in values)
    return prepared, single, dtype


def _outputs(func, inputs, dtype):
    if not callable(func):
        raise TypeError('func must be callable')
    values, single = _tuple(func(*inputs), 'outputs')
    _check(values)
    return values, single


def _finish(value, create_graph):
    if isinstance(value, Tensor):
        return value if create_graph else value.detach()
    return tuple(_finish(item, create_graph) for item in value)


def _structure(values, single):
    return values[0] if single else tuple(values)


def _strict(outputs, inputs):
    for index, output in enumerate(outputs):
        if not output.requires_grad:
            raise RuntimeError(f'output {index} does not require gradients in strict mode')
    for index, value in enumerate(inputs):
        if not any(output._depends_on(value) for output in outputs):
            raise RuntimeError(f'outputs are independent of input {index} in strict mode')


def _zeros(value):
    return full(value.shape, 0., dtype=value.dtype)


def _vector(v, values, single, create_graph, name):
    if v is None:
        if len(values) != 1 or values[0].numel() != 1:
            raise RuntimeError(name + ' can be omitted only for a single one-element Tensor')
        return (full(values[0].shape, 1., dtype=values[0].dtype),)
    vectors, _ = _tuple(v, name)
    if len(vectors) != len(values):
        raise RuntimeError(name + ' must contain one direction per corresponding Tensor')
    for vector in vectors:
        if vector.device != 'cpu':
            raise NotImplementedError('functional derivative directions currently require CPU tensors')
        if vector.dtype not in ('float32', 'float64', 'int64', 'bool'):
            raise ValueError('unsupported functional derivative direction dtype')
    if any(vector.shape != value.shape for vector, value in zip(vectors, values)):
        raise RuntimeError(name + ' shapes must match')
    return tuple((vector.reshape(vector.shape) if create_graph and vector.requires_grad else vector.detach()).to(dtype=value.dtype)
                 for vector, value in zip(vectors, values))


def _vjp(outputs, inputs, vectors, create_graph, strict, preserve_passthrough=False, freeze_created_after=None, freeze_excluded=(), _forward=False, _functional=False, _caller_grad_mode=False):
    if _forward_ad_is_active() and not _forward:
        raise ValueError('reverse queries inside an active forward AD level are not supported')
    if strict:
        _strict(outputs, inputs)
    if len(outputs) != len(vectors):
        raise RuntimeError('one cotangent is required per output')
    active = [(output, vector) for output, vector in zip(outputs, vectors) if output.requires_grad]
    roots, seeds = [pair[0] for pair in active], [pair[1] for pair in active]
    if _forward:
        totals = _forward_ad_vjp_many(roots, list(inputs), seeds)
    else:
        frozen = create_graph and freeze_created_after is not None and not strict
        totals = Tensor._vjp_many_wrt_transform(roots, list(inputs), seeds, create_graph,
                                               preserve_passthrough and not strict,
                                               freeze_created_after if frozen else None,
                                               list(freeze_excluded) if frozen else None, _functional, _caller_grad_mode)
    if strict and any(total is None for total in totals):
        raise RuntimeError('an input has no derivative in strict mode')
    results = tuple(_zeros(value) if total is None else total for total, value in zip(totals, inputs))
    if strict and create_graph and any(not value.requires_grad for value in results):
        raise RuntimeError('the derivative is independent of the inputs in strict create_graph mode')
    return results


def _jacobian(outputs, inputs, create_graph, strict, _forward=False, _functional=False, _caller_grad_mode=False):
    blocks = []
    for output in outputs:
        if strict:
            _strict((output,), inputs)
            if not output.numel():
                _vjp((output,), inputs, (_zeros(output),), create_graph, True, _forward=_forward, _functional=_functional, _caller_grad_mode=_caller_grad_mode)
        for value in inputs:
            _shape([*output.shape, *value.shape], value.dtype)
        _shape([output.numel()], 'int64')
        rows = [[] for _ in inputs]
        for index in range(output.numel()):
            data = [0.] * output.numel()
            data[index] = 1.
            seed = Tensor(data, output.shape) if output.dtype == 'float32' else Tensor.from_f64(data, output.shape)
            derivatives = _vjp((output,), inputs, (seed,), create_graph, strict, _forward=_forward, _functional=_functional, _caller_grad_mode=_caller_grad_mode)
            for entries, derivative in zip(rows, derivatives):
                entries.append(derivative)
        if rows and not output.numel() and (create_graph or _forward):
            empty_derivatives = _vjp((output,), inputs, (_zeros(output),), True, strict, _forward=_forward, _functional=_functional, _caller_grad_mode=_caller_grad_mode)
            blocks.append(tuple(full([*output.shape, *value.shape], 0., dtype=value.dtype)
                                + derivative.reshape([derivative.numel()]).narrow(0, 0, 0).sum()
                                for derivative, value in zip(empty_derivatives, inputs)))
        else:
            blocks.append(tuple(stack(entries, 0).reshape([*output.shape, *value.shape]) if entries
                                else full([*output.shape, *value.shape], 0., dtype=value.dtype)
                                for entries, value in zip(rows, inputs)))
    return tuple(blocks)


def jacobian(func, inputs, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode'):
    """Return output-shape + input-shape blocks, preserving Tensor/tuple structure."""
    _options(create_graph, strict, vectorize, strategy)
    with enable_grad():
        values, input_single, dtype = _prepare(inputs, create_graph)
        outputs, output_single = _outputs(func, values, dtype)
        blocks = _jacobian(outputs, values, create_graph, strict)
        result = _structure(tuple(_structure(row, input_single) for row in blocks), output_single)
        return _finish(result, create_graph)


def vjp(func, inputs, v=None, create_graph=False, strict=False):
    """Return func(inputs) and a cotangent-weighted derivative without changing grad slots."""
    _options(create_graph, strict)
    with enable_grad():
        values, input_single, dtype = _prepare(inputs, create_graph)
        outputs, output_single = _outputs(func, values, dtype)
        vectors = _vector(v, outputs, output_single, create_graph, 'v')
        result = _vjp(outputs, values, vectors, create_graph, strict)
        return _finish((_structure(outputs, output_single), _structure(result, input_single)), create_graph)


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    """Contract reverse Jacobians; strict mode uses reverse-over-reverse connectivity."""
    _options(create_graph, strict)
    with enable_grad():
        values, input_single, dtype = _prepare(inputs, create_graph)
        vectors = _vector(v, values, input_single, create_graph, 'v')
        outputs, output_single = _outputs(func, values, dtype)
        if strict:
            if any(value.dtype != 'float32' for value in (*values, *outputs)):
                raise NotImplementedError('strict JVP requires float32 native higher-order rules')
            seeds = tuple(_zeros(output).requires_grad_(True) for output in outputs)
            first = _vjp(outputs, values, seeds, True, True)
            results = _vjp(first, seeds, vectors, create_graph, True)
            return _finish((_structure(outputs, output_single), _structure(results, output_single)), create_graph)
        blocks = _jacobian(outputs, values, create_graph, False)
        results = []
        for output, row in zip(outputs, blocks):
            result = None
            for block, value, vector in zip(row, values, vectors):
                term = (block.reshape([output.numel(), value.numel()]) * vector.reshape([1, value.numel()])).sum_dim(1).reshape(output.shape)
                if term.dtype != output.dtype:
                    term = term.to(dtype=output.dtype)
                result = term if result is None else result + term
            results.append(result)
        return _finish((_structure(outputs, output_single), _structure(results, output_single)), create_graph)


def hessian(func, inputs, create_graph=False, strict=False, vectorize=False, outer_jacobian_strategy='reverse-mode'):
    """Differentiate scalar first gradients; float32 native higher-order rules apply."""
    _options(create_graph, strict, vectorize, outer_jacobian_strategy)
    original, single = _tuple(inputs, 'inputs')
    _check(original)
    if any(value.dtype != 'float32' for value in original):
        raise NotImplementedError('Hessian computation currently supports float32 only')
    def first(*values):
        outputs, output_single = _outputs(func, values, 'float32')
        if not output_single or outputs[0].numel() != 1:
            raise RuntimeError('hessian func must return a single one-element Tensor')
        result = _vjp(outputs, values, (full(outputs[0].shape, 1.),), True, strict)
        return _structure(result, single)
    return jacobian(first, inputs, create_graph=create_graph, strict=strict)
