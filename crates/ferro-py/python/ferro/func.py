"""Eager CPU float32 function transforms over the native reverse-mode engine.

Transform entry and VJP closures honor their invocation-time grad mode,
including identity returns under no_grad. jacrev and hessian use individual
reverse sweeps; jacfwd propagates native forward columns in one evaluation.
JVP uses a scoped native forward level on the same Op recorder;
opaque operators may use their local reverse rule without reconstructing an
outer backward tape. No transform implements vmap. Inputs/outputs/auxiliary
values may be Tensor trees built from tuples, lists and dictionaries.
Derivative graphs retain native higher-order operator limits; unsupported
paths raise explicitly. Each selected argument and output tree must contain
at least one Tensor; auxiliary trees may be empty.
JVP supports one forward level and composes with grad/grad_and_value through
the same graph-building reverse rules. Sequential reverse transforms are
supported; nested reverse transforms inside one forward level reject.
Nested levels, other reverse queries
inside the level, and in-place mutation reject. Returned tangents remain
differentiable after the level closes.
Direct local rules preserve their derivative graphs; opaque graph-building
fallbacks retain the native higher-order support boundary.
JVP directions may be float32 or float64. VJP cotangents also accept int64 and
bool, converting to the output dtype in the pullback invocation's grad mode.
"""
from functools import wraps
from contextlib import nullcontext

from ._native import Tensor, cat, enable_grad, no_grad, is_grad_enabled, stack
from . import _native
from ._tensor_api import full, _shape
from .autograd import functional as af


def _flatten(value):
    if isinstance(value, Tensor):
        return [value], ('tensor',)
    if type(value) in (tuple, list, dict):
        keys = tuple(value) if isinstance(value, dict) else None
        children = list(value.values()) if keys is not None else value
        leaves, specs = [], []
        for child in children:
            values, spec = _flatten(child)
            leaves.extend(values)
            specs.append(spec)
        return leaves, (type(value).__name__, keys, tuple(specs))
    raise TypeError('function-transform trees must contain only Tensors and tuple/list/dict containers')


def _unflatten(values, spec):
    iterator = iter(values)
    def build(current):
        if current[0] == 'tensor':
            return next(iterator)
        children = [build(child) for child in current[2]]
        if current[0] == 'dict':
            return dict(zip(current[1], children))
        return tuple(children) if current[0] == 'tuple' else children
    return build(spec)


def _post(value, excluded, enabled=True, aliases=None, cutoff=None):
    leaves, spec = _flatten(value)
    aliases = {} if aliases is None else aliases
    return _unflatten([aliases[leaf._id()] if leaf._id() in aliases
                       else leaf if cutoff is not None and leaf._id() < cutoff
                       or enabled and leaf._has_grad_leaf_except(excluded) else leaf.detach() for leaf in leaves], spec)


def _entry():
    af._options(False, False)
    return is_grad_enabled(), _native._tensor_id_watermark()


def _argnums(argnums, count):
    single = isinstance(argnums, int)
    values = (int(argnums),) if single else argnums
    if not isinstance(values, tuple) or not values or any(not isinstance(index, int) for index in values):
        raise TypeError('argnums must be an integer or a nonempty tuple of integers')
    if any(not -count <= index < count for index in values):
        raise RuntimeError('argnums index is outside positional arguments')
    normalized = tuple(index % count for index in values)
    if len(set(normalized)) != len(normalized):
        raise RuntimeError('argnums must not contain duplicate positions')
    return normalized, single


def _prepare(args, argnums, forward=False):
    positions, single = _argnums(argnums, len(args))
    originals, specifications, counts = [], [], []
    for position in positions:
        leaves, spec = _flatten(args[position])
        if not leaves:
            raise ValueError('each selected argument must contain at least one Tensor')
        originals.extend(leaves)
        specifications.append(spec)
        counts.append(len(leaves))
    values, _, _ = af._prepare(tuple(originals), True)
    if forward:
        for original, value in zip(originals, values):
            _native._forward_ad_prepare(original, value)
    excluded = [value for value, original in zip(values, originals) if not original.requires_grad]
    replaced, offset = list(args), 0
    for position, spec, count in zip(positions, specifications, counts):
        replaced[position] = _unflatten(values[offset:offset + count], spec)
        offset += count
    aliases = {value._id(): original for value, original in zip(values, originals)}
    return tuple(replaced), values, excluded, specifications, counts, single, aliases


def _post_dual(value, excluded, enabled=True, aliases=None, cutoff=None):
    leaves, spec = _flatten(value)
    tangents = [_native._forward_ad_tangents(leaf) for leaf in leaves]
    rebased = iter(_native._forward_ad_unwrap(leaves + [tangent for columns in tangents if columns is not None for tangent in columns]))
    def preserve_object(original):
        result = next(rebased)
        return original if result._id() == original._id() else result
    actual_leaves = [preserve_object(leaf) for leaf in leaves]
    tangents = [None if columns is None else [preserve_object(tangent) for tangent in columns] for columns in tangents]
    aliases = {} if aliases is None else aliases
    aliases = {**{value._id(): value for value in aliases.values()}, **aliases}
    def process(leaf):
        if leaf._id() in aliases:
            return aliases[leaf._id()]
        if not enabled and cutoff is not None:
            result = _native._forward_ad_freeze(leaf, cutoff, excluded)
            return leaf if result._id() == leaf._id() else result
        if enabled and not leaf.requires_grad:
            return leaf
        return _post(leaf, excluded, enabled)
    result = _unflatten([process(leaf) for leaf in actual_leaves], spec)
    processed, _ = _flatten(result)
    for original, target, columns in zip(leaves, processed, tangents):
        columns = None if columns is None else [process(tangent) for tangent in columns]
        _native._forward_ad_post_columns(original, target, columns)
    return result


def _pack(values, specifications, counts, single):
    result, offset = [], 0
    for spec, count in zip(specifications, counts):
        result.append(_unflatten(values[offset:offset + count], spec))
        offset += count
    return result[0] if single else tuple(result)


def _evaluate(func, args, kwargs, has_aux, output_dtype='float32'):
    result = func(*args, **kwargs)
    auxiliary = None
    if has_aux:
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError('has_aux=True requires an (output, auxiliary) pair')
        result, auxiliary = result
        _flatten(auxiliary)
    outputs, output_spec = _flatten(result)
    if not outputs:
        raise RuntimeError('function output must contain at least one Tensor')
    af._check(outputs, output_dtype)
    return result, tuple(outputs), output_spec, auxiliary


def _needed(outputs, excluded):
    return any(output._has_grad_leaf_except(excluded) or output._has_current_grad_leaf_except(excluded) for output in outputs)


def _shared_blocks(blocks):
    # Torch assembles output rows together before splitting their public trees.
    # Keep those shared zero-gradient paths, including constant output blocks.
    results = [[] for _ in blocks]
    for column in zip(*blocks):
        _shape([sum(block.numel() for block in column)])
        combined = cat([block.reshape([block.numel()]) for block in column], 0)
        offset = 0
        for row, block in zip(results, column):
            row.append(combined.narrow(0, offset, block.numel()).reshape(block.shape))
            offset += block.numel()
    return results


def _gradient(func, argnums, has_aux, with_value):
    if not callable(func) or type(has_aux) is not bool:
        raise TypeError('func must be callable and has_aux must be bool')
    @wraps(func)
    def transformed(*args, **kwargs):
        enabled, cutoff = _entry()
        forward = _native._forward_ad_is_active()
        with (_native._ForwardReverse() if forward else nullcontext()), enable_grad():
            arguments, values, excluded, specs, counts, single, aliases = _prepare(args, argnums, forward)
            result, outputs, _, auxiliary = _evaluate(func, arguments, kwargs, has_aux)
            if not isinstance(result, Tensor) or result.ndim != 0:
                raise RuntimeError('grad requires a scalar (zero-dimensional) Tensor output')
            with enable_grad() if enabled else no_grad():
                derivatives = af._vjp(outputs, values, (full([], 1.),), enabled and _needed(outputs, excluded), False, True, _forward=forward, _functional=True)
            post = _post_dual if forward else _post
            derivative = post(_pack(derivatives, specs, counts, single), excluded, enabled if forward else True)
            value = post(result, excluded, enabled, aliases, None if enabled else cutoff)
            if has_aux:
                aux = post(auxiliary, excluded, enabled, aliases, None if enabled else cutoff)
                return (derivative, (value, aux)) if with_value else (derivative, aux)
            return (derivative, value) if with_value else derivative
    return transformed


def grad(func, argnums=0, has_aux=False):
    """Return a scalar-gradient transform; argnums preserves selected argument trees."""
    return _gradient(func, argnums, has_aux, False)


def grad_and_value(func, argnums=0, has_aux=False):
    return _gradient(func, argnums, has_aux, True)


def jacrev(func, argnums=0, *, has_aux=False, chunk_size=None, _preallocate_and_copy=False):
    """Return output-tree Jacobian blocks using one row per reverse sweep."""
    if not callable(func) or type(has_aux) is not bool:
        raise TypeError('func must be callable and has_aux must be bool')
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError('chunk_size must be a positive integer or None')
    if _preallocate_and_copy:
        raise NotImplementedError('preallocated Jacobian assembly is unsupported')
    @wraps(func)
    def transformed(*args, **kwargs):
        enabled, cutoff = _entry()
        forward = _native._forward_ad_is_active()
        with (_native._ForwardReverse() if forward else nullcontext()), enable_grad():
            arguments, values, excluded, specs, counts, single, aliases = _prepare(args, argnums, forward)
            _, outputs, output_spec, auxiliary = _evaluate(func, arguments, kwargs, has_aux)
            blocks = _shared_blocks(af._jacobian(outputs, values, enabled and _needed(outputs, excluded), False, _forward=forward, _caller_grad_mode=enabled))
            post = _post_dual if forward else _post
            result = post(_unflatten([_pack(row, specs, counts, single) for row in blocks], output_spec), excluded, enabled)
            return (result, post(auxiliary, excluded, enabled, aliases, None if enabled else cutoff)) if has_aux else result
    return transformed


def jacfwd(func, argnums=0, has_aux=False, *, randomness='error'):
    """Eager CPU forward Jacobian for float32 inputs and float32/float64 outputs.

    This is a native physical-batch dual level, not vmap. `same` shares a primal
    random draw across columns; `different` and nested forward levels reject.
    Dense basis and result capacities are checked before materialization.
    Opaque local fallback retains numerical coverage with its existing
    higher-order limits; its layout is not a native batched-rule guarantee.
    Stateful module mutations (including active nn.Dropout counter updates)
    remain unsupported; use seeded functional dropout or random factories.
    """
    if not callable(func) or type(has_aux) is not bool:
        raise TypeError('func must be callable and has_aux must be bool')
    if randomness not in ('error', 'same'):
        raise NotImplementedError("randomness must be 'error' or 'same'; different draws per column are unsupported")
    @wraps(func)
    def transformed(*args, **kwargs):
        _entry()
        if _native._forward_ad_is_active():
            raise NotImplementedError('nested forward AD levels are not supported')
        positions, single = _argnums(argnums, len(args))
        originals, specs, counts = [], [], []
        for position in positions:
            leaves, spec = _flatten(args[position])
            if not leaves:
                raise ValueError('each selected argument must contain at least one Tensor')
            originals.extend(leaves)
            specs.append(spec)
            counts.append(len(leaves))
        af._check(originals, 'float32')
        sizes = [value.numel() for value in originals]
        count = sum(sizes)
        _shape([count, count])
        _shape([max(count, 1), len(originals)], 'int64')
        with enable_grad():
            values = [value.reshape(value.shape) for value in originals]
        directions, offset = [], 0
        for value, size in zip(values, sizes):
            data = [0.] * (count * size)
            for index in range(size):
                data[(offset + index) * size + index] = 1.
            directions.append(Tensor(data, [count, size]).view([count, *value.shape]))
            offset += size
        arguments, offset = list(args), 0
        for position, spec, length in zip(positions, specs, counts):
            arguments[position] = _unflatten(values[offset:offset + length], spec)
            offset += length
        with _native._ForwardBatched(values, directions, count, randomness == 'same') as level:
            _, outputs, output_spec, auxiliary = _evaluate(func, arguments, kwargs, has_aux, output_dtype=None)
            _shape([max(count, 1), sum(output.numel() * (8 if output.dtype == 'float64' else 4) for output in outputs)], 'bool')
            directions = level.tangents(outputs)
            if has_aux:
                aux_leaves, aux_spec = _flatten(auxiliary)
                auxiliary = _unflatten(level.primals(aux_leaves, False), aux_spec)
        rows = []
        for output, direction in zip(outputs, directions):
            _shape([count, *output.shape], output.dtype)
            if direction is None:
                direction = output._zeros_like_layout().expand([count, *output.shape])
            combined = direction.permute([*range(1, direction.ndim), 0])
            offset, blocks = 0, []
            for value, size in zip(values, sizes):
                block = combined.narrow(combined.ndim - 1, offset, size)
                blocks.append(block.squeeze(block.ndim - 1) if value.ndim == 0 else block.view([*output.shape, *value.shape]))
                offset += size
            rows.append(_pack(blocks, specs, counts, single))
        result = _unflatten(rows, output_spec)
        return (result, auxiliary) if has_aux else result
    return transformed


def hessian(func, argnums=0):
    """Return vector-output Hessian blocks without materializing an intermediate Jacobian graph."""
    if not callable(func):
        raise TypeError('func must be callable')
    @wraps(func)
    def transformed(*args, **kwargs):
        enabled, _ = _entry()
        with enable_grad():
            arguments, values, excluded, specs, counts, single, _ = _prepare(args, argnums)
            _, outputs, output_spec, _ = _evaluate(func, arguments, kwargs, False)
            create_graph = enabled and _needed(outputs, excluded)
            all_blocks = []
            for output in outputs:
                _shape([output.numel()], 'int64')
                for left in values:
                    for right in values:
                        _shape([*output.shape, *left.shape, *right.shape])
                rows = [[[] for _ in values] for _ in values]
                for index in range(output.numel()):
                    basis = [0.] * output.numel()
                    basis[index] = 1.
                    first = af._vjp((output,), values, (Tensor(basis, output.shape),), True, False, _functional=True)
                    second = af._jacobian(first, values, create_graph, False, _functional=True)
                    for i, row in enumerate(second):
                        for j, block in enumerate(row):
                            if create_graph and any(first[i]._depends_on(value) for value in values):
                                # Preserve constant higher derivatives without reading values
                                # (multiplication by zero would contaminate infinities).
                                block = block + first[i].reshape([first[i].numel()]).narrow(0, 0, 0).sum()
                            rows[i][j].append(block)
                blocks = [[stack(row, 0).reshape([*output.shape, *left.shape, *right.shape]) if row
                           else full([*output.shape, *left.shape, *right.shape], 0.)
                           for row, right in zip(entries, values)] for entries, left in zip(rows, values)]
                if not output.numel() and create_graph:
                    first = af._vjp((output,), values, (full(output.shape, 0.),), True, False, _functional=True)
                    for i, derivative in enumerate(first):
                        if any(derivative._depends_on(value) for value in values):
                            bridge = derivative.reshape([derivative.numel()]).narrow(0, 0, 0).sum()
                            blocks[i] = [block + bridge for block in blocks[i]]
                all_blocks.append([block for row in blocks for block in row])
            results = []
            for row in _shared_blocks(all_blocks):
                blocks = [row[index:index + len(values)] for index in range(0, len(row), len(values))]
                results.append(_pack([_pack(entries, specs, counts, single) for entries in blocks], specs, counts, single))
            return _post(_unflatten(results, output_spec), excluded, enabled)
    return transformed


def jvp(func, primals, tangents, *, strict=False, has_aux=False):
    """Forward tangents through the native Op recorder, including no_grad regions.

    Primals are float32; outputs may be float32 or float64.
    One eager CPU level is supported. Untagged operators contract their own
    backward rule; graph-building requires that operator's differentiable VJP.
    grad and grad_and_value compose through graph-building reverse rules.
    Nested levels, other reverse queries, and mutation reject explicitly.
    Reverse differentiation after JVP remains supported.
    """
    _entry()
    if not callable(func):
        raise TypeError('func must be callable')
    if not isinstance(primals, tuple) or not isinstance(tangents, tuple) or len(primals) != len(tangents) or not primals:
        raise TypeError('primals and tangents must be nonempty matching tuples')
    if type(strict) is not bool or type(has_aux) is not bool:
        raise TypeError('strict and has_aux must be bool')
    originals, primal_spec = _flatten(primals)
    vector_values, vector_spec = _flatten(tangents)
    if not originals or any(not _flatten(argument)[0] for argument in primals):
        raise ValueError('each primal argument must contain at least one Tensor')
    af._check(originals, 'float32')
    af._check(vector_values)
    if vector_spec != primal_spec or any(vector.shape != value.shape for vector, value in zip(vector_values, originals)):
        raise RuntimeError('tangent tree and Tensor shapes must match primals')
    with enable_grad():
        values = [value.reshape(value.shape) for value in originals]
    arguments = _unflatten(values, primal_spec)
    with _native._ForwardAD(values, vector_values) as level:
        result, outputs, output_spec, auxiliary = _evaluate(func, arguments, {}, has_aux, output_dtype=None)
        directions = level.tangents(outputs)
        output_values = level.primals(outputs, False)
        if has_aux:
            aux_leaves, aux_spec = _flatten(auxiliary)
            aux_values = level.primals(aux_leaves, True)
            auxiliary = _unflatten([original if value._id() == original._id() else value
                                    for original, value in zip(aux_leaves, aux_values)], aux_spec)
    if strict and any(direction is None for direction in directions):
        raise RuntimeError('an output is independent of all primals in strict JVP mode')
    vector_aliases = {value._id(): value for value in vector_values}
    directions = [output._zeros_like_layout() if direction is None else vector_aliases.get(direction._id(), direction)
                  for output, direction in zip(outputs, directions)]
    pair = (_unflatten(output_values, output_spec), _unflatten(directions, output_spec))
    return (*pair, auxiliary) if has_aux else pair


def vjp(func, *primals, has_aux=False):
    """Evaluate once and return a reusable native-VJP closure; tapes remain retained."""
    enabled, cutoff = _entry()
    if type(has_aux) is not bool:
        raise TypeError('has_aux must be bool')
    with enable_grad():
        arguments, values, excluded, specs, counts, _, aliases = _prepare(primals, tuple(range(len(primals))))
        result, outputs, output_spec, auxiliary = _evaluate(func, arguments, {}, has_aux)
    frozen_forward = None if enabled else cutoff
    def pullback(cotangents, retain_graph=True, create_graph=None):
        af._options(False, False)
        if retain_graph is not True:
            raise NotImplementedError('native VJP tapes are retained; retain_graph=False is unsupported')
        if create_graph is not None and type(create_graph) is not bool:
            raise TypeError('create_graph must be bool or None')
        enabled = is_grad_enabled() if create_graph is None else create_graph
        vectors, vector_spec = _flatten(cotangents)
        if vector_spec != output_spec or any(vector.shape != output.shape for vector, output in zip(vectors, outputs)):
            raise RuntimeError('cotangent tree and Tensor shapes must match outputs')
        if any(vector.device != 'cpu' or vector.dtype not in ('float32', 'float64', 'int64', 'bool') for vector in vectors):
            raise ValueError('cotangents require CPU float32, float64, int64 or bool tensors')
        vectors = [vector.to(dtype=output.dtype) for vector, output in zip(vectors, outputs)]
        with enable_grad() if enabled else no_grad():
            derivatives = af._vjp(outputs, values, vectors, enabled and _needed((*outputs, *vectors), excluded), False, True, frozen_forward, excluded, _caller_grad_mode=enabled)
            vector_aliases = {vector._id(): vector for vector in vectors}
            return _post(_pack(derivatives, specs, counts, False), excluded, True, vector_aliases)
    value = _post(result, excluded, enabled, aliases, None if enabled else cutoff)
    return (value, pullback, _post(auxiliary, excluded, enabled, aliases, None if enabled else cutoff)) if has_aux else (value, pullback)
