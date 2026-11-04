"""CPU spatial compositions; Conv1d/2d accept f32/f64, other paths require f32."""
import math
import sys
import itertools

from . import _native
from ._native import Tensor


def _tuple(value, dimensions, name, minimum=0):
    if type(value) is int:
        values = (value,) * dimensions
    else:
        try:
            values = tuple(value)
        except TypeError:
            raise TypeError(f'{name} must be an integer or a {dimensions}-tuple') from None
    if len(values) != dimensions or any(type(item) is not int or item < minimum for item in values):
        raise ValueError(f'{name} must contain {dimensions} integers >= {minimum}')
    return values


def _capacity(shape, dtype='float32'):
    # Even empty outputs need representable contiguous strides.
    if math.prod(max(size, 1) for size in shape) > sys.maxsize or math.prod(shape) > sys.maxsize // (8 if dtype == 'float64' else 4):
        raise ValueError('spatial tensor shape exceeds addressable ' + dtype + ' capacity')


def _tensor(value, name, allow_float64=False):
    if not isinstance(value, Tensor):
        raise TypeError(name + ' must be a Tensor')
    if value.device != 'cpu':
        raise NotImplementedError('spatial compositions currently support CPU tensors only')
    if value.dtype not in (('float32', 'float64') if allow_float64 else ('float32',)):
        raise ValueError(name + (' must have float32 or float64 dtype' if allow_float64 else ' must have float32 dtype'))


def _input(value, dimensions, allow_float64=False):
    _tensor(value, 'input', allow_float64)
    if value.ndim not in (dimensions + 1, dimensions + 2):
        raise ValueError(f'input must have {dimensions + 1} or {dimensions + 2} dimensions')
    unbatched = value.ndim == dimensions + 1
    result = value.unsqueeze(0) if unbatched else value
    if any(size == 0 for size in result.shape[1:]):
        raise ValueError('input channels and spatial dimensions must be positive')
    return result, unbatched


def pad(input, pad, mode='constant', value=None):
    """CPU f32 Torch-order signed padding and cropping.

    Nonconstant modes support one to three trailing spatial axes. Circular
    padding wraps the cropped source at most once; padding an empty cropped
    source rejects instead of returning uninitialized values.
    """
    _tensor(input, 'input')
    try:
        pads = tuple(pad)
    except TypeError:
        raise TypeError('pad must be a sequence of signed integer pairs') from None
    if len(pads) % 2 or len(pads) > 2 * input.ndim or any(type(p) is not int for p in pads):
        raise ValueError('pad requires signed integer pairs for trailing input axes')
    if mode not in ('constant', 'reflect', 'replicate', 'circular'):
        raise NotImplementedError('padding mode must be constant, reflect, replicate or circular')
    if value is not None and (type(value) not in (int, float) or mode != 'constant' and value != 0):
        raise ValueError('value must be numeric and is supported only for constant padding')
    count = len(pads) // 2
    if mode != 'constant' and (count not in (1, 2, 3) or input.ndim not in (count + 1, count + 2)):
        raise NotImplementedError('nonconstant padding requires 1-3 spatial axes and one or two leading axes')
    pairs = [(input.ndim - index - 1, pads[2 * index], pads[2 * index + 1]) for index in range(count)]
    shape = list(input.shape)
    for axis, before, after in pairs:
        shape[axis] += before + after
        if shape[axis] < 0 or mode in ('reflect', 'replicate') and shape[axis] == 0:
            raise ValueError('padding produces a nonpositive spatial dimension')
    _capacity(shape)
    if mode == 'constant':
        result = input
        full_pad = [0] * (2 * input.ndim)
        for axis, before, after in pairs:
            start, end = max(-before, 0), input.shape[axis] - max(-after, 0)
            if end < start:
                raise ValueError('negative padding crops beyond the input dimension')
            result = result.narrow(axis, start, end - start)
            full_pad[2 * axis:2 * axis + 2] = [max(before, 0), max(after, 0)]
        return result.pad_constant(full_pad, 0. if value is None else float(value))
    result = input
    for axis, before, after in pairs:
        length = input.shape[axis]
        output_length = length + before + after
        _capacity([*result.shape[:axis], output_length, *result.shape[axis + 1:]])
        if output_length > sys.maxsize // 8:
            raise ValueError('padding indices exceed addressable int64 capacity')
        if length == 0:
            raise ValueError('nonconstant padding requires nonempty input spatial dimensions')
        if mode == 'reflect' and (before >= length or after >= length):
            raise ValueError('reflection padding must be smaller than the original input dimension')
        if mode == 'circular':
            start, end = max(-before, 0), length - max(-after, 0)
            cropped = end - start
            if cropped < 0 or max(before, after, 0) > cropped:
                raise ValueError('circular padding requires a nonempty cropped source and at most one wrap')
            indices = [] if output_length == 0 else [start + (position - max(before, 0)) % cropped for position in range(output_length)]
        elif mode == 'replicate':
            indices = [min(max(position - before, 0), length - 1) for position in range(output_length)]
        else:
            indices = []
            for position in range(output_length):
                index = position - before
                indices.append(-index if index < 0 else 2 * length - index - 2 if index >= length else index)
        result = result.index_select(axis, indices)
    return result


def _conv_padding(input, kernel, padding, dilation, mode):
    if mode == 'zeros':
        return input, padding
    if mode not in ('reflect', 'replicate', 'circular'):
        raise ValueError('padding_mode must be zeros, reflect, replicate or circular')
    if isinstance(padding, str):
        totals = [spacing * (width - 1) if padding == 'same' else 0 for width, spacing in zip(kernel, dilation)]
        pairs = [(total // 2, total - total // 2) for total in totals]
    else:
        pairs = [(amount, amount) for amount in padding]
    return pad(input, tuple(value for pair in reversed(pairs) for value in pair), mode), (0,) * len(kernel)


def _convolution(input, weight, bias, stride, padding, dilation, groups, dimensions, transposed=False):
    allow_float64 = dimensions in (1, 2) and not transposed
    x, unbatched = _input(input, dimensions, allow_float64)
    _tensor(weight, 'weight', allow_float64)
    if x.dtype != weight.dtype:
        raise ValueError(f'convolution input dtype {x.dtype} must match weight dtype {weight.dtype}')
    if x.dtype == 'float64' and (_native._is_capturing() or _native._forward_ad_is_active()):
        raise NotImplementedError('float64 convolution supports eager ordinary reverse gradients only')
    if weight.ndim != dimensions + 2 or any(size <= 0 for size in weight.shape):
        raise ValueError(f'weight must have {dimensions + 2} positive dimensions')
    if type(groups) is not int or groups <= 0:
        raise ValueError('groups must be a positive integer')
    channels = x.shape[1]
    outputs = weight.shape[1] * groups if transposed else weight.shape[0]
    expected = weight.shape[0] if transposed else weight.shape[1] * groups
    if channels != expected or channels % groups or outputs % groups:
        raise ValueError('input channels, weight shape and groups are incompatible')
    if bias is not None:
        _tensor(bias, 'bias', allow_float64)
        if bias.dtype != x.dtype:
            raise ValueError('convolution bias must match input dtype')
        if bias.shape != [outputs]:
            raise ValueError(f'bias must have shape [{outputs}]')
    strides = _tuple(stride, dimensions, 'stride', 1)
    dilations = _tuple(dilation, dimensions, 'dilation', 1)
    kernel = tuple(weight.shape[2:])
    if isinstance(padding, str):
        if transposed or padding not in ('same', 'valid'):
            raise ValueError('padding must be numeric, or same/valid for ordinary convolution')
        resolved_padding = (0,) * dimensions
        if padding == 'same':
            if any(step != 1 for step in strides):
                raise ValueError("padding='same' requires stride=1")
            pads = [0, 0, 0, 0]
            for width, spacing in zip(kernel, dilations):
                total = spacing * (width - 1)
                pads.extend((total // 2, total - total // 2))
            padded = [x.shape[0], channels] + [size + spacing * (width - 1)
                      for size, width, spacing in zip(x.shape[2:], kernel, dilations)]
            _capacity(padded, x.dtype)
            if x.dtype == 'float64' and all(pads[2 * axis] == pads[2 * axis + 1] for axis in range(2, x.ndim)):
                resolved_padding = tuple(pads[2 * axis] for axis in range(2, x.ndim))
            elif x.dtype == 'float64':
                from ._tensor_api import zeros
                for axis in range(2, x.ndim):
                    before, after = pads[2 * axis:2 * axis + 2]
                    shape = list(x.shape)
                    pieces = []
                    if before:
                        shape[axis] = before
                        pieces.append(zeros(shape, dtype=x.dtype))
                    pieces.append(x)
                    if after:
                        shape[axis] = after
                        pieces.append(zeros(shape, dtype=x.dtype))
                    if len(pieces) > 1:
                        x = _native.cat(pieces, axis)
            else:
                x = x.pad_constant(pads, 0.)
        padding = resolved_padding
    pads = _tuple(padding, dimensions, 'padding')
    return x, unbatched, strides, pads, dilations, kernel, outputs


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """NCL/OIK grouped convolution, including unbatched CL and same/valid padding."""
    x, unbatched, s, p, d, _, _ = _convolution(input, weight, bias, stride, padding, dilation, groups, 1)
    result = _native.conv2d_options(x.unsqueeze(2), weight.unsqueeze(2), bias,
                                    (1, s[0]), (0, p[0]), (1, d[0]), groups).squeeze(2)
    return result.squeeze(0) if unbatched else result


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """CPU NCHW/OIHW grouped convolution with CHW and same/valid support."""
    x, unbatched, s, p, d, _, _ = _convolution(input, weight, bias, stride, padding, dilation, groups, 2)
    result = _native.conv2d_options(x, weight, bias, s, p, d, groups)
    return result.squeeze(0) if unbatched else result


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """NCDHW/OIDHW grouped convolution composed from native depth-slice Conv2d."""
    x, unbatched, s, p, d, kernel, outputs = _convolution(input, weight, bias, stride, padding, dilation, groups, 3)
    spatial = tuple((size + 2 * pad - spacing * (width - 1) - 1) // step + 1
                    for size, pad, spacing, width, step in zip(x.shape[2:], p, d, kernel, s))
    if any(size <= 0 for size in spatial):
        raise ValueError('convolution kernel exceeds padded input')
    shape = [x.shape[0], outputs, *spatial]
    _capacity(shape)
    # Empty slices preserve zero gradient paths without evaluating unused NaNs.
    zero = x.narrow(0, 0, 0).sum() + weight.narrow(0, 0, 0).sum()
    if x.shape[0] == 0:
        result = Tensor.zeros(shape) + zero
    else:
        planes = []
        for out_depth in range(spatial[0]):
            plane = None
            for kernel_depth in range(kernel[0]):
                source_depth = out_depth * s[0] - p[0] + kernel_depth * d[0]
                if 0 <= source_depth < x.shape[2]:
                    term = _native.conv2d_options(x.narrow(2, source_depth, 1).squeeze(2),
                                                  weight.narrow(2, kernel_depth, 1).squeeze(2), None,
                                                  s[1:], p[1:], d[1:], groups)
                    plane = term if plane is None else plane + term
            if plane is None:
                plane = Tensor.zeros([x.shape[0], outputs, *spatial[1:]]) + zero
            planes.append(plane)
        result = _native.stack(planes, 2)
    if bias is not None:
        result = result + bias.reshape([1, outputs, 1, 1, 1])
    return result.squeeze(0) if unbatched else result


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """NCHW/IOHW transposed convolution using native matrix products and fold2d."""
    x, unbatched, s, p, d, kernel, outputs = _convolution(input, weight, bias, stride, padding, dilation, groups, 2, True)
    extra = _tuple(output_padding, 2, 'output_padding')
    if any(pad >= step and pad >= spacing for pad, step, spacing in zip(extra, s, d)):
        raise ValueError('output_padding must be smaller than stride or dilation on each axis')
    spatial = tuple((size - 1) * step - 2 * pad + spacing * (width - 1) + extra_pad + 1
                    for size, step, pad, spacing, width, extra_pad in zip(x.shape[2:], s, p, d, kernel, extra))
    if any(size <= 0 for size in spatial):
        raise ValueError('transposed convolution output dimensions must be positive')
    shape = [x.shape[0], outputs, *spatial]
    _capacity(shape)
    if x.shape[0] == 0:
        result = Tensor.zeros(shape) + x.narrow(0, 0, 0).sum() + weight.narrow(0, 0, 0).sum()
    else:
        # Torch permits output_padding >= stride when it is below dilation.
        # fold's enlarged window grid then needs trailing zero input windows.
        trailing = tuple(pad // step for pad, step in zip(extra, s))
        if any(trailing):
            _capacity([x.shape[0], x.shape[1], x.shape[2] + trailing[0], x.shape[3] + trailing[1]])
            x = x.pad_constant([0, 0, 0, 0, 0, trailing[0], 0, trailing[1]], 0.)
        batch, channels, height, width = x.shape
        columns, input_group, output_group = height * width, channels // groups, outputs // groups
        _capacity([batch, outputs, *kernel, height, width])
        pieces = []
        for group in range(groups):
            inputs = x.narrow(1, group * input_group, input_group).transpose(0, 1).reshape([input_group, batch * columns])
            weights = weight.narrow(0, group * input_group, input_group).reshape([input_group, output_group * math.prod(kernel)])
            projected = weights.transpose(0, 1) @ inputs
            projected = projected.reshape([output_group * math.prod(kernel), batch, columns]).transpose(0, 1)
            pieces.append(_native.fold2d(projected, spatial, kernel, s, p, d))
        result = pieces[0] if groups == 1 else _native.cat(pieces, 1)
    if bias is not None:
        result = result + bias.reshape([1, outputs, 1, 1])
    return result.squeeze(0) if unbatched else result


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """NCL/IOK transposed convolution with PyTorch's output_padding convention."""
    x, unbatched, s, p, d, _, _ = _convolution(input, weight, bias, stride, padding, dilation, groups, 1, True)
    extra = _tuple(output_padding, 1, 'output_padding')
    result = conv_transpose2d(x.unsqueeze(2), weight.unsqueeze(2), bias, (1, s[0]), (0, p[0]),
                              (0, extra[0]), groups, (1, d[0])).squeeze(2)
    return result.squeeze(0) if unbatched else result


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """NCDHW/IODHW transpose convolution composed from grouped 2D depth slices."""
    x, unbatched, s, p, d, kernel, outputs = _convolution(input, weight, bias, stride, padding, dilation, groups, 3, True)
    extra = _tuple(output_padding, 3, 'output_padding')
    if any(pad >= step and pad >= spacing for pad, step, spacing in zip(extra, s, d)):
        raise ValueError('output_padding must be smaller than stride or dilation on each axis')
    spatial = tuple((size - 1) * step - 2 * pad + spacing * (width - 1) + added + 1
                    for size, step, pad, spacing, width, added in zip(x.shape[2:], s, p, d, kernel, extra))
    if any(size <= 0 for size in spatial):
        raise ValueError('transposed convolution output dimensions must be positive')
    shape = [x.shape[0], outputs, *spatial]
    _capacity(shape)
    zero = x.narrow(0, 0, 0).sum() + weight.narrow(0, 0, 0).sum()
    if x.shape[0] == 0:
        result = Tensor.zeros(shape) + zero
    else:
        planes = [None] * spatial[0]
        for source_depth in range(x.shape[2]):
            for kernel_depth in range(kernel[0]):
                destination = source_depth * s[0] - p[0] + kernel_depth * d[0]
                if 0 <= destination < spatial[0]:
                    term = conv_transpose2d(x.narrow(2, source_depth, 1).squeeze(2),
                                            weight.narrow(2, kernel_depth, 1).squeeze(2), None,
                                            s[1:], p[1:], extra[1:], groups, d[1:])
                    planes[destination] = term if planes[destination] is None else planes[destination] + term
        planes = [Tensor.zeros([x.shape[0], outputs, *spatial[1:]]) + zero if plane is None else plane for plane in planes]
        result = _native.stack(planes, 2)
    if bias is not None:
        result = result + bias.reshape([1, outputs, 1, 1, 1])
    return result.squeeze(0) if unbatched else result


def _pool_options(kernel_size, stride, padding, dimensions, dilation=1, ceil_mode=False):
    if type(ceil_mode) is not bool:
        raise TypeError('ceil_mode must be bool')
    kernel = _tuple(kernel_size, dimensions, 'kernel_size', 1)
    steps = kernel if stride is None else _tuple(stride, dimensions, 'stride', 1)
    pads = _tuple(padding, dimensions, 'padding')
    spacing = _tuple(dilation, dimensions, 'dilation', 1)
    if any(pad > size // 2 for pad, size in zip(pads, kernel)):
        raise ValueError('pooling padding must be at most half of kernel_size')
    return kernel, steps, pads, spacing


def _pool_geometry(x, kernel, steps, pads, spacing, ceil_mode):
    spatial = []
    for length, width, step, pad, dilation in zip(x.shape[2:], kernel, steps, pads, spacing):
        size = (length + 2 * pad - dilation * (width - 1) - 1 + (step - 1 if ceil_mode else 0)) // step + 1
        if ceil_mode and (size - 1) * step >= length + pad:
            size -= 1
        if size <= 0:
            raise ValueError('pooling output spatial dimensions must be positive')
        spatial.append(size)
    shape = [*x.shape[:2], *spatial]
    _capacity(shape)
    return spatial, shape


def _offsets(axes, spatial):
    strides = [math.prod(spatial[axis + 1:]) for axis in range(len(spatial))]
    return [sum(position * step for position, step in zip(point, strides)) for point in itertools.product(*axes)]


def _max_windows(x, spatial, windows, return_indices, unbatched):
    shape = [*x.shape[:2], *spatial]
    _capacity(shape)
    if math.prod(shape) > sys.maxsize // 8:
        raise ValueError('pooling indices exceed addressable int64 capacity')
    if x.shape[0] == 0:
        result = Tensor.zeros(shape) + x.sum()
        indices = Tensor.from_i64([], shape)
    else:
        planes, volume, count = math.prod(x.shape[:2]), math.prod(x.shape[2:]), math.prod(spatial)
        values = x.reshape([planes * volume]).tolist()
        selected, positions = [0] * (planes * count), [0] * (planes * count)
        for output, axes in enumerate(windows):
            offsets = _offsets(axes, x.shape[2:])
            if not offsets:
                raise ValueError('dilated max pooling windows must contain an input element')
            for plane in range(planes):
                best = offsets[0]
                maximum = values[plane * volume + best]
                for offset in offsets[1:]:
                    value = values[plane * volume + offset]
                    if math.isnan(value) or value > maximum:
                        best, maximum = offset, value
                positions[plane * count + output] = best
                selected[plane * count + output] = plane * volume + best
        # Selection is discrete; native gather owns the differentiable value path.
        result = x.reshape([planes * volume]).gather(0, Tensor.from_i64(selected, [len(selected)])).reshape(shape)
        indices = Tensor.from_i64(positions, shape)
    if unbatched:
        result, indices = result.squeeze(0), indices.squeeze(0)
    return (result, indices) if return_indices else result


def _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, dimensions):
    if type(return_indices) is not bool:
        raise TypeError('return_indices must be bool')
    x, unbatched = _input(input, dimensions)
    kernel, steps, pads, spacing = _pool_options(kernel_size, stride, padding, dimensions, dilation, ceil_mode)
    spatial, _ = _pool_geometry(x, kernel, steps, pads, spacing, ceil_mode)
    def windows():
        for point in itertools.product(*(range(size) for size in spatial)):
            axes = []
            for index, length, width, step, pad, dilation in zip(point, x.shape[2:], kernel, steps, pads, spacing):
                start = index * step - pad
                end = min(start + dilation * (width - 1) + 1, length)
                start += max(0, -start + dilation - 1) // dilation * dilation
                axes.append(range(start, end, dilation))
            yield axes
    return _max_windows(x, spatial, windows(), return_indices, unbatched)


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """CPU max pooling; first maximum and last NaN win, with int64 spatial indices."""
    return _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, 1)


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """CPU max pooling with rectangular windows, dilation, padding and ceil mode."""
    return _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, 2)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """CPU volumetric max pooling. All-padding dilated windows explicitly reject."""
    return _max_pool(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices, 3)


def _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dimensions):
    if type(count_include_pad) is not bool:
        raise TypeError('count_include_pad must be bool')
    if divisor_override is not None and (type(divisor_override) is not int or divisor_override == 0):
        raise ValueError('divisor_override must be a nonzero integer or None')
    x, unbatched = _input(input, dimensions)
    kernel, steps, pads, spacing = _pool_options(kernel_size, stride, padding, dimensions, ceil_mode=ceil_mode)
    if dimensions == 3 and any(length < width for length, width in zip(x.shape[2:], kernel)):
        raise ValueError('average pooling 3d input spatial dimensions must be at least kernel_size')
    spatial, shape = _pool_geometry(x, kernel, steps, pads, spacing, ceil_mode)
    if x.shape[0] == 0:
        result = Tensor.zeros(shape) + x.sum()
    else:
        flattened = x.reshape([math.prod(x.shape[:2]), math.prod(x.shape[2:])])
        outputs = []
        for point in itertools.product(*(range(size) for size in spatial)):
            axes, padded_count = [], 1
            for index, length, width, step, pad in zip(point, x.shape[2:], kernel, steps, pads):
                start = index * step - pad
                end = min(start + width, length + pad)
                padded_count *= end - start
                axes.append(range(max(start, 0), min(end, length)))
            offsets = _offsets(axes, x.shape[2:])
            divisor = divisor_override if divisor_override is not None else padded_count if count_include_pad else len(offsets)
            outputs.append(flattened.index_select(1, offsets).sum_dim(1) / divisor)
        result = _native.stack(outputs, 1).reshape(shape)
    return result.squeeze(0) if unbatched else result


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    """CPU average pooling with ceil mode and correct clipped-padding divisor."""
    return _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, None, 1)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 2)


def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return _avg_pool(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, 3)


def _adaptive_max_pool(input, output_size, return_indices, dimensions):
    if type(return_indices) is not bool:
        raise TypeError('return_indices must be bool')
    x, unbatched = _input(input, dimensions)
    requested = _pool_size(output_size, dimensions)
    spatial = tuple(current if wanted is None else wanted for current, wanted in zip(x.shape[2:], requested))
    def windows():
        for point in itertools.product(*(range(size) for size in spatial)):
            yield [range(index * length // size, ((index + 1) * length + size - 1) // size)
                   for index, length, size in zip(point, x.shape[2:], spatial)]
    return _max_windows(x, spatial, windows(), return_indices, unbatched)


def adaptive_max_pool1d(input, output_size, return_indices=False):
    return _adaptive_max_pool(input, output_size, return_indices, 1)


def adaptive_max_pool2d(input, output_size, return_indices=False):
    return _adaptive_max_pool(input, output_size, return_indices, 2)


def adaptive_max_pool3d(input, output_size, return_indices=False):
    return _adaptive_max_pool(input, output_size, return_indices, 3)


def _pool_size(output_size, dimensions):
    if type(output_size) is int:
        values = (output_size,) * dimensions
    else:
        try:
            values = tuple(output_size)
        except TypeError:
            raise TypeError('output_size must be an integer or a spatial tuple') from None
    if len(values) != dimensions or any(value is not None and (type(value) is not int or value <= 0) for value in values):
        raise ValueError('adaptive output_size must contain positive integers or None')
    if dimensions == 1 and values[0] is None:
        raise ValueError('adaptive_avg_pool1d requires a positive output size')
    return values


def _adaptive_avg_pool(input, output_size, dimensions):
    x, unbatched = _input(input, dimensions)
    requested = _pool_size(output_size, dimensions)
    sizes = tuple(current if wanted is None else wanted for current, wanted in zip(x.shape[2:], requested))
    _capacity([*x.shape[:2], *sizes])
    if x.shape[0] == 0:
        result = Tensor.zeros([*x.shape[:2], *sizes]) + x.sum()
    else:
        result = x
        for axis, size in enumerate(sizes, 2):
            _capacity([*result.shape[:axis], size, *result.shape[axis + 1:]])
            length = result.shape[axis]
            bins = []
            for index in range(size):
                start = index * length // size
                end = ((index + 1) * length + size - 1) // size
                bins.append(result.narrow(axis, start, end - start).mean_dim(axis, True))
            result = _native.cat(bins, axis)
    return result.squeeze(0) if unbatched else result


def adaptive_avg_pool1d(input, output_size):
    """Adaptive mean over Torch's overlapping floor/ceil bins; positive output sizes."""
    return _adaptive_avg_pool(input, output_size, 1)


def adaptive_avg_pool2d(input, output_size):
    """Adaptive mean with positive output sizes and None to preserve an axis."""
    return _adaptive_avg_pool(input, output_size, 2)


def adaptive_avg_pool3d(input, output_size):
    """Adaptive mean with positive output sizes and None to preserve an axis."""
    return _adaptive_avg_pool(input, output_size, 3)
