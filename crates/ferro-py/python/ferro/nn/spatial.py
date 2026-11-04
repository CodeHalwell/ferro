"""CPU spatial modules with PyTorch parameter layouts and output conventions.

Conv1d/2d also support float64 with zero padding; other spatial modules use f32.

Convolutions support zero/reflect/replicate/circular padding; transposes use zeros.
Adaptive pooling requires positive
channel, input-spatial and output-spatial sizes; empty batches are supported.
These eager compositions favor compatibility over fused-kernel performance.
"""
import math

from .. import _spatial
from .._native import Tensor, no_grad
from .module import Module, Parameter


class _Convolution(Module):
    def _setup(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
               groups, bias, padding_mode, output_padding, device, dtype):
        dimensions = self._dimensions
        if any(type(size) is not int or size <= 0 for size in (in_channels, out_channels, groups)):
            raise ValueError('in_channels, out_channels and groups must be positive integers')
        if in_channels % groups or out_channels % groups:
            raise ValueError('groups must divide both input and output channels')
        if type(bias) is not bool:
            raise TypeError('bias must be bool')
        if padding_mode not in ('zeros', 'reflect', 'replicate', 'circular') or self._transposed and padding_mode != 'zeros':
            raise NotImplementedError('convolutions support zeros/reflect/replicate/circular; transposes require zeros')
        if device not in (None, 'cpu'):
            raise NotImplementedError('spatial convolution modules currently support CPU only')
        from .. import _native
        from .._tensor_api import _dtype, zeros
        dtype = _dtype(dtype)
        if dtype not in ('float32', 'float64'):
            raise NotImplementedError('convolution parameters require float32 or float64')
        if dtype == 'float64' and (dimensions not in (1, 2) or self._transposed or padding_mode != 'zeros'):
            raise NotImplementedError('float64 currently supports Conv1d/2d with zero padding mode')
        if _native._is_capturing() or _native._forward_ad_is_active():
            raise NotImplementedError('convolution initialization during capture or forward AD is unsupported')
        kernel = _spatial._tuple(kernel_size, dimensions, 'kernel_size', 1)
        strides = _spatial._tuple(stride, dimensions, 'stride', 1)
        dilations = _spatial._tuple(dilation, dimensions, 'dilation', 1)
        if isinstance(padding, str):
            if self._transposed or padding not in ('same', 'valid'):
                raise ValueError('padding must be numeric, or same/valid for ordinary convolution')
            if padding == 'same' and any(step != 1 for step in strides):
                raise ValueError("padding='same' requires stride=1")
            pads = padding
        else:
            pads = _spatial._tuple(padding, dimensions, 'padding')
        extra = _spatial._tuple(output_padding, dimensions, 'output_padding')
        if self._transposed and any(pad >= step and pad >= spacing for pad, step, spacing in zip(extra, strides, dilations)):
            raise ValueError('output_padding must be smaller than stride or dilation on each axis')
        shape = [in_channels, out_channels // groups, *kernel] if self._transposed else [out_channels, in_channels // groups, *kernel]
        _spatial._capacity(shape, dtype)
        self._initialize()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding, self.dilation = kernel, strides, pads, dilations
        self.groups, self.padding_mode = groups, padding_mode
        self.transposed, self.output_padding = self._transposed, extra
        self.weight = Parameter(Tensor.zeros(shape) if dtype == 'float32' else zeros(shape, dtype=dtype))
        self.bias = Parameter(Tensor.zeros([out_channels]) if dtype == 'float32' else zeros([out_channels], dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self, *, generator=None):
        """Torch's uniform initialization, optionally using an explicit Ferro Generator."""
        from .. import _native
        if _native._is_capturing() or _native._forward_ad_is_active():
            raise NotImplementedError('convolution reset during capture or forward AD is unsupported')
        values = [parameter.tensor() for parameter in self.parameters()]
        for value in values:
            if value.device != 'cpu' or value.dtype not in ('float32', 'float64'):
                raise NotImplementedError('convolution reset requires CPU floating parameters')
            _spatial._capacity(value.shape, value.dtype)
        bound = 1 / math.sqrt(self.weight.tensor().shape[1] * math.prod(self.kernel_size))
        with no_grad():
            for value in values:
                value.copy_((Tensor.rand(list(value.shape), generator=generator, dtype=value.dtype) * 2 - 1) * bound)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding
        x, unbatched = _spatial._input(input, self._dimensions)
        try:
            size = tuple(output_size)
        except TypeError:
            raise TypeError('output_size must be a spatial or full output-shape sequence') from None
        prefix = 1 if unbatched else 2
        if len(size) == self._dimensions + prefix:
            size = size[prefix:]
        size = _spatial._tuple(size, self._dimensions, 'output_size', 1)
        minimum = tuple((width - 1) * step - 2 * pad + dilation * (kernel - 1) + 1
                        for width, step, pad, dilation, kernel in zip(x.shape[2:], self.stride, self.padding, self.dilation, self.kernel_size))
        if any(wanted < low or wanted >= low + step for wanted, low, step in zip(size, minimum, self.stride)):
            maximum = tuple(low + step - 1 for low, step in zip(minimum, self.stride))
            raise ValueError(f'output_size must be between {minimum} and {maximum}, got {size}')
        return tuple(wanted - low for wanted, low in zip(size, minimum))


class Conv1d(_Convolution):
    """Grouped NCL convolution, including CL unbatched input and same/valid padding."""
    _builtin_init, _dimensions, _transposed = True, 1, False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        self._setup(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                    bias, padding_mode, 0, device, dtype)

    def forward(self, input):
        input, padding = _spatial._conv_padding(input, self.kernel_size, self.padding, self.dilation, self.padding_mode)
        return _spatial.conv1d(input, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                               self.stride, padding, self.dilation, self.groups)


class Conv3d(_Convolution):
    """Grouped NCDHW convolution composed from native Conv2d depth slices."""
    _builtin_init, _dimensions, _transposed = True, 3, False

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        self._setup(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                    bias, padding_mode, 0, device, dtype)

    def forward(self, input):
        input, padding = _spatial._conv_padding(input, self.kernel_size, self.padding, self.dilation, self.padding_mode)
        return _spatial.conv3d(input, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                               self.stride, padding, self.dilation, self.groups)


class ConvTranspose1d(_Convolution):
    """Grouped NCL transposed convolution with IOK weight layout."""
    _builtin_init, _dimensions, _transposed = True, 1, True

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        self._setup(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                    bias, padding_mode, output_padding, device, dtype)

    def forward(self, input, output_size=None):
        return _spatial.conv_transpose1d(input, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                                         self.stride, self.padding, self._output_padding(input, output_size), self.groups, self.dilation)


class ConvTranspose2d(_Convolution):
    """Grouped NCHW transposed convolution with IOHW weight layout."""
    _builtin_init, _dimensions, _transposed = True, 2, True

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        self._setup(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                    bias, padding_mode, output_padding, device, dtype)

    def forward(self, input, output_size=None):
        return _spatial.conv_transpose2d(input, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                                         self.stride, self.padding, self._output_padding(input, output_size), self.groups, self.dilation)


class ConvTranspose3d(_Convolution):
    """Grouped NCDHW transposed convolution with IODHW weight layout."""
    _builtin_init, _dimensions, _transposed = True, 3, True

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        self._setup(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                    bias, padding_mode, output_padding, device, dtype)

    def forward(self, input, output_size=None):
        return _spatial.conv_transpose3d(input, self.weight.tensor(), None if self.bias is None else self.bias.tensor(),
                                         self.stride, self.padding, self._output_padding(input, output_size), self.groups, self.dilation)


class _MaxPool(Module):
    _builtin_init = True

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        kernel, steps, pads, spacing = _spatial._pool_options(kernel_size, stride, padding, self._dimensions, dilation, ceil_mode)
        if type(return_indices) is not bool:
            raise TypeError('return_indices must be bool')
        self._initialize()
        self.kernel_size, self.stride, self.padding, self.dilation = kernel, steps, pads, spacing
        self.return_indices, self.ceil_mode = return_indices, ceil_mode

    def forward(self, input):
        return _spatial._max_pool(input, self.kernel_size, self.stride, self.padding, self.dilation,
                                  self.ceil_mode, self.return_indices, self._dimensions)


class MaxPool1d(_MaxPool):
    """CPU max pooling with padding, dilation, ceil mode and optional int64 indices."""
    _dimensions = 1


class MaxPool3d(_MaxPool):
    """CPU volumetric max pooling; windows containing only padding reject."""
    _dimensions = 3


class _AvgPool(Module):
    _builtin_init = True

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        kernel, steps, pads, _ = _spatial._pool_options(kernel_size, stride, padding, self._dimensions, ceil_mode=ceil_mode)
        if type(count_include_pad) is not bool:
            raise TypeError('count_include_pad must be bool')
        if divisor_override is not None and (type(divisor_override) is not int or divisor_override == 0):
            raise ValueError('divisor_override must be a nonzero integer or None')
        self._initialize()
        self.kernel_size, self.stride, self.padding = kernel, steps, pads
        self.ceil_mode, self.count_include_pad, self.divisor_override = ceil_mode, count_include_pad, divisor_override

    def forward(self, input):
        return _spatial._avg_pool(input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                  self.count_include_pad, self.divisor_override, self._dimensions)


class AvgPool1d(_AvgPool):
    """CPU average pooling with ceil mode and optional exclusion of padding."""
    _builtin_init, _dimensions = True, 1

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)


class AvgPool3d(_AvgPool):
    """CPU volumetric average pooling with optional nonzero divisor_override."""
    _dimensions = 3


class _AdaptiveMaxPool(Module):
    _builtin_init = True

    def __init__(self, output_size, return_indices=False):
        if type(return_indices) is not bool:
            raise TypeError('return_indices must be bool')
        self._initialize()
        self.output_size = _spatial._pool_size(output_size, self._dimensions)
        self.return_indices = return_indices

    def forward(self, input):
        return _spatial._adaptive_max_pool(input, self.output_size, self.return_indices, self._dimensions)


class AdaptiveMaxPool1d(_AdaptiveMaxPool):
    """Adaptive floor/ceil max bins with optional per-channel spatial indices."""
    _dimensions = 1


class AdaptiveMaxPool2d(_AdaptiveMaxPool):
    """Adaptive max pooling; None preserves a spatial dimension."""
    _dimensions = 2


class AdaptiveMaxPool3d(_AdaptiveMaxPool):
    """Adaptive volumetric max pooling with exact int64 selected positions."""
    _dimensions = 3


class _Pad(Module):
    _builtin_init = True

    def __init__(self, padding):
        if type(padding) is int:
            padding = (padding,) * (2 * self._dimensions)
        else:
            try:
                padding = tuple(padding)
            except TypeError:
                raise TypeError('padding must be an integer or signed spatial pairs') from None
        if len(padding) != 2 * self._dimensions or any(type(value) is not int for value in padding):
            raise ValueError('padding must contain one signed integer pair per spatial dimension')
        self._initialize()
        self.padding, self.mode, self.value = padding, self._mode, None

    def forward(self, input):
        return _spatial.pad(input, self.padding, self.mode, self.value)


class _ConstantPad(_Pad):
    _builtin_init, _mode = True, 'constant'

    def __init__(self, padding, value):
        if type(value) not in (int, float):
            raise TypeError('constant padding value must be numeric')
        super().__init__(padding)
        self.value = float(value)


class ConstantPad1d(_ConstantPad):
    _dimensions = 1


class ConstantPad2d(_ConstantPad):
    _dimensions = 2


class ConstantPad3d(_ConstantPad):
    _dimensions = 3


class _ZeroPad(_ConstantPad):
    _builtin_init = True

    def __init__(self, padding):
        super().__init__(padding, 0.)


class ZeroPad1d(_ZeroPad):
    _dimensions = 1


class ZeroPad2d(_ZeroPad):
    _dimensions = 2


class ZeroPad3d(_ZeroPad):
    _dimensions = 3


class ReflectionPad1d(_Pad):
    _dimensions, _mode = 1, 'reflect'


class ReflectionPad2d(_Pad):
    _dimensions, _mode = 2, 'reflect'


class ReflectionPad3d(_Pad):
    _dimensions, _mode = 3, 'reflect'


class ReplicationPad1d(_Pad):
    _dimensions, _mode = 1, 'replicate'


class ReplicationPad2d(_Pad):
    _dimensions, _mode = 2, 'replicate'


class ReplicationPad3d(_Pad):
    _dimensions, _mode = 3, 'replicate'


class CircularPad1d(_Pad):
    _dimensions, _mode = 1, 'circular'


class CircularPad2d(_Pad):
    _dimensions, _mode = 2, 'circular'


class CircularPad3d(_Pad):
    _dimensions, _mode = 3, 'circular'


class _AdaptiveAvgPool(Module):
    _builtin_init = True

    def __init__(self, output_size):
        self._initialize()
        self.output_size = _spatial._pool_size(output_size, self._dimensions)

    def forward(self, input):
        return _spatial._adaptive_avg_pool(input, self.output_size, self._dimensions)


class AdaptiveAvgPool1d(_AdaptiveAvgPool):
    """Adaptive mean pooling of NCL/CL input into a positive output length."""
    _dimensions = 1


class AdaptiveAvgPool2d(_AdaptiveAvgPool):
    """Adaptive mean pooling of NCHW/CHW; None preserves an input spatial axis."""
    _dimensions = 2


class AdaptiveAvgPool3d(_AdaptiveAvgPool):
    """Adaptive mean pooling of NCDHW/CDHW; None preserves an input spatial axis."""
    _dimensions = 3
