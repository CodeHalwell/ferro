"""Typed shuffles, gates, views and channel softmax against actual PyTorch."""
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    raw, shape = value.detach().reshape(-1).tolist(), list(value.shape)
    if value.dtype == torch.float64:
        result = fr.Tensor.from_f64(raw, shape)
    elif value.dtype == torch.int64:
        result = fr.Tensor.from_i64(raw, shape)
    elif value.dtype == torch.bool:
        result = fr.tensor(raw, dtype='bool').reshape(shape)
    else:
        result = fr.Tensor(raw, shape)
    return result.requires_grad_(grad)


class RearrangementTests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=3e-6, rtol=4e-5)

    def compare(self, model, reference, data, strided=False):
        parent = native(data, data.dtype.is_floating_point)
        target = data.detach().requires_grad_(data.dtype.is_floating_point)
        left, right = (parent.transpose(-1, -2), target.transpose(-1, -2)) if strided else (parent, target)
        output, expected = model(left), reference(right)
        self.close(output, expected)
        if data.dtype.is_floating_point:
            probe = torch.linspace(-.7, .9, expected.numel(), dtype=data.dtype).reshape(expected.shape)
            (output * native(probe)).sum().backward()
            (expected * probe).sum().backward()
            self.close(parent.grad, target.grad)

    def test_pixel_shuffles_typed_layouts_and_gradients(self):
        for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
            for factor in (1, 2):
                for leading in ((), (2,), (2, 1)):
                    raw = torch.arange(torch.tensor(leading + (12, 2, 4)).prod().item()).reshape(leading + (12, 2, 4))
                    data = (raw % 3 == 0) if dtype == torch.bool else (raw + 2**53 if dtype == torch.int64 else raw * .02).to(dtype)
                    self.compare(fr.nn.PixelShuffle(factor), torch.nn.PixelShuffle(factor), data, True)
                    self.compare(fr.nn.PixelUnshuffle(factor), torch.nn.PixelUnshuffle(factor), data, True)

    def test_channel_shuffle_arbitrary_spatial_rank(self):
        for shape in ((2, 6, 3), (2, 6, 2, 3), (1, 6, 2, 1, 3)):
            for groups in (1, 2, 3, 6):
                for dtype in (torch.float32, torch.float64, torch.int64):
                    data = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(dtype) / 10 if dtype.is_floating_point else torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
                    self.compare(fr.nn.ChannelShuffle(groups), torch.nn.ChannelShuffle(groups), data, len(shape) > 3)

    def test_shuffle_output_owns_storage_even_for_identity_factor(self):
        data = torch.arange(32.).reshape(2, 4, 2, 2)
        for model, reference in ((fr.nn.PixelShuffle(1), torch.nn.PixelShuffle(1)),
                                 (fr.nn.PixelUnshuffle(1), torch.nn.PixelUnshuffle(1)),
                                 (fr.nn.ChannelShuffle(1), torch.nn.ChannelShuffle(1)),
                                 (fr.nn.ChannelShuffle(4), torch.nn.ChannelShuffle(4))):
            parent, target = native(data), data.clone()
            actual, expected = model(parent), reference(target)
            actual.copy_(fr.zeros(actual.shape))
            expected.zero_()
            self.close(parent, target)
            self.close(actual, expected)

    def test_glu_all_axes_empty_and_float64_gradients(self):
        for dtype in (torch.float32, torch.float64):
            data = torch.linspace(-2., 2., 48, dtype=dtype).reshape(2, 4, 6)
            for dim in (-3, -2, -1, 0, 1, 2):
                self.compare(fr.nn.GLU(dim), torch.nn.GLU(dim), data, True)
            self.compare(fr.nn.GLU(), torch.nn.GLU(), torch.empty(2, 0, dtype=dtype))

    def test_unflatten_strided_alias_and_gradients(self):
        for dim, sizes in ((1, (2, 3)), (-1, (2, -1))):
            model, reference = fr.nn.Unflatten(dim, sizes), torch.nn.Unflatten(dim, sizes)
            data = torch.arange(24., dtype=torch.float64).reshape(6, 4)
            parent, target = native(data, True), data.detach().requires_grad_()
            left, right = parent.transpose(0, 1), target.T
            actual, expected = model(left), reference(right)
            self.close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            self.close(parent.grad, target.grad)
            with fr.no_grad(), torch.no_grad():
                parent.copy_(native(torch.full_like(data, 7.)))
                target.fill_(7.)
            self.close(actual, expected)

    def test_softmax2d_channel_axis(self):
        for shape in ((3, 2, 4), (2, 3, 2, 4), (0, 3, 2, 4)):
            self.compare(fr.nn.Softmax2d(), torch.nn.Softmax2d(), torch.randn(shape))

    def test_functional_exports_empty_shapes_and_invalid_metadata(self):
        self.close(F.pixel_shuffle(fr.ones(4, 2, 3), 2), torch.nn.functional.pixel_shuffle(torch.ones(4, 2, 3), 2))
        self.close(F.pixel_unshuffle(fr.ones(1, 2, 4), 2), torch.nn.functional.pixel_unshuffle(torch.ones(1, 2, 4), 2))
        self.close(F.channel_shuffle(fr.ones(2, 4, 3), 2), torch.nn.functional.channel_shuffle(torch.ones(2, 4, 3), 2))
        self.close(F.glu(fr.ones(2, 4)), torch.nn.functional.glu(torch.ones(2, 4)))
        for shape in ((0, 4, 2, 4), (2, 0, 2, 4), (2, 4, 0, 4)):
            value = fr.zeros(shape)
            target = torch.empty(shape)
            self.close(F.pixel_shuffle(value, 2), torch.nn.functional.pixel_shuffle(target, 2))
            self.close(F.pixel_unshuffle(value, 2), torch.nn.functional.pixel_unshuffle(target, 2))
        for action in (lambda: fr.nn.PixelShuffle(0), lambda: fr.nn.PixelUnshuffle(True),
                       lambda: fr.nn.ChannelShuffle(-1), lambda: fr.nn.GLU('x'),
                       lambda: F.pixel_shuffle(fr.ones(3, 2, 3), 2),
                       lambda: F.pixel_unshuffle(fr.ones(2, 3, 3), 2),
                       lambda: F.channel_shuffle(fr.ones(2, 3, 4), 2),
                       lambda: F.glu(fr.ones(3)), lambda: fr.nn.Softmax2d()(fr.ones(2, 3)),
                       lambda: fr.nn.Unflatten(1, (2, 2))(fr.ones(2, 6))):
            with self.assertRaises((ValueError, TypeError)):
                action()


if __name__ == '__main__':
    unittest.main()
