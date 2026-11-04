"""Spatial compositions versus actual PyTorch, including every parameter gradient."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import ferro as fr
from ferro import _spatial as functional
from ferro.nn.spatial import (Conv1d, Conv3d, ConvTranspose1d, ConvTranspose2d,
                              AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d)
import torch


def tensor(value, requires_grad=False):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def close(actual, expected, name):
    if actual.shape != list(expected.shape):
        raise AssertionError(f'{name}: shape {actual.shape} differs from {list(expected.shape)}')
    got = torch.tensor(actual.tolist(), dtype=torch.float32).reshape(expected.shape)
    torch.testing.assert_close(got, expected.detach(), atol=1e-5, rtol=1e-4,
                               msg=lambda message: name + ': ' + message)


def copy_weights(model, reference):
    actual, expected = dict(model.named_parameters()), dict(reference.named_parameters())
    if list(actual) != list(expected):
        raise AssertionError('spatial parameter names/order differ from PyTorch')
    with fr.no_grad():
        for name, parameter in actual.items():
            parameter.tensor().copy_(tensor(expected[name]))


def gradients(result, expected, leaves):
    close(result, expected, 'output')
    coefficients = torch.randn(expected.shape, generator=torch.Generator().manual_seed(83))
    loss = (result * tensor(coefficients)).sum() + (result * result).sum() * 0.07
    reference = (expected * coefficients).sum() + (expected * expected).sum() * 0.07
    close(loss, reference, 'loss')
    loss.backward()
    reference.backward()
    for name, actual, wanted in leaves:
        if actual.grad is None or wanted.grad is None:
            raise AssertionError('missing gradient: ' + name)
        close(actual.grad, wanted.grad, name + ' gradient')


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_convolution(self, actual_cls, shape, options, output_size=None, permute=False):
        reference_cls = getattr(torch.nn, actual_cls.__name__)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            reference = reference_cls(4, 6, **options)
        model = actual_cls(4, 6, **options)
        copy_weights(model, reference)
        tx = torch.randn(shape, generator=torch.Generator().manual_seed(41), requires_grad=True)
        x = tensor(tx, True)
        inputs, torch_inputs = (x.transpose(-1, -2), tx.transpose(-1, -2)) if permute else (x, tx)
        arguments = {} if output_size is None else {'output_size': output_size}
        result, expected = model(inputs, **arguments), reference(torch_inputs, **arguments)
        leaves = [('input', x, tx)] + [(name, parameter.tensor(), dict(reference.named_parameters())[name])
                                      for name, parameter in model.named_parameters()]
        gradients(result, expected, leaves)

    def test_convolution_outputs_input_and_all_parameter_gradients(self):
        cases = (
            (Conv1d, [2, 4, 9], {'kernel_size': 3, 'stride': 2, 'padding': 2, 'dilation': 2, 'groups': 2}),
            (Conv1d, [4, 8], {'kernel_size': 4, 'padding': 'same', 'bias': False}),
            (Conv1d, [2, 4, 7], {'kernel_size': 2, 'padding': 'valid', 'groups': 2}),
            (Conv3d, [2, 4, 4, 5, 6], {'kernel_size': (2, 2, 3), 'stride': (1, 2, 1), 'padding': (1, 0, 2), 'dilation': (2, 1, 2), 'groups': 2}),
            (Conv3d, [4, 3, 4, 5], {'kernel_size': (2, 2, 2), 'padding': 'same', 'bias': False}),
            (Conv3d, [1, 4, 3, 4, 4], {'kernel_size': 2, 'padding': 'valid', 'groups': 2}),
            (ConvTranspose1d, [2, 4, 5], {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 2}),
            (ConvTranspose1d, [4, 4], {'kernel_size': 2, 'stride': 1, 'dilation': 3, 'output_padding': 2, 'bias': False}),
            (ConvTranspose2d, [2, 4, 3, 5], {'kernel_size': (3, 2), 'stride': (2, 3), 'padding': (1, 1), 'output_padding': (1, 2), 'groups': 2}),
            (ConvTranspose2d, [4, 4, 3], {'kernel_size': (2, 3), 'stride': (1, 2), 'dilation': (3, 2), 'output_padding': (2, 1), 'bias': False}),
            (ConvTranspose2d, [2, 4, 3, 3], {'kernel_size': 2, 'stride': 1, 'dilation': 3, 'output_padding': 2, 'groups': 2}),
        )
        for cls, shape, options in cases:
            with self.subTest(cls=cls.__name__, shape=shape, options=options):
                self.check_convolution(cls, shape, options)

    def test_noncontiguous_inputs_and_functional_weights(self):
        for cls, shape in ((Conv3d, [2, 4, 3, 4, 5]), (ConvTranspose2d, [2, 4, 3, 4])):
            with self.subTest(cls=cls.__name__):
                self.check_convolution(cls, shape, {'kernel_size': 2, 'padding': 1, 'groups': 2}, permute=True)
        generator = torch.Generator().manual_seed(39)
        for name, input_shape, weight_shape, kwargs in (
            ('conv1d', [2, 4, 7], [3, 4, 2], {'padding': 1}),
            ('conv3d', [1, 4, 3, 4, 4], [3, 4, 2, 2, 2], {'padding': 1}),
            ('conv_transpose1d', [2, 4, 5], [3, 4, 2], {'stride': 2}),
            ('conv_transpose2d', [1, 4, 3, 4], [3, 4, 2, 2], {'stride': (2, 1)}),
        ):
            with self.subTest(function=name):
                tx = torch.randn(input_shape, generator=generator, requires_grad=True)
                tw = torch.randn(weight_shape, generator=generator, requires_grad=True)
                x, w = tensor(tx, True), tensor(tw, True)
                transposed = 'transpose' in name
                weight, torch_weight = (w.transpose(0, 1), tw.transpose(0, 1)) if transposed else (w, tw)
                actual = getattr(functional, name)(x, weight, **kwargs)
                expected = getattr(torch.nn.functional, name)(tx, torch_weight, **kwargs)
                gradients(actual, expected, [('input', x, tx), ('weight', w, tw)])

    def test_empty_batches_keep_all_gradient_paths(self):
        for cls, shape in ((Conv1d, [0, 4, 5]), (Conv3d, [0, 4, 3, 4, 5]),
                           (ConvTranspose1d, [0, 4, 5]), (ConvTranspose2d, [0, 4, 3, 4])):
            with self.subTest(cls=cls.__name__):
                self.check_convolution(cls, shape, {'kernel_size': 2, 'padding': 1, 'groups': 2})

    def test_transpose_explicit_spatial_and_full_output_sizes(self):
        for cls, shape, output_sizes in (
            (ConvTranspose1d, [2, 4, 4], ((8,), (2, 6, 8))),
            (ConvTranspose1d, [4, 4], ((8,), (6, 8))),
            (ConvTranspose2d, [2, 4, 3, 4], ((6, 8), (2, 6, 6, 8))),
            (ConvTranspose2d, [4, 3, 4], ((6, 8), (6, 6, 8))),
        ):
            for size in output_sizes:
                with self.subTest(cls=cls.__name__, shape=shape, output_size=size):
                    self.check_convolution(cls, shape, {'kernel_size': 3, 'stride': 2, 'padding': 1}, output_size=size)

    def test_fully_padded_conv3d_depth_keeps_zero_input_and_weight_gradients(self):
        self.check_convolution(Conv3d, [2, 4, 1, 3, 3], {'kernel_size': 1, 'padding': (1, 0, 0), 'stride': (2, 1, 1)})
        x = fr.Tensor([float('nan')], [1, 1, 1, 1, 1]).requires_grad_(True)
        weight = fr.Tensor.ones([1, 1, 1, 1, 1]).requires_grad_(True)
        result = functional.conv3d(x, weight, stride=(2, 1, 1), padding=(1, 0, 0))
        self.assertEqual(result.tolist(), [[[[[0.]], [[0.]]]]])
        result.sum().backward()
        self.assertEqual(x.grad.tolist(), [[[[[0.]]]]])
        self.assertEqual(weight.grad.tolist(), [[[[[0.]]]]])

    def test_adaptive_pool_bins_overlap_upsampling_none_and_gradients(self):
        cases = (
            (AdaptiveAvgPool1d, [2, 3, 7], 3), (AdaptiveAvgPool1d, [3, 3], 7),
            (AdaptiveAvgPool2d, [2, 3, 5, 7], (3, 4)), (AdaptiveAvgPool2d, [3, 2, 3], (5, 4)),
            (AdaptiveAvgPool2d, [2, 3, 5, 7], (None, 3)),
            (AdaptiveAvgPool3d, [2, 3, 5, 4, 7], (3, 2, 4)), (AdaptiveAvgPool3d, [3, 2, 3, 2], (3, 5, 4)),
            (AdaptiveAvgPool3d, [2, 3, 5, 4, 3], (None, 2, None)),
            (AdaptiveAvgPool1d, [0, 3, 4], 2), (AdaptiveAvgPool2d, [0, 3, 4, 5], 2),
            (AdaptiveAvgPool3d, [0, 3, 4, 5, 6], 2),
        )
        for cls, shape, size in cases:
            with self.subTest(cls=cls.__name__, shape=shape, size=size):
                tx = torch.randn(shape, generator=torch.Generator().manual_seed(17), requires_grad=True)
                x = tensor(tx, True)
                gradients(cls(size)(x), getattr(torch.nn, cls.__name__)(size)(tx), [('input', x, tx)])

    def test_invalid_options_shapes_dtypes_and_bounds(self):
        classes = (Conv1d, Conv3d, ConvTranspose1d, ConvTranspose2d)
        for cls in classes:
            for args in ((0, 3, 2), (2, 0, 2), (2, 3, 0), (True, 3, 2)):
                with self.subTest(cls=cls.__name__, args=args), self.assertRaises(ValueError):
                    cls(*args)
            for options in ({'groups': 3}, {'groups': 0}, {'stride': 0}, {'dilation': 0}, {'padding': -1}, {'kernel_size': (2, 2, 2, 2)}):
                with self.subTest(cls=cls.__name__, options=options), self.assertRaises(ValueError):
                    cls(2, 4, **{'kernel_size': 2, **options})
            for options in ({'padding_mode': 'symmetric'}, {'dtype': 'float64'}, {'device': 'cuda'}):
                with self.subTest(cls=cls.__name__, options=options), self.assertRaises(NotImplementedError):
                    cls(2, 3, 2, **options)
            with self.assertRaises(TypeError):
                cls(2, 3, 2, bias=1)
            dimensions = cls._dimensions
            model = cls(2, 3, 2)
            for shape in ([2], [1, 3] + [3] * dimensions, [1, 2] + [0] * dimensions):
                with self.subTest(cls=cls.__name__, shape=shape), self.assertRaises(ValueError):
                    model(fr.Tensor.zeros(shape))
            bad_shape = [1, 2] + [3] * dimensions
            with self.assertRaisesRegex(ValueError, 'float32'):
                model(fr.Tensor.from_f64([0.] * math.prod(bad_shape), bad_shape))
        for cls in (Conv1d, Conv3d):
            with self.assertRaises(ValueError):
                cls(2, 3, 2, stride=2, padding='same')
        for cls in (ConvTranspose1d, ConvTranspose2d):
            for options in ({'output_padding': 1}, {'output_padding': -1}, {'padding': 'same'}):
                with self.subTest(cls=cls.__name__, options=options), self.assertRaises(ValueError):
                    cls(2, 3, 2, **options)
            model = cls(2, 3, 3, stride=2, padding=1)
            x = fr.Tensor.zeros([1, 2] + [4] * cls._dimensions)
            for size in ((3,) * cls._dimensions, (10,) * cls._dimensions, (), (True,) * cls._dimensions):
                with self.subTest(cls=cls.__name__, size=size), self.assertRaises(ValueError):
                    model(x, output_size=size)
        for cls in (AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d):
            for size in (0, -1, True, (1, 2, 3, 4)):
                with self.subTest(cls=cls.__name__, size=size), self.assertRaises((ValueError, TypeError)):
                    cls(size)
        with self.assertRaises(ValueError):
            AdaptiveAvgPool1d((None,))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            Conv3d(2, 3, sys.maxsize)
        empty = fr.Tensor.zeros([0, 1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            functional.conv3d(empty, fr.Tensor.ones([1, 1, 1, 1, 1]), padding=sys.maxsize)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            functional.adaptive_avg_pool3d(empty, sys.maxsize)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            functional.adaptive_avg_pool2d(fr.Tensor.ones([1, 1, 1, 8]), (sys.maxsize // 8, 1))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            functional.conv_transpose2d(fr.Tensor.zeros([0, 1, 2, 2]), fr.Tensor.ones([1, 1, 1, 1]), stride=sys.maxsize)

    def test_reset_parameter_identity_bounds_and_train_eval(self):
        for cls in (Conv1d, Conv3d, ConvTranspose1d, ConvTranspose2d):
            with self.subTest(cls=cls.__name__):
                left, right = cls(4, 6, 2, groups=2), cls(4, 6, 2, groups=2)
                identities = [id(parameter) for parameter in left.parameters()]
                left.reset_parameters(generator=fr.Generator(7))
                right.reset_parameters(generator=fr.Generator(7))
                self.assertEqual([id(parameter) for parameter in left.parameters()], identities)
                bound = 1 / math.sqrt(left.weight.tensor().shape[1] * 2 ** cls._dimensions)
                for actual, expected in zip(left.parameters(), right.parameters()):
                    self.assertEqual(actual.tensor().tolist(), expected.tensor().tolist())
                    self.assertLessEqual(float(torch.tensor(actual.tensor().tolist()).abs().max()), bound)
                x = fr.Tensor.randn([2, 4] + [3] * cls._dimensions, seed=5)
                self.assertIs(left.eval(), left)
                evaluated = left(x).tolist()
                self.assertIs(left.train(), left)
                self.assertEqual(left(x).tolist(), evaluated)

    def test_checkpoint_config_modes_and_exact_adamw_restart(self):
        def model(dilation=1, output=(3, 4)):
            return fr.nn.Sequential(ConvTranspose2d(2, 3, 2, stride=2, dilation=dilation), AdaptiveAvgPool2d(output))
        network = model()
        network[0].reset_parameters(generator=fr.Generator(7))
        optimizer, rng = fr.optim.AdamW(network.parameters(), lr=0.01), fr.Generator(13)
        def step(module, opt, stream):
            opt.zero_grad()
            x = fr.Tensor.randn([2, 2, 3, 4], generator=stream)
            y = module(x)
            loss = (y * y).mean()
            value = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return value
        def snapshot(module, opt, stream, number):
            return fr.checkpoint.snapshot_training(module, optimizers={'main': opt}, generator=stream, step=number)
        def values(checkpoint):
            return {name: (value.dtype, value.shape, value.tolist()) for name, value in checkpoint.tensors().items()}
        for _ in range(2):
            step(network, optimizer, rng)
        network.eval()
        checkpoint = snapshot(network, optimizer, rng, 2)
        expected_losses = [step(network, optimizer, rng) for _ in range(3)]
        expected_state = values(snapshot(network, optimizer, rng, 5))
        expected_draws = fr.Tensor.randn([9], generator=rng).tolist()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint.save(Path(directory) / 'spatial')
            saved = fr.checkpoint.load(Path(directory) / 'spatial')
            restored = model()
            target_opt, target_rng = fr.optim.AdamW(restored.parameters(), lr=0.7), fr.Generator(99)
            self.assertEqual(saved.restore(restored, optimizers={'main': target_opt}, generator=target_rng), 2)
            self.assertFalse(restored.training)
            self.assertFalse(restored[1].training)
            self.assertEqual([step(restored, target_opt, target_rng) for _ in range(3)], expected_losses)
            self.assertEqual(values(snapshot(restored, target_opt, target_rng, 5)), expected_state)
            self.assertEqual(fr.Tensor.randn([9], generator=target_rng).tolist(), expected_draws)
            for mismatch in (model(dilation=2), model(output=(4, 4))):
                bad_opt, bad_rng = fr.optim.AdamW(mismatch.parameters()), fr.Generator(3)
                before = values(snapshot(mismatch, bad_opt, bad_rng, 0))
                with self.assertRaises(ValueError):
                    saved.restore(mismatch, optimizers={'main': bad_opt}, generator=bad_rng)
                self.assertEqual(values(snapshot(mismatch, bad_opt, bad_rng, 0)), before)


if __name__ == '__main__':
    unittest.main()
