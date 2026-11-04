"""Installed spatial options versus real PyTorch, including selected-index gradients."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def tensor(value, grad=False):
    return fr.Tensor(value.detach().flatten().tolist(), list(value.shape)).requires_grad_(grad)


def close(actual, expected):
    if actual.shape != list(expected.shape):
        raise AssertionError(f'shapes differ: {actual.shape} != {list(expected.shape)}')
    torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(expected.shape), expected.detach(),
                               atol=1e-5, rtol=1e-4, equal_nan=True)


class SpatialOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def gradients(self, actual, expected, leaves):
        close(actual, expected)
        probe = torch.randn(expected.shape, generator=torch.Generator().manual_seed(307))
        actual.backward_with(tensor(probe))
        expected.backward(probe)
        for name, native, reference in leaves:
            with self.subTest(gradient=name):
                self.assertIsNotNone(native.grad)
                self.assertIsNotNone(reference.grad)
                close(native.grad, reference.grad)

    def convolution(self, shape, options, output_size=None, noncontiguous=False):
        torch.manual_seed(503)
        reference = torch.nn.ConvTranspose3d(4, 6, **options)
        model = fr.nn.ConvTranspose3d(4, 6, **options)
        refs = dict(reference.named_parameters())
        self.assertEqual(list(dict(model.named_parameters())), list(refs))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.tensor().shape, list(refs[name].shape))
                parameter.tensor().copy_(tensor(refs[name]))
        tx = torch.randn(shape, generator=torch.Generator().manual_seed(505), requires_grad=True)
        sx = tensor(tx, True)
        x, y = (sx.transpose(-1, -2), tx.transpose(-1, -2)) if noncontiguous else (sx, tx)
        kwargs = {} if output_size is None else {'output_size': output_size}
        self.gradients(model(x, **kwargs), reference(y, **kwargs),
                       [('input', sx, tx)] + [(name, p.tensor(), refs[name]) for name, p in model.named_parameters()])

    def test_transpose3d_groups_padding_dilation_and_all_gradients(self):
        for shape, options, permute in (
            ([2, 4, 2, 3, 4], dict(kernel_size=(2, 3, 2), stride=(2, 1, 3), padding=(1, 1, 0), output_padding=(1, 0, 2), groups=2), False),
            ([4, 2, 3, 2], dict(kernel_size=2, stride=(1, 2, 1), dilation=(3, 2, 4), output_padding=(2, 1, 3), bias=False), False),
            ([1, 4, 2, 3, 2], dict(kernel_size=2, dilation=3, output_padding=2, groups=2), True),
            ([0, 4, 2, 3, 2], dict(kernel_size=2, stride=2, padding=1, groups=2), False),
            ([1, 4, 1, 2, 3], dict(kernel_size=1, stride=3, output_padding=2, groups=2), False),
        ):
            with self.subTest(shape=shape, options=options):
                self.convolution(shape, options, noncontiguous=permute)

    def test_transpose3d_explicit_output_sizes_and_noncontiguous_functional_weight(self):
        for shape, output in (([2, 4, 2, 3, 4], (4, 6, 8)), ([2, 4, 2, 3, 4], (2, 6, 4, 6, 8)),
                              ([4, 2, 3, 4], (6, 4, 6, 8))):
            with self.subTest(shape=shape, output=output):
                self.convolution(shape, dict(kernel_size=3, stride=2, padding=1), output)
        tx = torch.randn(2, 4, 2, 3, 2, requires_grad=True)
        tw = torch.randn(3, 4, 2, 2, 2, requires_grad=True)
        sx, sw = tensor(tx, True), tensor(tw, True)
        self.gradients(F.conv_transpose3d(sx, sw.transpose(0, 1), stride=2),
                       torch.nn.functional.conv_transpose3d(tx, tw.transpose(0, 1), stride=2),
                       [('input', sx, tx), ('weight', sw, tw)])
        self.convolution([1, 4, 1, 2, 2], dict(kernel_size=1, stride=(5, 1, 1), padding=(1, 0, 0), output_padding=(3, 0, 0)))

    def pool(self, name, shape, options, *, values=None, noncontiguous=False):
        tx = torch.randn(shape, generator=torch.Generator().manual_seed(509)) if values is None else torch.tensor(values).reshape(shape)
        tx.requires_grad_(True)
        sx = tensor(tx, True)
        x, y = (sx.transpose(-1, -2), tx.transpose(-1, -2)) if noncontiguous else (sx, tx)
        result = getattr(fr.nn, name)(**options)(x)
        expected = getattr(torch.nn, name)(**options)(y)
        if isinstance(expected, tuple):
            actual, indices = result
            wanted, reference_indices = expected
            self.assertEqual(indices.dtype, 'int64')
            self.assertFalse(indices.requires_grad)
            self.assertEqual(indices.tolist(), reference_indices.tolist())
        else:
            actual, wanted = result, expected
        self.gradients(actual, wanted, [('input', sx, tx)])

    def test_max_pool_all_dimensions_options_indices_and_gradients(self):
        cases = (
            ('MaxPool1d', [2, 3, 9], dict(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True)),
            ('MaxPool1d', [3, 6], dict(kernel_size=2, stride=3, padding=1, ceil_mode=True)),
            ('MaxPool2d', [2, 3, 5, 7], dict(kernel_size=(2, 3), stride=(2, 2), padding=(1, 1), dilation=(2, 1), ceil_mode=True)),
            ('MaxPool2d', [3, 4, 5], dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))),
            ('MaxPool3d', [2, 3, 5, 4, 7], dict(kernel_size=(3, 2, 3), stride=(2, 1, 3), padding=1, dilation=(1, 2, 1), ceil_mode=True)),
            ('MaxPool3d', [3, 4, 5, 3], dict(kernel_size=2, stride=2, padding=1, ceil_mode=True)),
        )
        for name, shape, options in cases:
            for indices in (False, True):
                with self.subTest(name=name, indices=indices, options=options):
                    self.pool(name, shape, {**options, 'return_indices': indices})

    def test_max_ties_negative_infinity_nan_and_overlapping_scatter_gradients(self):
        for dimensions in (1, 2, 3):
            shape = [1, 2] + [3] * dimensions
            count = torch.tensor(shape).prod().item()
            for values in ([1.] * count, [float('-inf')] * count,
                           ([float('nan'), 2., float('nan')] * ((count + 2) // 3))[:count]):
                with self.subTest(dimensions=dimensions, first=values[0]):
                    self.pool(f'MaxPool{dimensions}d', shape,
                              dict(kernel_size=2, stride=1, padding=1, return_indices=True), values=values)

    def test_average_pool_all_dimensions_padding_ceil_divisors_and_gradients(self):
        for dimensions in (1, 2, 3):
            name = f'AvgPool{dimensions}d'
            for batched in (False, True):
                shape = ([2] if batched else []) + [3] + [5, 4, 7][:dimensions]
                for include in (False, True):
                    for ceil in (False, True):
                        options = dict(kernel_size=3, stride=2, padding=1, ceil_mode=ceil, count_include_pad=include)
                        with self.subTest(name=name, shape=shape, options=options):
                            self.pool(name, shape, options)
            if dimensions > 1:
                for divisor in (5, -3):
                    with self.subTest(name=name, divisor=divisor):
                        self.pool(name, [2, 3] + [4, 5, 6][:dimensions],
                                  dict(kernel_size=tuple([2, 3, 2][:dimensions]), stride=tuple([3, 2, 3][:dimensions]),
                                       padding=1, ceil_mode=True, divisor_override=divisor))

    def test_adaptive_max_overlap_upsampling_none_and_selected_indices(self):
        for dimensions, shape, size in (
            (1, [2, 3, 7], 3), (1, [3, 3], 7),
            (2, [2, 3, 5, 7], (3, 4)), (2, [3, 2, 3], (5, 4)), (2, [2, 3, 4, 5], (None, 3)),
            (3, [2, 3, 5, 4, 7], (3, 2, 4)), (3, [3, 2, 3, 2], (3, 5, 4)), (3, [2, 3, 4, 5, 3], (None, 2, None)),
        ):
            with self.subTest(dimensions=dimensions, shape=shape, output_size=size):
                self.pool(f'AdaptiveMaxPool{dimensions}d', shape, dict(output_size=size, return_indices=True))
        for dimensions in (1, 2, 3):
            shape = [1, 1] + [3] * dimensions
            for value in (2., float('-inf'), float('nan')):
                self.pool(f'AdaptiveMaxPool{dimensions}d', shape, dict(output_size=2, return_indices=True),
                          values=[value] * 3 ** dimensions)

    def test_pool_empty_batches_and_noncontiguous_inputs(self):
        for dimensions in (1, 2, 3):
            for kind in ('Max', 'Avg', 'AdaptiveMax'):
                name = f'{kind}Pool{dimensions}d'
                options = dict(output_size=2, return_indices=True) if kind == 'AdaptiveMax' else dict(kernel_size=2, ceil_mode=True)
                if kind == 'Max':
                    options['return_indices'] = True
                with self.subTest(name=name):
                    self.pool(name, [0, 2] + [5] * dimensions, options)
                if dimensions > 1:
                    self.pool(name, [2, 3] + [3, 4, 5][:dimensions], options, noncontiguous=True)

    def test_functional_wrappers_and_functional_flag_order(self):
        for dimensions in (1, 2, 3):
            tx = torch.randn([2, 3] + [4, 5, 6][:dimensions], requires_grad=True)
            for prefix in ('max_pool', 'avg_pool', 'adaptive_max_pool'):
                with self.subTest(dimensions=dimensions, prefix=prefix):
                    sx = tensor(tx, True)
                    reference = tx.detach().clone().requires_grad_(True)
                    name = f'{prefix}{dimensions}d'
                    args = (2, None, 1, 1, True, True) if prefix == 'max_pool' else (2, None, 1, True, False) if prefix == 'avg_pool' else (2, True)
                    actual = getattr(F, name)(sx, *args)
                    expected = getattr(torch.nn.functional, name)(reference, *args)
                    if isinstance(expected, tuple):
                        self.assertEqual(actual[1].tolist(), expected[1].tolist())
                        actual, expected = actual[0], expected[0]
                    self.gradients(actual, expected, [('input', sx, reference)])

    def test_invalid_metadata_all_padding_windows_and_capacity(self):
        for dimensions in (1, 2, 3):
            for name in (f'MaxPool{dimensions}d', f'AvgPool{dimensions}d'):
                for options in (dict(kernel_size=0), dict(kernel_size=True), dict(kernel_size=2, stride=0),
                                dict(kernel_size=2, padding=2), dict(kernel_size=2, ceil_mode=1)):
                    with self.subTest(name=name, options=options), self.assertRaises((TypeError, ValueError)):
                        getattr(fr.nn, name)(**options)
                layer = getattr(fr.nn, name)(2)
                with self.assertRaisesRegex(ValueError, 'float32'):
                    layer(fr.tensor([[[1, 2]]], dtype='int64'))
                with self.assertRaises(ValueError):
                    layer(fr.Tensor.zeros([1, 2] + [0] * dimensions))
                with self.assertRaises(ValueError):
                    layer(fr.Tensor.ones([1, 2] + [1] * dimensions))
            with self.assertRaisesRegex(ValueError, 'contain an input'):
                getattr(fr.nn, f'MaxPool{dimensions}d')(2, 1, 1, 3)(fr.Tensor.ones([1, 1] + [2] * dimensions))
            with self.assertRaises(ValueError):
                getattr(fr.nn, f'AdaptiveMaxPool{dimensions}d')(0)
        for name in ('AvgPool2d', 'AvgPool3d'):
            for divisor in (0, True, .2):
                with self.assertRaises(ValueError):
                    getattr(fr.nn, name)(2, divisor_override=divisor)
        with self.assertRaisesRegex(ValueError, 'at least kernel_size'):
            fr.nn.AvgPool3d(3, padding=1)(fr.Tensor.ones([1, 1, 2, 2, 2]))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            fr.nn.ConvTranspose3d(2, 3, sys.maxsize)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            fr.nn.AdaptiveMaxPool3d(sys.maxsize)(fr.Tensor.zeros([0, 1, 1, 1, 1]))
        model = fr.nn.ConvTranspose3d(2, 3, 2, stride=2)
        with self.assertRaises(ValueError):
            model(fr.Tensor.ones([1, 2, 2, 2, 2]), output_size=(99, 99, 99))
        for options in ({'groups': 0}, {'output_padding': 1}, {'padding': 'same'}, {'dilation': 0}):
            with self.assertRaises(ValueError):
                fr.nn.ConvTranspose3d(2, 3, 2, **options)

    def test_checkpoint_all_configuration_fields_and_exact_optimizer_restart(self):
        def build(dilation=1, include=False, size=2):
            return fr.nn.Sequential(fr.nn.ConvTranspose3d(2, 3, 2, stride=2, dilation=dilation),
                                    fr.nn.AvgPool3d(2, 1, 1, True, include), fr.nn.AdaptiveMaxPool3d(size))
        model = build()
        model[0].reset_parameters(generator=fr.Generator(71))
        opt, rng = fr.optim.AdamW(model.parameters(), lr=.01), fr.Generator(73)
        def step(layer, optimizer, stream):
            output = layer(fr.Tensor.randn([2, 2, 2, 3, 2], generator=stream))
            loss = output.square().mean()
            value = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return value
        def state(layer, optimizer, stream):
            return fr.checkpoint.snapshot_training(layer, optimizers={'main': optimizer}, generator=stream, step=2)
        def contents(checkpoint):
            return {key: (value.dtype, value.shape, value.tolist()) for key, value in checkpoint.tensors().items()}
        step(model, opt, rng)
        model.eval()
        saved = state(model, opt, rng)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(Path(directory) / 'state')
            loaded = fr.checkpoint.load(Path(directory) / 'state')
            other = build()
            other_opt, other_rng = fr.optim.AdamW(other.parameters(), lr=.9), fr.Generator(3)
            loaded.restore(other, optimizers={'main': other_opt}, generator=other_rng)
            self.assertFalse(other[0].training)
            self.assertEqual(step(model, opt, rng), step(other, other_opt, other_rng))
            self.assertEqual(contents(state(model, opt, rng)), contents(state(other, other_opt, other_rng)))
            self.assertEqual(fr.Tensor.rand([7], generator=rng).tolist(), fr.Tensor.rand([7], generator=other_rng).tolist())
            for bad in (build(dilation=2), build(include=True), build(size=3)):
                bad_opt, bad_rng = fr.optim.AdamW(bad.parameters()), fr.Generator(9)
                before = contents(state(bad, bad_opt, bad_rng))
                with self.assertRaises(ValueError):
                    loaded.restore(bad, optimizers={'main': bad_opt}, generator=bad_rng)
                self.assertEqual(contents(state(bad, bad_opt, bad_rng)), before)
        for factory, changed in ((lambda: fr.nn.MaxPool2d(2, dilation=1), lambda: fr.nn.MaxPool2d(2, dilation=2)),
                                 (lambda: fr.nn.MaxPool1d(2, return_indices=True), lambda: fr.nn.MaxPool1d(2, return_indices=False))):
            snapshot = fr.checkpoint.snapshot_training(factory(), optimizers={}, generator=fr.Generator(2))
            with self.assertRaises(ValueError):
                snapshot.restore(changed(), optimizers={}, generator=fr.Generator(5))

    def test_signed_padding_modes_values_and_saved_view_gradients(self):
        for dimensions in (1, 2, 3):
            for batched in (False, True):
                shape = ([2] if batched else []) + [3] + [4, 5, 6][:dimensions]
                for mode in ('constant', 'reflect', 'replicate', 'circular'):
                    for pair in ((1, 2), (-1, 2), (2, -1), (-2, -1)):
                        with self.subTest(dimensions=dimensions, shape=shape, mode=mode, pair=pair):
                            tx = torch.randn(shape, generator=torch.Generator().manual_seed(523), requires_grad=True)
                            sx = tensor(tx, True)
                            pads = pair * dimensions
                            actual_x, expected_x = (sx.transpose(-1, -2), tx.transpose(-1, -2)) if dimensions > 1 else (sx, tx)
                            options = {'value': .75} if mode == 'constant' else {}
                            self.gradients(F.pad(actual_x, pads, mode, **options),
                                           torch.nn.functional.pad(expected_x, pads, mode, **options), [('input', sx, tx)])
        for mode in ('reflect', 'replicate'):
            tx = torch.arange(4., requires_grad=True).reshape(1, 1, 4).detach().requires_grad_(True)
            sx = tensor(tx, True)
            self.gradients(F.pad(sx, (-4, 1), mode), torch.nn.functional.pad(tx, (-4, 1), mode), [('input', sx, tx)])
        for mode in ('constant', 'circular'):
            tx = torch.randn(1, 2, 4, requires_grad=True)
            sx = tensor(tx, True)
            self.gradients(F.pad(sx, (-4, 0), mode), torch.nn.functional.pad(tx, (-4, 0), mode), [('input', sx, tx)])
        for mode in ('constant', 'reflect', 'replicate', 'circular'):
            tx = torch.empty(0, 2, 4, requires_grad=True)
            sx = tensor(tx, True)
            self.gradients(F.pad(sx, (1, 1), mode), torch.nn.functional.pad(tx, (1, 1), mode), [('input', sx, tx)])

    def test_convolution_padding_modes_all_dimensions_and_unbatched(self):
        for dimensions in (1, 2, 3):
            for mode in ('reflect', 'replicate', 'circular'):
                for padding in (1, 'same', 'valid'):
                    with self.subTest(dimensions=dimensions, mode=mode, padding=padding):
                        name = f'Conv{dimensions}d'
                        torch.manual_seed(541)
                        options = dict(kernel_size=2, padding=padding, groups=2, padding_mode=mode)
                        model, reference = getattr(fr.nn, name)(4, 6, **options), getattr(torch.nn, name)(4, 6, **options)
                        parameters = dict(reference.named_parameters())
                        with fr.no_grad():
                            for key, parameter in model.named_parameters():
                                parameter.tensor().copy_(tensor(parameters[key]))
                        shape = ([2] if padding == 1 else []) + [4] + [3, 4, 5][:dimensions]
                        tx = torch.randn(shape, generator=torch.Generator().manual_seed(547), requires_grad=True)
                        sx = tensor(tx, True)
                        self.gradients(model(sx), reference(tx), [('input', sx, tx)] +
                                       [(key, p.tensor(), parameters[key]) for key, p in model.named_parameters()])
                        for parameter in model.parameters():
                            parameter.zero_grad()
                        different = getattr(fr.nn, name)(4, 6, **{**options, 'padding_mode': 'zeros'})
                        checkpoint = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(2))
                        with self.assertRaises(ValueError):
                            checkpoint.restore(different, optimizers={}, generator=fr.Generator(3))
        tx, tw = torch.randn(4, 3, 5, requires_grad=True), torch.randn(6, 2, 2, 3, requires_grad=True)
        sx, sw = tensor(tx, True), tensor(tw, True)
        self.gradients(F.conv2d(sx, sw, padding='same', groups=2), torch.nn.functional.conv2d(tx, tw, padding='same', groups=2),
                       [('input', sx, tx), ('weight', sw, tw)])

    def test_padding_invalid_shapes_modes_and_preallocation_bounds(self):
        x = fr.Tensor.ones([1, 1, 4])
        for pads, mode, kwargs in (((1,), 'constant', {}), ((True, 0), 'constant', {}),
                                   ((-5, 2), 'constant', {}), ((4, 0), 'reflect', {}),
                                   ((-4, 0), 'replicate', {}), ((-4, 1), 'circular', {}),
                                   ((-3, 3), 'circular', {}), ((5, 0), 'circular', {}),
                                   ((1, 1), 'reflect', {'value': 1.})):
            with self.subTest(pads=pads, mode=mode), self.assertRaises((TypeError, ValueError)):
                F.pad(x, pads, mode, **kwargs)
        with self.assertRaises(NotImplementedError):
            F.pad(fr.Tensor.ones([4]), (1, 1), 'reflect')
        with self.assertRaises(NotImplementedError):
            F.pad(x, (1, 1), 'symmetric')
        with self.assertRaisesRegex(ValueError, 'capacity'):
            F.pad(x, (sys.maxsize, 0))
        for dimensions in (1, 2, 3):
            with self.assertRaises(NotImplementedError):
                getattr(fr.nn, f'ConvTranspose{dimensions}d')(2, 3, 2, padding_mode='reflect')

    def test_padding_modules_values_gradients_and_checkpoint_configuration(self):
        for prefix in ('Constant', 'Zero', 'Reflection', 'Replication', 'Circular'):
            for dimensions in (1, 2, 3):
                name = f'{prefix}Pad{dimensions}d'
                for padding in (1, (-1, 2) * dimensions):
                    with self.subTest(name=name, padding=padding):
                        arguments = (padding, .75) if prefix == 'Constant' else (padding,)
                        model = getattr(fr.nn, name)(*arguments)
                        reference = getattr(torch.nn, name)(*arguments)
                        tx = torch.randn([2, 3] + [4, 5, 6][:dimensions], requires_grad=True)
                        sx = tensor(tx, True)
                        self.gradients(model(sx), reference(tx), [('input', sx, tx)])
                        checkpoint = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1))
                        restored = getattr(fr.nn, name)(*arguments)
                        checkpoint.restore(restored, optimizers={}, generator=fr.Generator(2))
                        self.assertEqual(model(sx).tolist(), restored(sx).tolist())
                        changed_args = (2, .75) if prefix == 'Constant' else (2,)
                        with self.assertRaises(ValueError):
                            checkpoint.restore(getattr(fr.nn, name)(*changed_args), optimizers={}, generator=fr.Generator(3))
                with self.assertRaises((TypeError, ValueError)):
                    args = ((1,), .75) if prefix == 'Constant' else ((1,),)
                    getattr(fr.nn, name)(*args)
        checkpoint = fr.checkpoint.snapshot_training(fr.nn.ConstantPad2d(1, .5), optimizers={}, generator=fr.Generator(1))
        with self.assertRaises(ValueError):
            checkpoint.restore(fr.nn.ConstantPad2d(1, .7), optimizers={}, generator=fr.Generator(2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
