"""Independent PyTorch values/adjoints for public classification losses."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, requires_grad=False):
    shape, data = list(value.shape), value.detach().reshape(-1).tolist()
    result = fr.Tensor.from_i64(data, shape) if value.dtype == torch.int64 else fr.Tensor(data, shape)
    return result.requires_grad_(True) if requires_grad else result


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


def state_bytes(model, rng):
    state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=rng)
    return json.dumps({key: [value.shape, value.tolist()] for key, value in state.tensors().items()},
                      sort_keys=True, allow_nan=False).encode('ascii')


class ClassificationLossParityTests(unittest.TestCase):
    def compare(self, name, values, target, *, weight=None, reduction='mean', ignore_index=-100,
                smoothing=0.0, target_grad=False, weight_grad=False, module=False):
        reference = values.detach().clone().requires_grad_(True)
        expected_target = target.detach().clone().requires_grad_(target_grad)
        expected_weight = None if weight is None else weight.detach().clone().requires_grad_(weight_grad)
        actual, labels = native(values, True), native(target, target_grad)
        weights = None if weight is None else native(weight, weight_grad)
        options = dict(weight=weights, reduction=reduction, ignore_index=ignore_index)
        reference_options = dict(weight=expected_weight, reduction=reduction, ignore_index=ignore_index)
        if name == 'cross_entropy':
            options['label_smoothing'] = smoothing
            reference_options['label_smoothing'] = smoothing
        if module:
            cls = fr.nn.CrossEntropyLoss if name == 'cross_entropy' else fr.nn.NLLLoss
            result = cls(**options)(actual, labels)
        else:
            result = getattr(F, name)(actual, labels, **options)
        expected = getattr(torch.nn.functional, name)(reference, expected_target, **reference_options)
        torch.testing.assert_close(torch_value(result), expected.detach(), atol=3e-6, rtol=3e-5, equal_nan=True)
        # Nonuniform upstream gradients also exercise the unreduced path.
        seed = torch.arange(expected.numel(), dtype=torch.float32).reshape(expected.shape) * .17 + .3
        (result * native(seed)).sum().backward()
        (expected * seed).sum().backward()
        pairs = [(actual, reference)]
        if target_grad:
            pairs.append((labels, expected_target))
        if weight_grad:
            pairs.append((weights, expected_weight))
        for found, wanted in pairs:
            self.assertIsNotNone(found.grad)
            self.assertIsNotNone(wanted.grad)
            torch.testing.assert_close(torch_value(found.grad), wanted.grad, atol=4e-6, rtol=4e-5, equal_nan=True)

    def test_index_losses_spatial_unbatched_weights_ignores_and_smoothing(self):
        generator = torch.Generator().manual_seed(43)
        for shape in ((3,), (5, 3), (2, 3, 2, 3)):
            values = torch.randn(shape, generator=generator)
            target_shape = () if len(shape) == 1 else (shape[0],) + shape[2:]
            labels = torch.arange(math.prod(target_shape), dtype=torch.int64).reshape(target_shape) % 3
            if labels.numel() > 1:
                labels.reshape(-1)[1] = -7
            for weight in (None, torch.tensor([.25, 1.5, 2.])):
                for reduction in ('none', 'sum', 'mean'):
                    for name, smoothing in (('nll_loss', 0.), ('cross_entropy', 0.), ('cross_entropy', .2), ('cross_entropy', 1.)):
                        with self.subTest(shape=shape, weight=weight, reduction=reduction, loss=name, smoothing=smoothing):
                            self.compare(name, values, labels, weight=weight, reduction=reduction,
                                         ignore_index=-7, smoothing=smoothing)

    def test_probability_targets_preserve_input_target_and_weight_gradients(self):
        generator = torch.Generator().manual_seed(11)
        for shape in ((4,), (3, 4), (2, 4, 2, 3)):
            values = torch.randn(shape, generator=generator)
            # PyTorch intentionally allows unnormalized and negative entries.
            target = torch.randn(shape, generator=generator) * .4 + .2
            for weight in (None, torch.tensor([.2, 1., 1.7, .6])):
                for smoothing in (0., .3, 1.):
                    for reduction in ('none', 'sum', 'mean'):
                        with self.subTest(shape=shape, weight=weight, smoothing=smoothing, reduction=reduction):
                            self.compare('cross_entropy', values, target, weight=weight, smoothing=smoothing,
                                         reduction=reduction, target_grad=True, weight_grad=weight is not None)

    def test_empty_and_all_ignored_have_torch_values_and_zero_gradients(self):
        for shape, labels in [((0, 3), torch.empty(0, dtype=torch.int64)),
                              ((2, 3, 0), torch.empty(2, 0, dtype=torch.int64)),
                              ((2, 3), torch.full((2,), -100, dtype=torch.int64)),
                              ((3,), torch.tensor(-100))]:
            for reduction in ('none', 'sum', 'mean'):
                for name in ('cross_entropy', 'nll_loss'):
                    with self.subTest(shape=shape, reduction=reduction, name=name):
                        self.compare(name, torch.zeros(shape), labels, reduction=reduction,
                                     weight=torch.tensor([.5, 1., 2.]), smoothing=.2 if name == 'cross_entropy' else 0.)
        for shape in ((0, 3), (2, 3, 0)):
            for reduction in ('none', 'sum', 'mean'):
                self.compare('cross_entropy', torch.empty(shape), torch.empty(shape), reduction=reduction,
                             smoothing=.2, weight=torch.ones(3), target_grad=True, weight_grad=True)

    def test_zero_weight_denominator_differs_from_all_ignored(self):
        for weights in (torch.tensor([0., 0., 1.]), torch.tensor([1., -1., 1.])):
            for name in ('cross_entropy', 'nll_loss'):
                for reduction in ('none', 'sum', 'mean'):
                    for smoothing in ((0., .2, 1.) if name == 'cross_entropy' else (0.,)):
                        with self.subTest(weights=weights, name=name, reduction=reduction, smoothing=smoothing):
                            self.compare(name, torch.zeros(2, 3), torch.tensor([0, 1]), weight=weights,
                                         reduction=reduction, smoothing=smoothing)

    def test_positive_ignore_index_and_weighted_smoothing_denominator(self):
        values = torch.tensor([[.3, -1., 2.], [.5, 1., -2.], [3., 1., 2.], [1., 3., 2.]])
        target = torch.tensor([0, 1, 2, 0])
        for name in ('cross_entropy', 'nll_loss'):
            self.compare(name, values, target, weight=torch.tensor([.2, 5., 1.7]), ignore_index=1,
                         smoothing=.4 if name == 'cross_entropy' else 0.)

    def test_extreme_finite_logits_and_common_offsets(self):
        for values in (torch.tensor([[1e8, 1e8, 1e8], [3e38, 3e38, 3e38]]),
                       torch.tensor([[1000., -1000., 0.], [-1000., 1000., 5.]])):
            for smoothing in (0., .25):
                self.compare('cross_entropy', values, torch.tensor([0, 2]), smoothing=smoothing,
                             weight=torch.tensor([.5, 1.2, 2.]))

    def test_noncontiguous_input_and_target_views(self):
        base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        values = base.transpose(0, 2)  # [4,3,2], class axis remains one.
        targets = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]]).transpose(0, 1)
        x = native(base, True)
        actual = F.cross_entropy(x.transpose(0, 2), native(targets.transpose(0, 1)).transpose(0, 1), reduction='sum', label_smoothing=.2)
        ref = base.clone().requires_grad_(True)
        expected = torch.nn.functional.cross_entropy(ref.transpose(0, 2), targets, reduction='sum', label_smoothing=.2)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(torch_value(x.grad), ref.grad, atol=3e-6, rtol=3e-5)
        self.assertEqual(list(values.shape), [4, 3, 2])

    def test_modules_and_legacy_positional_reduction(self):
        x, target = torch.tensor([[.3, 1., -2.], [3., -.5, 1.]]), torch.tensor([1, 0])
        for name in ('cross_entropy', 'nll_loss'):
            for reduction in ('none', 'sum', 'mean'):
                self.compare(name, x, target, weight=torch.tensor([.2, 1.5, 2.]), reduction=reduction,
                             smoothing=.2 if name == 'cross_entropy' else 0., module=True)
                actual = getattr(F, name)(native(x), native(target), reduction)
                expected = getattr(torch.nn.functional, name)(x, target, reduction=reduction)
                torch.testing.assert_close(torch_value(actual), expected)

    def test_invalid_requests_reject_before_differentiable_computation(self):
        x, target = native(torch.zeros(2, 3)), native(torch.tensor([0, 1]))
        invalid = [(x, native(torch.tensor([0., 1.])), {}),
                   (x, native(torch.tensor([0, 3])), {}),
                   (x, native(torch.tensor([0, 2**40])), {}),
                   (x, native(torch.tensor([0, -2])), {}),
                   (x, native(torch.tensor([[0], [1]])), {}),
                   (x, target, {'weight': native(torch.ones(2))}),
                   (x, target, {'weight': native(torch.ones(3), True)}),
                   (x, target, {'weight': fr.Tensor.from_i64([1, 1, 1], [3])}),
                   (x, target, {'weight': [1., 1., 1.]}),
                   (x, target, {'reduction': 'batchmean'}),
                   (x, target, {'ignore_index': 1.5}),
                   (fr.Tensor.zeros([]), fr.Tensor.from_i64([0], []), {}),
                   (fr.Tensor.zeros([2, 0]), target, {}),
                   (fr.Tensor.from_f64([0.] * 6, [2, 3]), target, {})]
        with patch.object(F, '_classification_rows', side_effect=AssertionError('invalid request reached computation')):
            for name in ('cross_entropy', 'nll_loss'):
                for a, b, options in invalid:
                    with self.subTest(name=name, options=options), self.assertRaises((ValueError, TypeError)):
                        getattr(F, name)(a, b, **options)
            for smoothing in (-.1, 1.1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    F.cross_entropy(x, target, label_smoothing=smoothing)
            for prob, options in [(native(torch.ones(2, 3)), {'ignore_index': 0}),
                                  (fr.Tensor.from_i64([0] * 6, [2, 3]), {}),
                                  (fr.Tensor.from_f64([.3] * 6, [2, 3]), {})]:
                with self.assertRaises(ValueError):
                    F.cross_entropy(x, prob, **options)

    def test_module_buffers_and_configuration_round_trip_atomically(self):
        for cls in (fr.nn.CrossEntropyLoss, fr.nn.NLLLoss):
            options = dict(ignore_index=-7, reduction='sum')
            if cls is fr.nn.CrossEntropyLoss:
                options['label_smoothing'] = .2
            source = cls(weight=fr.Tensor([.2, 1.5, 2.], [3]), **options).eval()
            self.assertEqual([name for name, _ in source.named_buffers()], ['weight'])
            self.assertEqual(list(source.parameters()), [])
            for name in options:
                with self.assertRaises(AttributeError):
                    setattr(source, name, options[name])
            rng = fr.Generator(7)
            state = fr.checkpoint.snapshot_training(source, optimizers={}, generator=rng)
            with tempfile.TemporaryDirectory() as directory:
                state.save(Path(directory) / 'loss')
                state = fr.checkpoint.load(Path(directory) / 'loss')
                target = cls(weight=fr.Tensor.ones([3]), **options)
                state.restore(target, optimizers={}, generator=fr.Generator(11))
                self.assertEqual(target.weight.tolist(), source.weight.tolist())
                self.assertFalse(target.training)
                for changes in ({'reduction': 'mean'}, {'ignore_index': -8}, {'weight': None},
                                {'weight': fr.Tensor.ones([2])}):
                    bad = cls(**{'weight': fr.Tensor.ones([3]), **options, **changes})
                    other_rng = fr.Generator(43)
                    before = state_bytes(bad, other_rng)
                    with self.assertRaises(ValueError):
                        state.restore(bad, optimizers={}, generator=other_rng)
                    self.assertEqual(state_bytes(bad, other_rng), before)
                if cls is fr.nn.CrossEntropyLoss:
                    bad = cls(weight=fr.Tensor.ones([3]), **{**options, 'label_smoothing': .3})
                    other_rng = fr.Generator(43)
                    before = state_bytes(bad, other_rng)
                    with self.assertRaises(ValueError):
                        state.restore(bad, optimizers={}, generator=other_rng)
                    self.assertEqual(state_bytes(bad, other_rng), before)


if __name__ == '__main__':
    unittest.main()
