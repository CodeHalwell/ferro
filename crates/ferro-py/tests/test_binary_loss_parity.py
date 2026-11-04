"""CPU binary loss values, adjoints, weights and module state against Torch."""
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.'), requires_grad=grad)


class BinaryLossParity(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        actual = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(actual, expected.detach(), atol=2e-6, rtol=3e-6, equal_nan=True)

    def test_weighted_reductions_and_both_input_adjoints(self):
        for dtype in (torch.float32, torch.float64):
            for logits in (False, True):
                for reduction in ('none', 'sum', 'mean'):
                    for weighted in (False, True):
                        with self.subTest(dtype=dtype, logits=logits, reduction=reduction, weighted=weighted):
                            values = [[-.7, 0., 1.2], [-2., .5, .9]] if logits else [[.12, .5, .91], [.3, .7, .4]]
                            x = torch.tensor(values, dtype=dtype, requires_grad=True)
                            target = torch.tensor([[.2, .9, .6], [.7, .4, .1]], dtype=dtype, requires_grad=True)
                            a, t = native(x, True), native(target, True)
                            weight = torch.tensor([.7, 1.2, .4], dtype=dtype) if weighted else None
                            positive = torch.tensor([[.4], [2.3]], dtype=dtype) if weighted and logits else None
                            options = {'weight': weight, 'reduction': reduction}
                            actual_options = {'weight': None if weight is None else native(weight), 'reduction': reduction}
                            if logits:
                                options['pos_weight'] = positive
                                actual_options['pos_weight'] = None if positive is None else native(positive)
                            reference = torch.nn.functional.binary_cross_entropy_with_logits if logits else torch.nn.functional.binary_cross_entropy
                            apply = F.binary_cross_entropy_with_logits if logits else F.binary_cross_entropy
                            expected, actual = reference(x, target, **options), apply(a, t, **actual_options)
                            self.same(actual, expected)
                            (actual.square().sum()).backward()
                            (expected.square().sum()).backward()
                            self.same(a.grad, x.grad)
                            self.same(t.grad, target.grad)

    def test_probability_boundaries_and_tiny_values(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([0., 1., 0., 1., 1e-30, .4], dtype=dtype, requires_grad=True)
            target = torch.tensor([0., 1., 1., 0., .2, .7], dtype=dtype, requires_grad=True)
            a, t = native(x, True), native(target, True)
            expected = torch.nn.functional.binary_cross_entropy(x, target, reduction='none')
            actual = F.binary_cross_entropy(a, t, reduction='none')
            self.same(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
            self.same(a.grad, x.grad)
            self.same(t.grad, target.grad)

    def test_nonfinite_logits_and_target_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for positive in (None, .5, 1., 2.):
                x = torch.tensor([-float('inf'), -1000., 0., 1000., float('inf')], dtype=dtype, requires_grad=True)
                target = torch.tensor([1., 0., .3, 1., 0.], dtype=dtype, requires_grad=True)
                p = None if positive is None else torch.tensor(positive, dtype=dtype)
                a, t = native(x, True), native(target, True)
                expected = torch.nn.functional.binary_cross_entropy_with_logits(x, target, pos_weight=p, reduction='none')
                actual = F.binary_cross_entropy_with_logits(a, t, pos_weight=None if p is None else native(p), reduction='none')
                self.same(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                self.same(a.grad, x.grad)
                self.same(t.grad, target.grad)

    def test_empty_reductions_keep_empty_gradients(self):
        for dtype in ('float32', 'float64'):
            for reduction in ('none', 'sum', 'mean'):
                x = fr.zeros(0, 3, dtype=dtype, requires_grad=True)
                target = fr.zeros(0, 3, dtype=dtype)
                expected_x = torch.zeros(0, 3, dtype=getattr(torch, dtype), requires_grad=True)
                expected = torch.nn.functional.binary_cross_entropy_with_logits(expected_x, torch.zeros_like(expected_x), reduction=reduction)
                actual = F.binary_cross_entropy_with_logits(x, target, reduction=reduction)
                self.same(actual, expected)
                actual.sum().backward()
                self.assertEqual(x.grad.shape, [0, 3])

    def test_weighted_modules_register_state_and_support_unreduced_outputs(self):
        weight, positive = torch.tensor([.7, 1.3]), torch.tensor([.4, 2.])
        for logits in (False, True):
            x = torch.tensor([[.2, .8], [.6, .4]])
            target = torch.tensor([[0., 1.], [1., 0.]])
            if logits:
                actual = fr.nn.BCEWithLogitsLoss(native(weight), reduction='none', pos_weight=native(positive))
                expected = torch.nn.BCEWithLogitsLoss(weight, reduction='none', pos_weight=positive)
                self.assertEqual([name for name, _ in actual.named_buffers()], ['weight', 'pos_weight'])
            else:
                actual, expected = fr.nn.BCELoss(native(weight), reduction='none'), torch.nn.BCELoss(weight, reduction='none')
                self.assertEqual([name for name, _ in actual.named_buffers()], ['weight'])
            self.same(actual(native(x), native(target)), expected(x, target))
        self.assertEqual(list(fr.nn.BCEWithLogitsLoss().named_buffers()), [])

    def test_invalid_probabilities_shapes_and_differentiable_weights_reject(self):
        target = fr.tensor([0., 1.])
        for values in ([-.1, .2], [1.1, .2], [float('nan'), .2]):
            with self.assertRaises(ValueError):
                F.binary_cross_entropy(fr.tensor(values), target)
        with self.assertRaises(ValueError):
            F.binary_cross_entropy(fr.ones(1, 2), target)
        for keyword in ('weight', 'pos_weight'):
            with self.assertRaises(RuntimeError):
                F.binary_cross_entropy_with_logits(fr.ones(2), target, **{keyword: fr.ones(2, requires_grad=True)})
        with self.assertRaises(ValueError):
            F.binary_cross_entropy_with_logits(fr.ones(2), target, weight=fr.ones(2, 2))
        expected = torch.nn.functional.binary_cross_entropy(torch.ones(2), torch.tensor([0., 1.]),
                                                            weight=torch.ones(2, dtype=torch.float64))
        self.same(F.binary_cross_entropy(fr.ones(2), target, weight=fr.ones(2, dtype='float64')), expected)
        with self.assertRaises(TypeError):
            F.binary_cross_entropy_with_logits(fr.ones(2), target, pos_weight=fr.ones(2, dtype='bool'))
