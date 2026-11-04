"""Mixed CPU binary losses against actual Torch, including scalar promotion."""
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    if value is None:
        return None
    result = fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).removeprefix('torch.'))
    return result.reshape(list(value.shape)).requires_grad_(grad)


def option(kind):
    if kind == 'none': return None
    if kind == 'f32': return torch.tensor([.3, 2., 1.7], dtype=torch.float32)
    if kind == 'f64': return torch.tensor([.3, 2., 1.7], dtype=torch.float64)
    if kind == 'f64scalar': return torch.tensor(1.7, dtype=torch.float64)
    if kind == 'i64': return torch.tensor([0, 2, 3], dtype=torch.int64)
    if kind == 'bool': return torch.tensor([False, True, True])
    raise AssertionError(kind)


class BinaryMixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def same(self, actual, expected, *, wide=False, exact=False):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        atol, rtol = (0., 0.) if exact else ((3e-12, 3e-12) if wide else (2e-6, 3e-6))
        torch.testing.assert_close(value, expected.detach(), atol=atol, rtol=rtol, equal_nan=True)

    def compare(self, x, target, weight=None, positive=None, reduction='none', logits=True, seed=None, transpose=False):
        x, target = x.detach().requires_grad_(), target.detach().requires_grad_()
        if transpose:
            a = native(x.transpose(0, 1)).transpose(0, 1).requires_grad_(True)
            t = native(target.transpose(0, 1)).transpose(0, 1).requires_grad_(True)
            self.assertFalse(a.is_contiguous()); self.assertFalse(t.is_contiguous())
        else:
            a, t = native(x, True), native(target, True)
        apply = F.binary_cross_entropy_with_logits if logits else F.binary_cross_entropy
        reference = torch.nn.functional.binary_cross_entropy_with_logits if logits else torch.nn.functional.binary_cross_entropy
        kwargs = dict(weight=weight, reduction=reduction)
        args = dict(weight=native(weight), reduction=reduction)
        if logits:
            kwargs['pos_weight'] = positive; args['pos_weight'] = native(positive)
        expected, actual = reference(x, target, **kwargs), apply(a, t, **args)
        wide = x.dtype == target.dtype == torch.float64
        self.same(actual, expected, wide=wide)
        if seed is None:
            seed = torch.randn(expected.shape, dtype=expected.dtype, generator=torch.Generator().manual_seed(1213))
        actual.backward_with(native(seed)); expected.backward(seed)
        self.same(a.grad, x.grad, wide=wide)
        self.same(t.grad, target.grad, wide=wide)
        return actual, a.grad, t.grad, expected, x.grad, target.grad

    def test_396_mixed_weight_target_positive_and_reduction_configurations(self):
        count = 0
        for logits in (False, True):
            for xd in (torch.float32, torch.float64):
                for td in (torch.float32, torch.float64):
                    if not logits and xd != td: continue
                    for wk in ('none', 'f32', 'f64', 'f64scalar', 'i64', 'bool'):
                        for pk in ('none', 'f32', 'f64', 'f64scalar', 'i64') if logits else ('none',):
                            for reduction in ('none', 'sum', 'mean'):
                                with self.subTest(logits=logits, input=xd, target=td, weight=wk, positive=pk, reduction=reduction):
                                    x = torch.tensor([[-2., 0., 1.3], [.2, -.7, 2.1]] if logits else [[.12, .5, .91], [.3, .7, .4]], dtype=xd)
                                    target = torch.tensor([[.2, .9, .6], [.7, .4, .1]], dtype=td)
                                    self.compare(x, target, option(wk), option(pk), reduction, logits)
                                    count += 1
        self.assertEqual(count, 396)

    def test_large_weights_preserve_scalar_vector_promotion_and_finite_products(self):
        for logits in (False, True):
            for scalar in (False, True):
                weight = torch.tensor(1e40 if scalar else [1e40], dtype=torch.float64)
                x = torch.tensor([-70.] if logits else [1e-30], dtype=torch.float32)
                target = torch.zeros_like(x)
                self.compare(x, target, weight, logits=logits, seed=torch.ones_like(x))
        for scalar in (False, True):
            positive = torch.tensor(1e40 if scalar else [1e40], dtype=torch.float64)
            self.compare(torch.tensor([1.]), torch.tensor([1e-30]), positive=positive, seed=torch.ones(1))
        self.compare(torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64),
                     positive=torch.tensor([-(1 << 63), (1 << 63)-1], dtype=torch.int64))

    def test_target_adjoint_retains_torch_intermediate_rounding(self):
        for positive in (None, torch.tensor([.3, 2.3], dtype=torch.float64)):
            for reduction in ('none', 'sum', 'mean'):
                x = torch.ones(2) if positive is None else torch.zeros(2)
                target = torch.tensor([.2, .7], dtype=torch.float64)
                shape = (2,) if reduction == 'none' else ()
                seed = torch.full(shape, .123456789123, dtype=torch.float64)
                actual, _, dt, expected, _, wanted = self.compare(x, target, positive=positive, reduction=reduction, seed=seed)
                self.same(dt, wanted, exact=True)
                self.assertEqual(actual.dtype, 'float64')

    def test_strided_scalar_empty_and_broadcast_metadata(self):
        for reduction in ('none', 'sum', 'mean'):
            self.compare(torch.tensor(.3, dtype=torch.float32), torch.tensor(.2, dtype=torch.float64),
                         torch.tensor(2., dtype=torch.float64), torch.tensor(1.7), reduction)
            x = torch.tensor([[-.7, .3], [1.2, 0.], [-.4, 2.]], dtype=torch.float64).T
            target = torch.tensor([[.2, .6], [.9, .7], [.4, .1]], dtype=torch.float32).T
            self.compare(x, target, torch.tensor([[.3], [1.7]], dtype=torch.float64), torch.tensor([1, 2, 3]), reduction, transpose=True)
            self.compare(torch.empty(0, 3), torch.empty(0, 3, dtype=torch.float64), torch.ones(3, dtype=torch.float64), reduction=reduction)
        a = fr.zeros(1 << 60, 0, requires_grad=True)
        target = fr.zeros(1 << 60, 0, dtype='float64', requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(a, target, weight=fr.tensor(2, dtype='int64'), reduction='sum')
        loss.backward()
        self.assertEqual(a.grad.shape, [1 << 60, 0]); self.assertEqual(target.grad.shape, [1 << 60, 0])

    def test_every_original_operand_version_is_checked_without_partial_gradients(self):
        from ferro.autograd import grad
        for selected in ('input', 'target', 'weight', 'pos_weight'):
            for functional in (False, True):
                a = fr.tensor([-.7, .3], requires_grad=True)
                target = fr.tensor([.2, .6], dtype='float64', requires_grad=True)
                weight = fr.tensor([.3, 1.7], dtype='float64')
                positive = fr.tensor([1, 2], dtype='int64')
                loss = F.binary_cross_entropy_with_logits(a, target, weight=weight, pos_weight=positive, reduction='sum')
                modified = dict(input=a, target=target, weight=weight, pos_weight=positive)[selected]
                with fr.no_grad(): modified.copy_(fr.ones(modified.shape, dtype=modified.dtype))
                with self.assertRaises((ValueError, RuntimeError)):
                    grad(loss, [a, target]) if functional else loss.backward()
                self.assertIsNone(a.grad); self.assertIsNone(target.grad)
        a = fr.tensor([.3], requires_grad=True); target = fr.tensor([.2], dtype='float64', requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(a, target, reduction='none')
        loss.detach().copy_(fr.ones(1, dtype='float64'))
        loss.sum().backward()
        self.assertIsNotNone(a.grad); self.assertIsNotNone(target.grad)

    def test_domains_nondifferentiable_weights_and_no_grad(self):
        a, target = fr.tensor([.3]), fr.tensor([.2])
        with self.assertRaises(ValueError): F.binary_cross_entropy(a, target.to(dtype='float64'))
        with self.assertRaises(TypeError): F.binary_cross_entropy_with_logits(a, target, pos_weight=fr.tensor([True], dtype='bool'))
        for keyword in ('weight', 'pos_weight'):
            value = fr.tensor([2.], dtype='float64', requires_grad=True)
            with self.assertRaises(RuntimeError): F.binary_cross_entropy_with_logits(a, target, **{keyword: value})
            with fr.no_grad(): result = F.binary_cross_entropy_with_logits(a, target, **{keyword: value})
            self.assertFalse(result.requires_grad)
        for weight in (fr.ones(2), fr.ones(1, 1)):
            with self.assertRaises(ValueError): F.binary_cross_entropy_with_logits(a, target, weight=weight)
        for value in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): F.binary_cross_entropy(fr.tensor([value]), target)
            with self.assertRaises(ValueError): F.binary_cross_entropy(a, fr.tensor([value]))
        with self.assertRaisesRegex(ValueError, 'capture'):
            fr.capture(lambda: F.binary_cross_entropy_with_logits(a, target, weight=fr.ones(1, dtype='float64')))
        from ferro.autograd import grad
        a = a.requires_grad_(True)
        with self.assertRaises(ValueError): grad(F.binary_cross_entropy_with_logits(a, target), [a], create_graph=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
