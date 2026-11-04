"""Actual-Torch coverage for Cat/Slice higher-order rules and frozen operands."""
import unittest
import torch
import ferro as fr
import ferro.func as ff
from ferro.autograd import functional as af


def native(value):
    make = fr.Tensor.from_f64 if value.dtype == torch.float64 else fr.Tensor
    return make(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(value.requires_grad)


def close(actual, expected):
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise AssertionError('derivative tuple structure differs')
        for a, b in zip(actual, expected):
            close(a, b)
        return
    torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape), expected.detach(), atol=2e-5, rtol=2e-4)
    if actual.shape != list(expected.shape) or actual.dtype != str(expected.dtype).removeprefix('torch.') or actual.requires_grad != expected.requires_grad:
        raise AssertionError('derivative shape/graph flag differs')


class HigherOrderLayouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_empty_slices_match_torch_offsets_strides_and_zero_gradients(self):
        for dtype in (torch.float32, torch.float64):
            tx = torch.arange(5, dtype=dtype).requires_grad_()
            x = native(tx)
            for start, end, step in ((3, 3, 1), (4, 2, 1), (99, 100, 1), (99, 0, 1),
                                     (-2, -2, 3), (-1, -3, 2), (-99, -98, 4), (2, 2, 3)):
                actual, expected = x[start:end:step], tx[start:end:step]
                close(actual, expected)
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.reshape([0, 1]).storage_offset(), expected.reshape(0, 1).storage_offset())
                actual.sigmoid().sum().backward()
                expected.sigmoid().sum().backward()
                close(x.grad, tx.grad)
            actual, expected = x[1::2][1:1], tx[1::2][1:1]
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.stride(), expected.stride())
            close(actual.clone(), expected.clone())

    def test_four_derivatives_through_shared_strided_slices_cat_and_stack(self):
        tx = torch.linspace(-0.5, 0.6, 10).reshape(5, 2).clone().requires_grad_()
        tw = torch.linspace(0.4, 0.9, 6).reshape(1, 6).clone().requires_grad_()
        x, w = native(tx), native(tw)
        def network(a, weight, lib):
            a = a.transpose(0, 1)
            pieces = [a[:, ::2], a[:, 1::2], a[:, 1:2]]
            z = lib.cat(pieces, 1) * weight
            both = lib.stack([z.tanh(), (z * 0.7).tanh()], 0)
            return (both * both).sum()
        loss, tloss = network(x, w, fr), network(tx, tw, torch)
        for order in range(1, 5):
            actual = fr.autograd.grad(loss, (x, w), create_graph=order < 4)
            expected = torch.autograd.grad(tloss, (tx, tw), create_graph=order < 4)
            with self.subTest(order=order):
                close(actual, expected)
            if order < 4:
                seeds = [torch.linspace(-0.3 + order * 0.02, 0.6, a.numel()).reshape(a.shape) for a in expected]
                loss = sum((a * native(seed)).sum() for a, seed in zip(actual, seeds))
                tloss = sum((a * seed).sum() for a, seed in zip(expected, seeds))
        self.assertIsNone(x.grad)
        self.assertIsNone(w.grad)

    def test_assembled_hessian_and_jacobian_can_be_differentiated_again(self):
        for name in ('jacobian', 'hessian'):
            tx = torch.tensor([0.2, -0.4], requires_grad=True)
            x = native(tx)
            function = (lambda a: a.tanh()) if name == 'jacobian' else (lambda a: (a.tanh() * a).sum())
            actual = getattr(af, name)(function, x, create_graph=True)
            expected = getattr(torch.autograd.functional, name)(function, tx, create_graph=True)
            close(actual, expected)
            for graph in (True, False):
                actual = fr.autograd.grad(actual.sum(), (x,), create_graph=graph)[0]
                expected = torch.autograd.grad(expected.sum(), tx, create_graph=graph)[0]
                close(actual, expected)

    def test_frozen_forward_cat_slice_pullback_keeps_capture_and_live_seed_only(self):
        tx = torch.tensor([0.2, -0.4, 0.3, 0.7], requires_grad=True)
        tw = torch.tensor([0.6, 0.5, 0.7, 0.4], requires_grad=True)
        tv = torch.linspace(0.2, 0.6, 4).requires_grad_()
        x, w, v = native(tx), native(tw), native(tv)
        def fn(a, weight, lib):
            z = (a * weight).tanh()
            pieces = [z[::2], z[1::2]]
            return lib.cat(pieces)
        with fr.no_grad():
            _, closure = ff.vjp(lambda a: fn(a, w, fr), x)
        with torch.no_grad():
            _, reference = torch.func.vjp(lambda a: fn(a, tw, torch), tx)
        actual, expected = closure(v, create_graph=True)[0], reference(tv, create_graph=True)[0]
        close(actual, expected)
        got = fr.autograd.grad(actual.sum(), (x, w, v), allow_unused=True)
        wanted = torch.autograd.grad(expected.sum(), (tx, tw, tv), allow_unused=True)
        for a, b, original in zip(got, wanted, (tx, tw, tv)):
            if b is None: self.assertIsNone(a)
            else: close(a, b)


if __name__ == '__main__':
    unittest.main()
