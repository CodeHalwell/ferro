"""Functional normalization numerics and fail-before-write buffer boundaries."""
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    dtype = str(value.dtype).removeprefix('torch.')
    return fr.tensor(value.detach().tolist(), dtype=dtype, requires_grad=grad)


class FunctionalBatchNorm(unittest.TestCase):
    def same(self, actual, expected):
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), atol=3e-6, rtol=4e-5, equal_nan=True)

    def test_tiny_positive_epsilon_constant_inputs_keep_finite_adjoint(self):
        for dtype, epsilon in ((torch.float32, 1e-40), (torch.float32, 1e-60), (torch.float64, 1e-100)):
            x = torch.ones(2, 2, dtype=dtype, requires_grad=True)
            weight = torch.tensor([.7, -.3], dtype=dtype, requires_grad=True)
            bias = torch.zeros(2, dtype=dtype, requires_grad=True)
            inputs = [native(value, True) for value in (x, weight, bias)]
            actual = F.batch_norm(inputs[0], None, None, inputs[1], inputs[2], True, eps=epsilon)
            expected = torch.nn.functional.batch_norm(x, None, None, weight, bias, True, eps=epsilon)
            self.same(actual, expected)
            seed = torch.tensor([[.3, -.6], [.1, .8]], dtype=dtype)
            (actual * native(seed)).sum().backward()
            (expected * seed).sum().backward()
            for actual_input, reference in zip(inputs, (x, weight, bias)):
                self.same(actual_input.grad, reference.grad)

    def test_tiny_epsilon_preserves_affine_bias(self):
        for dtype, epsilon in ((torch.float32, 1e-40), (torch.float64, 1e-100)):
            x = torch.ones(2, 2, dtype=dtype)
            weight = torch.tensor([.7, -.3], dtype=dtype)
            bias = torch.tensor([.2, .4], dtype=dtype)
            actual = F.batch_norm(native(x), None, None, native(weight), native(bias), True, eps=epsilon)
            self.same(actual, bias.expand_as(x))
            # Torch 2.13 CPU loses the bias through cancellation in its fused formula.
            upstream = torch.nn.functional.batch_norm(x, None, None, weight, bias, True, eps=epsilon)
            if torch.__version__.startswith('2.13.'):
                torch.testing.assert_close(upstream, torch.zeros_like(x))

    def test_running_buffer_layout_rejection_is_atomic(self):
        for dtype in ('float32', 'float64'):
            x = fr.tensor([[1., 3.], [2., 7.]], dtype=dtype)
            average = fr.zeros(2, dtype=dtype)
            owner = fr.ones(4, dtype=dtype)
            variance = owner.narrow(0, 0, 2)
            before = (average.tolist(), owner.tolist())
            with self.assertRaises(ValueError):
                F.batch_norm(x, average, variance, training=True)
            self.assertEqual((average.tolist(), owner.tolist()), before)
            wanted_mean = torch.zeros(2, dtype=getattr(torch, dtype))
            wanted_var = torch.ones(4, dtype=getattr(torch, dtype))
            torch.nn.functional.batch_norm(torch.tensor(x.tolist(), dtype=getattr(torch, dtype)),
                                           wanted_mean, wanted_var[:2], training=True)
            self.assertNotEqual(wanted_mean.tolist(), before[0])

    def test_aliased_running_statistics_reject_before_updates(self):
        for dtype in ('float32', 'float64'):
            state = fr.ones(2, dtype=dtype)
            other = state.reshape([2])
            self.assertIsNot(state, other)
            self.assertTrue(state._shares_storage(other))
            with self.assertRaisesRegex(NotImplementedError, 'independent storage'):
                F.batch_norm(fr.tensor([[1., 3.], [2., 7.]], dtype=dtype), state, other, training=True)
            self.assertEqual(state.tolist(), [1., 1.])
            wanted = torch.ones(2, dtype=getattr(torch, dtype))
            torch.nn.functional.batch_norm(torch.tensor([[1., 3.], [2., 7.]], dtype=getattr(torch, dtype)),
                                           wanted, wanted.view(2), training=True)
            self.same(fr.tensor([.995, 2.06], dtype=dtype), wanted)
