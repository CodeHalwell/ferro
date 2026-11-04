"""Concatenation saves metadata; surrounding nonlinear operations retain guards."""
import unittest

import torch
import ferro as fr
from ferro.autograd import grad, vjp


def native(value, requires_grad=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(list(value.shape)).requires_grad_(requires_grad)


def close(test, actual, expected):
    test.assertEqual(actual.shape, list(expected.shape))
    test.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
    value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(value, expected.detach(), atol=0, rtol=0)


class CatSavedValueTests(unittest.TestCase):
    def test_linear_cat_permits_input_and_output_mutation_in_both_backward_modes(self):
        for dtype in (torch.float32, torch.float64):
            for functional in (False, True):
                for modified in ('input', 'output'):
                    ta = torch.arange(6, dtype=dtype).reshape(2, 3).requires_grad_(); a = native(ta, True)
                    expected = torch.cat([ta.T, ta.T], dim=1); actual = fr.cat([a.T, a.T], dim=1)
                    seed = torch.arange(1, 13, dtype=dtype).reshape(3, 4)
                    value, wanted = (a, ta) if modified == 'input' else (actual, expected)
                    value.detach().copy_(fr.full(value.shape, 9., dtype=value.dtype)); wanted.detach().fill_(9.)
                    if functional:
                        result = vjp(actual, [a], native(seed))[0]
                        reference = torch.autograd.grad(expected, ta, seed)[0]
                        self.assertIsNone(a.grad); self.assertIsNone(ta.grad)
                    else:
                        actual.backward_with(native(seed)); expected.backward(seed)
                        result, reference = a.grad, ta.grad
                    close(self, result, reference)

    def test_nonlinear_producers_and_consumers_keep_their_saved_value_guards(self):
        for dtype in (torch.float32, torch.float64):
            for producer in (False, True):
                for source in (False, True):
                    for functional in (False, True):
                        ta = torch.tensor([.5, -2.], dtype=dtype, requires_grad=True); a = native(ta, True)
                        if producer:
                            middle, wanted = a.square(), ta.square()
                            actual, expected = fr.cat([middle, middle]).sum(), torch.cat([wanted, wanted]).sum()
                        else:
                            middle, wanted = fr.cat([a, a]), torch.cat([ta, ta])
                            actual, expected = middle.square().sum(), wanted.square().sum()
                        value, reference = (a, ta) if source else (middle, wanted)
                        value.detach().copy_(fr.full(value.shape, 9., dtype=value.dtype)); reference.detach().fill_(9.)
                        if producer == source:
                            with self.assertRaises((ValueError, RuntimeError)):
                                grad(actual, [a]) if functional else actual.backward()
                            with self.assertRaises(RuntimeError):
                                torch.autograd.grad(expected, ta) if functional else expected.backward()
                            self.assertIsNone(a.grad)
                        elif functional:
                            close(self, grad(actual, [a])[0], torch.autograd.grad(expected, ta)[0])
                            self.assertIsNone(a.grad)
                        else:
                            actual.backward(); expected.backward()
                            close(self, a.grad, ta.grad)

    def test_higher_derivatives_and_live_cotangent_survive_input_mutation(self):
        ta = torch.tensor([.5, -2.], requires_grad=True); a = native(ta, True)
        actual, expected = fr.cat([a, a]).square().sum(), torch.cat([ta, ta]).square().sum()
        a.detach().copy_(fr.full(a.shape, 9.)); ta.detach().fill_(9.)
        first = grad(actual, [a], create_graph=True)[0]
        reference = torch.autograd.grad(expected, ta, create_graph=True)[0]
        close(self, first, reference)
        close(self, grad(first.sum(), [a])[0], torch.autograd.grad(reference.sum(), ta)[0])
        ta = torch.tensor([.5, -2.], requires_grad=True); a = native(ta, True)
        tb = torch.tensor([3., 4.], requires_grad=True); b = native(tb, True)
        seed = torch.tensor([1., 2., 3., 4.], requires_grad=True); v = native(seed, True)
        actual, expected = fr.cat([a, b]), torch.cat([ta, tb])
        a.detach().copy_(fr.full(a.shape, 9.)); ta.detach().fill_(9.)
        first = vjp(actual, [a], v, create_graph=True)[0]
        reference = torch.autograd.grad(expected, ta, seed, create_graph=True)[0]
        self.assertTrue(first.requires_grad)
        close(self, first, reference)
        close(self, grad(first.sum(), [v])[0], torch.autograd.grad(reference.sum(), seed)[0])
        self.assertIsNone(a.grad); self.assertIsNone(b.grad); self.assertIsNone(v.grad)

    def test_stack_views_and_empty_cat_retain_correct_gradient_edges(self):
        for dtype in (torch.float32, torch.float64):
            ta = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, requires_grad=True); a = native(ta, True)
            actual, expected = fr.stack([a.T, a.T], dim=1), torch.stack([ta.T, ta.T], dim=1)
            a.detach().copy_(fr.full(a.shape, 9., dtype=a.dtype)); ta.detach().fill_(9.)
            actual.sum().backward(); expected.sum().backward()
            close(self, a.grad, ta.grad)
            ta = torch.empty(0, 2, dtype=dtype, requires_grad=True); a = native(ta, True)
            actual, expected = fr.cat([a, a], dim=0), torch.cat([ta, ta], dim=0)
            self.assertTrue(actual.requires_grad)
            close(self, grad(actual.sum(), [a])[0], torch.autograd.grad(expected.sum(), ta)[0])
            self.assertIsNone(a.grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
