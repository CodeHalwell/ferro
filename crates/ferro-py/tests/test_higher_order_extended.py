"""Higher derivatives, mixed partials and live cotangents against actual Torch."""
import unittest

import torch
import ferro as fr
from ferro.autograd import grad, vjp


def native(value):
    return fr.tensor(value.detach().tolist(), requires_grad=value.requires_grad)


class HigherOrderExtended(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(expected.shape),
                                   expected.detach(), atol=5e-5, rtol=7e-5)

    def test_log_power_silu_third_derivatives(self):
        for name in ('log', 'power', 'silu'):
            with self.subTest(name=name):
                reference = torch.tensor([.4, .8, 1.3], requires_grad=True)
                actual = native(reference)
                left = actual.log() if name == 'log' else actual.pow(3.4) if name == 'power' else actual.silu()
                right = reference.log() if name == 'log' else reference.pow(3.4) if name == 'power' else torch.nn.functional.silu(reference)
                for order in range(1, 4):
                    left, = grad(left.sum(), [actual], create_graph=order < 3)
                    right, = torch.autograd.grad(right.sum(), reference, create_graph=order < 3)
                    self.close(left, right)
                    self.assertIsNone(actual.grad)

    def test_piecewise_vjp_live_seed_and_zero_input_derivative(self):
        for name in ('relu', 'abs', 'clamp'):
            with self.subTest(name=name):
                reference = torch.tensor([-1.1, -.3, .2, 1.2], requires_grad=True)
                cotangent = torch.tensor([.2, -.4, .7, 1.3], requires_grad=True)
                actual, seed = native(reference), native(cotangent)
                if name == 'clamp':
                    output, expected = actual.clamp(-.7, .8), reference.clamp(-.7, .8)
                else:
                    output, expected = getattr(actual, name)(), getattr(reference, name)()
                derivative, = vjp(output, [actual], seed, create_graph=True)
                wanted, = torch.autograd.grad(expected, reference, cotangent, create_graph=True)
                self.close(derivative, wanted)
                second = grad(derivative.sum(), [actual, seed], allow_unused=True)
                reference_second = torch.autograd.grad(wanted.sum(), (reference, cotangent), allow_unused=True)
                for left, right, target in zip(second, reference_second, (reference, cotangent)):
                    if right is None:
                        self.assertIsNone(left)
                    else:
                        self.close(left, right)
                self.assertIsNone(actual.grad)
                self.assertIsNone(seed.grad)

    def test_bmm_softmax_mixed_partials_and_third_derivatives(self):
        generator = torch.Generator().manual_seed(915)
        references = [torch.randn(shape, generator=generator).requires_grad_()
                      for shape in ((2, 3, 4), (2, 4, 2), (2, 3, 2))]
        actual, weights, seed = [native(value) for value in references]
        left = actual.bmm(weights).softmax(-1)
        right = references[0].bmm(references[1]).softmax(-1)
        first = vjp(left, [actual, weights], seed, create_graph=True)
        expected_first = torch.autograd.grad(right, references[:2], references[2], create_graph=True)
        for value, expected in zip(first, expected_first):
            self.close(value, expected)
        penalty = (first[0] * first[0]).sum() + (first[1] * first[1]).sum()
        wanted = expected_first[0].square().sum() + expected_first[1].square().sum()
        second = grad(penalty, [actual, weights, seed], create_graph=True)
        expected_second = torch.autograd.grad(wanted, references, create_graph=True)
        for value, expected in zip(second, expected_second):
            self.close(value, expected)
        third = grad(second[0].sum(), [weights, seed])
        expected_third = torch.autograd.grad(expected_second[0].sum(), references[1:])
        for value, expected in zip(third, expected_third):
            self.close(value, expected)
        self.assertTrue(all(value.grad is None for value in (actual, weights, seed)))

    def test_zero_and_unit_powers_at_zero_and_nonfinite_seed(self):
        for power in (0., 1.):
            actual = fr.tensor([0., -2., 3.], requires_grad=True)
            reference = torch.tensor([0., -2., 3.], requires_grad=True)
            seed = fr.tensor([float('inf'), 2., -1.], requires_grad=True)
            expected_seed = torch.tensor([float('inf'), 2., -1.], requires_grad=True)
            derivative, = vjp(actual.pow(power), [actual], seed, create_graph=True)
            expected, = torch.autograd.grad(reference.pow(power), reference, expected_seed, create_graph=True)
            self.close(derivative, expected)


if __name__ == '__main__':
    unittest.main()
