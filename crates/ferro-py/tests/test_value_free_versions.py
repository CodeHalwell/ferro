"""Saved-value checks follow actual adjoint dependencies, including live seeds."""
import unittest

import ferro as fr
import torch


CASES = ('reshape', 'transpose', 'permute', 'unsqueeze', 'squeeze', 'expand',
         'slice', 'sum', 'mean', 'sum_dim', 'mean_dim', 'materialize')


def transform(name, x):
    if name == 'reshape': return x.reshape(6)
    if name == 'transpose': return x.transpose(1, 2)
    if name == 'permute': return x.permute([2, 0, 1])
    if name == 'unsqueeze': return x.unsqueeze(1)
    if name == 'squeeze': return x.squeeze(0)
    if name == 'expand': return x.expand([4, 2, 3])
    if name == 'slice': return x[:, :, 1:]
    if name == 'sum': return x.sum()
    if name == 'mean': return x.mean()
    if name == 'sum_dim': return x.sum(2)
    if name == 'mean_dim': return x.mean(2)
    return x.transpose(1, 2).reshape(6)


class ValueFreeVersions(unittest.TestCase):
    def leaves(self, dtype):
        values = [[[1., 2., 3.], [4., 5., 6.]]]
        return (fr.tensor(values, dtype=dtype, requires_grad=True),
                torch.tensor(values, dtype=getattr(torch, dtype), requires_grad=True))

    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype), expected.detach(), atol=1e-6, rtol=1e-6)

    def change(self, x, reference, value):
        with fr.no_grad(), torch.no_grad():
            x.copy_(fr.full(x.shape, value, dtype=x.dtype))
            reference.fill_(value)

    def test_view_and_reduction_adjoints_allow_base_mutation(self):
        for dtype in ('float32', 'float64'):
            for name in CASES:
                with self.subTest(dtype=dtype, operation=name):
                    x, reference = self.leaves(dtype)
                    loss, expected_loss = transform(name, x).sum(), transform(name, reference).sum()
                    self.change(x, reference, 9.)
                    actual = fr.autograd.grad(loss, [x])[0]
                    expected = torch.autograd.grad(expected_loss, reference, retain_graph=True)[0]
                    self.same(actual, expected)
                    self.assertIsNone(x.grad)
                    loss.backward(); expected_loss.backward()
                    self.same(x.grad, reference.grad)

    def test_live_cotangent_higher_derivatives_need_no_original_input_values(self):
        for name in CASES:
            with self.subTest(operation=name):
                x, reference = self.leaves('float32')
                output, expected_output = transform(name, x).sum(), transform(name, reference).sum()
                seed = fr.tensor(0.25, requires_grad=True)
                expected_seed = torch.tensor(0.25, requires_grad=True)
                self.change(x, reference, 9.)
                gradient = fr.autograd.vjp(output, [x], seed, create_graph=True)[0]
                expected = torch.autograd.grad(expected_output, reference, expected_seed, create_graph=True)[0]
                self.same(gradient, expected)
                self.change(x, reference, 11.)
                second = fr.autograd.grad(gradient.sum(), [seed, x], allow_unused=True)
                expected_second = torch.autograd.grad(expected.sum(), [expected_seed, reference], allow_unused=True)
                self.same(second[0], expected_second[0])
                self.assertIsNone(expected_second[1])
                self.assertIsNone(second[1])
                self.assertIsNone(x.grad)
                self.assertIsNone(seed.grad)

    def test_saved_nonlinear_consumers_and_outputs_still_reject(self):
        for dtype in ('float32', 'float64'):
            for kind in ('consumer', 'parent', 'output'):
                with self.subTest(dtype=dtype, kind=kind):
                    x, reference = self.leaves(dtype)
                    if kind == 'consumer':
                        y, expected_y = x.transpose(1, 2), reference.transpose(1, 2)
                        loss, expected_loss = y.square().sum(), expected_y.square().sum()
                        self.change(x, reference, 9.)
                    elif kind == 'parent':
                        loss, expected_loss = x.square().reshape(6).sum(), reference.square().reshape(6).sum()
                        self.change(x, reference, 9.)
                    else:
                        y, expected_y = x.exp(), reference.exp()
                        y.detach().copy_(fr.zeros(y.shape, dtype=dtype)); expected_y.detach().zero_()
                        loss, expected_loss = y.sum(), expected_y.sum()
                    with self.assertRaises(ValueError): fr.autograd.grad(loss, [x])
                    with self.assertRaises(RuntimeError): torch.autograd.grad(expected_loss, reference)
                    self.assertIsNone(x.grad)

    def test_materialized_reshape_keeps_its_saved_values_after_base_changes(self):
        for dtype in ('float32', 'float64'):
            x, reference = self.leaves(dtype)
            copied = x.transpose(1, 2).reshape(6)
            expected_copy = reference.transpose(1, 2).reshape(6)
            self.assertFalse(copied._shares_storage(x))
            loss, expected_loss = copied.square().sum(), expected_copy.square().sum()
            self.change(x, reference, 9.)
            self.same(fr.autograd.grad(loss, [x])[0], torch.autograd.grad(expected_loss, reference, retain_graph=True)[0])
            copied.detach().copy_(fr.zeros(copied.shape, dtype=dtype)); expected_copy.detach().zero_()
            with self.assertRaises(ValueError): fr.autograd.grad(loss, [x])
            with self.assertRaises(RuntimeError): torch.autograd.grad(expected_loss, reference)


if __name__ == '__main__':
    unittest.main()
