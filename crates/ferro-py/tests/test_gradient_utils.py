"""Gradient clipping values, slots, aliases and updates against actual Torch."""
import math
import unittest

import torch
import ferro as fr


def native(value):
    data, shape = value.detach().reshape(-1).tolist(), list(value.shape)
    return fr.Tensor.from_f64(data, shape) if value.dtype == torch.float64 else fr.Tensor(data, shape)


class GradientUtils(unittest.TestCase):
    def close(self, value, expected):
        self.assertEqual(value.shape, list(expected.shape))
        actual = torch.tensor(value.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(actual, expected.detach(), atol=2e-6, rtol=7e-6, equal_nan=True)

    def parameters(self, dtype=torch.float32, duplicate=False):
        references = [torch.ones(shape, dtype=dtype, requires_grad=True) for shape in ((2, 3), (2,), ())]
        parameters = [native(value).requires_grad_(True) for value in references]
        gradients = [torch.tensor([[.7, -.9, 2.1], [-.2, 1.3, .4]], dtype=dtype),
                     torch.tensor([-.3, .8], dtype=dtype)]
        for left, right, gradient in zip(parameters, references, gradients):
            (left * native(gradient)).sum().backward()
            (right * gradient).sum().backward()
        if duplicate:
            parameters.append(parameters[0])
            references.append(references[0])
        return parameters, references

    def test_norm_orders_mixed_slots_and_original_norm(self):
        for dtype in (torch.float32, torch.float64):
            for order in (0., .5, 1., 2., 3., -1., -2., math.inf, -math.inf):
                for duplicate in (False, True):
                    with self.subTest(dtype=dtype, order=order, duplicate=duplicate):
                        parameters, references = self.parameters(dtype, duplicate)
                        result = fr.nn.utils.clip_grad_norm_(iter(parameters), .7, order)
                        expected = torch.nn.utils.clip_grad_norm_(references, .7, order, foreach=False)
                        self.close(result, expected)
                        self.assertFalse(result.requires_grad)
                        for left, right in zip(parameters, references):
                            if right.grad is None:
                                self.assertIsNone(left.grad)
                            else:
                                self.close(left.grad, right.grad)
                            self.close(left, right)

    def test_value_clipping_and_tensor_parameter_inputs(self):
        for dtype in (torch.float32, torch.float64):
            for limit in (.4, 0., -.4, math.inf, math.nan):
                parameters, references = self.parameters(dtype)
                fr.nn.utils.clip_grad_value_(parameters, limit)
                torch.nn.utils.clip_grad_value_(references, limit, foreach=False)
                for left, right in zip(parameters[:2], references[:2]):
                    self.close(left.grad, right.grad)
        parameter = fr.nn.Parameter(fr.ones(2))
        parameter.tensor().sum().backward()
        fr.nn.utils.clip_grad_value_(parameter, .2)
        self.assertEqual(parameter.grad.tolist(), [float(torch.tensor(.2))] * 2)

    def test_get_total_norm_precomputed_clipping_and_no_grad(self):
        parameters, references = self.parameters()
        gradients = [value.grad for value in parameters if value.grad is not None]
        ref_gradients = [value.grad for value in references if value.grad is not None]
        total = fr.nn.utils.get_total_norm(gradients)
        expected = torch.nn.utils.get_total_norm(ref_gradients, foreach=False)
        self.close(total, expected)
        self.assertFalse(total.requires_grad)
        fr.nn.utils.clip_grads_with_norm_(parameters, .3, total)
        torch.nn.utils.clip_grads_with_norm_(references, .3, expected, foreach=False)
        for left, right in zip(parameters[:2], references[:2]):
            self.close(left.grad, right.grad)
        leaf = fr.tensor([2., 3.], requires_grad=True)
        self.assertFalse(fr.nn.utils.get_total_norm([leaf]).requires_grad)
        self.assertIsNone(leaf.grad)

    def test_nonfinite_error_is_atomic_and_nonerror_matches_torch(self):
        for nonfinite in (math.inf, math.nan):
            for reject in (False, True):
                parameter = fr.tensor([1., 2.], requires_grad=True)
                reference = torch.tensor([1., 2.], requires_grad=True)
                coefficients = torch.tensor([nonfinite, .3])
                (parameter * native(coefficients)).sum().backward()
                (reference * coefficients).sum().backward()
                if reject:
                    before = parameter.grad.tolist()
                    with self.assertRaises(RuntimeError):
                        fr.nn.utils.clip_grad_norm_(parameter, 1., error_if_nonfinite=True)
                    self.close(parameter.grad, torch.tensor(before))
                else:
                    norm = fr.nn.utils.clip_grad_norm_(parameter, 1.)
                    expected = torch.nn.utils.clip_grad_norm_([reference], 1., foreach=False)
                    self.close(norm, expected)
                    self.close(parameter.grad, reference.grad)

    def test_negative_value_limit_preserves_nan_gradients(self):
        for dtype in (torch.float32, torch.float64):
            reference = torch.ones(3, dtype=dtype, requires_grad=True)
            parameter = native(reference).requires_grad_(True)
            weights = torch.tensor([math.nan, math.inf, 1.], dtype=dtype)
            (parameter * native(weights)).sum().backward()
            (reference * weights).sum().backward()
            fr.nn.utils.clip_grad_value_(parameter, -.4)
            torch.nn.utils.clip_grad_value_([reference], -.4, foreach=False)
            self.close(parameter.grad, reference.grad)

    def test_empty_gradients_and_rejections_before_mutation(self):
        self.assertEqual(fr.nn.utils.clip_grad_norm_([], 1.).item(), 0.)
        empty = fr.zeros(0, requires_grad=True)
        empty.sum().backward()
        self.assertEqual(fr.nn.utils.clip_grad_norm_(empty, 1.).item(), 0.)
        self.assertEqual(empty.grad.shape, [0])
        parameter = fr.tensor([1., 2.], requires_grad=True)
        parameter.sum().backward()
        before = parameter.grad.tolist()
        for action in (lambda: fr.nn.utils.clip_grad_norm_([parameter, object()], 1.),
                       lambda: fr.nn.utils.clip_grad_norm_(parameter, 1., foreach=True),
                       lambda: fr.nn.utils.clip_grad_value_(parameter, 'bad'),
                       lambda: fr.nn.utils.clip_grads_with_norm_(parameter, 1., fr.ones(2)),
                       lambda: fr.capture(lambda: fr.nn.utils.clip_grad_norm_(parameter, 1.))):
            with self.assertRaises((TypeError, ValueError, NotImplementedError)):
                action()
            self.assertEqual(parameter.grad.tolist(), before)

    def test_clipped_native_optimizer_trajectory(self):
        parameter = fr.nn.Parameter(fr.tensor([1., -2., .7]))
        reference = torch.tensor([1., -2., .7], requires_grad=True)
        optimizer = fr.optim.SGD([parameter], lr=.2, momentum=.7)
        oracle = torch.optim.SGD([reference], lr=.2, momentum=.7)
        for _ in range(8):
            parameter.tensor().square().sum().backward()
            reference.square().sum().backward()
            fr.nn.utils.clip_grad_norm_([parameter], .4)
            torch.nn.utils.clip_grad_norm_([reference], .4, foreach=False)
            optimizer.step()
            optimizer.zero_grad()
            oracle.step()
            oracle.zero_grad()
            self.close(parameter.tensor(), reference)


if __name__ == '__main__':
    unittest.main()
