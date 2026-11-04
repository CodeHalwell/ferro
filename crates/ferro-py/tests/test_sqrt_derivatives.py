"""Square-root denominator staging and exact saved-output dependency semantics."""
import unittest
import torch
import ferro as fr


def native(value, grad=False):
    result = fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


def same(test, actual, expected, exact=True):
    test.assertEqual(actual.shape, list(expected.shape))
    test.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
    result = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(result, expected.detach(), rtol=0 if exact else 3e-6, atol=0 if exact else 1e-7, equal_nan=True)
    zeros = expected.detach() == 0
    test.assertEqual(torch.signbit(result[zeros]).tolist(), torch.signbit(expected.detach()[zeros]).tolist())


def same_saved_output_gradient(test, actual, expected, saved_output, transpose=False):
    dtype = expected.dtype
    saved = torch.tensor(saved_output.tolist(), dtype=dtype).reshape(saved_output.shape)
    own_reference = torch.ones_like(saved) / (2 * saved)
    if transpose:
        own_reference = own_reference.T
    same(test, actual, own_reference)
    if dtype == torch.float64:
        # Torch's AVX512 sqrt may round one ULP below Rust's correctly-rounded
        # scalar sqrt. The denominator staging above remains an exact check.
        result = torch.tensor(actual.tolist(), dtype=dtype).reshape(expected.shape)
        torch.testing.assert_close(result, expected.detach(), rtol=2 * torch.finfo(dtype).eps, atol=0)
    else:
        same(test, actual, expected)


class SqrtDerivativeTests(unittest.TestCase):
    def test_subnormal_seeds_and_extreme_values_match_exact_denominator_order(self):
        for dtype in (torch.float32, torch.float64):
            tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)).item()
            values = [tiny, 1., torch.finfo(dtype).max, 0., -0., float('inf'), float('nan')]
            for seed_value in (tiny, 1., torch.finfo(dtype).max):
                for graph in ((False, True) if dtype == torch.float32 else (False,)):
                    x = torch.tensor(values, dtype=dtype, requires_grad=True)
                    seed = torch.full(x.shape, seed_value, dtype=dtype, requires_grad=True)
                    a, v = native(x, True), native(seed, True)
                    actual = fr.autograd.vjp(fr.sqrt(a), [a], v, create_graph=graph)[0]
                    expected = torch.autograd.grad(torch.sqrt(x), [x], seed, create_graph=graph)[0]
                    same(self, actual, expected)
                    self.assertEqual(actual.requires_grad, expected.requires_grad)
                    self.assertIsNone(a.grad)
                    self.assertIsNone(v.grad)

    def test_input_edits_preserve_original_output_higher_derivatives_and_guard_writes(self):
        for dtype in (torch.float32, torch.float64):
            for graph in ((False, True) if dtype == torch.float32 else (False,)):
                x = torch.tensor([.5, 1.5], dtype=dtype, requires_grad=True)
                a = native(x, True)
                expected_output, output = x.sqrt(), fr.sqrt(a)
                with torch.no_grad(), fr.no_grad():
                    x.fill_(8.)
                    a.fill_(8.)
                expected = torch.autograd.grad(expected_output.sum(), [x], create_graph=graph, retain_graph=True)[0]
                actual = fr.autograd.grad(output.sum(), [a], create_graph=graph)[0]
                same_saved_output_gradient(self, actual, expected, output)
                if graph:
                    same(self, fr.autograd.grad(actual.sum(), [a])[0], torch.autograd.grad(expected.sum(), [x], retain_graph=True)[0], False)
                output.detach().fill_(1.)
                expected_output.detach().fill_(1.)
                with self.assertRaises(ValueError):
                    fr.autograd.grad(output.sum(), [a], create_graph=graph)
                with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
                    torch.autograd.grad(expected_output.sum(), [x], create_graph=graph)
                self.assertIsNone(a.grad)
        x = torch.tensor([.5, 1.5], requires_grad=True)
        v = torch.ones(2, requires_grad=True)
        a, seed = native(x, True), native(v, True)
        with torch.no_grad():
            _, expected_pullback = torch.func.vjp(lambda value: value.sqrt(), x)
        with fr.no_grad():
            _, pullback = fr.func.vjp(lambda value: value.sqrt(), a)
        actual, expected = pullback(seed)[0], expected_pullback(v)[0]
        same(self, actual, expected)
        self.assertEqual(actual.sum().grad_wrt([a])[0].tolist(), [0., 0.])
        self.assertIsNone(torch.autograd.grad(expected.sum(), [x], allow_unused=True)[0])

    def test_direct_jvp_preserves_denominator_order_direction_dtype_and_grad_mode(self):
        tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
        for dtype in (torch.float32, torch.float64):
            for enabled in (False, True):
                x = torch.tensor([tiny, .5, 1.5, float('inf')], requires_grad=True)
                v = torch.tensor([tiny, 1., 2., 1.], dtype=dtype, requires_grad=True)
                with torch.set_grad_enabled(enabled):
                    expected = torch.func.jvp(torch.sqrt, (x,), (v,))
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = fr.func.jvp(fr.sqrt, (native(x, True),), (native(v, True),))
                for value, reference in zip(actual, expected):
                    same(self, value, reference)
                    self.assertEqual(value.requires_grad, reference.requires_grad)

    def test_strides_empty_offsets_wrapper_and_capture_replay(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.arange(1., 13., dtype=dtype).reshape(3, 4).requires_grad_(True)
            a = native(x, True)
            expected = x.T.sqrt().sum()
            saved_output = fr.sqrt(a.T)
            actual = saved_output.sum()
            same_saved_output_gradient(self, fr.autograd.grad(actual, [a])[0], torch.autograd.grad(expected, [x])[0], saved_output, transpose=True)
            empty = fr.zeros(0, 8, dtype=str(dtype).removeprefix('torch.'), requires_grad=True)
            fr.sqrt(empty[:, 4:]).sum().backward()
            self.assertEqual(empty.grad.shape, [0, 8])
        with self.assertRaises(TypeError):
            fr.sqrt([1.])
        with self.assertRaises(NotImplementedError):
            fr.sqrt(fr.ones(2), out=fr.zeros(2))
        value = fr.tensor([1., 4.])
        captured = fr.capture(lambda: fr.sqrt(value)).compile_fused()
        value.copy_(fr.tensor([4., 9.]))
        same(self, captured.replay(), torch.tensor([2., 3.]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
