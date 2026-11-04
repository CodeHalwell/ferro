"""Native forward transforms with float32 primals and mixed floating outputs."""
import itertools
import unittest
from unittest.mock import patch

import torch
import ferro as fr


def native(value):
    data = value.detach().clone().reshape(-1).tolist()
    result = fr.Tensor.from_f64(data, list(value.shape)) if value.dtype == torch.float64 else fr.Tensor(data, list(value.shape))
    return result.requires_grad_(value.requires_grad)


def same(actual, expected):
    if actual.shape != list(expected.shape) or actual.dtype != str(expected.dtype).split('.')[-1]:
        raise AssertionError((actual.shape, actual.dtype, expected.shape, expected.dtype))
    torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape),
                               expected.detach().clone(), atol=1e-6, rtol=1e-5, equal_nan=True)
    if actual.requires_grad != expected.requires_grad:
        raise AssertionError(('requires_grad', actual.requires_grad, expected.requires_grad))


def tree_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        same(actual, expected)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError((actual.keys(), expected.keys()))
        for key in expected:
            tree_same(actual[key], expected[key])
    else:
        if len(actual) != len(expected):
            raise AssertionError((len(actual), len(expected)))
        for a, b in zip(actual, expected):
            tree_same(a, b)


def gradients(test, actual, expected, pairs):
    pairs = [(a, b) for a, b in pairs if b.requires_grad]
    if expected.requires_grad and pairs:
        got = fr.autograd.grad(actual.sum(), [a for a, _ in pairs], allow_unused=True)
        want = torch.autograd.grad(expected.sum(), [b for _, b in pairs], allow_unused=True, retain_graph=True)
        for a, b in zip(got, want):
            if b is None:
                test.assertIsNone(a)
            else:
                same(a, b)
    for a, b in pairs:
        test.assertIsNone(a.grad)
        test.assertIsNone(b.grad)


class ForwardOutputDtypeTests(unittest.TestCase):
    def test_jvp_output_dtype_flags_and_input_direction_gradients_48_cases(self):
        for enabled, xg, vg, dtype, kind in itertools.product(
                (False, True), (False, True), (False, True), (torch.float32, torch.float64), ('cast', 'square', 'linear')):
            with self.subTest(enabled=enabled, xg=xg, vg=vg, dtype=dtype, kind=kind):
                tx = torch.tensor([.2, -.4], requires_grad=xg)
                tv = torch.tensor([.3, .7], dtype=dtype, requires_grad=vg)
                x, v = native(tx), native(tv)
                def objective(a, target):
                    value = a if kind == 'cast' else a * a if kind == 'square' else a * 2
                    return value.to(dtype=target)
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = fr.func.jvp(lambda a: objective(a, 'float64'), (x,), (v,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected = torch.func.jvp(lambda a: objective(a, torch.float64), (tx,), (tv,))
                tree_same(actual, expected)
                if kind == 'cast':
                    self.assertFalse(actual[1]._shares_storage(v))
                    self.assertNotEqual(expected[1].data_ptr(), tv.data_ptr())
                for a, b in zip(actual, expected):
                    gradients(self, a, b, [(x, tx), (v, tv)])

    def test_mixed_output_tree_constants_auxiliary_and_strict_jvp(self):
        for enabled in (False, True):
            tx = torch.tensor([.2, -.4], requires_grad=True)
            tv = torch.tensor([.3, .7], dtype=torch.float64, requires_grad=True)
            x, v = native(tx), native(tv)
            calls = []
            def objective(library, a, dtype):
                calls.append(library.__name__)
                output = {'wide': (a * a).to(dtype=dtype), 'narrow': a * 2,
                          'constant': library.ones(2, dtype=dtype), 'empty': library.zeros([0, 2], dtype=dtype)}
                return output, {'wide': a.to(dtype=dtype), 'input': a}
            with fr.enable_grad() if enabled else fr.no_grad():
                actual = fr.func.jvp(lambda a: objective(fr, a, 'float64'), (x,), (v,), has_aux=True)
            with torch.enable_grad() if enabled else torch.no_grad():
                expected = torch.func.jvp(lambda a: objective(torch, a, torch.float64), (tx,), (tv,), has_aux=True)
            tree_same(actual, expected)
            self.assertEqual(calls, ['ferro', 'torch'])
            gradients(self, actual[1]['wide'], expected[1]['wide'], [(x, tx), (v, tv)])
        with self.assertRaises(RuntimeError):
            fr.func.jvp(lambda a: fr.ones(2, dtype='float64'), (x,), (v,), strict=True)
        with self.assertRaises(RuntimeError):
            torch.func.jvp(lambda a: torch.ones(2, dtype=torch.float64), (tx,), (tv,), strict=True)

    def test_jacfwd_mixed_output_blocks_flags_and_real_reverse_gradients(self):
        for enabled, live in itertools.product((False, True), repeat=2):
            tx = torch.tensor([.2, -.4], requires_grad=live)
            x = native(tx)
            calls = []
            def objective(library, a, dtype):
                calls.append(library.__name__)
                return {'wide': (a * a).to(dtype=dtype), 'cast': a.to(dtype=dtype),
                        'narrow': a * a, 'promoted': (a * a) * library.tensor([.5, .7], dtype=dtype),
                        'constant': library.ones(2, dtype=dtype)}
            with fr.enable_grad() if enabled else fr.no_grad():
                actual = fr.func.jacfwd(lambda a: objective(fr, a, 'float64'))(x)
            with torch.enable_grad() if enabled else torch.no_grad():
                expected = torch.func.jacfwd(lambda a: objective(torch, a, torch.float64))(tx)
            tree_same(actual, expected)
            for key in actual:
                gradients(self, actual[key], expected[key], [(x, tx)])
            self.assertEqual(calls, ['ferro', 'torch'])

    def test_empty_input_output_and_independent_float64_tangents(self):
        for size, enabled, live in itertools.product((0, 2), (False, True), (False, True)):
            tx = torch.tensor([.2, -.4][:size], requires_grad=live)
            tv = torch.ones_like(tx, dtype=torch.float64, requires_grad=True)
            x, v = native(tx), native(tv)
            def objective(library, a, dtype):
                return (a * a).to(dtype=dtype)[:0], library.ones(2, dtype=dtype), library.zeros([0, 2], dtype=dtype)
            with fr.enable_grad() if enabled else fr.no_grad():
                actual_jvp = fr.func.jvp(lambda a: objective(fr, a, 'float64'), (x,), (v,))
                actual_jac = fr.func.jacfwd(lambda a: objective(fr, a, 'float64'))(x)
            with torch.enable_grad() if enabled else torch.no_grad():
                expected_jvp = torch.func.jvp(lambda a: objective(torch, a, torch.float64), (tx,), (tv,))
                expected_jac = torch.func.jacfwd(lambda a: objective(torch, a, torch.float64))(tx)
            tree_same(actual_jvp, expected_jvp)
            tree_same(actual_jac, expected_jac)
            gradients(self, actual_jvp[1][0], expected_jvp[1][0], [(x, tx), (v, tv)])

    def test_float64_result_capacity_checked_before_unused_zero_materialization(self):
        huge = fr.ones(1, dtype='float64').expand([1 << 60])
        calls = []
        def objective(a):
            calls.append(True)
            return {'small': a * a, 'huge': huge}
        with patch('ferro.func.full', side_effect=AssertionError('must preflight before zero allocation')):
            with self.assertRaises(ValueError):
                fr.func.jacfwd(objective)(fr.tensor(.5))
        self.assertEqual(calls, [True])

    def test_reverse_transform_and_primal_dtype_boundaries_remain_explicit(self):
        x = fr.tensor([.2, -.4], requires_grad=True)
        wide = x.to(dtype='float64')
        for transform in (fr.func.grad, fr.func.jacrev, fr.func.hessian, fr.func.vjp):
            with self.subTest(transform=transform.__name__):
                objective = lambda a: (a * a).sum().to(dtype='float64')
                with self.assertRaises(ValueError):
                    transform(objective, x) if transform is fr.func.vjp else transform(objective)(x)
        with self.assertRaises(ValueError):
            fr.func.jvp(lambda a: a, (wide,), (fr.ones(2, dtype='float64'),))
        with self.assertRaises(ValueError):
            fr.func.jacfwd(lambda a: a)(wide)
        self.assertTrue(fr.is_grad_enabled())
        self.assertIsNone(x.grad)


if __name__ == '__main__':
    unittest.main()
