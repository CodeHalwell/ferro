"""Absent forward directions materialize ordinary zeros in the output layout."""
import unittest
import torch
import ferro as fr


class AbsentForwardLayoutTests(unittest.TestCase):
    def compare(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertFalse(actual._is_structural_zero())
        torch.testing.assert_close(torch.tensor(actual.reshape([-1]).tolist(), dtype=expected.dtype), expected.reshape(-1), rtol=0, atol=0)

    def test_twelve_captured_layouts_match_actual_torch(self):
        for input_shape in ((2,), ()):
            for kind in ('transpose', 'gap_transpose', 'empty'):
                def make(lib):
                    value = lib.ones(2, 3).T
                    if kind == 'gap_transpose':
                        value = lib.ones(2, 6)[:, ::2].T
                    if kind == 'empty':
                        value = value[:0]
                    return value
                x, tx = fr.ones(input_shape), torch.ones(input_shape)
                captured, reference = make(fr), make(torch)
                for transform in ('jvp', 'jacfwd'):
                    with self.subTest(input_shape=input_shape, kind=kind, transform=transform):
                        if transform == 'jvp':
                            actual = fr.func.jvp(lambda unused: captured, (x,), (fr.ones(input_shape),))[1]
                            expected = torch.func.jvp(lambda unused: reference, (tx,), (torch.ones(input_shape),))[1]
                        else:
                            actual = fr.func.jacfwd(lambda unused: captured)(x)
                            expected = torch.func.jacfwd(lambda unused: reference)(tx)
                        self.compare(actual, expected)
                        self.assertIsNone(x.grad)

    def test_scalar_basis_squeeze_keeps_empty_layout_across_argument_blocks(self):
        for other_shape in ((), (1,), (2,)):
            for scalar_first in (False, True):
                with self.subTest(other_shape=other_shape, scalar_first=scalar_first):
                    shapes = [(), other_shape] if scalar_first else [other_shape, ()]
                    actual_args = tuple(fr.ones(shape) for shape in shapes)
                    expected_args = tuple(torch.ones(shape) for shape in shapes)
                    captured = fr.ones(2, 3).T[:0]
                    reference = torch.ones(2, 3).T[:0]
                    actual = fr.func.jacfwd(lambda a, b: (captured, captured.T), argnums=(0, 1))(*actual_args)
                    expected = torch.func.jacfwd(lambda a, b: (reference, reference.T), argnums=(0, 1))(*expected_args)
                    for actual_row, expected_row in zip(actual, expected):
                        for actual_block, expected_block in zip(actual_row, expected_row):
                            self.compare(actual_block, expected_block)
                            self.assertFalse(actual_block._shares_storage(captured))

    def test_wide_live_capture_stays_unused_and_strict_rejects(self):
        x, tx = fr.ones(2).requires_grad_(), torch.ones(2, requires_grad=True)
        captured = fr.ones(2, 3, dtype='float64').requires_grad_().T
        reference = torch.ones(2, 3, dtype=torch.float64, requires_grad=True).T
        for enabled in (False, True):
            with fr.enable_grad() if enabled else fr.no_grad():
                result = fr.func.jvp(lambda unused: captured, (x,), (fr.ones(2),))
            with torch.enable_grad() if enabled else torch.no_grad():
                expected = torch.func.jvp(lambda unused: reference, (tx,), (torch.ones(2),))
            self.compare(result[1], expected[1])
            self.assertFalse(result[1]._shares_storage(captured))
        with self.assertRaises(RuntimeError):
            fr.func.jvp(lambda unused: captured, (x,), (fr.ones(2),), strict=True)
        with self.assertRaises(RuntimeError):
            torch.func.jvp(lambda unused: reference, (tx,), (torch.ones(2),), strict=True)
        self.assertIsNone(x.grad)


if __name__ == '__main__':
    unittest.main()
