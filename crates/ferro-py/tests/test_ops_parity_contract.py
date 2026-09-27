"""The standalone torch parity gate must preserve integer value/type checks."""
import importlib.util
from pathlib import Path
import unittest

import ferro
import torch

spec = importlib.util.spec_from_file_location(
    'ops_vs_torch', Path(__file__).resolve().parents[3] / 'examples/ops_vs_torch.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class OpsParityContract(unittest.TestCase):
    def test_i64_values_are_exact_beyond_float_precision(self):
        values = [-(1 << 63), (1 << 53) + 1, (1 << 63) - 1]
        actual = ferro.Tensor.from_i64(values, [3])
        ops.check('exact i64', actual, torch.tensor(values, dtype=torch.int64))
        wrong = torch.tensor([values[0], values[1] - 1, values[2]], dtype=torch.int64)
        with self.assertRaises(AssertionError):
            ops.check('adjacent i64 mismatch', actual, wrong)

    def test_float_output_cannot_stand_in_for_indices(self):
        with self.assertRaises(AssertionError):
            ops.check('float is not an index', ferro.Tensor([1.0], [1]), torch.tensor([1]))

    def test_index_output_cannot_stand_in_for_floats(self):
        with self.assertRaises(AssertionError):
            ops.check('index is not float', ferro.Tensor.from_i64([1], [1]), torch.tensor([1.0]))

    def test_scalar_and_nested_indices(self):
        ops.check('scalar index', ferro.Tensor.from_i64([2], []), torch.tensor(2))
        ops.check('nested indices', ferro.Tensor.from_i64([0, 2], [2, 1]), torch.tensor([[0], [2]]))

    def test_float_tolerance_is_unchanged(self):
        actual = ferro.Tensor([1.0], [1])
        ops.check('float tolerance', actual, torch.tensor([1.000001]))
        with self.assertRaises(AssertionError):
            ops.check('float mismatch', actual, torch.tensor([1.01]))


if __name__ == '__main__':
    unittest.main()
