"""Public CPU model gates, including independent process restoration."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples'))
from training_restart import SEEDS, prove
import ferro as fr


class ArchitectureRestart(unittest.TestCase):
    def test_regression(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                prove('regression', seed)

    def test_classification(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                prove('classifier', seed)

    def test_cross_entropy_contract(self):
        import torch
        for values in ([2., -3., 0., 4., 5., -2.], [100., -100., 0., -80., 90., 10.]):
            x = fr.Tensor(values, [2, 3]).requires_grad_(True)
            target = fr.Tensor.from_i64([2, 0], [2])
            loss = fr.nn.functional.cross_entropy(x, target)
            ref = torch.tensor(values).reshape(2, 3).requires_grad_(True)
            expected = torch.nn.functional.cross_entropy(ref, torch.tensor([2, 0]))
            loss.backward()
            expected.backward()
            torch.testing.assert_close(torch.tensor(loss.item()), expected.detach(), atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(torch.tensor(x.grad.tolist()), ref.grad, atol=1e-5, rtol=1e-4)
        x = fr.Tensor([1., 2., 3., 4.], [2, 2])
        for target in (fr.Tensor([0., 1.], [2]), fr.Tensor.from_i64([0, 2], [2]),
                       fr.Tensor.from_i64([-1, 1], [2]), fr.Tensor.from_i64([0, 1], [2, 1]),
                       fr.Tensor.from_i64([0], [1]), fr.Tensor.from_i64([0, 2**40], [2])):
            with self.assertRaises(ValueError):
                fr.nn.functional.cross_entropy(x, target)
        with self.assertRaises(ValueError):
            fr.nn.functional.cross_entropy(fr.Tensor([], [0, 2]), fr.Tensor.from_i64([], [0]))


if __name__ == '__main__':
    unittest.main()
