"""Actual PyG graph BatchNorm and learned group normalization state references."""
import inspect
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.norm import BatchNorm as TorchBatch, DiffGroupNorm as TorchDiff
import ferro as fr
from ferro.graph import BatchNorm, DiffGroupNorm
import test_pyg_modulated as helpers

native = helpers.native


class StatefulNormTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self): torch.manual_seed(2111)

    def buffers(self, model, oracle):
        actual, expected = dict(model.named_buffers()), dict(oracle.named_buffers())
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            if expected[name].dtype == torch.int64:
                self.assertEqual((actual[name].dtype, actual[name].shape, actual[name].item()), ('int64', list(expected[name].shape), expected[name].item()))
            else: self.close(actual[name], expected[name])

    def compare(self, model, oracle, rows, width=3):
        for parameter in model.parameters(): parameter.tensor().zero_grad()
        oracle.zero_grad()
        base = torch.randn(width, rows, requires_grad=True)
        nb = native(base, True)
        self.gradients(model, oracle, model(nb.transpose(0, 1)), oracle(base.transpose(0, 1)), [('x', nb, base)])
        self.buffers(model, oracle)

    def test_graph_batch_norm_training_eval_affine_tracking_and_cumulative_momentum(self):
        for options in ({}, dict(momentum=None), dict(affine=False), dict(track_running_stats=False), dict(affine=False, track_running_stats=False)):
            model, oracle = BatchNorm(3, **options), TorchBatch(3, **options)
            self.copy(model, oracle)
            for training in (True, True, False):
                model.train(training); oracle.train(training)
                self.compare(model, oracle, 7)
            model.reset_running_stats(); oracle.reset_running_stats(); self.buffers(model, oracle)
            model.reset_parameters(); oracle.reset_parameters(); self.buffers(model, oracle)

    def test_allowed_singleton_and_empty_batches_preserve_saved_counter(self):
        for affine in (False, True):
            model = BatchNorm(3, momentum=None, affine=affine, allow_single_element=True)
            oracle = TorchBatch(3, momentum=None, affine=affine, allow_single_element=True)
            self.copy(model, oracle)
            for rows in (0, 1, 5, 1, 0): self.compare(model, oracle, rows)
            self.assertEqual(model.module.num_batches_tracked.item(), 1)
            model.eval(); oracle.eval()
            for rows in (0, 1, 4): self.compare(model, oracle, rows)
            self.assertEqual(model.module.num_batches_tracked.item(), 1)
        with self.assertRaises(ValueError): BatchNorm(3, track_running_stats=False, allow_single_element=True)
        with self.assertRaises(ValueError): TorchBatch(3, track_running_stats=False, allow_single_element=True)

    def test_diff_group_assignments_norm_buffers_and_all_gradients(self):
        cases = [(1, {}), (3, dict(lamda=.2)), (2, dict(lamda=0.)), (2, dict(lamda=-.1, momentum=None)),
                 (3, dict(affine=False)), (2, dict(track_running_stats=False))]
        for groups, options in cases:
            model, oracle = DiffGroupNorm(3, groups, **options), TorchDiff(3, groups, **options)
            self.copy(model, oracle)
            for training in (True, True, False):
                model.train(training); oracle.train(training)
                self.compare(model, oracle, 7)
            model.reset_parameters(); oracle.reset_parameters(); self.buffers(model, oracle)
            if model.norm.weight is not None:
                self.assertEqual(model.norm.weight.tensor().tolist(), [1.] * (3 * groups))
                self.assertEqual(model.norm.bias.tensor().tolist(), [0.] * (3 * groups))

    def test_empty_and_singleton_error_counter_behavior_matches_torch(self):
        for model, oracle, name in ((BatchNorm(3), TorchBatch(3), 'module'), (DiffGroupNorm(3, 2), TorchDiff(3, 2), 'norm')):
            self.copy(model, oracle)
            with self.assertRaises(ValueError): model(fr.Tensor.ones([1, 3]))
            with self.assertRaises(ValueError): oracle(torch.ones(1, 3))
            self.assertEqual(getattr(model, name).num_batches_tracked.item(), 1)
            self.buffers(model, oracle)
            self.compare(model, oracle, 0)
            self.assertEqual(getattr(model, name).num_batches_tracked.item(), 2)

    def test_scalar_group_distance_ratio_matches_pyg_both_euclidean_paths(self):
        for nodes, width, classes in ((9, 3, 3), (90, 4, 3), (6, 0, 2), (60, 3, 2)):
            x = torch.ones(nodes, width) if nodes == 60 else torch.randn(nodes, width)
            y = torch.arange(nodes) % classes
            nx = native(x, True)
            actual = DiffGroupNorm.group_distance_ratio(nx, native(y))
            expected = TorchDiff.group_distance_ratio(x, y)
            self.assertAlmostEqual(actual, expected, delta=3e-5 + 4e-4 * abs(expected))
            self.assertIsNone(nx.grad)
        x, y = fr.Tensor.ones([4, 3]), fr.Tensor.from_i64([0, 0, 1, 1], [4])
        with self.assertRaises(ZeroDivisionError): DiffGroupNorm.group_distance_ratio(x, y, eps=0.)
        with self.assertRaises(ZeroDivisionError): TorchDiff.group_distance_ratio(torch.ones(4, 3), torch.tensor([0, 0, 1, 1]), eps=0.)

    def test_serialized_counters_running_statistics_optimizer_and_rng_continuation(self):
        for kind in ('batch', 'diff'):
            def factory(changed=False):
                return BatchNorm(3, momentum=None, allow_single_element=not changed) if kind == 'batch' else DiffGroupNorm(3, 3 if changed else 2, momentum=None)
            def step(model, optimizer, generator, rows):
                output = model(fr.Tensor.randn([rows, 3], generator=generator))
                loss = output.square().sum() / max(1, rows * 3)
                values = output.tolist(), loss.item()
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                return values
            model, rng = factory(), fr.Generator(331)
            optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
            step(model, optimizer, rng, 5)
            saved = fr.checkpoint.snapshot_training(model, optimizers={'norm': optimizer}, generator=rng, step=1)
            with tempfile.TemporaryDirectory() as directory:
                saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
            sequence = (1, 0, 5) if kind == 'batch' else (5, 0, 5)
            expected = [step(model, optimizer, rng, rows) for rows in sequence]
            other, other_rng = factory(), fr.Generator(1)
            other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
            saved.restore(other, optimizers={'norm': other_opt}, generator=other_rng)
            self.assertEqual(expected, [step(other, other_opt, other_rng, rows) for rows in sequence])
            left = fr.checkpoint.snapshot_training(model, optimizers={'norm': optimizer}, generator=rng, step=4).tensors()
            right = fr.checkpoint.snapshot_training(other, optimizers={'norm': other_opt}, generator=other_rng, step=4).tensors()
            self.assertEqual(set(left), set(right))
            for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
            changed = factory(True)
            before = [(name, value.tolist()) for name, value in changed.named_buffers()]
            with self.assertRaises(ValueError): saved.restore(changed, optimizers={'norm': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
            self.assertEqual(before, [(name, value.tolist()) for name, value in changed.named_buffers()])

    def test_signatures_metadata_capacity_and_strict_boundaries(self):
        for cls, reference in ((BatchNorm, TorchBatch), (DiffGroupNorm, TorchDiff)):
            self.assertEqual(list(inspect.signature(cls).parameters), list(inspect.signature(reference).parameters))
        for build in (lambda: BatchNorm(0), lambda: BatchNorm(3, allow_single_element=1), lambda: DiffGroupNorm(3, 0),
                      lambda: DiffGroupNorm(3, 2, lamda=float('nan')), lambda: BatchNorm(3, device='cuda:0')):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)): build()
        for build in (lambda: BatchNorm(sys.maxsize), lambda: DiffGroupNorm(3, sys.maxsize)):
            with self.assertRaisesRegex(ValueError, 'capacity'): build()
        for model in (BatchNorm(3), DiffGroupNorm(3, 2)):
            before = [(name, value.tolist()) for name, value in model.named_buffers()]
            for value in (fr.Tensor.ones([4, 2]), fr.Tensor.ones([4, 3]).to(dtype='float64')):
                with self.assertRaises(ValueError): model(value)
            with self.assertRaises(NotImplementedError): fr.capture(lambda: model(fr.Tensor.ones([4, 3])))
            self.assertEqual(before, [(name, value.tolist()) for name, value in model.named_buffers()])
        model = DiffGroupNorm(3, 2)
        huge = fr.Tensor.ones([1, 3]).expand(sys.maxsize // 12 + 1, 3)
        with patch.object(fr.nn.Linear, 'forward', side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(huge)
        for labels in (torch.tensor([0, 0, 0, 0]), torch.tensor([0, 2, 0, 2]), torch.tensor([-1, 0, 1, 1])):
            with self.assertRaises(ValueError): DiffGroupNorm.group_distance_ratio(fr.Tensor.ones([4, 3]), native(labels))


if __name__ == '__main__':
    unittest.main(verbosity=2)
