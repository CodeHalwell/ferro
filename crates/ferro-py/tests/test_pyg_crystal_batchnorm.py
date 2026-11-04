"""Actual PyG CGConv BatchNorm gradients, counters and saved native state."""
import tempfile
import unittest

import torch
from torch_geometric.nn import CGConv as TorchCG
import ferro as fr
from ferro.graph import CGConv
import test_pyg_modulated as helpers

native = helpers.native


class CrystalBatchNormTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1831)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def state(self, model, oracle):
        self.close(model.bn.running_mean, oracle.bn.running_mean)
        self.close(model.bn.running_var, oracle.bn.running_var)
        self.assertEqual(model.bn.num_batches_tracked.dtype, 'int64')
        self.assertEqual(model.bn.num_batches_tracked.item(), oracle.bn.num_batches_tracked.item())

    def test_training_eval_bipartite_edge_and_all_parameter_gradients(self):
        for bipartite, bias, aggr in ((False, True, 'add'), (True, False, 'mean')):
            channels = (3, 2) if bipartite else 3
            model, oracle = CGConv(channels, 2, aggr=aggr, batch_norm=True, bias=bias), TorchCG(channels, 2, aggr=aggr, batch_norm=True, bias=bias)
            self.copy(model, oracle)
            for training in (True, True, False):
                model.train(training); oracle.train(training)
                for parameter in model.parameters(): parameter.tensor().zero_grad()
                oracle.zero_grad()
                x = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else x
                attrs = torch.randn(9, 2, requires_grad=True)
                nx, nt, na = native(x, True), native(target, True) if bipartite else None, native(attrs, True)
                actual = model((nx, nt) if bipartite else nx, native(self.edge), na)
                expected = oracle((x, target) if bipartite else x, self.edge, attrs)
                inputs = [('x', nx, x), ('attrs', na, attrs)] + ([('target', nt, target)] if bipartite else [])
                self.gradients(model, oracle, actual, expected, inputs)
                self.state(model, oracle)

    def test_empty_graphs_zero_messages_and_reset_state(self):
        empty = torch.empty(2, 0, dtype=torch.int64)
        model, oracle = CGConv(3, batch_norm=True), TorchCG(3, batch_norm=True)
        self.copy(model, oracle)
        for nodes in (4, 0):
            for parameter in model.parameters(): parameter.tensor().zero_grad()
            oracle.zero_grad()
            x = torch.randn(nodes, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, native(empty)), oracle(x, empty), [('x', nx, x)])
            self.state(model, oracle)
        model.reset_parameters()
        self.assertEqual(model.bn.running_mean.tolist(), [0.] * 3)
        self.assertEqual(model.bn.running_var.tolist(), [1.] * 3)
        self.assertEqual(model.bn.num_batches_tracked.item(), 0)
        self.assertEqual(model.bn.weight.tensor().tolist(), [1.] * 3)
        self.assertEqual(model.bn.bias.tensor().tolist(), [0.] * 3)

    def test_singleton_matches_counter_and_invalid_topology_precedes_statistics(self):
        model, oracle = CGConv(3, batch_norm=True), TorchCG(3, batch_norm=True)
        self.copy(model, oracle)
        empty = torch.empty(2, 0, dtype=torch.int64)
        with self.assertRaises(ValueError): model(fr.Tensor.ones([1, 3]), native(empty))
        with self.assertRaises(ValueError): oracle(torch.ones(1, 3), empty)
        self.state(model, oracle)
        before = [(name, value.tolist()) for name, value in model.named_buffers()]
        with self.assertRaises(ValueError): model(fr.Tensor.ones([6, 3]), fr.Tensor.from_i64([0, 6], [2, 1]))
        self.assertEqual(before, [(name, value.tolist()) for name, value in model.named_buffers()])
        model.eval(); oracle.eval()
        self.close(model(fr.Tensor.ones([1, 3]), native(empty)), oracle(torch.ones(1, 3), empty))

    def test_saved_typed_counter_optimizer_configuration_and_rng_continuation(self):
        def factory(momentum=None):
            model = CGConv(3, 2, batch_norm=True)
            model.bn.momentum = momentum
            return model
        model, generator = factory(), fr.Generator(193)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        edge = native(self.edge)
        def step(model, optimizer, generator):
            x, attrs = fr.Tensor.randn([6, 3], generator=generator), fr.Tensor.randn([9, 2], generator=generator)
            output = model(x, edge, attrs)
            loss = output.square().mean()
            result = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return result
        step(model, optimizer, generator)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'cg': optimizer}, generator=generator, step=1)
        self.assertEqual(saved.tensors()['buffers.bn.num_batches_tracked'].dtype, 'int64')
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state')
            saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, generator) for _ in range(3)]
        other, rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.7)
        saved.restore(other, optimizers={'cg': other_opt}, generator=rng)
        self.assertEqual(expected, [step(other, other_opt, rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'cg': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'cg': other_opt}, generator=rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertEqual((left[name].dtype, left[name].shape, left[name].tolist()),
                             (right[name].dtype, right[name].shape, right[name].tolist()), name)
        changed = factory(.1)
        before = [(name, value.tolist()) for name, value in changed.named_buffers()]
        with self.assertRaises(ValueError):
            saved.restore(changed, optimizers={'cg': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [(name, value.tolist()) for name, value in changed.named_buffers()])


if __name__ == '__main__':
    unittest.main(verbosity=2)
