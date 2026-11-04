"""Actual PyG light/simple propagation and reusable aggregation references."""
import sys
import tempfile
import unittest

import torch
from torch_geometric.nn import LGConv as TorchLG, SimpleConv as TorchSimple
from torch_geometric.nn.aggr import SoftmaxAggregation as TorchSoftmax
import ferro as fr
from ferro.graph import LGConv, SimpleConv
import test_pyg_modulated as helpers

native = helpers.native


class SimpleTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(1861)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def test_light_normalized_raw_reductions_and_edge_input_gradients(self):
        for normalize, weighted, aggr in ((True, True, 'add'), (True, False, 'sum'), (False, True, 'mean'), (False, True, 'max'), (False, False, 'min')):
            model, oracle = LGConv(normalize, aggr=aggr), TorchLG(normalize, aggr=aggr)
            base = torch.randn(3, 6, requires_grad=True)
            x = base.transpose(0, 1)
            weight = (torch.rand(9) + .5).requires_grad_() if weighted else None
            nb, nw = native(base, True), native(weight, True) if weighted else None
            nx = nb.transpose(0, 1)
            inputs = [('x', nb, base)] + ([('weight', nw, weight)] if weighted else [])
            self.gradients(model, oracle, model(nx, native(self.edge), nw), oracle(x, self.edge, weight), inputs)

    def test_simple_root_modes_bipartite_and_edge_gradients(self):
        for combine, bipartite, aggr, weighted in ((None, False, 'sum', True), ('sum', True, 'mean', True),
                                                  ('cat', True, 'max', False), ('self_loop', False, 'mean', True),
                                                  ('self_loop', False, 'min', False)):
            model, oracle = SimpleConv(aggr, combine), TorchSimple(aggr, combine)
            x = torch.randn(6, 3, requires_grad=True)
            target = torch.randn(5, 2 if combine == 'cat' else 3, requires_grad=True) if bipartite else x
            weight = (torch.rand(9) + .5).requires_grad_() if weighted else None
            nx, nt, nw = native(x, True), native(target, True) if bipartite else None, native(weight, True) if weighted else None
            inputs = [('x', nx, x)] + ([('target', nt, target)] if bipartite else []) + ([('weight', nw, weight)] if weighted else [])
            self.gradients(model, oracle, model((nx, nt) if bipartite else nx, native(self.edge), nw), oracle((x, target) if bipartite else x, self.edge, weight), inputs)

    def test_missing_targets_broadcast_root_and_learned_multi_aggregation(self):
        options = dict(aggr_kwargs={'aggrs_kwargs': [{}, {'learn': True, 't': .7}]})
        for missing, root_width in ((True, None), (False, 1)):
            model = SimpleConv(['mean', 'softmax'], 'sum', **options)
            oracle = TorchSimple(['mean', 'softmax'], 'sum', **options)
            self.copy(model, oracle)
            x = torch.randn(6, 3, requires_grad=True)
            target = None if missing else torch.randn(7, root_width, requires_grad=True)
            nx, nt = native(x, True), None if missing else native(target, True)
            actual = model((nx, nt), native(self.edge), size=(6, 7))
            expected = oracle((x, target), self.edge, size=(6, 7))
            self.gradients(model, oracle, actual, expected, [('x', nx, x)] + ([] if missing else [('target', nt, target)]))

    def test_zero_nodes_empty_features_and_native_topology(self):
        empty = torch.empty(2, 0, dtype=torch.int64)
        for nodes, width in ((0, 3), (4, 0), (4, 3)):
            for model, oracle in ((LGConv(), TorchLG()), (SimpleConv('mean', 'self_loop'), TorchSimple('mean', 'self_loop')),
                                 (SimpleConv('sum', 'cat'), TorchSimple('sum', 'cat'))):
                x = torch.randn(nodes, width, requires_grad=True)
                nx = native(x, True)
                edge = fr.graph.EdgeIndex(nodes, [], [])
                self.gradients(model, oracle, model(nx, edge), oracle(x, empty), [('x', nx, x)])

    def test_saved_learned_aggregation_optimizer_reset_configuration(self):
        def factory(combine='sum'):
            return SimpleConv(fr.graph.SoftmaxAggregation(.7, learn=True), combine)
        model, generator = factory(), fr.Generator(199)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        edge = native(self.edge)
        def step(model, optimizer, generator):
            x = fr.Tensor.randn([6, 3], generator=generator)
            output = model(x, edge)
            loss = output.square().mean()
            result = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return result
        step(model, optimizer, generator)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'simple': optimizer}, generator=generator, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, generator) for _ in range(3)]
        other, rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'simple': other_opt}, generator=rng)
        self.assertEqual(expected, [step(other, other_opt, rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'simple': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'simple': other_opt}, generator=rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left: self.assertEqual(left[name].tolist(), right[name].tolist(), name)
        changed = factory(None)
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'simple': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        model.reset_parameters()
        self.assertAlmostEqual(model.aggr_module.t.tensor().item(), .7, places=6)

    def test_validation_capacity_and_incompatible_root_precedes_attention_stream(self):
        for constructor in (lambda: LGConv(1), lambda: SimpleConv(combine_root='bad')):
            with self.assertRaises((ValueError, TypeError)): constructor()
        for constructor in (lambda: LGConv(flow='target_to_source'), lambda: SimpleConv('mul'), lambda: SimpleConv(None)):
            with self.assertRaises(NotImplementedError): constructor()
        x, edge = fr.Tensor.ones([6, 3]), native(self.edge)
        for call in (lambda: LGConv()(x, edge, fr.Tensor.ones([8])), lambda: SimpleConv('sum', 'self_loop')((x, x), edge),
                     lambda: SimpleConv('sum', 'self_loop')(x, edge, size=[6]), lambda: SimpleConv('sum', 'sum')((x, fr.Tensor.ones([6, 2])), edge)):
            with self.assertRaises((ValueError, TypeError)): call()
        model = SimpleConv(['mean', 'sum'], 'sum', aggr_kwargs={'mode': 'attn', 'mode_kwargs': {'in_channels': 3, 'out_channels': 4, 'dropout': .2}})
        stream = model.aggr_module.multihead_attn.attn_dropout.rng_state.tolist()
        with self.assertRaises(ValueError): model((x, fr.Tensor.ones([6, 2])), edge)
        self.assertEqual(stream, model.aggr_module.multihead_attn.attn_dropout.rng_state.tolist())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: LGConv()(x, edge))
        empty = fr.Tensor.from_i64([], [2, 0])
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8, 1)
        with self.assertRaisesRegex(ValueError, 'capacity'): LGConv()(huge, empty)
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 16 + 1, 1)
        with self.assertRaisesRegex(ValueError, 'capacity'): SimpleConv('sum', 'self_loop')(huge, empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
