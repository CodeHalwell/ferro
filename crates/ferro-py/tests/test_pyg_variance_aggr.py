"""Actual PyG VPA values, input gradients and registered graph composition."""
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import VariancePreservingAggregation as TorchVPA, MultiAggregation as TorchMulti
from torch_geometric.nn import SimpleConv as TorchSimple
import ferro as fr
from ferro.graph import VariancePreservingAggregation
import test_pyg_modulated as helpers

native = helpers.native


class VariancePreservingTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2399)

    def test_unsorted_csr_single_group_strides_empty_and_input_gradients(self):
        layouts = [dict(index=torch.tensor([2, 0, 2, 1, 0, 2]), dim_size=4),
                   dict(ptr=torch.tensor([0, 2, 3, 3, 6]), dim_size=4), {},
                   dict(index=torch.tensor([], dtype=torch.long), dim_size=3),
                   dict(ptr=torch.tensor([0, 0, 0])), dict(index=torch.tensor([], dtype=torch.long))]
        for kwargs in layouts:
            rows = len(kwargs['index']) if 'index' in kwargs else int(kwargs['ptr'][-1]) if 'ptr' in kwargs else 6
            for width in (3, 0):
                x = torch.randn(width, rows, requires_grad=True); nx = native(x, True)
                model, oracle = VariancePreservingAggregation(), TorchVPA()
                options = {key: native(value) if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
                self.gradients(model, oracle, model(nx.transpose(0, 1), **options), oracle(x.transpose(0, 1), **kwargs), [('x', nx, x)])
                self.assertEqual(model.get_out_channels(width), width)

    def test_multi_projection_and_simple_edge_weight_gradients(self):
        model = fr.graph.MultiAggregation([fr.graph.SumAggregation(), VariancePreservingAggregation()], mode='proj', mode_kwargs=dict(in_channels=3, out_channels=4))
        oracle = TorchMulti(['sum', TorchVPA()], mode='proj', mode_kwargs=dict(in_channels=3, out_channels=4))
        self.copy(model, oracle)
        ids = torch.tensor([0, 1, 0, 3, 1, 3])
        x = torch.randn(6, 3, requires_grad=True); nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(ids), dim_size=5), oracle(x, ids, dim_size=5), [('x', nx, x)])
        model, oracle = fr.graph.SimpleConv(aggr=VariancePreservingAggregation(), combine_root='cat'), TorchSimple(aggr=TorchVPA(), combine_root='cat')
        edge = torch.tensor([[0, 1, 2, 2, 4, 1], [1, 2, 1, 3, 3, 2]])
        x, weight = torch.randn(5, 3, requires_grad=True), torch.randn(6, requires_grad=True)
        nx, nw = native(x, True), native(weight, True)
        self.gradients(model, oracle, model(nx, native(edge), nw), oracle(x, edge, weight), [('x', nx, x), ('edge_weight', nw, weight)])

    def test_named_resolvers_multi_simple_and_gen_all_gradients(self):
        from torch_geometric.nn import GENConv as TorchGEN
        edge = torch.tensor([[0, 1, 2, 2, 4, 1], [1, 2, 1, 3, 3, 2]])
        for name in ('variance_preserving', 'variancepreserving'):
            multi, reference = fr.graph.MultiAggregation([name, 'sum']), TorchMulti([name, 'sum'])
            x = torch.randn(5, 3, requires_grad=True); nx = native(x, True)
            self.gradients(multi, reference, multi(nx, [0, 1, 0, 2, 1]), reference(x, torch.tensor([0, 1, 0, 2, 1])), [('x', nx, x)])
            for model, oracle in ((fr.graph.SimpleConv(name), TorchSimple(name)),
                                  (fr.graph.GENConv(3, 3, aggr=name, norm=None), TorchGEN(3, 3, aggr=name, norm=None))):
                self.copy(model, oracle)
                x = torch.randn(5, 3, requires_grad=True); nx = native(x, True)
                self.gradients(model, oracle, model(nx, native(edge)), oracle(x, edge), [('x', nx, x)])

    def test_registered_shared_aggregation_and_projection_checkpoint(self):
        def factory():
            aggregation = VariancePreservingAggregation()
            return fr.graph.MultiAggregation([aggregation, aggregation], mode='proj', mode_kwargs=dict(in_channels=3, out_channels=2))
        model, rng = factory(), fr.Generator(433)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        def step(model, optimizer, rng):
            out = model(fr.Tensor.randn([5, 3], generator=rng), [0, 1, 0, 2, 1], dim_size=4)
            loss = out.square().mean(); values = out.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad(); return values
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'projection': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(2)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.7)
        saved.restore(other, optimizers={'projection': other_opt}, generator=other_rng)
        self.assertIs(other.aggrs[0], other.aggrs[1])
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(2)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'projection': optimizer}, generator=rng, step=3).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'projection': other_opt}, generator=other_rng, step=3).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual(left[key].tolist(), right[key].tolist(), key)

    def test_metadata_and_derived_count_capacity_before_allocation(self):
        model, x = VariancePreservingAggregation(), fr.Tensor.ones([3, 2])
        for kwargs in (dict(index=[0, 1]), dict(index=[0, -1, 1]), dict(index=[0, 1, 2], dim_size=2),
                       dict(ptr=fr.Tensor.from_i64([0, 2], [2])), dict(dim=1), dict(index=[0, 0, 0], ptr=fr.Tensor.from_i64([0, 3], [2]))):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)): model(x, **kwargs)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, [0, 0, 0]))
        with patch.object(fr.Tensor, 'ones', side_effect=AssertionError('allocated counts before validation')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(x, [0, 0, 0], dim_size=sys.maxsize // 8 + 1)
        ptr = fr.Tensor.from_i64([0, 2, 3], [3])
        with patch.object(fr.Tensor, 'ones', side_effect=AssertionError('CSR counts must use pointer differences')):
            self.close(model(x, ptr=ptr), TorchVPA()(torch.ones(3, 2), ptr=torch.tensor([0, 2, 3])))
        self.assertEqual(list(model.parameters()), [])
        self.assertEqual(list(model.named_buffers()), [])
        model.reset_parameters()


if __name__ == '__main__': unittest.main(verbosity=2)
