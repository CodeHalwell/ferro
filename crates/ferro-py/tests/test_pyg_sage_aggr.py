"""Actual PyG learned GraphSAGE aggregation and native continuation references."""
import sys
import tempfile
import unittest

import torch
from torch_geometric.nn import SAGEConv as TorchSAGE
from torch_geometric.nn.aggr import SoftmaxAggregation as TorchSoftmax
import ferro as fr
from ferro.graph import SAGEConv
import test_pyg_modulated as helpers

native = helpers.native


class SAGEAggregationTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2011)
        self.edge = torch.tensor([[0, 1, 2, 3, 4, 5, 1], [0, 0, 1, 2, 2, 2, 4]])

    def test_actual_pyg_learned_recurrent_and_multi_aggregation_all_gradients(self):
        cases = [('lstm', {}), ('lstm', dict(aggr_kwargs=dict(num_layers=2))),
                 ('gru', dict(aggr_kwargs=dict(in_channels=3, out_channels=3, num_layers=2))),
                 ('softmax', dict(aggr_kwargs=dict(learn=True, t=.7, channels=3))),
                 (['mean', 'max', 'std'], {}),
                 (['mean', 'lstm'], dict(aggr_kwargs=dict(aggrs_kwargs=[{}, dict(in_channels=3, out_channels=5)], mode='proj', mode_kwargs=dict(in_channels=[3, 5], out_channels=4))))]
        for aggr, options in cases:
            for root in (True, False):
                model = SAGEConv((3, 2), 4, aggr, project=True, normalize=True, root_weight=root, **options)
                oracle = TorchSAGE((3, 2), 4, aggr, project=True, normalize=True, root_weight=root, **options)
                self.copy(model, oracle)
                base = torch.randn(3, 6, requires_grad=True)
                target = torch.randn(6, 2, requires_grad=True)
                nb, nt = native(base, True), native(target, True)
                self.gradients(model, oracle, model((nb.transpose(0, 1), nt), native(self.edge)),
                               oracle((base.transpose(0, 1), target), self.edge), [('source', nb, base), ('target', nt, target)])

    def test_registered_instance_missing_targets_and_empty_reduction_messages(self):
        for empty in (False, True):
            model = SAGEConv((3, 2), 4, fr.graph.SoftmaxAggregation(.7, learn=True), bias=False)
            oracle = TorchSAGE((3, 2), 4, TorchSoftmax(.7, learn=True), bias=False)
            self.copy(model, oracle)
            x = torch.randn(6, 3, requires_grad=True)
            nx = native(x, True)
            edge = torch.empty(2, 0, dtype=torch.long) if empty else self.edge
            self.gradients(model, oracle, model((nx, None), native(edge), size=(6, 7)), oracle((x, None), edge, size=(6, 7)), [('x', nx, x)])
            model.reset_parameters()
            self.assertAlmostEqual(model.aggr_module.t.tensor().item(), .7, places=6)
        for aggr in ('min', ['mean', 'max']):
            model, oracle = SAGEConv(3, 2, aggr), TorchSAGE(3, 2, aggr)
            self.copy(model, oracle)
            x = torch.empty(0, 3, requires_grad=True)
            nx, edge = native(x, True), torch.empty(2, 0, dtype=torch.long)
            self.gradients(model, oracle, model(nx, native(edge)), oracle(x, edge), [('x', nx, x)])

    def test_changed_recurrent_width_extension_matches_torch_projection_algebra(self):
        cases = [('lstm', dict(aggr_kwargs=dict(out_channels=5)), 5),
                 ('lstm', dict(aggr_kwargs=dict(out_channels=5, proj_size=2, bidirectional=True)), 4),
                 (['mean', 'lstm'], dict(aggr_kwargs=dict(aggrs_kwargs=[{}, dict(in_channels=3, out_channels=5)])), 8)]
        for aggr, options, width in cases:
            model, oracle = SAGEConv(3, 2, aggr, **options), TorchSAGE(3, 2, aggr, **options)
            self.assertEqual(model.lin_l.weight.tensor().shape, [2, width])
            # PyG's get_out_channels currently assumes input width on these
            # paths. Correct only that declared projection size; the actual
            # PyG recurrent aggregation/message/root algebra remains intact.
            if oracle.lin_l.weight.shape[1] != width: oracle.lin_l = torch.nn.Linear(width, 2)
            self.copy(model, oracle)
            x = torch.randn(6, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, native(self.edge)), oracle(x, self.edge), [('x', nx, x)])

    def test_serialized_lstm_dropout_optimizer_and_generator_continuation(self):
        def factory(dropout=.2):
            return SAGEConv((3, 2), 4, 'lstm', project=True, aggr_kwargs=dict(num_layers=2, dropout=dropout))
        def step(model, optimizer, generator):
            x, target = fr.Tensor.randn([6, 3], generator=generator), fr.Tensor.randn([6, 2], generator=generator)
            output = model((x, target), native(self.edge))
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(293)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'sage': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'sage': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'sage': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'sage': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'sage': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_validation_and_nested_metadata_precede_attention_streams(self):
        for build in (lambda: SAGEConv(3, 4, 'invalid'), lambda: SAGEConv(3, 4, 'lstm', aggr_kwargs=[]),
                      lambda: SAGEConv(3, 4, 'lstm', aggr_kwargs=dict(in_channels=2)), lambda: SAGEConv(3, 4, 'mean', aggr_kwargs=dict(unknown=True))):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)): build()
        options = dict(aggrs_kwargs=[{}, dict(in_channels=3, out_channels=3, num_layers=2, dropout=.2)],
                       mode='attn', mode_kwargs=dict(in_channels=[3, 3], out_channels=4, dropout=.2))
        model = SAGEConv(3, 2, ['mean', 'lstm'], project=True, aggr_kwargs=options)
        def streams(): return [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)]
        before = streams()
        invalid = self.edge.clone(); invalid[1, :3] = torch.tensor([0, 1, 0])
        with self.assertRaisesRegex(ValueError, 'sorted'): model(fr.Tensor.ones([6, 3]), native(invalid))
        self.assertEqual(before, streams())
        with self.assertRaises(ValueError): model(fr.Tensor.ones([6, 2]), native(self.edge))
        self.assertEqual(before, streams())
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(fr.Tensor.ones([6, 3]), native(self.edge)))
        self.assertEqual(before, streams())
        with self.assertRaisesRegex(ValueError, 'capacity'): SAGEConv(sys.maxsize, 2)
        with self.assertRaisesRegex(ValueError, 'capacity'): SAGEConv(3, 2)(fr.Tensor.ones([1, 3]).expand(sys.maxsize // 12 + 1, 3), fr.Tensor.from_i64([], [2, 0]))
        recurrent = SAGEConv(3, 2, 'lstm', aggr_kwargs=dict(num_layers=2, dropout=.2))
        stream = recurrent.aggr_module.lstm.interlayer_dropout[0].rng_state.tolist()
        with self.assertRaises(ValueError): recurrent(fr.Tensor.ones([6, 3]), fr.Tensor.from_i64([], [2, 0]))
        self.assertEqual(stream, recurrent.aggr_module.lstm.interlayer_dropout[0].rng_state.tolist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
