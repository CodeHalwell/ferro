"""Actual PyG MetaLayer callbacks, graph feature gradients, and native state."""
import tempfile
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import MetaLayer as TorchMeta
from torch_geometric.utils import scatter as torch_scatter
import ferro as fr
from ferro.graph import MetaLayer
import test_pyg_modulated as helpers

native = helpers.native


class Update(fr.nn.Module):
    kind: str
    in_width: int
    out_width: int
    p: float = 0.

    def build(self):
        self.lin = fr.nn.Linear(self.in_width, self.out_width)
        self.drop = fr.nn.Dropout(self.p, seed=41)
        self.scale = fr.nn.Parameter(fr.Tensor.ones([1]))
        self._calls, self._resets = 0, 0

    def reset_parameters(self):
        self._resets += 1
        with fr.no_grad():
            self.scale.tensor().copy_(fr.Tensor.ones([1]))
            self.lin.weight.tensor().copy_((fr.Tensor.rand([self.out_width, self.in_width]) * 2 - 1) * self.in_width ** -.5)
            self.lin.bias.tensor().copy_((fr.Tensor.rand([self.out_width]) * 2 - 1) * self.in_width ** -.5)

    def forward(self, *args):
        self._calls += 1
        def globals_for(u, batch, rows):
            if u is None: return fr.Tensor.zeros([rows, 2])
            return u.index_select(0, [0] * rows if batch is None else batch.tolist())
        if self.kind == 'edge':
            source, target, attrs, u, batch = args
            attrs = fr.Tensor.zeros([source.shape[0], 2]) if attrs is None else attrs
            value = fr.cat([source, target, attrs, globals_for(u, batch, source.shape[0])], 1)
        elif self.kind == 'node':
            x, edge, attrs, u, batch = args
            if not isinstance(edge, fr.Tensor): raise TypeError('callback requires COO Tensor')
            attrs = fr.Tensor.zeros([edge.shape[1], 3]) if attrs is None else attrs
            aggregate = fr.graph.scatter(attrs, edge.narrow(0, 1, 1).reshape([edge.shape[1]]).tolist(), x.shape[0], 'mean')
            value = fr.cat([x, aggregate, globals_for(u, batch, x.shape[0])], 1)
        else:
            x, edge, attrs, u, batch = args
            ids = [0] * x.shape[0] if batch is None else batch.tolist()
            count = u.shape[0] if u is not None else max(ids, default=-1) + 1
            previous = fr.Tensor.zeros([count, 2]) if u is None else u
            value = fr.cat([previous, fr.graph.scatter(x, ids, count, 'mean')], 1)
        return self.drop(self.lin(value).tanh() * self.scale.tensor())


class TorchUpdate(torch.nn.Module):
    def __init__(self, kind, in_width, out_width, p=0.):
        super().__init__()
        self.kind = kind
        self.lin = torch.nn.Linear(in_width, out_width)
        self.drop = torch.nn.Dropout(p)
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.calls, self.resets = 0, 0

    def reset_parameters(self):
        self.resets += 1
        self.lin.reset_parameters()
        with torch.no_grad(): self.scale.fill_(1.)

    def forward(self, *args):
        self.calls += 1
        def globals_for(u, batch, rows):
            if u is None: return torch.zeros(rows, 2)
            return u[torch.zeros(rows, dtype=torch.long) if batch is None else batch]
        if self.kind == 'edge':
            source, target, attrs, u, batch = args
            attrs = torch.zeros(source.shape[0], 2) if attrs is None else attrs
            value = torch.cat([source, target, attrs, globals_for(u, batch, source.shape[0])], 1)
        elif self.kind == 'node':
            x, edge, attrs, u, batch = args
            attrs = torch.zeros(edge.shape[1], 3) if attrs is None else attrs
            aggregate = torch_scatter(attrs, edge[1], dim=0, dim_size=x.shape[0], reduce='mean')
            value = torch.cat([x, aggregate, globals_for(u, batch, x.shape[0])], 1)
        else:
            x, edge, attrs, u, batch = args
            ids = torch.zeros(x.shape[0], dtype=torch.long) if batch is None else batch
            count = u.shape[0] if u is not None else int(ids.max()) + 1 if ids.numel() else 0
            previous = torch.zeros(count, 2) if u is None else u
            value = torch.cat([previous, torch_scatter(x, ids, dim=0, dim_size=count, reduce='mean')], 1)
        return self.drop(self.lin(value).tanh() * self.scale)


def factory(constructor, update, edge=True, node=True, global_=True, dropout=0.):
    return constructor(update('edge', 10, 3, dropout) if edge else None,
                       update('node', 8, 4, dropout) if node else None,
                       update('global', 6 if node else 5, 2, dropout) if global_ else None)


class MetaLayerTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2081)
        self.edge = torch.tensor([[0, 1, 2, 3, 4, 5, 1], [1, 2, 0, 4, 5, 3, 4]])
        self.batch = torch.tensor([0, 0, 0, 2, 2, 2])

    def compare_tuple(self, model, oracle, actual, expected, inputs):
        for value, reference in zip(actual, expected):
            if reference is None: self.assertIsNone(value)
            else: self.assertEqual(value.shape, list(reference.shape))
        self.gradients(model, oracle, fr.cat([value.flatten() for value in actual if value is not None], 0),
                       torch.cat([value.reshape(-1) for value in expected if value is not None], 0), inputs)

    def test_actual_pyg_update_order_source_batch_ids_and_all_feature_parameter_gradients(self):
        for edge_enabled, node_enabled, global_enabled in ((True, True, True), (True, False, False), (False, True, True), (False, False, True)):
            for native_topology in (False, True):
                model = factory(MetaLayer, lambda kind, width, out, p: Update(kind=kind, in_width=width, out_width=out, p=p), edge_enabled, node_enabled, global_enabled)
                oracle = factory(TorchMeta, TorchUpdate, edge_enabled, node_enabled, global_enabled)
                self.copy(model, oracle)
                base = torch.randn(3, 6, requires_grad=True)
                attrs = torch.randn(7, 2 if edge_enabled else 3, requires_grad=True)
                u = torch.randn(4, 2, requires_grad=True)
                nb, na, nu = native(base, True), native(attrs, True), native(u, True)
                topology = fr.graph.EdgeIndex(6, self.edge[0].tolist(), self.edge[1].tolist()) if native_topology else native(self.edge)
                actual = model(nb.transpose(0, 1), topology, na, nu, native(self.batch))
                expected = oracle(base.transpose(0, 1), self.edge, attrs, u, self.batch)
                self.compare_tuple(model, oracle, actual, expected, [('x', nb, base), ('edge_attr', na, attrs), ('u', nu, u)])

    def test_optional_attributes_batches_empty_edges_and_empty_graphs(self):
        for nodes, edges, has_u, has_batch in ((6, 7, False, True), (6, 7, True, False), (6, 0, True, True), (0, 0, True, True)):
            model = factory(MetaLayer, lambda kind, width, out, p: Update(kind=kind, in_width=width, out_width=out, p=p))
            oracle = factory(TorchMeta, TorchUpdate)
            self.copy(model, oracle)
            x = torch.randn(nodes, 3, requires_grad=True)
            edge = self.edge if edges else torch.empty(2, 0, dtype=torch.long)
            batch = self.batch if nodes and has_batch else torch.empty(0, dtype=torch.long) if has_batch else None
            u = torch.randn(4 if has_batch else 1, 2, requires_grad=True) if has_u else None
            nx, nu = native(x, True), None if u is None else native(u, True)
            actual = model(nx, native(edge), None, nu, None if batch is None else native(batch))
            expected = oracle(x, edge, None, u, batch)
            self.compare_tuple(model, oracle, actual, expected, [('x', nx, x)] + ([] if u is None else [('u', nu, u)]))
        x, attrs, u = fr.Tensor.ones([6, 3]), fr.Tensor.ones([7, 2]), fr.Tensor.ones([4, 2])
        actual = MetaLayer()(x, native(self.edge), attrs, u, native(self.batch))
        self.assertIs(actual[0], x); self.assertIs(actual[1], attrs); self.assertIs(actual[2], u)
        self.assertEqual(MetaLayer()(x, native(self.edge))[1:], (None, None))

    def test_reset_uses_callback_contract_node_edge_global_order(self):
        events = []
        class Recorder(fr.nn.Module):
            name: str
            def reset_parameters(self): events.append(self.name)
            def forward(self, *args): return args[0]
        models = [Recorder(name=name) for name in ('edge', 'node', 'global')]
        model = MetaLayer(*models)
        self.assertEqual(events, ['node', 'edge', 'global'])
        self.assertIs(model.edge_model, models[0]); self.assertIs(model.node_model, models[1]); self.assertIs(model.global_model, models[2])
        model.reset_parameters()
        self.assertEqual(events, ['node', 'edge', 'global'] * 2)
        class NoReset(fr.nn.Module):
            def build(self): self.lin = fr.nn.Linear(3, 3)
            def forward(self, *args): return args[0]
        no_reset = NoReset()
        before = no_reset.lin.weight.tensor().tolist()
        MetaLayer(edge_model=no_reset).reset_parameters()
        self.assertEqual(before, no_reset.lin.weight.tensor().tolist())

    def test_serialized_graph_callbacks_optimizer_dropout_rng_and_config(self):
        def build(dropout=.2):
            model = factory(MetaLayer, lambda kind, width, out, p: Update(kind=kind, in_width=width, out_width=out, p=p), dropout=dropout)
            model.node_model.scale = model.edge_model.scale
            model.global_model.scale = model.edge_model.scale
            model.node_model.drop = model.edge_model.drop
            return model
        def step(model, optimizer, generator):
            inputs = [fr.Tensor.randn(shape, generator=generator) for shape in ([6, 3], [7, 2], [4, 2])]
            output = model(inputs[0], native(self.edge), inputs[1], inputs[2], native(self.batch))
            combined = fr.cat([value.flatten() for value in output], 0)
            loss = combined.square().mean()
            values = [value.tolist() for value in output], loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = build(), fr.Generator(313)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'meta': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = build(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'meta': other_opt}, generator=other_rng)
        self.assertIs(other.node_model.scale, other.edge_model.scale)
        self.assertIs(other.global_model.scale, other.edge_model.scale)
        self.assertIs(other.node_model.drop, other.edge_model.drop)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'meta': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'meta': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = build(.3)
        before = [p.tensor().tolist() for p in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'meta': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_invalid_metadata_precedes_callbacks_and_output_row_contracts(self):
        with self.assertRaises(TypeError): MetaLayer(edge_model=lambda *args: args[0])
        model = factory(MetaLayer, lambda kind, width, out, p: Update(kind=kind, in_width=width, out_width=out, p=p), dropout=.2)
        x, attrs, u = fr.Tensor.ones([6, 3]), fr.Tensor.ones([7, 2]), fr.Tensor.ones([4, 2])
        streams = [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)]
        calls = (lambda: model(x, native(self.edge), fr.Tensor.ones([6, 2]), u),
                 lambda: model(x, native(self.edge), attrs, fr.Tensor.ones([2, 2]), native(self.batch)),
                 lambda: model(x, native(self.edge), attrs, u, fr.Tensor.ones([6])),
                 lambda: model(x, fr.Tensor.from_i64([0, 6], [2, 1])),
                 lambda: model(x.to(dtype='float64'), native(self.edge)))
        for call in calls:
            with self.assertRaises(ValueError): call()
            self.assertEqual([model.edge_model._calls, model.node_model._calls, model.global_model._calls], [0, 0, 0])
            self.assertEqual(streams, [layer.rng_state.tolist() for _, layer in model.named_modules() if isinstance(layer, fr.nn.Dropout)])
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, native(self.edge), attrs, u, native(self.batch)))
        huge_edge = fr.Tensor.from_i64([0], [1, 1]).expand(2, sys.maxsize // 16 + 1)
        huge_batch = fr.Tensor.from_i64([0], [1]).expand(sys.maxsize // 8 + 1)
        with patch.object(fr.Tensor, 'tolist', side_effect=AssertionError('invalid topology materialized')):
            with self.assertRaisesRegex(ValueError, 'capacity'): model(x, huge_edge)
            with self.assertRaisesRegex(ValueError, 'entry per node'): model(x, native(self.edge), attrs, u, huge_batch)
        class BadRows(fr.nn.Module):
            kind: str
            def forward(self, *args): return (args[3] if self.kind == 'global' else args[0]).narrow(0, 0, 1)
        for name in ('edge', 'node', 'global'):
            broken = MetaLayer(**{name + '_model': BadRows(kind=name)})
            with self.assertRaisesRegex(ValueError, 'preserve'): broken(x, native(self.edge), attrs, u, native(self.batch))


if __name__ == '__main__':
    unittest.main(verbosity=2)
