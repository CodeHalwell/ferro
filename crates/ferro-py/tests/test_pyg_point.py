"""Actual PyG point convolutions with feature, position and parameter gradients."""
import sys
import unittest

import torch
from torch_geometric.nn import PointNetConv as TorchPointNet, PointTransformerConv as TorchPointTransformer
import ferro as fr
from ferro.graph import PointNetConv, PointTransformerConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class PointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(929)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2, 1], [1, 2, 0, 3, 4, 0, 1, 2, 2, 1]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4)

    def network(self, inputs, outputs):
        return (fr.nn.Sequential(fr.nn.Linear(inputs, 7), fr.nn.Tanh(), fr.nn.Linear(7, outputs)),
                torch.nn.Sequential(torch.nn.Linear(inputs, 7), torch.nn.Tanh(), torch.nn.Linear(7, outputs)))

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))

    def gradients(self, model, reference, actual, expected, inputs):
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward()
        (expected * probe).sum().backward()
        parameters = dict(reference.named_parameters())
        for name, actual, expected in inputs + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    self.assertIsNotNone(actual.grad)
                    self.close(actual.grad, expected.grad)

    def test_pointnet_local_global_and_bipartite_coordinate_gradients(self):
        for bipartite, features, loops, aggr in ((False, True, True, 'max'), (True, True, True, 'mean'),
                                                (True, False, False, 'add'), (False, False, False, 'max')):
            with self.subTest(bipartite=bipartite, features=features, loops=loops, aggr=aggr):
                local, tlocal = self.network(6 if features else 3, 5)
                glob, tglob = self.network(5, 4)
                model = PointNetConv(local, glob, loops, aggr=aggr)
                reference = TorchPointNet(tlocal, tglob, loops, aggr=aggr)
                self.copy(model, reference)
                pos_s = torch.randn(6, 3, requires_grad=True)
                pos_t = torch.randn(5, 3, requires_grad=True) if bipartite else pos_s
                ps, pt = native(pos_s, True), None
                pt = native(pos_t, True) if bipartite else ps
                source = torch.randn(6, 3, requires_grad=True) if features else None
                target = torch.randn(5, 2, requires_grad=True) if bipartite and features else None
                ns, nt = None if source is None else native(source, True), None if target is None else native(target, True)
                x, tx = ((ns, nt), (source, target)) if bipartite else (ns, source)
                pos, tpos = ((ps, pt), (pos_s, pos_t)) if bipartite else (ps, pos_s)
                inputs = [('source_pos', ps, pos_s)] + ([('target_pos', pt, pos_t)] if bipartite else [])
                if source is not None:
                    inputs.append(('source', ns, source))
                if target is not None:
                    inputs.append(('unused_target', nt, target))
                self.gradients(model, reference, model(x, pos, native(self.edge)), reference(tx, tpos, self.edge), inputs)

    def test_pointnet_without_networks_and_bipartite_self_loop_bound(self):
        for features, loops in ((False, False), (True, True)):
            model, reference = PointNetConv(add_self_loops=loops), TorchPointNet(add_self_loops=loops)
            source_pos = torch.randn(4, 2, requires_grad=True)
            target_pos = torch.randn(6, 2, requires_grad=True)
            edge = torch.tensor([[0, 1, 3, 1, 2], [1, 2, 5, 1, 2]])
            ps, pt = native(source_pos, True), native(target_pos, True)
            source = torch.randn(4, 3, requires_grad=True) if features else None
            ns = None if source is None else native(source, True)
            pairs = [('source_pos', ps, source_pos), ('target_pos', pt, target_pos)] + ([] if source is None else [('source', ns, source)])
            self.gradients(model, reference, model(ns, (ps, pt), native(edge)), reference(source, (source_pos, target_pos), edge), pairs)

    def test_pointtransformer_default_custom_networks_and_all_gradients(self):
        for bipartite, custom, loops in ((False, False, True), (True, True, True), (False, True, False)):
            with self.subTest(bipartite=bipartite, custom=custom, loops=loops):
                width = (3, 2) if bipartite else 3
                position_dim = 2 if custom else 3
                pos_nn, tpos_nn = self.network(position_dim, 4) if custom else (None, None)
                attn_nn, tattn_nn = self.network(4, 4) if custom else (None, None)
                model = PointTransformerConv(width, 4, pos_nn, attn_nn, loops)
                reference = TorchPointTransformer(width, 4, tpos_nn, tattn_nn, loops)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else source
                pos_s = torch.randn(6, position_dim, requires_grad=True)
                pos_t = torch.randn(5, position_dim, requires_grad=True) if bipartite else pos_s
                ns, ps = native(source, True), native(pos_s, True)
                nt, pt = (native(target, True), native(pos_t, True)) if bipartite else (ns, ps)
                x, tx = ((ns, nt), (source, target)) if bipartite else (ns, source)
                pos, tpos = ((ps, pt), (pos_s, pos_t)) if bipartite else (ps, pos_s)
                inputs = [('source', ns, source), ('source_pos', ps, pos_s)]
                if bipartite:
                    inputs += [('target', nt, target), ('target_pos', pt, pos_t)]
                self.gradients(model, reference, model(x, pos, native(self.edge)), reference(tx, tpos, self.edge), inputs)

    def test_empty_graphs_isolates_and_zero_width_positions(self):
        edge = torch.empty(2, 0, dtype=torch.int64)
        for nodes in (0, 4):
            for loops in (False, True):
                for model, reference in ((PointNetConv(fr.nn.Linear(6, 4), add_self_loops=loops),
                                          TorchPointNet(torch.nn.Linear(6, 4), add_self_loops=loops)),
                                         (PointTransformerConv(3, 4, add_self_loops=loops), TorchPointTransformer(3, 4, add_self_loops=loops))):
                    with self.subTest(nodes=nodes, loops=loops, kind=type(model).__name__):
                        self.copy(model, reference)
                        x, pos = torch.randn(nodes, 3, requires_grad=True), torch.randn(nodes, 3, requires_grad=True)
                        nx, np = native(x, True), native(pos, True)
                        self.gradients(model, reference, model(nx, np, native(edge)), reference(x, pos, edge), [('x', nx, x), ('pos', np, pos)])
        model, reference = PointNetConv(), TorchPointNet()
        pos = torch.empty(6, 0, requires_grad=True)
        np = native(pos, True)
        edges = fr.graph.EdgeIndex(6, self.edge[0].tolist(), self.edge[1].tolist())
        self.gradients(model, reference, model(None, np, edges), reference(None, pos, self.edge), [('pos', np, pos)])

    def test_checkpoint_aliases_coordinate_noise_optimizer_and_config(self):
        def factory(loops=True):
            model = PointTransformerConv(3, 4, add_self_loops=loops)
            model.lin_src.weight = model.lin.weight
            model.pos_nn.bias.trainable = False
            return model
        model, rng = factory(), fr.Generator(79)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.01)
        edge = native(self.edge)
        def step(current, opt, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            positions = fr.Tensor.rand([6, 3], generator=generator)
            result = current(x, positions, edge)
            loss = result.square().mean()
            values = result.tolist(), loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return values
        step(model, optimizer, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'point': optimizer}, generator=rng, step=1)
        expected = [step(model, optimizer, rng) for _ in range(2)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        state.restore(other, optimizers={'point': other_opt}, generator=other_rng)
        self.assertIs(other.lin.weight, other.lin_src.weight)
        self.assertFalse(other.pos_nn.bias.trainable)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(2)])
        for left, right in zip(model.parameters(), other.parameters()):
            self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        changed = factory(False)
        with self.assertRaises(ValueError):
            state.restore(changed, optimizers={'point': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))

    def test_strict_networks_shapes_coordinate_bounds_and_capture(self):
        for constructor in (lambda: PointNetConv(local_nn=lambda x: x), lambda: PointNetConv(add_self_loops=1),
                            lambda: PointTransformerConv(-1, 4), lambda: PointTransformerConv(3, 0),
                            lambda: PointTransformerConv(3, sys.maxsize), lambda: PointTransformerConv(3, 4, attn_nn=lambda x: x)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        with self.assertRaises(NotImplementedError):
            PointNetConv(flow='target_to_source')
        x, pos, edge = native(torch.randn(6, 3)), native(torch.randn(6, 3)), native(self.edge)
        model = PointTransformerConv(3, 4)
        for call in (lambda: model(x, native(torch.randn(6, 2)), edge), lambda: model(x, native(torch.randn(5, 3)), edge),
                     lambda: model((x, None), pos, edge), lambda: PointNetConv()(None, (pos, None), edge),
                     lambda: PointTransformerConv(3, 4, pos_nn=fr.nn.Linear(3, 5))(x, pos, edge),
                     lambda: PointTransformerConv(3, 4, attn_nn=fr.nn.Linear(4, 5))(x, pos, edge),
                     lambda: model(x, pos, native(torch.tensor([[6], [0]]))),
                     lambda: model(x, fr.tensor([[1., 2., 3.]], dtype='float64'), edge)):
            with self.assertRaises((ValueError, TypeError)):
                call()
        huge = fr.Tensor.zeros([1, 0]).expand(sys.maxsize // 8 + 1, 0)
        with self.assertRaises(ValueError):
            PointNetConv()(None, huge, fr.Tensor.from_i64([], [2, 0]))
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(x, pos, edge))
        with self.assertRaises(AttributeError):
            model.add_self_loops = False


if __name__ == '__main__':
    unittest.main(verbosity=2)
