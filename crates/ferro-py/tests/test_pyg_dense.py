"""Actual PyG dense graph references, including live adjacency derivatives."""
import sys
import unittest

import torch
from torch_geometric.nn import (DenseGCNConv as TorchGCN, DenseSAGEConv as TorchSAGE,
    DenseGINConv as TorchGIN, DenseGraphConv as TorchGraph, dense_diff_pool as torch_diff_pool)
import ferro as fr
from ferro.graph import DenseGCNConv, DenseSAGEConv, DenseGINConv, DenseGraphConv


def native(value, grad=False):
    if value.dtype == torch.bool:
        return fr.Tensor.from_bool(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class DenseGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(947)

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4)

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))
            buffers = dict(reference.named_buffers())
            self.assertEqual(set(dict(model.named_buffers())), set(buffers))
            for name, buffer in model.named_buffers():
                buffer.copy_(native(buffers[name]))

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

    def compare(self, model, reference, xb=2, ab=2, nodes=4, rank2=False, mask_kind='float', mask_batch=None, **kwargs):
        self.copy(model, reference)
        base = torch.randn(3, nodes, requires_grad=True) if rank2 else torch.randn(xb, 3, nodes, requires_grad=True)
        x = base.transpose(-1, -2)
        nb = native(base, True)
        nx = nb.transpose(-1, -2)
        data = torch.rand((nodes, nodes) if rank2 else (ab, nodes, nodes)) * .4
        if nodes:
            data[..., -1, :] = 0.
        adj = data.requires_grad_()
        na = native(adj, True)
        mb = (1 if rank2 else ab) if mask_batch is None else mask_batch
        mask = None if mask_kind is None else (torch.rand(mb, nodes) > .3) if mask_kind == 'bool' else torch.randn(mb, nodes).requires_grad_()
        nm = None if mask is None else native(mask, mask_kind == 'float')
        pairs = [('x', nb, base), ('adjacency', na, adj)] + ([] if mask_kind != 'float' else [('mask', nm, mask)])
        self.gradients(model, reference, model(nx, na, nm, **kwargs), reference(x, adj, mask, **kwargs), pairs)
        return na, adj

    def test_densegcn_diagonal_overwrite_improved_and_broadcast_gradients(self):
        for improved, bias, loops, xb, ab, rank2, mask in ((False, True, True, 2, 2, False, 'float'),
                                                         (True, False, True, 2, 1, False, 'bool'),
                                                         (False, False, False, 1, 2, False, 'float'),
                                                         (True, True, False, 1, 1, True, None)):
            with self.subTest(improved=improved, bias=bias, loops=loops, xb=xb, ab=ab, rank2=rank2):
                actual, reference = self.compare(DenseGCNConv(3, 4, improved, bias), TorchGCN(3, 4, improved, bias),
                                                  xb, ab, rank2=rank2, mask_kind=mask, add_loop=loops)
                if loops:
                    self.assertEqual(torch_value(actual.grad).diagonal(dim1=-2, dim2=-1).abs().sum().item(), 0.)

    def test_densesage_normalization_bias_masks_and_adjacency_gradients(self):
        for normalize, bias, xb, ab, rank2 in ((False, True, 2, 2, False), (True, False, 2, 1, False),
                                              (True, True, 1, 2, False), (False, False, 1, 1, True)):
            with self.subTest(normalize=normalize, bias=bias, xb=xb, ab=ab, rank2=rank2):
                self.compare(DenseSAGEConv(3, 4, normalize, bias), TorchSAGE(3, 4, normalize, bias), xb, ab, rank2=rank2)

    def test_densegin_epsilon_parameter_buffer_and_unused_root(self):
        for train, loops, xb, ab, rank2 in ((True, True, 2, 1, False), (False, True, 1, 2, False),
                                          (True, False, 2, 2, False), (False, False, 1, 1, True)):
            with self.subTest(train=train, loops=loops, xb=xb, ab=ab, rank2=rank2):
                network = fr.nn.Sequential(fr.nn.Linear(3, 6), fr.nn.Tanh(), fr.nn.Linear(6, 4))
                tnetwork = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Tanh(), torch.nn.Linear(6, 4))
                self.compare(DenseGINConv(network, .2, train), TorchGIN(tnetwork, .2, train), xb, ab, rank2=rank2, add_loop=loops)

    def test_densegraph_add_mean_max_orientation_and_first_winner_ties(self):
        for aggr, bias, xb, ab, rank2, mb in (('add', True, 2, 1, False, 2), ('mean', False, 1, 2, False, 1),
                                             ('max', True, 2, 1, False, 2), ('max', False, 1, 1, True, 3)):
            with self.subTest(aggr=aggr, bias=bias, xb=xb, ab=ab, rank2=rank2):
                self.compare(DenseGraphConv(3, 4, aggr, bias), TorchGraph(3, 4, aggr, bias), xb, ab, rank2=rank2, mask_batch=mb)
        model, reference = DenseGraphConv(2, 2, 'max', False), TorchGraph(2, 2, 'max', False)
        with torch.no_grad():
            reference.lin_rel.weight.copy_(torch.eye(2))
            reference.lin_root.weight.zero_()
        self.copy(model, reference)
        x = torch.tensor([[[-2., 1.], [5., -3.], [5., 7.]]], requires_grad=True)
        adjacency = torch.tensor([[[1., 0., 1.], [1., 0., 0.], [1., 1., 0.]]], requires_grad=True)
        nx, na = native(x, True), native(adjacency, True)
        self.gradients(model, reference, model(nx, na), reference(x, adjacency), [('x', nx, x), ('adjacency', na, adjacency)])

    def test_empty_batch_and_node_axes_and_binary_masks(self):
        for xb, ab, nodes in ((0, 1, 3), (1, 0, 3), (2, 2, 0)):
            for model, reference in ((DenseGCNConv(3, 4), TorchGCN(3, 4)), (DenseSAGEConv(3, 4), TorchSAGE(3, 4)),
                                     (DenseGINConv(fr.nn.Linear(3, 4)), TorchGIN(torch.nn.Linear(3, 4))),
                                     (DenseGraphConv(3, 4), TorchGraph(3, 4))):
                with self.subTest(xb=xb, ab=ab, nodes=nodes, kind=type(model).__name__):
                    self.compare(model, reference, xb, ab, nodes, mask_kind=None)
        self.compare(DenseGraphConv(3, 4, 'max'), TorchGraph(3, 4, 'max'), xb=0, ab=1, mask_kind=None)
        self.compare(DenseGCNConv(3, 4), TorchGCN(3, 4), mask_kind='bool')

    def test_diffpool_to_densegcn_preserves_assignments_adjacency_and_parameter_gradients(self):
        model, reference = DenseGCNConv(3, 4), TorchGCN(3, 4)
        self.copy(model, reference)
        x = torch.randn(2, 5, 3, requires_grad=True)
        adjacency = torch.rand(2, 5, 5, requires_grad=True)
        logits = torch.randn(2, 5, 3, requires_grad=True)
        nx, na, ns = native(x, True), native(adjacency, True), native(logits, True)
        pooled, edges, link, entropy = fr.graph.dense_diff_pool(nx, na, ns)
        tpooled, tedges, tlink, tentropy = torch_diff_pool(x, adjacency, logits)
        actual = model(pooled, edges) + (link + entropy) * .1
        expected = reference(tpooled, tedges) + (tlink + tentropy) * .1
        self.gradients(model, reference, actual, expected, [('x', nx, x), ('adjacency', na, adjacency), ('assignments', ns, logits)])

    def test_strided_adjacency_and_masks_preserve_base_gradients(self):
        for model, reference in ((DenseGCNConv(3, 4), TorchGCN(3, 4)), (DenseSAGEConv(3, 4), TorchSAGE(3, 4)),
                                 (DenseGINConv(fr.nn.Linear(3, 4)), TorchGIN(torch.nn.Linear(3, 4))),
                                 (DenseGraphConv(3, 4, 'mean'), TorchGraph(3, 4, 'mean'))):
            with self.subTest(kind=type(model).__name__):
                self.copy(model, reference)
                x = torch.randn(2, 4, 3, requires_grad=True)
                base_adj = torch.rand(2, 4, 4, requires_grad=True)
                base_mask = torch.randn(4, 2, requires_grad=True)
                nx, na, nm = native(x, True), native(base_adj, True), native(base_mask, True)
                actual = model(nx, na.transpose(1, 2), nm.transpose(0, 1))
                expected = reference(x, base_adj.transpose(1, 2), base_mask.transpose(0, 1))
                self.gradients(model, reference, actual, expected, [('x', nx, x), ('adjacency base', na, base_adj), ('mask base', nm, base_mask)])

    def test_native_optimizer_checkpoint_aliases_epsilon_and_normalization_config(self):
        def factory(normalize=True):
            model = DenseSAGEConv(3, 4, normalize)
            model.lin_root.weight = model.lin_rel.weight
            return model
        model, rng = factory(), fr.Generator(83)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.01)
        def step(current, opt, generator):
            x = fr.Tensor.rand([2, 4, 3], generator=generator)
            adjacency = fr.Tensor.rand([2, 4, 4], generator=generator)
            result = current(x, adjacency)
            loss = result.square().mean()
            values = result.tolist(), loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return values
        step(model, optimizer, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'dense': optimizer}, generator=rng, step=1)
        expected = [step(model, optimizer, rng) for _ in range(2)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        state.restore(other, optimizers={'dense': other_opt}, generator=other_rng)
        self.assertIs(other.lin_root.weight, other.lin_rel.weight)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(2)])
        for left, right in zip(model.parameters(), other.parameters()):
            self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        changed = factory(False)
        with self.assertRaises(ValueError):
            state.restore(changed, optimizers={'dense': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        for trained in (True, False):
            gin = DenseGINConv(fr.nn.Linear(3, 4), .2, trained)
            state = fr.checkpoint.snapshot_training(gin, optimizers={}, generator=fr.Generator(1))
            with self.assertRaises(ValueError):
                state.restore(DenseGINConv(fr.nn.Linear(3, 4), .3, trained), optimizers={}, generator=fr.Generator(1))

    def test_strict_shapes_types_caps_masks_and_capture(self):
        for constructor in (lambda: DenseGCNConv(0, 4), lambda: DenseSAGEConv(3, -1), lambda: DenseGINConv(lambda x: x),
                            lambda: DenseGINConv(fr.nn.Linear(3, 4), eps=float('nan')), lambda: DenseGraphConv(3, 4, 'min'),
                            lambda: DenseGCNConv(sys.maxsize, 1), lambda: DenseSAGEConv(3, 4, normalize=1)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        x, adj = native(torch.randn(2, 4, 3)), native(torch.rand(2, 4, 4))
        model = DenseGCNConv(3, 4)
        for call in (lambda: model(x, native(torch.rand(2, 4, 3))), lambda: model(x, native(torch.rand(3, 4, 4))),
                     lambda: model(x, adj, native(torch.ones(2, 5))), lambda: model(x, adj, fr.Tensor.from_i64([1] * 8, [2, 4])),
                     lambda: model(x, adj, add_loop=1), lambda: model(x, fr.tensor([[[1.]]], dtype='float64')),
                     lambda: DenseGraphConv(3, 4, 'max')(native(torch.rand(1, 4, 3)), adj),
                     lambda: DenseGraphConv(3, 4, 'max')(fr.Tensor.zeros([1, 0, 3]), fr.Tensor.zeros([1, 0, 0]))):
            with self.assertRaises((ValueError, TypeError)):
                call()
        huge_x = fr.Tensor.zeros([1, 1, 1024]).expand(1, 2**27, 1024)
        huge_adj = fr.Tensor.zeros([1, 1, 1]).expand(1, 2**27, 2**27)
        with self.assertRaises(ValueError):
            DenseGraphConv(1024, 1, 'max')(huge_x, huge_adj)
        diagonal_x = fr.Tensor.zeros([1, 1, 1]).expand(1, 2**30, 1)
        diagonal_adj = fr.Tensor.zeros([1, 1, 1]).expand(1, 2**30, 2**30)
        with self.assertRaises(ValueError):
            DenseGCNConv(1, 1)(diagonal_x, diagonal_adj)
        huge_mask = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaises(ValueError):
            DenseGraphConv(1, 4)(fr.Tensor.ones([1, 1]), fr.Tensor.ones([1, 1]), huge_mask)
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(x, adj))
        with self.assertRaises(AttributeError):
            model.improved = True


if __name__ == '__main__':
    unittest.main(verbosity=2)

