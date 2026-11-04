"""Actual PyG degree mixtures, feature-steered kernels and SSG references."""
import sys
import unittest

import torch
from torch_geometric.nn import MFConv as TorchMF, FeaStConv as TorchFeaSt, SSGConv as TorchSSG
import ferro as fr
from ferro.graph import MFConv, FeaStConv, SSGConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class MixtureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1237)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=3e-5, rtol=4e-4)

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
        for name, left, right in inputs + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if right.grad is None:
                    self.assertIsNone(left.grad)
                else:
                    self.assertIsNotNone(left.grad)
                    self.close(left.grad, right.grad)

    def test_degree_specific_maps_bipartite_missing_targets_clamps_and_gradients(self):
        for limit, target_kind, aggr, bias in ((0, 'same', 'add', True), (1, 'pair', 'mean', False),
                                               (4, 'same', 'max', False), (8, 'missing', 'min', True)):
            with self.subTest(limit=limit, target_kind=target_kind, aggr=aggr):
                width = (3, 2) if target_kind != 'same' else 3
                model, reference = MFConv(width, 4, limit, bias, aggr=aggr), TorchMF(width, 4, limit, bias, aggr=aggr)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if target_kind == 'pair' else None
                ns, nt = native(source, True), None if target is None else native(target, True)
                inputs = [('source', ns, source)] + ([] if target is None else [('target', nt, target)])
                actual_x = ns if target_kind == 'same' else (ns, nt)
                expected_x = source if target_kind == 'same' else (source, target)
                size = (6, 7) if target_kind == 'missing' else None
                self.gradients(model, reference, model(actual_x, native(self.edge), size),
                               reference(expected_x, self.edge, size), inputs)
                if target_kind == 'missing':
                    self.assertTrue(all(p.tensor().grad is None for p in model.lins_r.parameters()))

    def test_degree_empty_branches_keep_zero_gradients_and_zero_nodes(self):
        for nodes, edges in ((4, torch.empty(2, 0, dtype=torch.int64)), (0, torch.empty(2, 0, dtype=torch.int64))):
            model, reference = MFConv(3, 2, 3), TorchMF(3, 2, 3)
            self.copy(model, reference)
            x = torch.randn(nodes, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, reference, model(nx, native(edges)), reference(x, edges), [('x', nx, x)])
            for index in range(1, 4):
                self.assertEqual(model.lins_l[index].weight.tensor().grad.tolist(), [[0., 0., 0.], [0., 0., 0.]])

    def test_steered_head_assignments_bipartite_loops_reducers_all_gradients(self):
        for heads, source_nodes, target_nodes, loops, aggr, bias in ((1, 6, 6, True, 'mean', True),
                  (3, 6, 5, True, 'add', False), (2, 6, 7, False, 'max', True), (4, 6, 5, False, 'min', False)):
            with self.subTest(heads=heads, targets=target_nodes, loops=loops, aggr=aggr):
                model = FeaStConv(3, 4, heads, loops, bias, aggr=aggr)
                reference = TorchFeaSt(3, 4, heads, loops, bias, aggr=aggr)
                self.copy(model, reference)
                source = torch.randn(source_nodes, 3, requires_grad=True)
                target = source if source_nodes == target_nodes else torch.randn(target_nodes, 3, requires_grad=True)
                ns = native(source, True)
                nt = ns if target is source else native(target, True)
                inputs = [('source', ns, source)] + ([] if target is source else [('target', nt, target)])
                self.gradients(model, reference, model((ns, nt), native(self.edge)),
                               reference((source, target), self.edge), inputs)

    def test_steered_empty_edges_and_empty_nodes(self):
        edges = torch.empty(2, 0, dtype=torch.int64)
        for nodes, loops in ((0, False), (0, True), (4, False), (4, True)):
            model, reference = FeaStConv(3, 2, 2, loops), TorchFeaSt(3, 2, 2, loops)
            self.copy(model, reference)
            x = torch.randn(nodes, 3, requires_grad=True)
            nx = native(x, True)
            if not nodes or not loops:
                # PyG 2.8.1's view(0, heads, -1) rejects before aggregation.
                # Use its exact message algebra with an explicit empty width.
                with self.assertRaises(RuntimeError): reference(x, edges)
                empty = x[:0]
                q = (reference.u(empty - empty) + reference.c).softmax(1)
                messages = (reference.lin(empty).reshape(0, 2, 2) * q[:, :, None]).sum(1)
                expected = torch.zeros(nodes, 2).index_add(0, torch.empty(0, dtype=torch.int64), messages) + reference.bias
            else:
                expected = reference(x, edges)
            self.gradients(model, reference, model(nx, native(edges)), expected, [('x', nx, x)])

    def test_spectral_average_weighted_gradients_zero_hops_and_external_alpha(self):
        for hops, alpha, loops, weighted, bias in ((0, .3, True, True, True), (1, 0., False, False, False),
                   (3, .2, True, True, True), (4, 1., False, True, False), (2, -1.2, False, True, True)):
            with self.subTest(hops=hops, alpha=alpha, loops=loops, weighted=weighted):
                model, reference = SSGConv(3, 4, alpha, hops, add_self_loops=loops, bias=bias), TorchSSG(3, 4, alpha, hops, add_self_loops=loops, bias=bias)
                self.copy(model, reference)
                x = torch.randn(3, 6).T.detach().requires_grad_(True)
                nx = native(x.T).transpose(0, 1).requires_grad_(True)
                weight = (torch.rand(9) + .4).requires_grad_(True) if weighted else None
                nw = None if weight is None else native(weight, True)
                inputs = [('x', nx, x)] + ([] if weight is None else [('weight', nw, weight)])
                self.gradients(model, reference, model(nx, native(self.edge), nw), reference(x, self.edge, weight), inputs)

    def test_spectral_empties_and_loop_overwrite_gradient(self):
        for nodes, loops in ((0, False), (0, True), (4, False), (4, True)):
            model, reference = SSGConv(3, 2, .2, 3, add_self_loops=loops), TorchSSG(3, 2, .2, 3, add_self_loops=loops)
            self.copy(model, reference)
            x, edges = torch.randn(nodes, 3, requires_grad=True), torch.empty(2, 0, dtype=torch.int64)
            nx, weight = native(x, True), torch.empty(0, requires_grad=True)
            nw = native(weight, True)
            self.gradients(model, reference, model(nx, native(edges), nw), reference(x, edges, weight), [('x', nx, x), ('weight', nw, weight)])
        # Unique supplied diagonals survive remaining-loop normalization, including adjoints.
        edge = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 2]])
        x, weights = torch.randn(3, 3, requires_grad=True), torch.tensor([.4, .7, .8, 1.2], requires_grad=True)
        model, reference = SSGConv(3, 2, .4, 2), TorchSSG(3, 2, .4, 2)
        self.copy(model, reference)
        nx, nw = native(x, True), native(weights, True)
        self.gradients(model, reference, model(nx, native(edge), nw), reference(x, edge, weights), [('x', nx, x), ('weight', nw, weights)])

    def test_native_optimizer_rng_checkpoint_aliases_and_config(self):
        def factory(kind, changed=False):
            if kind == 'mf':
                model = MFConv(3, 2, 2, aggr='mean' if changed else 'add')
                model.lins_l[1].weight = model.lins_l[0].weight
            elif kind == 'feast':
                model = FeaStConv(3, 1, 2, add_self_loops=not changed)
                model.u.weight = model.lin.weight
            else:
                model = SSGConv(3, 2, .3 if changed else .2, 2)
            return model
        edge = native(self.edge)
        for kind in ('mf', 'feast', 'ssg'):
            with self.subTest(kind=kind):
                model, rng = factory(kind), fr.Generator(101)
                opt = fr.optim.AdamW(model.parameters(), lr=.01)
                def step(current, optimizer, generator):
                    x = fr.Tensor.rand([6, 3], generator=generator)
                    output = current(x, edge)
                    loss = output.square().mean()
                    result = output.tolist(), loss.item()
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    return result
                step(model, opt, rng)
                state = fr.checkpoint.snapshot_training(model, optimizers={'mixture': opt}, generator=rng, step=1)
                expected = [step(model, opt, rng) for _ in range(3)]
                other, other_rng = factory(kind), fr.Generator(999)
                other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
                state.restore(other, optimizers={'mixture': other_opt}, generator=other_rng)
                self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
                for left, right in zip(model.parameters(), other.parameters()):
                    self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
                if kind == 'mf': self.assertIs(other.lins_l[0].weight, other.lins_l[1].weight)
                if kind == 'feast': self.assertIs(other.lin.weight, other.u.weight)
                changed = factory(kind, True)
                before = [p.tensor().tolist() for p in changed.parameters()]
                with self.assertRaises(ValueError):
                    state.restore(changed, optimizers={'mixture': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
                self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_config_shape_bounds_capacity_and_capture_rejections(self):
        for constructor in (lambda: MFConv(3, 2, -1), lambda: MFConv(3, 2, True), lambda: MFConv(3, 2, sys.maxsize),
                            lambda: FeaStConv(3, 2, 0), lambda: FeaStConv(3, 2, sys.maxsize), lambda: FeaStConv(3, 2, bias=1),
                            lambda: SSGConv(3, 2, float('nan')), lambda: SSGConv(3, 2, True), lambda: SSGConv(3, 2, .2, -1)):
            with self.assertRaises((TypeError, ValueError)): constructor()
        for constructor in (lambda: MFConv(3, 2, flow='target_to_source'), lambda: FeaStConv(3, 2, aggr='mul'),
                            lambda: SSGConv(3, 2, .2, cached=True), lambda: SSGConv(3, 2, .2, aggr='mean')):
            with self.assertRaises(NotImplementedError): constructor()
        x, edge = native(torch.randn(6, 3)), native(self.edge)
        for call in (lambda: MFConv(3, 2)(x, edge, (6, 7)), lambda: FeaStConv(3, 2)((x, None), edge),
                     lambda: FeaStConv(3, 2)((x, native(torch.randn(7, 3))), edge),
                     lambda: SSGConv(3, 2, .2)(x, edge, native(torch.ones(8))),
                     lambda: SSGConv(3, 2, .2)(x, native(torch.tensor([[6], [0]])))):
            with self.assertRaises((TypeError, ValueError)): call()
        duplicate = native(torch.tensor([[0, 0], [0, 0]]))
        with self.assertRaises(NotImplementedError): SSGConv(3, 2, .2)(x, duplicate, native(torch.ones(2)))
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        empty = fr.Tensor.from_i64([], [2, 0])
        for model in (MFConv(1, 1), FeaStConv(1, 1), SSGConv(1, 1, .2)):
            with self.assertRaises(ValueError): model(huge, empty)
        for model in (MFConv(3, 2), FeaStConv(3, 2), SSGConv(3, 2, .2)):
            with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, edge))
            with self.assertRaises(AttributeError): model.in_channels = 4


if __name__ == '__main__':
    unittest.main(verbosity=2)
