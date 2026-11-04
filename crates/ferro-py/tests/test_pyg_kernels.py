"""Actual PyG Gaussian and learned-edge kernels, including all live adjoints."""
import sys
import unittest

import torch
from torch_geometric.nn import GMMConv as TorchGMM, PDNConv as TorchPDN
import ferro as fr
from ferro.graph import GMMConv, PDNConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1327)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=4e-5, rtol=5e-4)

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

    def test_gaussian_shared_and_separate_bipartite_all_parameter_coordinate_gradients(self):
        for separate, target_kind, kernels, dim, aggr, root, bias in (
                (False, 'same', 1, 1, 'mean', True, True), (False, 'pair', 3, 2, 'add', True, False),
                (True, 'same', 3, 2, 'mean', True, True), (True, 'pair', 2, 3, 'max', False, False),
                (True, 'missing', 2, 1, 'min', True, True), (False, 'missing', 2, 2, 'add', False, True)):
            with self.subTest(separate=separate, target=target_kind, aggr=aggr, root=root):
                widths = 3 if target_kind == 'same' else (3, 2)
                model = GMMConv(widths, 4, dim, kernels, separate, aggr, root, bias)
                reference = TorchGMM(widths, 4, dim, kernels, separate, aggr, root, bias)
                # PyG 2.8.1 omits self.root instead of registering None.
                if not root: reference.root = None
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if target_kind == 'pair' else None
                coordinate = torch.randn(9, dim, requires_grad=True)
                ns, nt, nc = native(source, True), None if target is None else native(target, True), native(coordinate, True)
                inputs = [('source', ns, source), ('coordinates', nc, coordinate)] + ([] if target is None else [('target', nt, target)])
                size = (6, 7) if target_kind == 'missing' else None
                actual_x = ns if target_kind == 'same' else (ns, nt)
                expected_x = source if target_kind == 'same' else (source, target)
                self.gradients(model, reference, model(actual_x, native(self.edge), nc, size),
                               reference(expected_x, self.edge, coordinate, size), inputs)

    def test_gaussian_empty_graphs_and_documented_disabled_root_reference_bug(self):
        x = torch.randn(6, 3)
        reference = TorchGMM(3, 2, 2, 2, root_weight=False)
        with self.assertRaisesRegex(AttributeError, 'root'):
            reference(x, self.edge, torch.randn(9, 2))
        for nodes in (0, 4):
            for separate in (False, True):
                model, reference = GMMConv(3, 2, 2, 2, separate), TorchGMM(3, 2, 2, 2, separate)
                self.copy(model, reference)
                x = torch.randn(nodes, 3, requires_grad=True)
                coordinate = torch.empty(0, 2, requires_grad=True)
                edges = torch.empty(2, 0, dtype=torch.int64)
                nx, nc = native(x, True), native(coordinate, True)
                self.gradients(model, reference, model(nx, native(edges), nc), reference(x, edges, coordinate), [('x', nx, x), ('coordinates', nc, coordinate)])

    def test_gaussian_zero_sigma_uses_finite_epsilon_and_empty_target_root(self):
        model, reference = GMMConv(2, 3, 1, 2), TorchGMM(2, 3, 1, 2)
        with torch.no_grad():
            reference.mu.zero_(); reference.sigma.zero_()
        self.copy(model, reference)
        x = torch.randn(3, 2, requires_grad=True)
        coords = torch.tensor([[0.], [.2]], requires_grad=True)
        edge = torch.tensor([[0, 1], [1, 2]])
        nx, nc = native(x, True), native(coords, True)
        self.gradients(model, reference, model(nx, native(edge), nc), reference(x, edge, coords), [('x', nx, x), ('coordinates', nc, coords)])

    def test_learned_edge_network_normalization_loop_and_all_gradients(self):
        for normalize, loops, attributes, bias in ((True, True, True, True), (True, False, True, False),
                                                   (False, True, True, True), (True, True, False, False)):
            with self.subTest(normalize=normalize, loops=loops, attributes=attributes):
                model = PDNConv(3, 4, 2, 5, loops, normalize, bias)
                reference = TorchPDN(3, 4, 2, 5, loops, normalize, bias)
                self.copy(model, reference)
                x = torch.randn(3, 6).T.detach().requires_grad_(True)
                nx = native(x.T).transpose(0, 1).requires_grad_(True)
                edge_attr = torch.randn(9, 2, requires_grad=True) if attributes else None
                na = None if edge_attr is None else native(edge_attr, True)
                inputs = [('x', nx, x)] + ([] if edge_attr is None else [('edge_attr', na, edge_attr)])
                self.gradients(model, reference, model(nx, native(self.edge), na), reference(x, self.edge, edge_attr), inputs)
                if not attributes:
                    self.assertTrue(all(parameter.tensor().grad is None for parameter in model.mlp.parameters()))

    def test_learned_edge_empty_attributes_and_topology(self):
        for nodes, normalize, loops in ((0, True, True), (0, False, False), (4, True, False), (4, True, True), (4, False, True)):
            model, reference = PDNConv(3, 2, 2, 4, loops, normalize), TorchPDN(3, 2, 2, 4, loops, normalize)
            self.copy(model, reference)
            x, attrs = torch.randn(nodes, 3, requires_grad=True), torch.empty(0, 2, requires_grad=True)
            nx, na = native(x, True), native(attrs, True)
            edge = torch.empty(2, 0, dtype=torch.int64)
            self.gradients(model, reference, model(nx, native(edge), na), reference(x, edge, attrs), [('x', nx, x), ('edge_attr', na, attrs)])

    def test_native_optimizer_checkpoint_aliases_and_configuration(self):
        def factory(kind, changed=False):
            if kind == 'gmm':
                model = GMMConv(3, 2, 3, 2, aggr='add' if changed else 'mean')
                model.sigma = model.mu
            else:
                model = PDNConv(3, 2, 3, 2, normalize=not changed)
                model.mlp[0].weight = model.lin.weight
            return model
        edge = native(self.edge)
        for kind in ('gmm', 'pdn'):
            model, rng = factory(kind), fr.Generator(79)
            opt = fr.optim.AdamW(model.parameters(), lr=.005)
            def step(current, optimizer, generator):
                x = fr.Tensor.rand([6, 3], generator=generator)
                attrs = fr.Tensor.rand([9, 3], generator=generator)
                result = current(x, edge, attrs)
                loss = result.square().mean()
                values = result.tolist(), loss.item()
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                return values
            step(model, opt, rng)
            state = fr.checkpoint.snapshot_training(model, optimizers={'kernel': opt}, generator=rng, step=1)
            expected = [step(model, opt, rng) for _ in range(3)]
            other, other_rng = factory(kind), fr.Generator(999)
            other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
            state.restore(other, optimizers={'kernel': other_opt}, generator=other_rng)
            self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
            for left, right in zip(model.parameters(), other.parameters()):
                self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
            if kind == 'gmm': self.assertIs(other.mu, other.sigma)
            else: self.assertIs(other.lin.weight, other.mlp[0].weight)
            changed = factory(kind, True)
            before = [parameter.tensor().tolist() for parameter in changed.parameters()]
            with self.assertRaises(ValueError):
                state.restore(changed, optimizers={'kernel': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
            self.assertEqual(before, [parameter.tensor().tolist() for parameter in changed.parameters()])

    def test_strict_shapes_configuration_capacity_and_capture(self):
        for constructor in (lambda: GMMConv(3, 2, 0, 2), lambda: GMMConv(3, 2, 2, 0), lambda: GMMConv(3, 2, 2, sys.maxsize),
                            lambda: GMMConv(3, 2, 2, 2, separate_gaussians=1), lambda: PDNConv(3, 2, 0, 4),
                            lambda: PDNConv(3, 2, 2, 0), lambda: PDNConv(3, 2, 2, sys.maxsize)):
            with self.assertRaises((ValueError, TypeError)): constructor()
        for constructor in (lambda: GMMConv(3, 2, 2, 2, aggr='mul'), lambda: PDNConv(3, 2, 2, 4, aggr='mean')):
            with self.assertRaises(NotImplementedError): constructor()
        x, edge, attrs = native(torch.randn(6, 3)), native(self.edge), native(torch.randn(9, 2))
        for call in (lambda: GMMConv(3, 2, 2, 2)(x, edge), lambda: GMMConv(3, 2, 2, 2)(x, edge, native(torch.randn(8, 2))),
                     lambda: GMMConv(3, 2, 2, 2)(x, edge, attrs, (6, 7)), lambda: PDNConv(3, 2, 2, 4, normalize=False)(x, edge),
                     lambda: PDNConv(3, 2, 2, 4)(x, edge, native(torch.randn(9, 3)))):
            with self.assertRaises((ValueError, TypeError)): call()
        duplicate = native(torch.tensor([[0, 0], [0, 0]]))
        with self.assertRaises(NotImplementedError): PDNConv(3, 2, 2, 4)(x, duplicate, native(torch.randn(2, 2)))
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        empty = fr.Tensor.from_i64([], [2, 0])
        for model in (GMMConv(1, 1, 1, 1), PDNConv(1, 1, 1, 1)):
            with self.assertRaises(ValueError): model(huge, empty, fr.Tensor.zeros([0, 1]))
        for model in (GMMConv(3, 2, 2, 2), PDNConv(3, 2, 2, 4)):
            with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, edge, attrs))
            with self.assertRaises(AttributeError): model.in_channels = 4


if __name__ == '__main__':
    unittest.main(verbosity=2)
