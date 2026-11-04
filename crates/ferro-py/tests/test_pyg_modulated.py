"""Actual PyG references for feature modulation, crystal filters and FA attention."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import FiLMConv as TorchFiLM, CGConv as TorchCG, FAConv as TorchFA
import ferro as fr
from ferro.graph import FiLMConv, CGConv, FAConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def torch_value(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class ModulatedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(967)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2, 1], [1, 2, 0, 3, 4, 0, 1, 2, 2, 1]])

    def close(self, actual, expected):
        torch.testing.assert_close(torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4)

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))

    def gradients(self, model, reference, actual, expected, inputs, alpha=None, reference_alpha=None):
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        loss, reference_loss = (actual * native(probe)).sum(), (expected * probe).sum()
        if alpha is not None:
            self.close(alpha, reference_alpha)
            ap = torch.randn(reference_alpha.shape)
            loss = loss + (alpha * native(ap)).sum()
            reference_loss = reference_loss + (reference_alpha * ap).sum()
        loss.backward()
        reference_loss.backward()
        parameters = dict(reference.named_parameters())
        for name, actual, expected in inputs + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    self.assertIsNotNone(actual.grad)
                    self.close(actual.grad, expected.grad)

    def test_film_relations_bipartite_activation_and_all_gradients(self):
        for relations, bipartite, aggr, act in ((1, False, 'mean', 'relu'), (3, True, 'add', None),
                                               (3, False, 'max', 'tanh'), (2, True, 'min', 'relu')):
            with self.subTest(relations=relations, bipartite=bipartite, aggr=aggr, act=act):
                width = (3, 2) if bipartite else 3
                activation = None if act is None else torch.nn.Tanh() if act == 'tanh' else torch.nn.ReLU()
                model = FiLMConv(width, 4, relations, act=act, aggr=aggr)
                reference = TorchFiLM(width, 4, relations, act=activation, aggr=aggr)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else source
                ns, nt = native(source, True), None
                nt = native(target, True) if bipartite else ns
                types = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0, 1, 0]) if relations > 1 else None
                ntypes = None if types is None else native(types)
                inputs = [('source', ns, source)] + ([('target', nt, target)] if bipartite else [])
                self.gradients(model, reference, model((ns, nt) if bipartite else ns, native(self.edge), ntypes),
                               reference((source, target) if bipartite else source, self.edge, types), inputs)

    def test_film_known_template_custom_modulation_gradients_and_independent_clones(self):
        template = fr.nn.Sequential(fr.nn.Linear(3, 6), fr.nn.Tanh(), fr.nn.Linear(6, 8))
        oracle_template = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Tanh(), torch.nn.Linear(6, 8))
        model, reference = FiLMConv(3, 4, 2, nn=template), TorchFiLM(3, 4, 2, nn=oracle_template)
        self.copy(model, reference)
        x = torch.randn(6, 3, requires_grad=True)
        nx, types = native(x, True), torch.tensor([0, 1] * 5)
        self.gradients(model, reference, model(nx, native(self.edge), native(types)), reference(x, self.edge, types), [('x', nx, x)])
        template = fr.nn.Sequential(fr.nn.Linear(3, 8), fr.nn.Dropout(.2, seed=17))
        template[0].weight.trainable = False
        template[1].eval()
        original = template[0].weight.tensor().tolist()
        model = FiLMConv(3, 4, 2, nn=template)
        self.assertEqual(original, template[0].weight.tensor().tolist())
        self.assertFalse(model.films[0][0].weight.trainable)
        self.assertFalse(model.films[0][1].training)
        before = model.films[1][0].weight.tensor().tolist()
        stream = model.films[1][1].rng_state.tolist()
        with fr.no_grad():
            model.films[0][0].weight.tensor().copy_(fr.Tensor.zeros([8, 3]))
            model.films[0][1].rng_state.copy_(fr.nn.Dropout(.2, seed=19).rng_state)
        self.assertEqual(before, model.films[1][0].weight.tensor().tolist())
        self.assertEqual(stream, model.films[1][1].rng_state.tolist())
        self.assertEqual(stream, template[1].rng_state.tolist())
        pending = fr.nn.Linear(3, 8)
        pending(fr.Tensor.ones([2, 3])).sum().backward()
        with self.assertRaises(ValueError):
            FiLMConv(3, 4, nn=pending)
        tied = fr.nn.Sequential(fr.nn.Linear(8, 8), fr.nn.Linear(8, 8))
        tied[1].weight = tied[0].weight
        with self.assertRaises(ValueError):
            FiLMConv(8, 4, nn=tied)

    def test_crystal_filter_edge_features_bipartite_reducers_and_all_gradients(self):
        for bipartite, dim, bias, aggr in ((False, 0, True, 'add'), (True, 2, False, 'mean'),
                                          (False, 2, True, 'max'), (True, 0, False, 'min')):
            with self.subTest(bipartite=bipartite, dim=dim, bias=bias, aggr=aggr):
                widths = (3, 2) if bipartite else 3
                model, reference = CGConv(widths, dim, aggr, bias=bias), TorchCG(widths, dim, aggr, bias=bias)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, 2, requires_grad=True) if bipartite else source
                attr = torch.randn(10, dim, requires_grad=True) if dim else None
                ns, nt, na = native(source, True), None, None if attr is None else native(attr, True)
                nt = native(target, True) if bipartite else ns
                inputs = [('source', ns, source)] + ([('target', nt, target)] if bipartite else []) + ([] if attr is None else [('edge_attr', na, attr)])
                self.gradients(model, reference, model((ns, nt) if bipartite else ns, native(self.edge), na),
                               reference((source, target) if bipartite else source, self.edge, attr), inputs)

    def fa_compare(self, model, reference, nodes=6, edge=None, flag=True):
        self.copy(model, reference)
        edge = self.edge if edge is None else edge
        x, initial = torch.randn(nodes, 3, requires_grad=True), torch.randn(nodes, 3, requires_grad=True)
        weight = None if model.normalize else torch.randn(edge.shape[1], requires_grad=True)
        nx, ni, nw = native(x, True), native(initial, True), None if weight is None else native(weight, True)
        actual = model(nx, ni, native(edge), nw, flag)
        expected = reference(x, initial, edge, weight, flag)
        inputs = [('x', nx, x), ('initial', ni, initial)] + ([] if weight is None else [('weights', nw, weight)])
        if flag is None:
            self.gradients(model, reference, actual, expected, inputs)
        else:
            self.assertEqual(actual[1][0].tolist(), expected[1][0].tolist())
            self.gradients(model, reference, actual[0], expected[0], inputs, actual[1][1], expected[1][1])

    def test_frequency_attention_raw_coefficients_weights_and_initial_feature_gradients(self):
        for normalize, loops, eps, flag in ((True, True, .1, True), (True, False, 0., False),
                                            (False, True, -.3, False), (False, False, .2, None)):
            with self.subTest(normalize=normalize, loops=loops, eps=eps, flag=flag):
                options = dict(normalize=normalize, add_self_loops=loops, eps=eps)
                self.fa_compare(FAConv(3, **options), TorchFA(3, **options), flag=flag)

    def test_frequency_nonzero_dropout_same_masks_and_unit_dropout(self):
        model, reference = FAConv(3, dropout=.35, normalize=False), TorchFA(3, dropout=.35, normalize=False)
        sampler = fr.nn.Dropout(.35, seed=1)
        with fr.no_grad():
            sampler.rng_state.copy_(model.attention_dropout.rng_state)
        mask = torch_value(sampler(fr.Tensor.ones([10])))
        calls = []
        def fixed_dropout(value, p=.5, training=True, inplace=False):
            self.assertEqual(p, .35)
            self.assertTrue(training)
            self.assertEqual(list(value.shape), [10])
            calls.append(1)
            return value * mask
        with patch('torch.nn.functional.dropout', side_effect=fixed_dropout):
            self.fa_compare(model, reference, flag=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sampler.rng_state.tolist(), model.attention_dropout.rng_state.tolist())
        self.fa_compare(FAConv(3, dropout=1.), TorchFA(3, dropout=1.))
        model, reference = FAConv(3, dropout=.4), TorchFA(3, dropout=.4)
        model.eval(); reference.eval()
        self.fa_compare(model, reference)

    def test_empty_graphs_and_isolated_nodes(self):
        edge = torch.empty(2, 0, dtype=torch.int64)
        for nodes in (0, 4):
            for model, reference in ((FiLMConv(3, 4), TorchFiLM(3, 4)), (CGConv(3), TorchCG(3))):
                with self.subTest(nodes=nodes, kind=type(model).__name__):
                    self.copy(model, reference)
                    x = torch.randn(nodes, 3, requires_grad=True)
                    nx = native(x, True)
                    self.gradients(model, reference, model(nx, native(edge)), reference(x, edge), [('x', nx, x)])
            self.fa_compare(FAConv(3), TorchFA(3), nodes, edge)
            self.fa_compare(FAConv(3, normalize=False), TorchFA(3, normalize=False), nodes, edge, flag=False)

    def test_native_dropout_optimizer_checkpoint_and_configuration(self):
        def factory(dropout=.25):
            model = FAConv(3, dropout=dropout, normalize=False)
            model.att_r.weight = model.att_l.weight
            return model
        model, rng = factory(), fr.Generator(101)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.01)
        edge = native(self.edge)
        def step(current, opt, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            initial = fr.Tensor.rand([6, 3], generator=generator)
            weights = fr.Tensor.rand([10], generator=generator)
            result = current(x, initial, edge, weights)
            loss = result.square().mean()
            values = result.tolist(), loss.item()
            loss.backward(); opt.step(); opt.zero_grad()
            return values
        step(model, optimizer, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'fa': optimizer}, generator=rng, step=1)
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        state.restore(other, optimizers={'fa': other_opt}, generator=other_rng)
        self.assertIs(other.att_l.weight, other.att_r.weight)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertEqual(model.attention_dropout.rng_state.tolist(), other.attention_dropout.rng_state.tolist())
        for left, right in zip(model.parameters(), other.parameters()):
            self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        changed = factory(.3)
        before = changed.attention_dropout.rng_state.tolist()
        with self.assertRaises(ValueError):
            state.restore(changed, optimizers={'fa': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, changed.attention_dropout.rng_state.tolist())

    def test_strict_config_relation_shape_capacity_and_invalid_stream_boundaries(self):
        for constructor in (lambda: FiLMConv(3, 4, 0), lambda: FiLMConv(-1, 4), lambda: CGConv(3, dim=-1),
                            lambda: FAConv(0), lambda: FAConv(3, eps=float('nan')), lambda: FAConv(3, dropout=float('inf')),
                            lambda: FiLMConv(3, 4, sys.maxsize), lambda: CGConv(sys.maxsize)):
            with self.assertRaises((ValueError, TypeError)):
                constructor()
        for constructor in (lambda: FAConv(3, cached=True),
                            lambda: FiLMConv(3, 4, nn=fr.nn.LayerNorm(3)), lambda: FAConv(3, flow='target_to_source')):
            with self.assertRaises(NotImplementedError):
                constructor()
        x, initial, edge = native(torch.randn(6, 3)), native(torch.randn(6, 3)), native(self.edge)
        for call in (lambda: FiLMConv(3, 4, 2)(x, edge), lambda: FiLMConv(3, 4, 2)(x, edge, native(torch.ones(10))),
                     lambda: FiLMConv(3, 4, 2)(x, edge, native(torch.tensor([2] * 10))),
                     lambda: CGConv(3, dim=2)(x, edge), lambda: CGConv(3, dim=2)(x, edge, native(torch.randn(9, 2)))):
            with self.assertRaises((ValueError, TypeError)):
                call()
        model = FAConv(3, dropout=.2)
        before = model.attention_dropout.rng_state.tolist()
        for call in (lambda: model(x, initial, edge, native(torch.ones(10))), lambda: model(x, initial, edge, return_attention_weights=1),
                     lambda: model(x, native(torch.ones(5, 3)), edge), lambda: model(x, initial, native(torch.tensor([[6], [0]])))):
            with self.assertRaises((ValueError, TypeError)):
                call()
            self.assertEqual(before, model.attention_dropout.rng_state.tolist())
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaises(ValueError):
            FAConv(1)(huge, huge, fr.Tensor.from_i64([], [2, 0]))
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(x, initial, edge))
        with self.assertRaises(AttributeError):
            model.normalize = False


if __name__ == '__main__':
    unittest.main(verbosity=2)
