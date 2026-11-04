"""Learned graph pooling and aggregation versus actual optional PyG."""
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.graph import (AttentionalAggregation, GraphConv, SAGPooling,
                         Set2Set, TopKPooling, dense_diff_pool)

try:
    from torch_geometric.nn import AttentionalAggregation as TorchAttention, GraphConv as TorchGraphConv
    from torch_geometric.nn import SAGPooling as TorchSAG, Set2Set as TorchSet2Set, TopKPooling as TorchTopK
    from torch_geometric.nn import dense_diff_pool as torch_diff_pool
except ImportError:
    TorchTopK = None


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def indices(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


@unittest.skipUnless(TorchTopK is not None, 'PyG is an optional test reference')
class PoolingTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected.detach(), atol=3e-5, rtol=3e-4)

    def copy_parameters(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(parameters[name].detach()))

    def gradients(self, model, reference, actual, expected, inputs):
        if not isinstance(actual, tuple):
            actual, expected = (actual,), (expected,)
        loss, target = None, None
        generator = torch.Generator().manual_seed(414)
        for result, wanted in zip(actual, expected):
            self.close(result, wanted)
            probe = torch.randn(wanted.shape, generator=generator)
            term, reference_term = (result * tensor(probe)).sum(), (wanted * probe).sum()
            loss = term if loss is None else loss + term
            target = reference_term if target is None else target + reference_term
        loss.backward()
        target.backward()
        parameters = {} if reference is None else dict(reference.named_parameters())
        pairs = list(inputs) + ([] if model is None else [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()])
        for name, result, wanted in pairs:
            with self.subTest(gradient=name):
                if wanted.grad is None:
                    self.assertIsNone(result.grad)
                else:
                    self.assertIsNotNone(result.grad)
                    self.close(result.grad, wanted.grad)

    def test_topk_and_sag_ratio_threshold_attention_edge_gradients(self):
        edges = torch.tensor([[0, 1, 2, 2, 3, 4, 5, 6, 7, 7], [1, 0, 3, 1, 2, 5, 4, 7, 6, 7]])
        batch = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
        for cls, reference_cls in ((TopKPooling, TorchTopK), (SAGPooling, TorchSAG)):
            for ratio, threshold, external in ((.6, None, False), (2, None, True), (None, .2, True), (1., None, False)):
                with self.subTest(layer=cls.__name__, ratio=ratio, threshold=threshold, external=external):
                    torch.manual_seed(311)
                    model, reference = cls(3, ratio, min_score=threshold, multiplier=1.3), reference_cls(3, ratio, min_score=threshold, multiplier=1.3)
                    self.copy_parameters(model, reference)
                    x, edge = torch.randn(8, 3, requires_grad=True), torch.randn(10, 2, requires_grad=True)
                    attention = torch.randn(8, 3, requires_grad=True) if external else None
                    sx, se, sa = tensor(x.detach(), True), tensor(edge.detach(), True), None if attention is None else tensor(attention.detach(), True)
                    got = model(sx, indices(edges), se, indices(batch), sa)
                    wanted = reference(x, edges, edge, batch, attention)
                    for i in (1, 3, 4):
                        self.assertEqual(got[i].tolist(), wanted[i].tolist())
                    self.gradients(model, reference, (got[0], got[2], got[5]), (wanted[0], wanted[2], wanted[5]),
                                   [('x', sx, x), ('edge', se, edge)] + ([] if attention is None else [('attn', sa, attention)]))

    def test_pooling_empty_ties_threshold_survivor_and_unsorted_batch(self):
        empty_edges = indices(torch.empty(2, 0, dtype=torch.int64))
        for cls in (TopKPooling, SAGPooling):
            model = cls(2)
            result = model(tensor(torch.empty(0, 2)), empty_edges, tensor(torch.empty(0, 3)))
            self.assertEqual([value.shape for value in result], [[0, 2], [2, 0], [0, 3], [0], [0], [0]])
        model = TopKPooling(2, ratio=2)
        with fr.no_grad():
            model.select.weight.tensor().copy_(tensor([[1., 0.]]))
        x = tensor([[.3, 1.], [.3, 2.], [.3, 3.], [.3, 4.]], True)
        result = model(x, empty_edges, batch=indices([1, 0, 1, 0]))
        self.assertEqual(result[4].tolist(), [1, 3, 0, 2])
        result[0].sum().backward()
        self.assertTrue(all(torch.isfinite(values(x.grad)).reshape(-1)))
        threshold = TopKPooling(2, ratio=None, min_score=1.)
        with fr.no_grad():
            threshold.select.weight.tensor().copy_(tensor([[1., 0.]]))
        retained = threshold(tensor([[.1, .3], [.4, .7], [.2, .9]]), empty_edges, batch=indices([0, 0, 1]))
        self.assertEqual(retained[4].tolist(), [1, 2])
        # PyG computes ceil(ratio * degree) in float32, not Python double.
        rounded = TopKPooling(2, ratio=.500000001)
        self.assertEqual(rounded(tensor([[.1, .3], [.4, .7]]), empty_edges)[0].shape[0], 1)

    def test_graphconv_bipartite_weighted_aggregations(self):
        edges = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 1, 1, 2]])
        for aggregation in ('add', 'mean', 'max'):
            with self.subTest(aggregation=aggregation):
                torch.manual_seed(319)
                model, reference = GraphConv((3, 2), 4, aggr=aggregation), TorchGraphConv((3, 2), 4, aggr=aggregation)
                self.copy_parameters(model, reference)
                source, target = torch.randn(5, 3, requires_grad=True), torch.randn(4, 2, requires_grad=True)
                weight = torch.rand(5, requires_grad=True)
                ss, st, sw = tensor(source.detach(), True), tensor(target.detach(), True), tensor(weight.detach(), True)
                self.gradients(model, reference, model((ss, st), indices(edges), sw), reference((source, target), edges, weight),
                               [('source', ss, source), ('target', st, target), ('weight', sw, weight)])

    def test_attention_aggregation_scalar_feature_gates_and_shared_alias(self):
        index = torch.tensor([2, 0, 2, 0, 3, 2])
        for feature_gate, shared in ((False, False), (True, False), (True, True)):
            with self.subTest(feature_gate=feature_gate, shared=shared):
                torch.manual_seed(331)
                gate = fr.nn.Linear(3, 4 if feature_gate else 1)
                torch_gate = torch.nn.Linear(3, 4 if feature_gate else 1)
                model = AttentionalAggregation(gate, gate if shared else fr.nn.Linear(3, 4))
                reference = TorchAttention(torch_gate, torch_gate if shared else torch.nn.Linear(3, 4))
                self.copy_parameters(model, reference)
                x = torch.randn(6, 3, requires_grad=True)
                sx = tensor(x.detach(), True)
                self.gradients(model, reference, model(sx, indices(index), dim_size=5), reference(x, index, dim_size=5), [('x', sx, x)])

    def test_attention_ptr_empty_inputs_and_groups(self):
        for nodes, ptr in ((4, [0, 2, 2, 4]), (0, [0, 0, 0])):
            with self.subTest(nodes=nodes):
                model, reference = AttentionalAggregation(fr.nn.Linear(3, 1)), TorchAttention(torch.nn.Linear(3, 1))
                self.copy_parameters(model, reference)
                x = torch.randn(nodes, 3, requires_grad=True)
                sx = tensor(x.detach(), True)
                self.gradients(model, reference, model(sx, ptr=indices(ptr)), reference(x, ptr=torch.tensor(ptr)), [('x', sx, x)])

    def test_set2set_stacked_lstm_gradients_and_empty_groups(self):
        for nodes, layers, bias in ((6, 1, True), (6, 2, False), (0, 2, True)):
            with self.subTest(nodes=nodes, layers=layers, bias=bias):
                torch.manual_seed(349)
                model, reference = Set2Set(3, 3, num_layers=layers, bias=bias), TorchSet2Set(3, 3, num_layers=layers, bias=bias)
                self.copy_parameters(model, reference)
                x = torch.randn(nodes, 3, requires_grad=True)
                index = torch.tensor([0, 2, 2, 3, 0, 2] if nodes else [], dtype=torch.int64)
                sx = tensor(x.detach(), True)
                self.gradients(model, reference, model(sx, indices(index), dim_size=5), reference(x, index, dim_size=5), [('x', sx, x)])

    def test_diffpool_dense_adjacency_assignment_mask_and_losses_gradients(self):
        for batched, normalized, masked in ((False, True, False), (True, True, True), (True, False, True)):
            with self.subTest(batched=batched, normalized=normalized, masked=masked):
                torch.manual_seed(359)
                shape = (2, 4) if batched else (4,)
                x = torch.randn(*shape, 3, requires_grad=True)
                adj = torch.rand(*shape, 4, requires_grad=True)
                assignment = torch.randn(*shape, 2, requires_grad=True)
                mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]], requires_grad=True) if masked else None
                sx, sadj, ss = tensor(x.detach(), True), tensor(adj.detach(), True), tensor(assignment.detach(), True)
                sm = None if mask is None else tensor(mask.detach(), True)
                self.gradients(None, None, dense_diff_pool(sx, sadj, ss, sm, normalized), torch_diff_pool(x, adj, assignment, mask, normalized),
                               [('x', sx, x), ('adj', sadj, adj), ('assignment', ss, assignment)] + ([] if mask is None else [('mask', sm, mask)]))

    def test_diffpool_bool_mask_and_exact_zero_link_subgradient(self):
        x, adj, assignment = torch.randn(3, 2, requires_grad=True), torch.ones(3, 3, requires_grad=True), torch.ones(3, 1, requires_grad=True)
        sx, sadj, ss = tensor(x.detach(), True), tensor(adj.detach(), True), tensor(assignment.detach(), True)
        self.gradients(None, None, dense_diff_pool(sx, sadj, ss), torch_diff_pool(x, adj, assignment), [('x', sx, x), ('adj', sadj, adj), ('s', ss, assignment)])
        mask = torch.tensor([True, True, False])
        native_mask = fr.Tensor.from_bool(mask.tolist(), [3])
        for got, wanted in zip(dense_diff_pool(tensor(x.detach()), tensor(adj.detach()), tensor(assignment.detach()), native_mask),
                               torch_diff_pool(x.detach(), adj.detach(), assignment.detach(), mask)):
            self.close(got, wanted)

    def test_pooling_checkpoint_aliases_and_configuration(self):
        def attention():
            layer = fr.nn.Linear(3, 3)
            return AttentionalAggregation(layer, layer)
        x, edge, batch = tensor(torch.arange(18).reshape(6, 3) / 10), indices([[0, 1, 2, 3, 4], [1, 0, 1, 4, 5]]), indices([0, 0, 0, 1, 1, 1])
        cases = ((lambda: TopKPooling(3), lambda m: m(x, edge, batch=batch)[0]),
                 (lambda: SAGPooling(3), lambda m: m(x, edge, batch=batch)[0]),
                 (attention, lambda m: m(x, batch)), (lambda: Set2Set(3, 2), lambda m: m(x, batch)))
        for factory, execute in cases:
            model = factory()
            with self.subTest(layer=type(model).__name__):
                optimizer = fr.optim.Adam(model.parameters(), lr=.01)
                execute(model).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
                snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=fr.Generator(11), step=1)
                with tempfile.TemporaryDirectory() as directory:
                    snapshot.save(directory)
                    other = factory()
                    other_optimizer = fr.optim.Adam(other.parameters(), lr=.9)
                    self.assertEqual(fr.checkpoint.load_training(directory, other, optimizers={'main': other_optimizer}, generator=fr.Generator(99)), 1)
                    self.assertEqual(execute(model).tolist(), execute(other).tolist())
                    for current, opt in ((model, optimizer), (other, other_optimizer)):
                        output = execute(current)
                        (output * output).sum().backward()
                        opt.step()
                        opt.zero_grad()
                    self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in other.parameters()])
        for source, target in ((TopKPooling(3, .5), TopKPooling(3, .7)), (SAGPooling(3, min_score=.1), SAGPooling(3, min_score=.2)),
                               (Set2Set(3, 2), Set2Set(3, 3)), (attention(), AttentionalAggregation(fr.nn.Linear(3, 3), fr.nn.Linear(3, 3)))):
            state = fr.checkpoint.snapshot_training(source, optimizers={}, generator=fr.Generator(1))
            before = [p.tensor().tolist() for p in target.parameters()]
            with self.assertRaises(ValueError):
                state.restore(target, optimizers={}, generator=fr.Generator(2))
            self.assertEqual(before, [p.tensor().tolist() for p in target.parameters()])

    def test_pooling_validation_dtype_bounds_and_unsupported_options(self):
        for factory in (lambda: TopKPooling(3, 0), lambda: TopKPooling(3, float('nan')), lambda: SAGPooling(3, min_score=2),
                        lambda: Set2Set(3, 0), lambda: TopKPooling(sys.maxsize), lambda: Set2Set(sys.maxsize, 2)):
            with self.assertRaises((ValueError, TypeError)):
                factory()
        with self.assertRaises(ValueError):
            Set2Set(3, 2, num_layers=sys.maxsize)
        with self.assertRaises(NotImplementedError):
            Set2Set(3, 2, bidirectional=True)
        x, edge = tensor(torch.ones(3, 3)), indices([[0], [1]])
        for model in (TopKPooling(3), SAGPooling(3)):
            with self.assertRaises(ValueError):
                model(x, indices([[0], [3]]))
            with self.assertRaises(ValueError):
                model(x, edge, batch=indices([0, 0]))
            with self.assertRaises(ValueError):
                model(fr.tensor([[1, 2, 3]] * 3, dtype='float64'), edge)
        with self.assertRaises(ValueError):
            AttentionalAggregation(fr.nn.Linear(3, 2))(x, indices([0, 0, 1]))
        with self.assertRaises(ValueError):
            Set2Set(3, 2)(x, ptr=indices([0, 2]))
        with self.assertRaises(ValueError):
            dense_diff_pool(tensor(torch.empty(0, 3)), tensor(torch.empty(0, 0)), tensor(torch.empty(0, 2)))
        with self.assertRaises(ValueError):
            dense_diff_pool(x, tensor(torch.ones(3, 3)), tensor(torch.ones(3, 2)), tensor([1., .5, 0.]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
