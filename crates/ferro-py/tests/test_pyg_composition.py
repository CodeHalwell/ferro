"""Actual PyG JumpingKnowledge and GeneralConv output and adjoint references."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import JumpingKnowledge as TorchJK, GeneralConv as TorchGeneral
import ferro as fr
from ferro.graph import JumpingKnowledge, GeneralConv


def native(value, grad=False):
    if value.dtype == torch.int64:
        return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class CompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1439)
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

    def test_jumping_cat_max_lstm_actual_reference_all_inputs_parameters(self):
        for mode, channels, layers, widths in (('cat', None, None, [2, 3, 4]), ('max', None, None, [3, 3, 3]),
                  ('lstm', 3, 3, [3, 3, 3]), ('lstm', 2, 3, [2, 2]), ('lstm', 3, 1, [3, 3, 3, 3])):
            with self.subTest(mode=mode, channels=channels, layers=layers):
                model, reference = JumpingKnowledge(mode, channels, layers), TorchJK(mode, channels, layers)
                self.copy(model, reference)
                xs = [torch.randn(width, 5).T.detach().requires_grad_(True) for width in widths]
                nxs = [native(x.T).transpose(0, 1).requires_grad_(True) for x in xs]
                self.gradients(model, reference, model(nxs), reference(xs), [(str(i), nx, x) for i, (nx, x) in enumerate(zip(nxs, xs))])

    def test_jumping_max_first_layer_ties_and_single_input(self):
        values = [torch.tensor([[1., 2.], [3., 4.]], requires_grad=True), torch.tensor([[1., 3.], [2., 4.]], requires_grad=True)]
        nvalues = [native(x, True) for x in values]
        model, reference = JumpingKnowledge('max'), TorchJK('max')
        model(nvalues).sum().backward(); reference(values).sum().backward()
        for nx, x in zip(nvalues, values): self.close(nx.grad, x.grad)
        self.assertEqual(nvalues[0].grad.tolist(), [[1., 0.], [1., 1.]])
        for mode in ('cat', 'max', 'lstm'):
            model, reference = JumpingKnowledge(mode, 3, 2), TorchJK(mode, 3, 2)
            self.copy(model, reference)
            x = torch.randn(4, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, reference, model([nx]), reference([x]), [('x', nx, x)])

    def test_jumping_empty_nodes_and_zero_feature_width(self):
        for mode, nodes, channels in (('cat', 0, 3), ('max', 0, 3), ('lstm', 0, 3), ('cat', 4, 0), ('max', 4, 0)):
            model, reference = JumpingKnowledge(mode, 3, 2), TorchJK(mode, 3, 2)
            self.copy(model, reference)
            xs = [torch.randn(nodes, channels, requires_grad=True) for _ in range(2)]
            nxs = [native(x, True) for x in xs]
            self.gradients(model, reference, model(nxs), reference(xs), [(str(i), nx, x) for i, (nx, x) in enumerate(zip(nxs, xs))])

    def test_general_directed_symmetric_additive_dot_all_input_edge_parameter_gradients(self):
        cases = [(3, 3, True, 1, False, 'additive', False, False, 'add', True, False),
                 ((3, 2), 4, True, 2, True, 'additive', False, True, 'mean', False, True),
                 ((3, 3), 4, False, 3, True, 'additive', True, False, 'add', True, True),
                 (3, 3, True, 2, True, 'dot_product', True, True, 'max', True, True),
                 ((3, 3), 2, False, 2, True, 'dot_product', False, False, 'min', False, False),
                 ((3, 3), 3, False, 1, False, 'additive', False, False, 'mean', True, True)]
        for widths, output, directed, heads, attention, kind, normalize, skip, aggr, bias, edges in cases:
            with self.subTest(widths=widths, directed=directed, heads=heads, attention=attention, kind=kind, aggr=aggr):
                options = dict(in_edge_channels=2 if edges else None, directed_msg=directed, heads=heads, attention=attention,
                               attention_type=kind, l2_normalize=normalize, skip_linear=skip, aggr=aggr, bias=bias)
                model, reference = GeneralConv(widths, output, **options), TorchGeneral(widths, output, **options)
                self.copy(model, reference)
                source = torch.randn(6, 3, requires_grad=True)
                target = torch.randn(5, widths[1], requires_grad=True) if isinstance(widths, tuple) else source
                attrs = torch.randn(9, 2, requires_grad=True) if edges else None
                ns, nt, na = native(source, True), None, None if attrs is None else native(attrs, True)
                nt = ns if target is source else native(target, True)
                inputs = [('source', ns, source)] + ([] if source is target else [('target', nt, target)]) + ([] if attrs is None else [('attributes', na, attrs)])
                self.gradients(model, reference, model((ns, nt), native(self.edge), na), reference((source, target), self.edge, attrs), inputs)

    def test_general_empty_messages_isolates_optional_edge_network_and_normalize_zero(self):
        edge = torch.empty(2, 0, dtype=torch.int64)
        for nodes, attention, kind, directed in ((0, False, 'additive', True), (0, True, 'dot_product', False),
                                                (4, True, 'additive', False), (4, True, 'dot_product', True)):
            options = dict(in_edge_channels=2, attention=attention, attention_type=kind, directed_msg=directed, heads=2, l2_normalize=True)
            model, reference = GeneralConv(3, 2, **options), TorchGeneral(3, 2, **options)
            self.copy(model, reference)
            x, attrs = torch.randn(nodes, 3, requires_grad=True), torch.empty(0, 2, requires_grad=True)
            nx, na = native(x, True), native(attrs, True)
            self.gradients(model, reference, model(nx, native(edge), na), reference(x, edge, attrs), [('x', nx, x), ('edge_attr', na, attrs)])
        model, reference = GeneralConv(3, 3, in_edge_channels=2, l2_normalize=True, bias=False), TorchGeneral(3, 3, in_edge_channels=2, l2_normalize=True, bias=False)
        self.copy(model, reference)
        x = torch.zeros(4, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, reference, model(nx, native(edge)), reference(x, edge), [('x', nx, x)])
        self.assertTrue(all(parameter.tensor().grad is None for parameter in model.lin_edge.parameters()))

    def test_general_symmetric_constructor_initializes_bias_and_later_reset_preserves_it(self):
        model = GeneralConv(3, 4, heads=2, directed_msg=False)
        values = model.lin_msg_i.bias.tensor().tolist()
        self.assertTrue(any(value != 0. for value in values))
        self.assertTrue(all(abs(value) <= 3 ** -.5 for value in values))
        with fr.no_grad():
            model.lin_msg_i.bias.tensor().copy_(fr.Tensor.ones([8]) * .17)
            model.lin_msg_i.weight.tensor().copy_(fr.Tensor.ones([8, 3]) * -.21)
        reference = TorchGeneral(3, 4, heads=2, directed_msg=False)
        with torch.no_grad():
            reference.lin_msg_i.bias.fill_(.17)
            reference.lin_msg_i.weight.fill_(-.21)
        model.reset_parameters(); reference.reset_parameters()
        self.close(model.lin_msg_i.bias.tensor(), reference.lin_msg_i.bias)
        self.close(model.lin_msg_i.weight.tensor(), reference.lin_msg_i.weight)

    def test_native_optimizer_checkpoint_recurrent_state_aliases_and_configuration(self):
        def factory(kind, changed=False):
            if kind == 'jk':
                return JumpingKnowledge('lstm', 3, 3 if changed else 2)
            model = GeneralConv(3, 3, heads=2, attention=True, attention_type='dot_product', directed_msg=False, aggr='mean' if changed else 'add')
            model.lin_msg_i.weight = model.lin_msg.weight
            return model
        edge = native(self.edge)
        for kind in ('jk', 'general'):
            model, rng = factory(kind), fr.Generator(103)
            opt = fr.optim.AdamW(model.parameters(), lr=.005)
            def step(current, optimizer, generator):
                x = fr.Tensor.rand([6, 3], generator=generator)
                output = current([x, x * .7, x * -.3]) if kind == 'jk' else current(x, edge)
                loss = output.square().mean()
                values = output.tolist(), loss.item()
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                return values
            step(model, opt, rng)
            state = fr.checkpoint.snapshot_training(model, optimizers={'compose': opt}, generator=rng, step=1)
            expected = [step(model, opt, rng) for _ in range(3)]
            other, other_rng = factory(kind), fr.Generator(999)
            other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
            state.restore(other, optimizers={'compose': other_opt}, generator=other_rng)
            self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
            for left, right in zip(model.parameters(), other.parameters()):
                self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
            if kind == 'general': self.assertIs(other.lin_msg.weight, other.lin_msg_i.weight)
            changed = factory(kind, True)
            before = [parameter.tensor().tolist() for parameter in changed.parameters()]
            with self.assertRaises(ValueError):
                state.restore(changed, optimizers={'compose': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
            self.assertEqual(before, [parameter.tensor().tolist() for parameter in changed.parameters()])

    def test_config_shape_capacity_and_capture_rejections(self):
        for constructor in (lambda: JumpingKnowledge('sum'), lambda: JumpingKnowledge('lstm'), lambda: JumpingKnowledge('lstm', 1, 1),
                            lambda: JumpingKnowledge('lstm', 3, sys.maxsize), lambda: GeneralConv(3, None), lambda: GeneralConv(3, 2, heads=0),
                            lambda: GeneralConv((3, 2), 2, directed_msg=False), lambda: GeneralConv((3, 2), 2, attention=True, attention_type='dot_product')):
            with self.assertRaises((ValueError, TypeError)): constructor()
        x, edge = native(torch.randn(6, 3)), native(self.edge)
        for call in (lambda: JumpingKnowledge('cat')([]), lambda: JumpingKnowledge('max')([x, native(torch.randn(6, 2))]),
                     lambda: JumpingKnowledge('cat')([x, native(torch.randn(5, 3))]), lambda: JumpingKnowledge('lstm', 2, 2)([x]),
                     lambda: GeneralConv(3, 2)((x, None), edge), lambda: GeneralConv(3, 2)(x, edge, native(torch.randn(9, 2))),
                     lambda: GeneralConv(3, 2, in_edge_channels=2)(x, edge, native(torch.randn(8, 2)))):
            with self.assertRaises((TypeError, ValueError)): call()
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaises(ValueError): GeneralConv(1, 1)(huge, fr.Tensor.from_i64([], [2, 0]))
        with self.assertRaises(ValueError): JumpingKnowledge('max')([huge, huge])
        wide = fr.Tensor.ones([1, 1]).expand(1, sys.maxsize // 8 + 1)
        with patch('ferro._graph_composition.stack', side_effect=AssertionError('capacity must reject before stack')):
            with self.assertRaises(ValueError): JumpingKnowledge('max')([wide])
        with self.assertRaises(NotImplementedError): fr.capture(lambda: JumpingKnowledge('cat')([x]))
        model = GeneralConv(3, 2)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, edge))
        with self.assertRaises(AttributeError): JumpingKnowledge('max').mode = 'cat'


if __name__ == '__main__':
    unittest.main(verbosity=2)
