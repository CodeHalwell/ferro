"""Parameter-as-Tensor graph fills retain learned registration and gradients."""
import tempfile
import unittest

import torch
from torch_geometric.nn import GATConv as TorchGAT, GATv2Conv as TorchGATv2
import ferro as fr
from ferro.graph import GATConv, GATv2Conv
import test_pyg_modulated as helpers

native = helpers.native


class GraphParameterFillTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2237)
        self.edge = torch.tensor([[0, 1, 2, 1, 3, 0, 2], [1, 2, 0, 3, 0, 0, 2]])

    def test_scalar_and_vector_parameter_fill_preservation_and_actual_pyg_all_gradients(self):
        self.assertTrue(issubclass(fr.nn.Parameter, fr.Tensor))
        for constructor, reference in ((GATConv, TorchGAT), (GATv2Conv, TorchGATv2)):
            for values in (torch.tensor(.37), torch.tensor([.37, -.28])):
                fill, torch_fill = fr.nn.Parameter(native(values)), torch.nn.Parameter(values.clone())
                model = constructor(3, 2, heads=2, edge_dim=2, fill_value=fill, residual=True)
                oracle = reference(3, 2, heads=2, edge_dim=2, fill_value=torch_fill, residual=True)
                self.assertIs(model.fill_value, fill)
                self.assertIn('fill_value', dict(model.named_parameters()))
                self.assertNotIn('fill_value', dict(model.named_buffers()))
                self.close(fill, values)
                model.reset_parameters(); oracle.reset_parameters()
                self.close(fill, values); torch.testing.assert_close(torch_fill, values)
                self.copy(model, oracle)
                x, attrs = torch.randn(4, 3, requires_grad=True), torch.randn(7, 2, requires_grad=True)
                nx, na = native(x, True), native(attrs, True)
                self.gradients(model, oracle, model(nx, native(self.edge), na), oracle(x, self.edge, attrs), [('x', nx, x), ('edge_attr', na, attrs)])
                self.assertGreater(float(fill.tensor().grad.abs().sum().item()), 0.)

    def test_plain_tensor_buffer_then_parameter_assignment_changes_registration_category(self):
        for constructor in (GATConv, GATv2Conv):
            tensor = fr.Tensor([.3, -.2], [2])
            model = constructor(3, 2, edge_dim=2, fill_value=tensor)
            self.assertIs(model.fill_value, tensor)
            self.assertIn('fill_value', dict(model.named_buffers()))
            self.assertNotIn('fill_value', dict(model.named_parameters()))
            model.reset_parameters()
            self.assertEqual(tensor.tolist(), fr.Tensor([.3, -.2], [2]).tolist())
            parameter = fr.nn.Parameter(fr.Tensor([.8, .1], [2]))
            model.fill_value = parameter
            self.assertIs(model.fill_value, parameter)
            self.assertNotIn('fill_value', dict(model.named_buffers()))
            self.assertIs(dict(model.named_parameters())['fill_value'], parameter)
            model.reset_parameters()
            self.assertEqual(parameter.tolist(), fr.Tensor([.8, .1], [2]).tolist())

    def test_fill_alias_to_builtin_bias_still_resets_through_its_bias_role(self):
        for constructor, reference in ((GATConv, TorchGAT), (GATv2Conv, TorchGATv2)):
            fill = fr.nn.Parameter(fr.Tensor([.6, -.4], [2]))
            torch_fill = torch.nn.Parameter(torch.tensor([.6, -.4]))
            model = constructor(3, 2, edge_dim=2, fill_value=fill)
            oracle = reference(3, 2, edge_dim=2, fill_value=torch_fill)
            model.bias = model.fill_value
            oracle.bias = oracle.fill_value
            model.reset_parameters(); oracle.reset_parameters()
            self.assertIs(model.bias, model.fill_value)
            self.close(model.fill_value, oracle.fill_value)
            self.assertEqual(model.fill_value.tolist(), [0., 0.])

    def test_saved_tied_learned_fill_optimizer_rng_and_buffer_schema_mismatch(self):
        def factory(learned=True):
            fill = fr.Tensor([.3, -.2], [2])
            model = GATConv(3, 1, heads=2, edge_dim=2, fill_value=fr.nn.Parameter(fill) if learned else fill)
            if learned: model.bias = model.fill_value
            return model
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([4, 3], generator=generator), native(self.edge), fr.Tensor.randn([7, 2], generator=generator))
            loss = output.square().mean() + .13 * output.mean()
            values = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return values
        model, rng = factory(), fr.Generator(347)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
        before = model.fill_value.tolist()
        step(model, optimizer, rng)
        self.assertNotEqual(before, model.fill_value.tolist())
        saved = fr.checkpoint.snapshot_training(model, optimizers={'graph': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'graph': other_opt}, generator=other_rng)
        self.assertIs(other.fill_value, other.bias)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'graph': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'graph': other_opt}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        changed = factory(False)
        before = [p.tensor().tolist() for p in changed.parameters()], changed.fill_value.tolist()
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'graph': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, ([p.tensor().tolist() for p in changed.parameters()], changed.fill_value.tolist()))


if __name__ == '__main__': unittest.main(verbosity=2)
