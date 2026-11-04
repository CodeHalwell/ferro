"""Native Parameter tensor inheritance, registration and stable optimizer leaves."""
import unittest

import torch
import ferro as fr


class Affine(fr.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = fr.nn.Parameter(fr.tensor([[.4, -.3], [.2, .7]]))
        self.bias = fr.nn.Parameter(fr.tensor([.1, -.2]))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class TorchAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.4, -.3], [.2, .7]]))
        self.bias = torch.nn.Parameter(torch.tensor([.1, -.2]))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class ParameterTensorTests(unittest.TestCase):
    def same(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(expected.shape), expected.detach(), atol=3e-6, rtol=3e-5)

    def test_inherited_tensor_operations_and_functional_arguments_match_torch(self):
        value = [[.4, -.3], [.2, .7]]
        parameter, reference = fr.nn.Parameter(fr.tensor(value)), torch.nn.Parameter(torch.tensor(value))
        self.assertIsInstance(parameter, fr.Tensor)
        self.assertTrue(fr.is_tensor(parameter))
        self.assertEqual(parameter._id(), parameter.tensor()._id())
        self.assertEqual(parameter.shape, list(reference.shape))
        self.assertEqual(parameter.dtype, 'float32')
        self.assertEqual(parameter.device, 'cpu')
        self.assertEqual(parameter.stride(), reference.stride())
        for actual, expected in ((parameter * 2 + 1, reference * 2 + 1),
                                 (parameter.T, reference.T), (parameter.square().sum(), reference.square().sum()),
                                 (fr.nn.functional.linear(fr.ones(3, 2), parameter), torch.nn.functional.linear(torch.ones(3, 2), reference)),
                                 (fr.cat((parameter, parameter), 0), torch.cat((reference, reference), 0))):
            self.same(actual, expected)
            self.assertNotIsInstance(actual, fr.nn.Parameter)
        self.assertIsInstance(hash(parameter), int)
        self.assertIs(fr.as_tensor(parameter), parameter)
        self.assertEqual(fr.ones_like(parameter).tolist(), [[1., 1.], [1., 1.]])
        self.same(parameter.detach(), reference.detach())

    def test_constructor_flags_empty_data_history_and_owning_compatibility(self):
        for requires_grad in (False, True):
            parameter = fr.nn.Parameter(requires_grad=requires_grad)
            reference = torch.nn.Parameter(requires_grad=requires_grad)
            self.assertEqual(parameter.shape, list(reference.shape))
            self.assertEqual(parameter.requires_grad, reference.requires_grad)
            self.assertTrue(parameter.is_leaf)
            source = fr.ones(2, requires_grad=True)
            interior = source.square()
            parameter = fr.nn.Parameter(data=interior, requires_grad=requires_grad)
            self.assertTrue(parameter.is_leaf)
            self.assertEqual(parameter.requires_grad, requires_grad)
            self.assertTrue(source.requires_grad)
            self.assertTrue(interior.requires_grad)
            self.assertFalse(parameter._shares_storage(interior))
            if requires_grad:
                parameter.sum().backward()
                self.assertEqual(parameter.grad.tolist(), [1., 1.])
                self.assertIsNone(source.grad)
        legacy = fr.nn.Parameter(tensor=fr.ones(2))
        self.assertEqual(legacy.tolist(), [1., 1.])
        with self.assertRaises(TypeError):
            fr.nn.Parameter(data=fr.ones(2), tensor=fr.ones(2))
        with self.assertRaises(TypeError):
            fr.nn.Parameter(fr.ones(2, dtype='int64'))

    def test_standard_custom_forward_all_gradients_and_optimizer_identity(self):
        model, reference = Affine(), TorchAffine()
        optimizer, target = fr.optim.AdamW(model.parameters(), lr=.03), torch.optim.AdamW(reference.parameters(), lr=.03)
        identities = [p._id() for p in model.parameters()]
        x = [[.3, -.5], [.2, .7], [-.4, .6]]
        for _ in range(8):
            optimizer.zero_grad(); target.zero_grad()
            actual, expected = model(fr.tensor(x)), reference(torch.tensor(x))
            self.same(actual, expected)
            actual.square().mean().backward(); expected.square().mean().backward()
            for parameter, oracle in zip(model.parameters(), reference.parameters()):
                self.same(parameter.grad, oracle.grad)
            optimizer.step(); target.step()
            for parameter, oracle in zip(model.parameters(), reference.parameters()):
                self.same(parameter, oracle)
                self.assertEqual(parameter._id(), parameter.tensor()._id())
            self.assertEqual([p._id() for p in model.parameters()], identities)

    def test_parameter_containers_retain_existing_objects_and_ties(self):
        parameter = fr.nn.Parameter(fr.ones(2))
        values = fr.nn.ParameterList((parameter, parameter))
        mapping = fr.nn.ParameterDict({'a': parameter, 'b': parameter})
        self.assertIs(values[0], parameter); self.assertIs(values[1], parameter)
        self.assertIs(mapping['a'], parameter); self.assertIs(mapping['b'], parameter)
        self.assertEqual(list(values.parameters()), [parameter])
        self.assertEqual(list(mapping.parameters()), [parameter])
        values.append(fr.ones(2))
        self.assertIsInstance(values[2], fr.nn.Parameter)
        kept = mapping.state_dict(keep_vars=True)
        self.assertIs(kept['a'], parameter)
        self.assertIs(kept['b'], parameter)
        self.assertNotIsInstance(mapping.state_dict()['a'], fr.nn.Parameter)
        shared = fr.ones(3, 2)
        embedding = fr.nn.Embedding.from_pretrained(shared, freeze=False)
        self.assertIsInstance(embedding.weight, fr.Tensor)
        self.assertTrue(embedding.weight._shares_storage(shared))
        self.assertEqual(embedding.weight._id(), embedding.weight.tensor()._id())

    def test_public_gradient_assignment_keeps_handles_and_rejects_alias_updates(self):
        for dtype in ('float32', 'float64'):
            leaf = fr.ones(2, dtype=dtype, requires_grad=True)
            source = fr.full([2], .5, dtype=dtype, requires_grad=True)
            gradient = source * 2.
            leaf.grad = gradient
            self.assertEqual(leaf.grad._id(), gradient._id())
            leaf.grad.sum().backward()
            self.assertEqual(source.grad.tolist(), [2., 2.])
            leaf.grad = None
            self.assertIsNone(leaf.grad)
        parameter = fr.nn.Parameter(fr.ones(2))
        parameter.grad = fr.full([2], .4)
        identity = parameter.grad._id()
        with self.assertRaises(ValueError): parameter.grad = fr.ones(3)
        self.assertEqual(parameter.grad._id(), identity)
        optimizer = fr.optim.SGD([parameter], lr=.1)
        optimizer.step()
        self.assertEqual(parameter.grad._id(), identity)
        self.assertEqual(parameter._id(), parameter.tensor()._id())
        parameter.grad = parameter.detach()
        before = parameter.tolist()
        with self.assertRaisesRegex(ValueError, 'share parameter storage'): optimizer.step()
        self.assertEqual(parameter.tolist(), before)
        parameter.grad = None
        self.assertIsNone(parameter.grad)

    def test_buffer_parameter_categories_match_torch(self):
        model, reference = fr.nn.Module(), torch.nn.Module()
        parameter, oracle = fr.nn.Parameter(fr.ones(2)), torch.nn.Parameter(torch.ones(2))
        model.register_buffer('held', parameter); reference.register_buffer('held', oracle)
        self.assertEqual(list(dict(model.named_parameters())), list(dict(reference.named_parameters())))
        self.assertEqual(list(dict(model.named_buffers())), list(dict(reference.named_buffers())))
        self.assertIs(model.get_buffer('held'), parameter)
        model.register_buffer('replacement', fr.zeros(2)); reference.register_buffer('replacement', torch.zeros(2))
        model.replacement = parameter; reference.replacement = oracle
        self.assertEqual(list(dict(model.named_parameters())), list(dict(reference.named_parameters())))
        self.assertEqual(list(dict(model.named_buffers())), list(dict(reference.named_buffers())))
        self.assertIs(model.get_parameter('replacement'), parameter)

    def test_checkpoint_and_inplace_copy_preserve_inherited_base_leaf(self):
        model, target = Affine(), Affine()
        with fr.no_grad():
            self.assertIs(model.weight.copy_(fr.full([2, 2], .6)), model.weight)
        model.requires_grad_(False)
        saved = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(7))
        identity = target.weight._id()
        saved.restore(target, optimizers={}, generator=fr.Generator(11))
        self.assertEqual(target.weight._id(), identity)
        self.assertEqual(target.weight._id(), target.weight.tensor()._id())
        self.assertFalse(target.weight.requires_grad)
        self.assertFalse(target(fr.ones(1, 2)).requires_grad)
        self.assertIs(target.weight.requires_grad_(), target.weight)
        self.assertEqual(target.weight._id(), identity)
        self.same(target.weight, torch.full([2, 2], .6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
