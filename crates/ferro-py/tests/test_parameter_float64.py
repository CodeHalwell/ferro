"""CPU Double parameters use the existing slots, AD engine and typed state."""
import math
import tempfile
import unittest

import torch
import ferro as fr


def native(value, requires_grad=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(requires_grad)


class Float64ParameterTests(unittest.TestCase):
    def same(self, actual, expected, exact=False):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        result = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(result, expected.detach(), rtol=0 if exact else 2e-12,
                                   atol=0 if exact else 2e-14)

    def test_constructor_precision_history_flags_and_existing_owning_contract(self):
        values = [math.nextafter(1., 2.), 1e-200, 1e200, -0.]
        for enabled in (False, True):
            source = fr.tensor(values, dtype='float64', requires_grad=True)
            interior = source * 1.
            parameter = fr.nn.Parameter(interior, requires_grad=enabled)
            oracle = torch.nn.Parameter(torch.tensor(values, dtype=torch.float64) * 1., requires_grad=enabled)
            self.same(parameter, oracle, exact=True)
            self.assertIsInstance(parameter, fr.Tensor)
            self.assertTrue(parameter.is_leaf)
            self.assertEqual(parameter.requires_grad, oracle.requires_grad)
            self.assertEqual(parameter._id(), parameter.tensor()._id())
            self.assertFalse(parameter._shares_storage(source))
            self.assertFalse(parameter._shares_storage(interior))
            self.assertTrue(source.requires_grad)
            if enabled:
                parameter.sum().backward(); oracle.sum().backward()
                self.same(parameter.grad, oracle.grad, exact=True)
                self.assertIsNone(source.grad)
            with fr.no_grad():
                parameter.zero_()
            self.assertEqual(source.tolist(), values)
        for dtype in ('int64', 'bool'):
            with self.assertRaises(TypeError):
                fr.nn.Parameter(fr.ones(2, dtype=dtype))
        for dtype in ('float32', 'float64'):
            source = fr.tensor([.7, -.2], dtype=dtype, requires_grad=True)
            identity, before = source._id(), source.tolist()
            with self.assertRaisesRegex(NotImplementedError, 'Parameter construction during capture'):
                fr.capture(lambda: fr.nn.Parameter(source))
            self.assertEqual(source._id(), identity)
            self.assertEqual(source.tolist(), before)
            self.assertTrue(source.requires_grad)
            self.assertIsNone(source.grad)
            self.assertTrue(fr.nn.Parameter(source).is_leaf)

    def test_containers_preserve_tied_double_parameters_and_typed_state(self):
        parameter = fr.nn.Parameter(fr.tensor([1.0000000000000002, 1e-200], dtype='float64'))
        values = fr.nn.ParameterList([parameter, parameter])
        mapping = fr.nn.ParameterDict({'left': parameter, 'right': parameter})
        self.assertIs(values[0], parameter); self.assertIs(values[1], parameter)
        self.assertIs(mapping['left'], mapping['right'])
        self.assertEqual(len(list(values.parameters())), 1)
        state = mapping.state_dict()
        self.assertEqual(list(state), ['left', 'right'])
        self.assertEqual(state['left'].dtype, 'float64')
        self.assertTrue(state['left']._shares_storage(parameter))
        parameter.sum().backward()
        previous_grad = parameter.grad
        new = fr.tensor([1.0000000000000004, 1e-201], dtype='float64')
        mapping.load_state_dict({'left': new, 'right': new})
        self.assertEqual(parameter.tolist(), new.tolist())
        self.assertEqual(parameter.grad._id(), previous_grad._id())

    def test_linear_values_input_and_all_parameter_gradients_match_torch(self):
        generator = torch.Generator().manual_seed(739)
        for bias in (False, True):
            for shape in ((3,), (4, 3), (2, 4, 3), (0, 3), (2, 0, 3)):
                with self.subTest(bias=bias, shape=shape):
                    oracle = torch.nn.Linear(3, 2, bias=bias, dtype=torch.float64)
                    layer = fr.nn.Linear(3, 2, bias=bias, dtype='float64')
                    layer.load_state_dict({name: native(value) for name, value in oracle.state_dict().items()})
                    tx = torch.randn(shape, dtype=torch.float64, generator=generator).requires_grad_()
                    x = native(tx, True)
                    result, expected = layer(x), oracle(tx)
                    self.same(result, expected)
                    seed = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
                    (result * native(seed)).sum().backward(); (expected * seed).sum().backward()
                    self.same(x.grad, tx.grad)
                    for parameter, reference in zip(layer.parameters(), oracle.parameters()):
                        self.same(parameter.grad, reference.grad)
        # Both input and weight are noncontiguous genuine native views.
        xbase = torch.randn(3, 4, dtype=torch.float64, generator=generator).requires_grad_()
        wbase = torch.randn(3, 2, dtype=torch.float64, generator=generator).requires_grad_()
        x, w = native(xbase, True), native(wbase, True)
        actual = fr.nn.functional.linear(x.T, w.T)
        expected = torch.nn.functional.linear(xbase.T, wbase.T)
        self.same(actual, expected)
        actual.sum().backward(); expected.sum().backward()
        self.same(x.grad, xbase.grad); self.same(w.grad, wbase.grad)

    def test_linear_retains_double_residuals_and_extreme_reciprocal_scales(self):
        for a, b in ((1. + 2.**-40, 1. + 2.**-45), (1e-150, 1e150), (1e150, 1e-150)):
            with self.subTest(a=a, b=b):
                layer = fr.nn.Linear(1, 1, bias=False, dtype='float64')
                oracle = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
                with fr.no_grad(): layer.weight.copy_(fr.tensor([[b]], dtype='float64'))
                with torch.no_grad(): oracle.weight.fill_(b)
                x = fr.tensor([[a]], dtype='float64', requires_grad=True)
                tx = torch.tensor([[a]], dtype=torch.float64, requires_grad=True)
                result, expected = layer(x), oracle(tx)
                self.same(result, expected, exact=True)
                result.sum().backward(); expected.sum().backward()
                self.same(x.grad, tx.grad, exact=True)
                self.same(layer.weight.grad, oracle.weight.grad, exact=True)

    def test_reset_preserves_live_slots_flags_gradients_and_random_precision(self):
        for dtype in ('float32', 'float64'):
            layer = fr.nn.Linear(7, 5, dtype=dtype)
            held = list(layer.parameters())
            ids = [value._id() for value in held]
            layer(fr.ones(2, 7, dtype=dtype)).sum().backward()
            grads = [value.grad._id() for value in held]
            layer.bias.requires_grad_(False)
            layer.weight.trainable = False
            # Freezing optimizer state clears its gradient by existing contract.
            grads[0] = None
            before = layer.weight.tolist()
            layer.reset_parameters()
            self.assertEqual(ids, [value._id() for value in layer.parameters()])
            self.assertFalse(layer.weight.trainable); self.assertFalse(layer.bias.requires_grad)
            self.assertNotEqual(before, layer.weight.tolist())
            self.assertIsNone(layer.weight.grad)
            self.assertEqual(layer.bias.grad._id(), grads[1])
            for value, old in zip(layer.parameters(), held):
                self.assertTrue(value._shares_storage(old))
                self.assertEqual(value.dtype, dtype)
                self.assertTrue(all(abs(x) <= 7**-.5 for x in value.reshape(-1).tolist()))
            if dtype == 'float64':
                flat = layer.weight.reshape(-1).tolist()
                self.assertTrue(any(float(torch.tensor(v, dtype=torch.float32)) != v for v in flat))
            self.assertTrue(any(v != 0. for v in layer.bias.tolist()))
            layer.bias = fr.nn.Parameter(fr.zeros(6, dtype=dtype))
            before = layer.weight.tolist()
            with self.assertRaises(ValueError): layer.reset_parameters()
            self.assertEqual(layer.weight.tolist(), before)
        with self.assertRaises(ValueError):
            fr.capture(lambda: (layer.reset_parameters(), fr.ones(1))[1])

    def test_double_training_snapshot_resumes_exact_values_flags_modes_and_rng(self):
        layer = fr.nn.Linear(3, 2, dtype='float64')
        layer.weight.requires_grad_(False); layer.bias.trainable = False; layer.eval()
        with fr.no_grad():
            layer.weight.copy_(fr.tensor([[1.0000000000000002, 1e-200, -0.], [.3, -.7, .9]], dtype='float64'))
        generator = fr.Generator(173)
        saved = fr.checkpoint.snapshot_training(layer, optimizers={}, generator=generator, step=9)
        def run(model, rng):
            return model(fr.randn(4, 3, dtype='float64', generator=rng)).tolist()
        expected = [run(layer, generator) for _ in range(4)]
        target, rng = fr.nn.Linear(3, 2, dtype='float64'), fr.Generator(811)
        identities = [p._id() for p in target.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory)
            checkpoint = fr.checkpoint.load(directory)
            self.assertEqual(checkpoint.restore(target, optimizers={}, generator=rng), 9)
        self.assertEqual([p._id() for p in target.parameters()], identities)
        self.assertFalse(target.training)
        self.assertFalse(target.weight.requires_grad); self.assertFalse(target.bias.trainable)
        self.assertEqual([run(target, rng) for _ in range(4)], expected)
        for name, value in saved.tensors().items():
            if name.startswith('model.'):
                self.assertEqual(value.dtype, 'float64')

    def test_dtype_device_shape_validation_rejects_before_implicit_promotion(self):
        layer = fr.nn.Linear(3, 2, dtype='float64')
        for value in (fr.ones(2, 3), fr.ones(2, 4, dtype='float64'), fr.tensor(1., dtype='float64')):
            with self.assertRaises((TypeError, ValueError)): layer(value)
        with self.assertRaises(TypeError):
            fr.nn.functional.linear(fr.ones(2, 3, dtype='float64'), layer.weight, fr.zeros(2))
        for dtype in ('int64', 'bool'):
            with self.assertRaises(TypeError): fr.nn.Linear(3, 2, dtype=dtype)
        with self.assertRaises(ValueError): fr.nn.Linear(3, 2, dtype='float64', device='cuda')
        for dimensions in ((True, 2), (2, 0), (-1, 2), (1 << 62, 2)):
            with self.assertRaises(ValueError): fr.nn.Linear(*dimensions, dtype='float64')
        # Functional linear retains the existing native broadcast-bias route;
        # the module's owned bias is still a vector of out_features values.
        for dtype in ('float32', 'float64'):
            for shape in ((), (1,), (2,), (1, 2), (3, 2)):
                with self.subTest(dtype=dtype, bias_shape=shape):
                    actual = fr.nn.functional.linear(fr.ones(3, 4, dtype=dtype), fr.ones(2, 4, dtype=dtype), fr.ones(shape, dtype=dtype))
                    expected = torch.nn.functional.linear(torch.ones(3, 4, dtype=getattr(torch, dtype)), torch.ones(2, 4, dtype=getattr(torch, dtype)), torch.ones(shape, dtype=getattr(torch, dtype)))
                    self.same(actual, expected, exact=True)


if __name__ == '__main__':
    unittest.main()
