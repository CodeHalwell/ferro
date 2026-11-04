"""Typed Softsign/LogSigmoid staging against actual Torch CPU arithmetic."""
import math
import sys
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    result = fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


class StableActivationTests(unittest.TestCase):
    def close(self, actual, expected, exact=False):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 0. if exact else 2e-14 if expected.dtype == torch.float64 else 2e-6
        torch.testing.assert_close(value, expected.detach(), atol=tolerance, rtol=tolerance * 10, equal_nan=True)
        zeros = expected.detach() == 0
        self.assertEqual(value[zeros].signbit().tolist(), expected.detach()[zeros].signbit().tolist())

    def test_typed_strided_values_and_all_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for name, cls in (('softsign', 'Softsign'), ('logsigmoid', 'LogSigmoid')):
                for layout in ('transpose', 'slice', 'broadcast'):
                    with self.subTest(dtype=dtype, name=name, layout=layout):
                        x = torch.tensor([[-3.1, -.4, .2], [.7, 2.3, 5.2]], dtype=dtype, requires_grad=True)
                        value = native(x, True)
                        tx, fx = (x.T, value.T) if layout == 'transpose' else (x[:, ::2], value[:, ::2]) if layout == 'slice' else (x[:1].expand(3, 3), value[:1].expand(3, 3))
                        expected = getattr(torch.nn.functional, name)(tx)
                        actual = getattr(F, name)(input=fx)
                        self.close(actual, expected)
                        self.assertEqual(actual.stride(), expected.stride())
                        seed = torch.linspace(-.8, 1.3, expected.numel(), dtype=dtype).reshape(expected.shape)
                        actual.backward_with(native(seed))
                        expected.backward(seed)
                        self.close(value.grad, x.grad)
                        self.close(getattr(fr.nn, cls)()(input=fx), getattr(torch.nn, cls)()(input=tx))

    def test_nonfinite_and_extreme_cotangent_staging(self):
        for dtype in (torch.float32, torch.float64):
            tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)).item()
            largest = torch.finfo(dtype).max
            data = [-math.inf, -largest, -100., -30., -2., -tiny, -0., 0., tiny, 2., 30., 100., largest, math.inf, math.nan]
            for name in ('softsign', 'logsigmoid'):
                for scalar in (tiny, -tiny, 0., -0., 1., -1., largest, math.inf, math.nan):
                    with self.subTest(dtype=dtype, name=name, seed=scalar):
                        x = torch.tensor(data, dtype=dtype, requires_grad=True)
                        value = native(x, True)
                        expected, actual = getattr(torch.nn.functional, name)(x), getattr(F, name)(value)
                        seed = torch.full(x.shape, scalar, dtype=dtype)
                        actual.backward_with(native(seed)); expected.backward(seed)
                        self.close(actual, expected)
                        self.close(value.grad, x.grad)
            value = fr.tensor([-math.inf, math.inf], dtype=str(dtype).removeprefix('torch.'), requires_grad=True)
            F.softsign(value).sum().backward()
            self.assertTrue(all(math.isnan(item) for item in value.grad.tolist()))

    def test_logarithmic_tails_and_signed_zero_document_cpu_vector_rounding(self):
        x = torch.tensor([-30., 30., 100., math.inf], dtype=torch.float64, requires_grad=True)
        value = native(x, True)
        expected, actual = torch.nn.functional.logsigmoid(x), F.logsigmoid(value)
        self.close(actual, expected)
        self.assertLess(actual.tolist()[0], -30.)
        self.assertLess(actual.tolist()[1], 0.)
        self.assertLess(actual.tolist()[2], 0.)
        self.assertEqual(math.copysign(1., actual.tolist()[3]), 1.)
        actual.sum().backward(); expected.sum().backward(); self.close(value.grad, x.grad)
        # Torch's CPU vector log1p differs from scalar libm by one minimum
        # subnormal here. Keep the analytic nonzero tail, not an underflow to 0.
        result = F.logsigmoid(fr.tensor([100.])).item()
        reference = torch.nn.functional.logsigmoid(torch.tensor([100.])).item()
        minimum = torch.nextafter(torch.tensor(0.), torch.tensor(1.)).item()
        self.assertLess(result, 0.)
        self.assertLessEqual(abs(result - reference), minimum)

    def test_original_storage_versions_and_first_order_mode_boundaries(self):
        for dtype in ('float32', 'float64'):
            for name in ('softsign', 'logsigmoid'):
                x = fr.tensor([-.5, .7], dtype=dtype, requires_grad=True)
                with fr.no_grad(): self.assertFalse(getattr(F, name)(x).requires_grad)
                with self.assertRaises(ValueError): fr.capture(lambda: getattr(F, name)(x))
                output = getattr(F, name)(x)
                with self.assertRaises(ValueError): fr.autograd.grad(output.sum(), [x], create_graph=True)
                with fr.no_grad(): output.detach().fill_(3.)
                actual = fr.autograd.grad(output.sum(), [x])[0]
                reference = torch.tensor([-.5, .7], dtype=getattr(torch, dtype), requires_grad=True)
                getattr(torch.nn.functional, name)(reference).sum().backward()
                self.close(actual, reference.grad)
                with fr.no_grad(): x.fill_(4.)
                with self.assertRaises(ValueError): fr.autograd.grad(output.sum(), [x])
                self.assertIsNone(x.grad)
        for name in ('softsign', 'logsigmoid'):
            x, seed = torch.tensor([-.5, .7]), torch.tensor([.3, -.8])
            actual = fr.func.jvp(lambda a: getattr(F, name)(a), (native(x),), (native(seed),))
            expected = torch.func.jvp(lambda a: getattr(torch.nn.functional, name)(a), (x,), (seed,))
            for value, target in zip(actual, expected): self.close(value, target)
            with self.assertRaises(ValueError): fr.func.jvp(lambda a: getattr(F, name)(a), (native(x, True),), (native(seed),))

    def test_typed_empty_offsets_capacity_and_invalid_inputs(self):
        for dtype in ('float32', 'float64'):
            for name in ('softsign', 'logsigmoid'):
                base = fr.zeros(0, 8, dtype=dtype, requires_grad=True)
                output = getattr(F, name)(base[:, 4:])
                self.assertEqual(output.shape, [0, 4])
                output.sum().backward(); self.assertEqual(base.grad.shape, [0, 8])
                with self.assertRaises(ValueError): getattr(F, name)(fr.ones(1, dtype=dtype).expand(sys.maxsize // 8 + 1))
        for dtype in ('int64', 'bool'):
            for name in ('softsign', 'logsigmoid'):
                with self.assertRaises((TypeError, ValueError)): getattr(F, name)(fr.ones(2, dtype=dtype))

    def test_public_module_exact_training_checkpoint_continuation(self):
        def make():
            model = fr.nn.Sequential(fr.nn.Linear(2, 3), fr.nn.Softsign(), fr.nn.Linear(3, 1), fr.nn.LogSigmoid())
            return model, fr.optim.AdamW(model.parameters(), lr=.01), fr.Generator(11)
        def step(model, optimizer):
            optimizer.zero_grad(); model(fr.tensor([[.2, -.4], [.7, .8]])).square().mean().backward()
            optimizer.step(); optimizer.zero_grad()
        model, optimizer, generator = make(); step(model, optimizer)
        snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=1)
        other, target, rng = make(); snapshot.restore(other, optimizers={'main': target}, generator=rng)
        for _ in range(3): step(model, optimizer); step(other, target)
        left = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'main': target}, generator=rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual(left[key].tolist(), right[key].tolist(), key)


if __name__ == '__main__': unittest.main(verbosity=2)
