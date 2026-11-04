"""Typed bilinear modules, Gaussian likelihood gradients and exact SGD restart."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import ferro as fr
import test_sgd_f64 as state_helpers
from test_nn_catalog import native


class BilinearFloat64Tests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        values = torch.tensor(actual.reshape([-1]).tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 2e-12 if expected.dtype == torch.float64 else 3e-5
        torch.testing.assert_close(values, expected.detach(), atol=tolerance, rtol=tolerance, equal_nan=True)

    def matched(self, dtype=torch.float64, bias=True, output=3):
        actual = fr.nn.Bilinear(2, 3, output, bias=bias, dtype=str(dtype).split('.')[-1])
        expected = torch.nn.Bilinear(2, 3, output, bias=bias, dtype=dtype)
        with torch.no_grad():
            expected.weight.copy_(torch.linspace(-.37, .43, output * 6, dtype=dtype).reshape(output, 2, 3))
            if bias:
                expected.bias.copy_(torch.linspace(.10000000000000002, -.21, output, dtype=dtype))
        with fr.no_grad():
            actual.weight.copy_(native(expected.weight))
            if bias:
                actual.bias.copy_(native(expected.bias))
        return actual, expected

    def test_factories_preserve_default_dtype_and_preflight_before_random_draws(self):
        self.assertEqual(fr.nn.Bilinear(2, 3, 4).weight.dtype, 'float32')
        for dtype in ('float32', 'float64'):
            for bias in (False, True):
                for output in (0, 4):
                    model = fr.nn.Bilinear(2, 3, output, bias=bias, dtype=dtype)
                    self.assertEqual(model.weight.dtype, dtype)
                    self.assertEqual(model.weight.shape, [output, 2, 3])
                    self.assertEqual(list(dict(model.named_parameters())), ['weight', 'bias'] if bias else ['weight'])
                    for param in model.parameters():
                        self.assertTrue(param.requires_grad)
                        self.assertTrue(all(abs(v) <= 2 ** -.5 for v in param.reshape([-1]).tolist()))
        for kwargs in ({'dtype':'int64'}, {'dtype':'bool'}, {'dtype':'float64', 'device':'cuda'}):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                fr.nn.Bilinear(2, 3, 4, **kwargs)
        with patch.object(fr.Tensor, 'rand', side_effect=AssertionError('draw before validation')):
            with self.assertRaises(ValueError):
                fr.nn.Bilinear(sys.maxsize // 8 + 1, 1, 1, dtype='float64')
            with self.assertRaises(NotImplementedError):
                fr.capture(lambda: fr.nn.Bilinear(2, 3, 4, dtype='float64'))
            with self.assertRaises(NotImplementedError):
                fr.func.jvp(lambda x: (fr.nn.Bilinear(2, 3, 4), x)[1], (fr.ones(1),), (fr.ones(1),))

    def test_copied_modules_match_torch_values_and_all_input_parameter_gradients(self):
        for prefix in ((), (4,), (2, 3), (0, 2)):
            for bias in (False, True):
                for output in (0, 3):
                    with self.subTest(prefix=prefix, bias=bias, output=output):
                        actual, expected = self.matched(bias=bias, output=output)
                        size = 1
                        for extent in prefix:
                            size *= extent
                        tx = torch.linspace(-.8, .9, size * 2, dtype=torch.float64).reshape(*prefix, 2).requires_grad_()
                        ty = torch.linspace(.7, -.3, size * 3, dtype=torch.float64).reshape(*prefix, 3).requires_grad_()
                        x, y = native(tx), native(ty)
                        result, reference = actual(x, y), expected(tx, ty)
                        self.same(result, reference)
                        self.assertEqual(result.stride(), reference.stride())
                        seed = torch.linspace(.2, .8, reference.numel(), dtype=torch.float64).reshape(reference.shape)
                        ours = (x, y, *actual.parameters())
                        theirs = (tx, ty, *expected.parameters())
                        got = fr.autograd.grad(result, ours, native(seed))
                        want = torch.autograd.grad(reference, theirs, seed)
                        for value, oracle in zip(got, want):
                            self.same(value, oracle)
                        self.assertTrue(all(value.grad is None for value in ours))
        actual, expected = self.matched()
        tx = torch.arange(12, dtype=torch.float64).reshape(2, 6).requires_grad_()
        ty = torch.arange(18, dtype=torch.float64).reshape(3, 6).requires_grad_()
        x, y = native(tx), native(ty)
        result, reference = actual(x.T, y.T), expected(tx.T, ty.T)
        self.same(result, reference)
        got = fr.autograd.grad(result.sum(), (x, y, *actual.parameters()))
        want = torch.autograd.grad(reference.sum(), (tx, ty, *expected.parameters()))
        for value, oracle in zip(got, want):
            self.same(value, oracle)

    def test_reset_preserves_leaf_storage_flags_gradients_and_version_guards(self):
        for dtype in ('float32', 'float64'):
            model = fr.nn.Bilinear(2, 3, 4, dtype=dtype)
            held = list(model.parameters())
            aliases = [value.detach() for value in held]
            gradients = [fr.ones(value.shape, dtype=dtype) for value in held]
            for value, gradient in zip(held, gradients):
                value.requires_grad_(False)
                value.grad = gradient
            model.reset_parameters()
            self.assertEqual(list(model.parameters()), held)
            for value, alias, gradient in zip(model.parameters(), aliases, gradients):
                self.assertTrue(value._shares_storage(alias))
                self.assertFalse(value.requires_grad)
                self.assertEqual(value.grad._id(), gradient._id())
                self.assertTrue(all(abs(v) <= 2 ** -.5 for v in value.reshape([-1]).tolist()))
            if dtype == 'float64':
                self.assertTrue(any(float(torch.tensor(v, dtype=torch.float32)) != v for v in model.weight.reshape([-1]).tolist()))
            before = [state_helpers.flat_bits(value) for value in held]
            with self.assertRaises(NotImplementedError):
                fr.capture(model.reset_parameters)
            with self.assertRaises(NotImplementedError):
                fr.func.jvp(lambda x: (model.reset_parameters(), x)[1], (fr.ones(1),), (fr.ones(1),))
            self.assertEqual(before, [state_helpers.flat_bits(value) for value in held])
            for value in held:
                value.requires_grad_(True)
            stale = model.weight.square().sum()
            model.reset_parameters()
            with self.assertRaises((ValueError, RuntimeError)):
                stale.backward()
            self.assertEqual([value.grad._id() for value in held], [value._id() for value in gradients])

    def test_gaussian_module_double_gradients_include_clamped_variance_identity(self):
        model, oracle = self.matched(output=2)
        tx = torch.tensor([[.2, -.7], [.4, .8]], dtype=torch.float64, requires_grad=True)
        ty = torch.tensor([[.1, .5, -.3], [.9, -.4, .2]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[.3, -.2]], dtype=torch.float64, requires_grad=True)
        variance = torch.tensor([[.1], [.7]], dtype=torch.float64, requires_grad=True)
        x, y, target_value, var_value = map(native, (tx, ty, target, variance))
        loss = fr.nn.GaussianNLLLoss(full=True, eps=.25, reduction='sum')
        expected_loss = torch.nn.GaussianNLLLoss(full=True, eps=.25, reduction='sum')
        result = loss(model(x, y), target_value, var_value)
        reference = expected_loss(oracle(tx, ty), target, variance)
        self.same(result, reference)
        got = fr.autograd.grad(result, (x, y, target_value, var_value, *model.parameters()))
        want = torch.autograd.grad(reference, (tx, ty, target, variance, *oracle.parameters()))
        for value, expected in zip(got, want):
            self.same(value, expected)
        self.assertNotEqual(got[3].tolist()[0][0], 0.)
        self.assertEqual(var_value.tolist(), [[.1], [.7]])
        self.assertEqual(list(loss.parameters()), [])

    def test_float32_higher_jvp_retained_and_float64_transform_boundaries_explicit(self):
        model, oracle = self.matched(dtype=torch.float32, output=2)
        tx = torch.tensor([[.2, -.7]], requires_grad=True)
        ty = torch.tensor([[.1, .5, -.3]], requires_grad=True)
        x, y = native(tx), native(ty)
        result, reference = model(x, y).square().sum(), oracle(tx, ty).square().sum()
        first = fr.autograd.grad(result, (x, y, *model.parameters()), create_graph=True)
        expected = torch.autograd.grad(reference, (tx, ty, *oracle.parameters()), create_graph=True)
        got = fr.autograd.grad(sum(v.sum() for v in first), (x, y, *model.parameters()))
        want = torch.autograd.grad(sum(v.sum() for v in expected), (tx, ty, *oracle.parameters()))
        for value, target in zip(got, want):
            self.same(value, target)
        _, tangent = fr.func.jvp(lambda a: model(a, y), (x,), (fr.ones_like(x),))
        _, expected_tangent = torch.func.jvp(lambda a: oracle(a, ty), (tx,), (torch.ones_like(tx),))
        self.same(tangent, expected_tangent)
        model = fr.nn.Bilinear(2, 3, 2, dtype='float64')
        x, y = fr.ones(1, 2, dtype='float64').requires_grad_(), fr.ones(1, 3, dtype='float64').requires_grad_()
        result = model(x, y).sum()
        with self.assertRaises((ValueError, TypeError, RuntimeError, NotImplementedError)):
            fr.autograd.grad(result, (x, model.weight), create_graph=True)
        with self.assertRaises((ValueError, TypeError, RuntimeError, NotImplementedError)):
            fr.func.jvp(lambda a: model(a, y), (x,), (fr.ones_like(x),))
        self.assertIsNone(x.grad)
        self.assertIsNone(model.weight.grad)

    def test_typed_sgd_gaussian_trajectory_retains_exact_slots_and_states(self):
        actual, expected = self.matched(output=2)
        settings = dict(lr=.030000000000000002, momentum=.9000000000000001, dampening=.10000000000000002, weight_decay=.020000000000000004)
        optimizer = fr.optim.SGD(actual.parameters(), **settings)
        oracle = torch.optim.SGD(expected.parameters(), **settings)
        loss = fr.nn.GaussianNLLLoss(full=True, eps=.2)
        reference_loss = torch.nn.GaussianNLLLoss(full=True, eps=.2)
        tx = torch.tensor([[.2, -.4], [.7, .1]], dtype=torch.float64)
        ty = torch.tensor([[.1, .3, -.2], [-.4, .2, .6]], dtype=torch.float64)
        ids = [p._id() for p in actual.parameters()]
        for step in range(9):
            optimizer.zero_grad(); oracle.zero_grad()
            target = torch.full((2, 2), .05 * step, dtype=torch.float64)
            variance = torch.tensor([[.1], [.7]], dtype=torch.float64)
            result = loss(actual(native(tx), native(ty)), native(target), native(variance))
            reference = reference_loss(expected(tx, ty), target, variance)
            self.same(result, reference)
            result.backward(); reference.backward()
            for value, wanted in zip(actual.parameters(), expected.parameters()):
                self.same(value.grad, wanted.grad)
            held = [p.grad._id() for p in actual.parameters()]
            optimizer.step(); oracle.step()
            self.assertEqual([p._id() for p in actual.parameters()], ids)
            self.assertEqual([p.grad._id() for p in actual.parameters()], held)
            for index, (value, wanted) in enumerate(zip(actual.parameters(), expected.parameters())):
                self.same(value, wanted)
                self.same(optimizer.state_dict()['state'][index]['momentum_buffer'], oracle.state[wanted]['momentum_buffer'])

    def test_checkpoint_dtype_and_bias_schema_reject_without_partial_restore(self):
        source = fr.nn.Bilinear(2, 3, 2, dtype='float64')
        optimizer = fr.optim.SGD(source.parameters(), lr=.03, momentum=.9)
        source(fr.ones(1, 2, dtype='float64'), fr.ones(1, 3, dtype='float64')).square().sum().backward()
        optimizer.step(); optimizer.zero_grad()
        saved = fr.checkpoint.snapshot_training(source, optimizers={'sgd': optimizer}, generator=fr.Generator(7), step=1)
        for dtype, bias in (('float32', True), ('float64', False)):
            target = fr.nn.Bilinear(2, 3, 2, dtype=dtype, bias=bias)
            opt = fr.optim.SGD(target.parameters(), lr=.017, momentum=.7)
            rng = fr.Generator(11)
            before = state_helpers.digest(fr.checkpoint.snapshot_training(target, optimizers={'sgd': opt}, generator=rng))
            identities = [p._id() for p in target.parameters()]
            with self.assertRaises(ValueError):
                saved.restore(target, optimizers={'sgd': opt}, generator=rng)
            self.assertEqual(state_helpers.digest(fr.checkpoint.snapshot_training(target, optimizers={'sgd': opt}, generator=rng)), before)
            self.assertEqual([p._id() for p in target.parameters()], identities)

    def test_fresh_process_restores_full_double_training_state_and_rng_continuation(self):
        script = r'''
import json,struct,sys
import ferro as fr
class Model(fr.nn.Module):
 def __init__(self):
  super().__init__();self.map=fr.nn.Bilinear(2,3,2,dtype='float64');self.loss=fr.nn.GaussianNLLLoss(full=True,eps=.2)
  with fr.no_grad():
   self.map.weight.copy_(fr.tensor([.10000000000000002,-.3,.2,.4,-.5,.6,-.7,.8,-.9,.11,.12,-.13],dtype='float64').reshape(2,2,3));self.map.bias.copy_(fr.tensor([.2,-.1],dtype='float64'))
m=Model();rng=fr.Generator(79);opt=fr.optim.SGD(m.parameters(),lr=.030000000000000002,momentum=.9000000000000001,dampening=.1,weight_decay=.02)
def step():
 x=fr.randn([3,2],dtype='float64',generator=rng);y=fr.randn([3,3],dtype='float64',generator=rng);target=fr.randn([3,2],dtype='float64',generator=rng)
 loss=m.loss(m.map(x,y),target,fr.tensor([[.1],[.5],[.9]],dtype='float64'));value=loss.item();loss.backward();opt.step();opt.zero_grad();return value
if sys.argv[1]=='save':
 for _ in range(4):step()
 fr.checkpoint.snapshot_training(m,optimizers={'sgd':opt},generator=rng,step=4).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(m,optimizers={'sgd':opt},generator=rng)
values=[step()for _ in range(5)];state=fr.checkpoint.snapshot_training(m,optimizers={'sgd':opt},generator=rng,step=9).tensors()
def bits(t):
 values=t.reshape([t.numel()]).tolist();kind={'float32':'f','float64':'d','int64':'q','bool':'?'}[t.dtype];return[t.dtype,t.shape,struct.pack('<'+kind*len(values),*values).hex()]
print(json.dumps({'losses':values,'state':{k:bits(v)for k,v in state.items()},'next':fr.randn([7],dtype='float64',generator=rng).tolist()},sort_keys=True))
'''
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            results = [subprocess.run([sys.executable, '-c', script, mode, str(Path(directory) / 'state')], env=env, capture_output=True, text=True, timeout=60, check=True).stdout for mode in ('save', 'load')]
        first, second = map(json.loads, results)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main(verbosity=2)
