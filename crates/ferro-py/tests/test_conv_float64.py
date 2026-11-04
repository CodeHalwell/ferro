"""CPU double convolution kernels, complete gradients and typed training state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ferro as fr
import torch
import test_sgd_f64 as states


def native(value, grad=None):
    result = fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(value.requires_grad if grad is None else grad)


def same(case, actual, expected):
    case.assertEqual(actual.shape, list(expected.shape))
    case.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
    values = torch.tensor(actual.reshape([actual.numel()]).tolist(), dtype=expected.dtype).reshape(expected.shape)
    tolerance = 3e-11 if expected.dtype == torch.float64 else 5e-5
    torch.testing.assert_close(values, expected.detach(), rtol=tolerance, atol=tolerance, equal_nan=True)


class ConvFloat64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_copied_module_values_all_inputs_and_parameters_grouped_depthwise_strided(self):
        configurations = [
            ('Conv1d', (2, 4, 9), dict(in_channels=4, out_channels=6, kernel_size=3, stride=2, padding=2, dilation=2, groups=2)),
            ('Conv1d', (1, 3, 7), dict(in_channels=3, out_channels=6, kernel_size=2, groups=3)),
            ('Conv2d', (2, 4, 5, 6), dict(in_channels=4, out_channels=6, kernel_size=(2, 3), stride=(2, 1), padding=(1, 2), dilation=(1, 2), groups=2)),
            ('Conv2d', (1, 3, 4, 5), dict(in_channels=3, out_channels=6, kernel_size=(2, 2), padding=1, groups=3)),
        ]
        rng = torch.Generator().manual_seed(723)
        for name, shape, options in configurations:
            for bias in (False, True):
                for strided in (False, True):
                    with self.subTest(name=name, bias=bias, strided=strided):
                        a = getattr(fr.nn, name)(**options, bias=bias, dtype='float64')
                        b = getattr(torch.nn, name)(**options, bias=bias, dtype=torch.float64)
                        with fr.no_grad(), torch.no_grad():
                            for key, p in a.named_parameters():
                                q = torch.randn(p.shape, dtype=torch.float64, generator=rng) * .2 + .10000000000000002
                                p.copy_(native(q)); dict(b.named_parameters())[key].copy_(q)
                        base = torch.randn(shape, dtype=torch.float64, generator=rng)
                        if strided:
                            source = base.transpose(0, -1).contiguous()
                            x = source.transpose(0, -1).detach().requires_grad_()
                            nx = native(source).transpose(0, len(shape) - 1).detach().requires_grad_(True)
                        else: x = base.requires_grad_(); nx = native(x)
                        y, z = a(nx), b(x); same(self, y, z)
                        self.assertEqual(tuple(y.stride()), tuple(z.stride()))
                        seed = torch.randn(z.shape, dtype=torch.float64, generator=rng)
                        (y * native(seed)).sum().backward(); (z * seed).sum().backward()
                        same(self, nx.grad, x.grad)
                        for key, p in a.named_parameters(): same(self, p.grad, dict(b.named_parameters())[key].grad)

    def test_functional_same_valid_unbatched_empty_and_ieee_padding(self):
        rng = torch.Generator().manual_seed(733)
        for dimensions in (1, 2):
            function = getattr(fr.nn.functional, 'conv' + str(dimensions) + 'd')
            reference = getattr(torch.nn.functional, 'conv' + str(dimensions) + 'd')
            for prefix in ((2,), (), (0,)):
                for padding in ('same', 'valid'):
                    shape = (*prefix, 2, *([5] * dimensions)); weight_shape = (4, 1, *([2] * dimensions))
                    with self.subTest(dimensions=dimensions, prefix=prefix, padding=padding):
                        x = torch.randn(shape, dtype=torch.float64, generator=rng).requires_grad_()
                        w = torch.randn(weight_shape, dtype=torch.float64, generator=rng).requires_grad_()
                        bias = torch.randn(4, dtype=torch.float64, generator=rng).requires_grad_()
                        nx, nw, nb = map(native, (x, w, bias))
                        y = function(nx, nw, nb, padding=padding, dilation=3, groups=2)
                        z = reference(x, w, bias, padding=padding, dilation=3, groups=2)
                        same(self, y, z); self.assertEqual(tuple(y.stride()), tuple(z.stride()))
                        y.square().sum().backward(); z.square().sum().backward()
                        for left, right in ((nx, x), (nw, w), (nb, bias)): same(self, left.grad, right.grad)
        x = torch.ones((1, 1, 1, 1), dtype=torch.float64)
        w = torch.full((1, 1, 2, 2), float('inf'), dtype=torch.float64)
        same(self, fr.nn.functional.conv2d(native(x), native(w), padding=1), torch.nn.functional.conv2d(x, w, padding=1))

    def test_factory_default_dtype_and_typed_reset_preserve_handles_versions_and_rng(self):
        for name in ('Conv1d', 'Conv2d'):
            for dtype in ('float32', 'float64'):
                module = getattr(fr.nn, name)(2, 4, 3, groups=2, dtype=dtype)
                ids = [p._id() for p in module.parameters()]; old = [states.flat_bits(p) for p in module.parameters()]
                grads = []
                for p in module.parameters():
                    p.requires_grad_(False); p.grad = fr.ones(p.shape, dtype=dtype); grads.append(p.grad._id())
                module.reset_parameters(generator=fr.Generator(81))
                self.assertEqual(ids, [p._id() for p in module.parameters()]); self.assertEqual(grads, [p.grad._id() for p in module.parameters()])
                self.assertTrue(all(not p.requires_grad and p.dtype == dtype for p in module.parameters()))
                self.assertNotEqual(old, [states.flat_bits(p) for p in module.parameters()])
                before = [states.flat_bits(p) for p in module.parameters()]
                with self.assertRaises((ValueError, NotImplementedError)):
                    fr.func.jvp(lambda x: (module.reset_parameters(generator=fr.Generator(7)), x)[1], (fr.ones(1),), (fr.ones(1),))
                self.assertEqual(before, [states.flat_bits(p) for p in module.parameters()])
                module.weight.requires_grad_(True); stale = module.weight.square().sum(); module.reset_parameters(generator=fr.Generator(17))
                with self.assertRaises((ValueError, RuntimeError)): stale.backward()
            self.assertEqual(getattr(fr.nn, name)(1, 2, 1).weight.dtype, 'float32')
            self.assertEqual(getattr(fr.nn, name)(1, 2, 1, dtype='float64').weight.dtype, 'float64')

    def test_saved_input_weight_mutations_reject_and_output_mutation_is_allowed(self):
        for changed in ('input', 'weight', 'output'):
            model = fr.nn.Conv2d(1, 1, 1, dtype='float64', bias=False)
            x = fr.tensor([1., 2., 3., 4.], dtype='float64').reshape([1, 1, 2, 2]).requires_grad_(True)
            y = model(x); seed = fr.ones(y.shape, dtype='float64')
            before = fr.autograd.grad(y, (x, model.weight), seed)
            target = {'input': x, 'weight': model.weight, 'output': y}[changed]
            with fr.no_grad(): target.detach().copy_(fr.zeros(target.shape, dtype='float64'))
            if changed == 'output':
                after = fr.autograd.grad(y, (x, model.weight), seed)
                self.assertEqual([states.flat_bits(v) for v in before], [states.flat_bits(v) for v in after])
            else:
                with self.assertRaises((ValueError, RuntimeError)): fr.autograd.grad(y, (x, model.weight), seed)
                self.assertIsNone(x.grad); self.assertIsNone(model.weight.grad)

        for kernel in ((1, 1), (2, 2), (3, 3), (2, 3)):
            x = torch.ones((1, 1, 5, 5), dtype=torch.float64, requires_grad=True)
            w = torch.ones((1, 1, *kernel), dtype=torch.float64, requires_grad=True)
            nx, nw = native(x), native(w)
            y = fr.nn.functional.conv2d(nx, nw, padding='same')
            z = torch.nn.functional.conv2d(x, w, padding='same')
            with fr.no_grad(), torch.no_grad():
                nx.detach().copy_(fr.full(nx.shape, 2., dtype='float64')); x.fill_(2.)
            if all(width % 2 for width in kernel):
                with self.assertRaises((ValueError, RuntimeError)): fr.autograd.grad(y, (nx, nw), fr.ones(y.shape, dtype='float64'))
                with self.assertRaises(RuntimeError): torch.autograd.grad(z, (x, w), torch.ones_like(z))
            else:
                for got, expected in zip(fr.autograd.grad(y, (nx, nw), fr.ones(y.shape, dtype='float64')), torch.autograd.grad(z, (x, w), torch.ones_like(z))):
                    same(self, got, expected)

    def test_capacity_dtype_unsupported_modes_and_transform_boundaries(self):
        for name in ('Conv1d', 'Conv2d'):
            for dtype in ('bool', 'int64'):
                with self.assertRaises((ValueError, NotImplementedError)): getattr(fr.nn, name)(1, 2, 1, dtype=dtype)
            for mode in ('reflect', 'replicate', 'circular'):
                with self.assertRaises(NotImplementedError): getattr(fr.nn, name)(1, 2, 1, dtype='float64', padding_mode=mode)
            with self.assertRaisesRegex(ValueError, 'capacity'): getattr(fr.nn, name)(1, sys.maxsize // 8 + 1, 1, dtype='float64')
        for name in ('Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'):
            with self.assertRaises(NotImplementedError): getattr(fr.nn, name)(1, 2, 1, dtype='float64')
        module = fr.nn.Conv2d(1, 1, 1, dtype='float64'); x = fr.ones(1, 1, 2, 2, dtype='float64').requires_grad_(True)
        with self.assertRaises((ValueError, NotImplementedError)): module(fr.ones(x.shape))
        with self.assertRaises((ValueError, NotImplementedError)): fr.nn.functional.conv2d(x, module.weight, fr.ones(1))
        with self.assertRaises((ValueError, NotImplementedError)): fr.capture(lambda: module(x))
        with self.assertRaises((ValueError, NotImplementedError)): fr.func.jvp(lambda t: module(t), (x,), (fr.ones(x.shape, dtype='float64'),))
        with self.assertRaises((ValueError, NotImplementedError)): fr.autograd.grad(module(x).sum(), x, create_graph=True)
        with self.assertRaises((ValueError, NotImplementedError)):
            fr.func.jvp(lambda t: (module(x), t)[1], (fr.ones(1),), (fr.ones(1),))

    def test_typed_sgd_training_matches_torch_parameters_momentum_and_input_gradients(self):
        rng = torch.Generator().manual_seed(817)
        for name, shape in (('Conv1d', (2, 2, 6)), ('Conv2d', (2, 2, 4, 5))):
            a = getattr(fr.nn, name)(2, 4, 2, padding=1, groups=2, dtype='float64')
            b = getattr(torch.nn, name)(2, 4, 2, padding=1, groups=2, dtype=torch.float64)
            with fr.no_grad(), torch.no_grad():
                for key, p in a.named_parameters(): dict(b.named_parameters())[key].copy_(torch.tensor(p.tolist(), dtype=torch.float64))
            config = dict(lr=.030000000000000002, momentum=.8000000000000002, weight_decay=.01)
            opt, ref = fr.optim.SGD(a.parameters(), **config), torch.optim.SGD(b.parameters(), **config)
            ids = [p._id() for p in a.parameters()]
            for _ in range(5):
                opt.zero_grad(); ref.zero_grad(); x = torch.randn(shape, dtype=torch.float64, generator=rng).requires_grad_(); nx = native(x)
                y, z = a(nx), b(x); target = torch.randn(z.shape, dtype=torch.float64, generator=rng)
                loss, expected = (y-native(target)).square().mean(), (z-target).square().mean(); same(self, loss, expected)
                loss.backward(); expected.backward(); same(self, nx.grad, x.grad)
                for p, q in zip(a.parameters(), b.parameters()): same(self, p.grad, q.grad)
                opt.step(); ref.step(); self.assertEqual(ids, [p._id() for p in a.parameters()])
                for i, (p, q) in enumerate(zip(a.parameters(), b.parameters())):
                    same(self, p, q); same(self, opt.state_dict()['state'][i]['momentum_buffer'], ref.state[q]['momentum_buffer'])

    def test_checkpoint_dtype_and_same_shape_geometry_mismatch_reject_atomically(self):
        source = fr.nn.Conv2d(2, 4, 2, padding=1, groups=2, dtype='float64'); opt = fr.optim.SGD(source.parameters(), lr=.1, momentum=.9); rng = fr.Generator(42)
        source(fr.randn([2, 2, 4, 5], dtype='float64', generator=rng)).square().mean().backward(); opt.step(); opt.zero_grad()
        saved = fr.checkpoint.snapshot_training(source, optimizers={'sgd': opt}, generator=rng)
        for dtype, stride in (('float32', 1), ('float64', 2)):
            target = fr.nn.Conv2d(2, 4, 2, padding=1, stride=stride, groups=2, dtype=dtype)
            other = fr.optim.SGD(target.parameters(), lr=.1, momentum=.9); generator = fr.Generator(4)
            before = states.digest(fr.checkpoint.snapshot_training(target, optimizers={'sgd': other}, generator=generator))
            with self.assertRaises(ValueError): saved.restore(target, optimizers={'sgd': other}, generator=generator)
            self.assertEqual(before, states.digest(fr.checkpoint.snapshot_training(target, optimizers={'sgd': other}, generator=generator)))

    def test_fresh_process_full_double_model_sgd_and_generator_continuation(self):
        script = r'''
import json,struct,sys
import ferro as fr
class Model(fr.nn.Module):
 def __init__(self):
  super().__init__();self.image=fr.nn.Conv2d(2,4,(2,3),padding='same',groups=2,dtype='float64');self.sequence=fr.nn.Conv1d(4,2,2,padding='same',dtype='float64')
 def forward(self,x):
  y=self.image(x).tanh();return self.sequence(y.reshape([x.shape[0],4,-1]))
m=Model();o=fr.optim.SGD(m.parameters(),lr=.020000000000000004,momentum=.9000000000000001,weight_decay=.01);r=fr.Generator(81)
def step(i):
 x=fr.randn([1+i%2,2,3,4],dtype='float64',generator=r);y=m(x);target=fr.randn(y.shape,dtype='float64',generator=r)
 loss=(y-target).square().mean();value=loss.item();loss.backward();o.step();o.zero_grad();return value
if sys.argv[1]=='save':
 for i in range(3):step(i)
 fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=3).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(m,optimizers={'sgd':o},generator=r)
losses=[step(i)for i in range(3,8)];state=fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=8).tensors()
def bits(t):
 values=t.reshape([t.numel()]).tolist();code={'float32':'f','float64':'d','int64':'q','bool':'?'}[t.dtype];return[t.dtype,t.shape,struct.pack('<'+code*len(values),*values).hex()]
print(json.dumps({'losses':losses,'state':{k:bits(v)for k,v in state.items()},'next':fr.randn([9],dtype='float64',generator=r).tolist()},sort_keys=True))
'''
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            results = [subprocess.run([sys.executable, '-c', script, mode, str(Path(directory)/'state')], env=env, capture_output=True, text=True, timeout=60, check=True).stdout for mode in ('save', 'load')]
        self.assertEqual(*map(json.loads, results))


if __name__ == '__main__': unittest.main(verbosity=2)
