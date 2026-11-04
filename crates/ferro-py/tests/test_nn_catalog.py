"""Learned scalar/channel maps and Gaussian likelihoods against actual Torch."""
import math
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, live=None):
    value_live = value.requires_grad if live is None else live
    return fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).requires_grad_(value_live)


class Oracle(unittest.TestCase):
    def same(self, actual, expected, exact=False):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 0 if exact else 2e-5 if expected.dtype == torch.float32 else 2e-13
        torch.testing.assert_close(got, expected.detach(), rtol=tolerance, atol=tolerance, equal_nan=True)

    def gradients(self, out, reference, inputs, expected_inputs):
        seed = torch.linspace(.2, .9, reference.numel(), dtype=reference.dtype).reshape(reference.shape)
        actual = fr.autograd.grad(out, inputs, native(seed))
        expected = torch.autograd.grad(reference, expected_inputs, seed)
        for got, wanted in zip(actual, expected):
            self.same(got, wanted)


class BilinearTests(Oracle):
    def test_typed_shapes_strides_values_and_every_operand_gradient(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((), (3,), (2, 3), (0, 3)):
                for bias in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, bias=bias):
                        count = math.prod(shape)
                        a = torch.linspace(-.8, .7, count * 2, dtype=dtype).reshape([*shape, 2]).requires_grad_()
                        b = torch.linspace(.9, -.4, count * 3, dtype=dtype).reshape([*shape, 3]).requires_grad_()
                        w = torch.linspace(-.6, .5, 4 * 2 * 3, dtype=dtype).reshape(4, 2, 3).requires_grad_()
                        c = torch.linspace(-.2, .1, 4, dtype=dtype).requires_grad_() if bias else None
                        args = (a, b, w) if c is None else (a, b, w, c)
                        ours = tuple(native(x) for x in args)
                        out = F.bilinear(*ours); expected = torch.nn.functional.bilinear(*args)
                        self.same(out, expected); self.assertEqual(out.stride(), expected.stride())
                        self.gradients(out, expected, ours, args)
            bases = [torch.arange(12, dtype=dtype).reshape(2, 6).requires_grad_(),
                     torch.arange(18, dtype=dtype).reshape(3, 6).requires_grad_(),
                     torch.arange(24, dtype=dtype).reshape(2, 4, 3).requires_grad_()]
            native_bases = [native(x) for x in bases]
            out = F.bilinear(native_bases[0].T, native_bases[1].T, native_bases[2].transpose(0, 1))
            expected = torch.nn.functional.bilinear(bases[0].T, bases[1].T, bases[2].transpose(0, 1))
            self.same(out, expected); self.gradients(out, expected, native_bases, bases)

    def test_empty_widths_higher_derivatives_and_direct_jvp(self):
        for first, second, output in ((0, 3, 2), (2, 0, 3), (2, 3, 0)):
            args = [torch.ones(2, first, requires_grad=True), torch.ones(2, second, requires_grad=True), torch.ones(output, first, second, requires_grad=True)]
            ours = [native(x) for x in args]
            if first == 0 or second == 0:
                with self.assertRaises(ValueError): F.bilinear(*ours)
                with self.assertRaises(RuntimeError): torch.nn.functional.bilinear(*args)
                continue
            out, expected = F.bilinear(*ours), torch.nn.functional.bilinear(*args)
            self.same(out, expected); self.gradients(out, expected, ours, args)
        args = [torch.tensor([[.2, -.3]], requires_grad=True), torch.tensor([[.4, -.7]], requires_grad=True), torch.arange(8.).reshape(2, 2, 2).requires_grad_()]
        ours = [native(x) for x in args]
        out, reference = F.bilinear(*ours).square().sum(), torch.nn.functional.bilinear(*args).square().sum()
        actual = fr.autograd.grad(out, ours, create_graph=True)
        expected = torch.autograd.grad(reference, args, create_graph=True)
        self.gradients(sum(x.sum() for x in actual), sum(x.sum() for x in expected), ours, args)
        out, tangent = fr.func.jvp(lambda x: F.bilinear(x, ours[1], ours[2]), (ours[0],), (fr.ones_like(ours[0]),))
        expected, direction = torch.func.jvp(lambda x: torch.nn.functional.bilinear(x, args[1], args[2]), (args[0],), (torch.ones_like(args[0]),))
        self.same(out, expected); self.same(tangent, direction)

    def test_module_initialization_reset_named_parameters_and_preflight(self):
        module = fr.nn.Bilinear(3, 2, 4)
        self.assertEqual(list(dict(module.named_parameters())), ['weight', 'bias'])
        before = {key: value.tensor()._id() for key, value in module.named_parameters()}
        for value in module.parameters():
            self.assertTrue(all(abs(x) <= 3 ** -.5 for x in value.tensor().reshape([-1]).tolist()))
        self.assertTrue(any(x != 0 for x in module.bias.tensor().tolist()))
        module.reset_parameters()
        self.assertEqual(before, {key: value.tensor()._id() for key, value in module.named_parameters()})
        for kwargs in ({'in1_features': 0, 'in2_features': 2, 'out_features': 3}, {'in1_features': 2, 'in2_features': 3, 'out_features': 1, 'dtype': 'int64'}):
            with self.assertRaises((ValueError, NotImplementedError)): fr.nn.Bilinear(**kwargs)
        with patch.object(fr.Tensor, 'rand', side_effect=AssertionError('allocation before dimension preflight')):
            with self.assertRaises(ValueError): fr.nn.Bilinear(sys.maxsize, 2, 2)
        a, b, w = fr.ones(2, 3), fr.ones(2, 4), fr.ones(5, 3, 4)
        for bad in (fr.ones(1, 4), fr.ones(2, 4, dtype='float64'), fr.ones(4)):
            with self.assertRaises((ValueError, NotImplementedError)): F.bilinear(a, bad, w)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: F.bilinear(a, b, w))


class GaussianNLLTests(Oracle):
    def test_variance_broadcast_reductions_and_all_live_tensor_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for variance_shape in ((2, 3, 4), (2, 3), (2, 3, 1), (2, 1, 4), (1, 3, 4)):
                for reduction in ('none', 'sum', 'mean'):
                    x = torch.linspace(-.8, .9, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
                    y = torch.linspace(.5, -.1, 4, dtype=dtype).requires_grad_()
                    var = torch.linspace(.1, .9, math.prod(variance_shape), dtype=dtype).reshape(variance_shape).requires_grad_()
                    args = (x, y, var); ours = tuple(native(v) for v in args)
                    output = F.gaussian_nll_loss(*ours, eps=.3, full=True, reduction=reduction)
                    expected = torch.nn.functional.gaussian_nll_loss(*args, eps=.3, full=True, reduction=reduction)
                    self.same(output, expected); self.gradients(output, expected, ours, args)
        x = torch.tensor([.7, -.4], requires_grad=True); target = torch.tensor([.1, .8], requires_grad=True)
        self.same(F.gaussian_nll_loss(native(x), native(target), .3), torch.nn.functional.gaussian_nll_loss(x, target, .3))
        for dtype in (torch.float32, torch.float64):
            bases = [torch.linspace(.1,.7,12,dtype=dtype).reshape(3,4).requires_grad_(),
                     torch.linspace(-.2,.3,12,dtype=dtype).reshape(3,4).requires_grad_(),
                     torch.linspace(.1,.8,12,dtype=dtype).reshape(3,4).requires_grad_()]
            ours = [native(value) for value in bases]
            output = F.gaussian_nll_loss(*(value.T for value in ours), eps=.4, reduction='none')
            expected = torch.nn.functional.gaussian_nll_loss(*(value.T for value in bases), eps=.4, reduction='none')
            self.same(output, expected); self.gradients(output, expected, ours, bases)

    def test_clamp_identity_reverse_higher_gradient_and_original_variance_ownership(self):
        args = [torch.tensor([.7, -.4], requires_grad=True), torch.tensor([.1, .8], requires_grad=True), torch.tensor([.1, .5], requires_grad=True)]
        ours = [native(x) for x in args]
        output = F.gaussian_nll_loss(*ours, eps=.3, reduction='sum')
        expected = torch.nn.functional.gaussian_nll_loss(*args, eps=.3, reduction='sum')
        actual_grads = fr.autograd.grad(output, ours, create_graph=True)
        expected_grads = torch.autograd.grad(expected, args, create_graph=True)
        self.assertNotEqual(actual_grads[2].tolist()[0], 0)
        for actual, wanted in zip(actual_grads, expected_grads): self.same(actual, wanted)
        self.gradients(sum(x.sum() for x in actual_grads), sum(x.sum() for x in expected_grads), ours, args)
        self.assertEqual(ours[2].tolist(), [.10000000149011612, .5])
        a = fr.tensor([.7], requires_grad=True); y = fr.tensor([.1]); v = fr.tensor([.2], requires_grad=True)
        loss = F.gaussian_nll_loss(a, y, v)
        with fr.no_grad(): v.fill_(9.)
        self.assertIsNotNone(fr.autograd.grad(loss, v)[0])
        with self.assertRaisesRegex(NotImplementedError, 'forward AD'):
            fr.func.jvp(lambda z: F.gaussian_nll_loss(a, y, z), (v,), (fr.ones_like(v),))

    def test_empty_nonfinite_eps_and_variance_controls(self):
        for dtype in (torch.float32, torch.float64):
            for eps in (0., -.2, math.inf, -math.inf, math.nan, .3):
                x = torch.tensor([.7, -.4, .1], dtype=dtype, requires_grad=True)
                y = torch.tensor([.1, .8, .4], dtype=dtype, requires_grad=True)
                var = torch.tensor([0., .5, math.inf], dtype=dtype, requires_grad=True)
                args = (x, y, var); ours = tuple(native(v) for v in args)
                output = F.gaussian_nll_loss(*ours, eps=eps, reduction='none')
                expected = torch.nn.functional.gaussian_nll_loss(*args, eps=eps, reduction='none')
                self.same(output, expected); self.gradients(output, expected, ours, args)
            for shape, vshape in (((0, 3), (0, 1)), ((2, 0), (2, 1)), ((), ())):
                for reduction in ('none', 'mean', 'sum'):
                    args = [torch.ones(shape, dtype=dtype, requires_grad=True), torch.zeros(shape, dtype=dtype, requires_grad=True), torch.ones(vshape, dtype=dtype, requires_grad=True)]
                    ours = [native(v) for v in args]
                    output = F.gaussian_nll_loss(*ours, reduction=reduction)
                    expected = torch.nn.functional.gaussian_nll_loss(*args, reduction=reduction)
                    self.same(output, expected); self.gradients(output, expected, ours, args)

    def test_options_shape_dtype_capture_and_capacity_rejection(self):
        x, target = fr.ones(2, 3, 4), fr.zeros(2, 3, 4)
        for var in (fr.ones(2, 1, 1), fr.ones(3, 4), fr.full((2, 3, 4), -.1), 1, True):
            with self.assertRaises((ValueError, TypeError)): F.gaussian_nll_loss(x, target, var)
        with self.assertRaises(ValueError): F.gaussian_nll_loss(x, target, fr.ones(2, 3, 4), reduction='median')
        with self.assertRaises(NotImplementedError): fr.capture(lambda: F.gaussian_nll_loss(x, target, .5))
        module = fr.nn.GaussianNLLLoss(full=True, eps=.2, reduction='sum')
        self.assertEqual(list(module.parameters()), [])
        with self.assertRaises(AttributeError): module.eps = .7
        with patch.object(fr.Tensor, 'clone', side_effect=AssertionError('allocation before capacity preflight')):
            huge = fr.ones(1, 1).expand([sys.maxsize // 4 + 1, 1])
            with self.assertRaises(ValueError): F.gaussian_nll_loss(huge, huge, fr.ones(1, 1))


class PReLUTests(Oracle):
    def test_typed_channel_scalar_strided_values_and_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for shape, weights in (((), ()), ((5,), (1,)), ((2, 3, 4), (3,)), ((2, 0, 3), (0,)), ((0, 3, 2), (3,))):
                x = torch.linspace(-.8, .9, math.prod(shape), dtype=dtype).reshape(shape).requires_grad_()
                a = torch.linspace(.2, .7, math.prod(weights), dtype=dtype).reshape(weights).requires_grad_()
                args = (x, a); ours = tuple(native(v) for v in args)
                output = F.prelu(*ours); expected = torch.nn.functional.prelu(*args)
                self.same(output, expected, True); self.gradients(output, expected, ours, args)
            x = torch.linspace(-.8, .9, 12, dtype=dtype).reshape(3, 4).requires_grad_(); a = torch.tensor([.2,.7,.4], dtype=dtype, requires_grad=True)
            xx, aa = native(x), native(a)
            out, ref = F.prelu(xx.T, aa), torch.nn.functional.prelu(x.T, a)
            self.same(out, ref, True); self.assertEqual(out.stride(), ref.stride()); self.gradients(out, ref, (xx, aa), (x, a))
            weights = torch.linspace(.1,.7,6,dtype=dtype).requires_grad_(); source_weights = native(weights)
            out, ref = F.prelu(xx.T, source_weights[::2]), torch.nn.functional.prelu(x.T, weights[::2])
            self.same(out,ref,True); self.gradients(out,ref,(xx,source_weights),(x,weights))
            for transposed in (False, True):
                tx = torch.full((4,3,2), -2., dtype=dtype).permute(2,1,0) if transposed else torch.full((2,3,4), -2., dtype=dtype)
                x = fr.full((4,3,2), -2., dtype=str(dtype).split('.')[-1]).permute(2,1,0) if transposed else fr.full((2,3,4), -2., dtype=str(dtype).split('.')[-1])
                tx.requires_grad_(); x.requires_grad_(True)
                tg = torch.ones((2,3,4), dtype=dtype) if transposed else torch.ones((4,3,2), dtype=dtype).permute(2,1,0)
                g = fr.ones((2,3,4), dtype=str(dtype).split('.')[-1]) if transposed else fr.ones((4,3,2), dtype=str(dtype).split('.')[-1]).permute(2,1,0)
                a = fr.full((3,), .3, dtype=str(dtype).split('.')[-1]); ta = torch.full((3,), .3, dtype=dtype)
                got = fr.autograd.grad(F.prelu(x,a), x, g)[0]
                wanted = torch.autograd.grad(torch.nn.functional.prelu(tx,ta), tx, tg)[0]
                self.same(got, wanted, True); self.assertEqual(got.stride(), wanted.stride())

    def test_nonfinite_branches_and_signed_zero_do_not_leak_false_products(self):
        for dtype in (torch.float32, torch.float64):
            for slope in (-.3, 0., math.inf, math.nan):
                for coefficient in (.7, -0., math.inf):
                    x = torch.tensor([math.inf, -math.inf, math.nan, -0., 0., -.7, .2], dtype=dtype, requires_grad=True)
                    a = torch.tensor([slope], dtype=dtype, requires_grad=True); xx, aa = native(x), native(a)
                    output, expected = F.prelu(xx, aa), torch.nn.functional.prelu(x, a)
                    self.same(output, expected, True)
                    mask = expected.detach() == 0
                    got = torch.tensor(output.tolist(), dtype=dtype)
                    self.assertEqual(torch.signbit(got[mask]).tolist(), torch.signbit(expected.detach()[mask]).tolist())
                    seed = torch.full_like(x, coefficient)
                    actual = fr.autograd.grad(output, (xx, aa), native(seed)); wanted = torch.autograd.grad(expected, (x, a), seed)
                    for value, reference in zip(actual, wanted): self.same(value, reference, True)

    def test_module_reset_versions_and_explicit_derivative_boundaries(self):
        layer = fr.nn.PReLU(3, -.2)
        self.assertEqual(layer.weight.shape, [3]); identity = layer.weight.tensor()._id()
        with fr.no_grad(): layer.weight.tensor().fill_(1.)
        layer.reset_parameters(); self.assertEqual(identity, layer.weight.tensor()._id())
        self.same(layer.weight, torch.full((3,), -.2), True)
        x = fr.tensor([-.2, .3], requires_grad=True); a = fr.tensor([.4], requires_grad=True)
        for changed in ('input', 'weight', 'output'):
            xx, aa = x.clone(), a.clone(); y = F.prelu(xx, aa)
            {'input':xx, 'weight':aa, 'output':y}[changed].detach().fill_(1.)
            if changed == 'output': self.assertIsNotNone(fr.autograd.grad(y, xx, fr.ones_like(y))[0])
            else:
                with self.assertRaisesRegex(ValueError, 'modified'): fr.autograd.grad(y, xx, fr.ones_like(y))
        self.assertEqual(fr.autograd.grad(F.prelu(x, a), x, fr.ones_like(x), create_graph=True)[0].shape, x.shape)
        self.assertEqual(fr.func.jvp(lambda z:F.prelu(z,a), (x,), (fr.ones_like(x),))[1].shape, x.shape)
        with self.assertRaises((ValueError, NotImplementedError)): fr.capture(lambda:F.prelu(x,a))
        with self.assertRaises(NotImplementedError): fr.nn.PReLU(dtype='float64')


class CatalogCheckpointTests(Oracle):
    def test_registered_configuration_and_atomic_mismatched_restore(self):
        for source, target in ((fr.nn.Bilinear(2,3,4), fr.nn.Bilinear(2,3,4,bias=False)),
                               (fr.nn.PReLU(3,.2), fr.nn.PReLU(3,.3)),
                               (fr.nn.GaussianNLLLoss(eps=.2), fr.nn.GaussianNLLLoss(eps=.3))):
            saved = fr.checkpoint.snapshot_training(source, optimizers={}, generator=fr.Generator(17))
            before = {name: value.tensor().tolist() for name,value in target.named_parameters()}
            with self.assertRaises(ValueError): saved.restore(target, optimizers={}, generator=fr.Generator(18))
            self.assertEqual(before, {name: value.tensor().tolist() for name,value in target.named_parameters()})
        layer = fr.nn.Bilinear(2,3,4)
        state = layer.state_dict()
        restored = fr.nn.Bilinear(2,3,4)
        restored.load_state_dict(state)
        for name, value in layer.named_parameters():
            self.assertEqual(value.tolist(), dict(restored.named_parameters())[name].tolist())
        for name, value in (('in1_features', 5), ('out_features', 8)):
            with self.assertRaises(AttributeError): setattr(layer,name,value)

    def test_fresh_process_complete_training_suffix_and_tied_parameters(self):
        script = r'''
import json,sys
import ferro as fr
class Model(fr.nn.Module):
    def __init__(self):
        super().__init__()
        self.map=fr.nn.Bilinear(2,3,3)
        self.slope=fr.nn.PReLU(3,.3)
        self.tied=self.slope
        self.loss=fr.nn.GaussianNLLLoss(full=True,eps=.15,reduction='mean')
        self.register_buffer('variance',fr.tensor([[.1,.4,.7],[.2,.5,.9]]))
    def forward(self,a,b,target):
        prediction=self.slope(self.map(a,b))+self.tied(self.map(a,b)*.2)
        return self.loss(prediction,target,self.variance)
model=Model();optimizer=fr.optim.AdamW(model.parameters(),lr=.01);rng=fr.Generator(71)
def step():
    a=fr.Tensor.rand([2,2],generator=rng)*2-1
    b=fr.Tensor.rand([2,3],generator=rng)*2-1
    target=fr.Tensor.rand([2,3],generator=rng)
    loss=model(a,b,target);value=loss.item()
    loss.backward();optimizer.step();optimizer.zero_grad();return value
if sys.argv[1]=='save':
    step();fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=rng,step=1).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(model,optimizers={'main':optimizer},generator=rng)
losses=[step() for _ in range(3)]
state=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=rng,step=4).tensors()
def encoded(value):
    flat=value.reshape([-1]).tolist()
    return [value.dtype,value.shape,[float.hex(x) for x in flat] if value.dtype in ('float32','float64') else flat]
print(json.dumps({'losses':[float.hex(x) for x in losses],'alias':model.slope is model.tied,'tensors':{k:encoded(v) for k,v in state.items()}},sort_keys=True))
'''
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as directory:
            results = [subprocess.run([sys.executable,'-c',script,mode,str(Path(directory)/'state')],env=env,capture_output=True,text=True,check=True,timeout=60).stdout for mode in ('save','load')]
        self.assertEqual(json.loads(results[0]), json.loads(results[1]))
        self.assertTrue(json.loads(results[0])['alias'])


if __name__ == '__main__': unittest.main()
