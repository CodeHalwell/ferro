"""Typed exponential-linear activation contracts against actual Torch."""
import math
import unittest
from fractions import Fraction
from decimal import Decimal

import torch
import ferro as fr
from ferro.nn import functional as F
from ferro import checkpoint


def native(value, grad=None):
    result = fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(list(value.shape))
    return result.requires_grad_(value.requires_grad if grad is None else grad)


class EluFamilyTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), rtol=4e-6 if expected.dtype == torch.float32 else 6e-14, atol=0, equal_nan=True)
        mask = expected.detach() == 0
        self.assertEqual(torch.signbit(value[mask]).tolist(), torch.signbit(expected.detach()[mask]).tolist())

    def test_dtypes_options_strides_and_original_input_gradients(self):
        for name in ('elu', 'celu', 'selu'):
            for dtype in (torch.float32, torch.float64):
                for alpha in ((.7, -2., 3.7) if name != 'selu' else (1.,)):
                    for layout in ('transpose', 'slice', 'broadcast'):
                        with self.subTest(name=name, dtype=dtype, alpha=alpha, layout=layout):
                            raw = torch.linspace(-1.4, 1.4, 12, dtype=dtype).reshape(3, 4).requires_grad_()
                            base = native(raw)
                            if layout == 'transpose':
                                x, y = base.T, raw.T
                            elif layout == 'slice':
                                x, y = base[:, ::2], raw[:, ::2]
                            else:
                                x, y = base[:1].expand([4, 4]), raw[:1].expand(4, 4)
                            options = {} if name == 'selu' else {'alpha': alpha}
                            out, reference = getattr(F, name)(input=x, **options), getattr(torch.nn.functional, name)(input=y, **options)
                            self.same(out, reference); self.assertEqual(out.stride(), reference.stride())
                            seed = torch.linspace(.2, 1.1, reference.numel(), dtype=dtype).reshape(reference.shape)
                            actual, = fr.autograd.grad(out, base, native(seed))
                            expected, = torch.autograd.grad(reference, raw, seed)
                            self.same(actual, expected)

    def test_zero_nan_subnormal_and_extreme_seed_staging(self):
        for name in ('elu', 'celu', 'selu'):
            for dtype in (torch.float32, torch.float64):
                for alpha in ((0., -2., math.inf, -math.inf, math.nan, 1e38) if name != 'selu' else (1.,)):
                    if name == 'celu' and alpha == 0: continue
                    for seed_value in (.3, -0., 3e38 if dtype == torch.float32 else 1e308):
                        with self.subTest(name=name, dtype=dtype, alpha=alpha, seed=seed_value):
                            x = torch.tensor([-math.inf,-100.,-2.,-0.,0.,.2,math.inf,math.nan], dtype=dtype, requires_grad=True)
                            a = native(x); opts = {} if name == 'selu' else {'alpha': alpha}
                            output, reference = getattr(F, name)(a, **opts), getattr(torch.nn.functional, name)(x, **opts)
                            self.same(output, reference)
                            seed = torch.full_like(x, seed_value)
                            self.same(fr.autograd.grad(output,a,native(seed))[0],torch.autograd.grad(reference,x,seed)[0])

    def test_celu_double_forward_and_float_backward_scalar_stages(self):
        for alpha in (.7, 3.7, 1e-40, 1e-100, 1e38):
            x = torch.tensor([-2.,-0.,0.,1.,math.nan],dtype=torch.float64,requires_grad=True);a=native(x)
            output,reference=F.celu(a,alpha),torch.nn.functional.celu(x,alpha)
            self.same(output,reference)
            seed=torch.full_like(x,.3)
            self.same(fr.autograd.grad(output,a,native(seed))[0],torch.autograd.grad(reference,x,seed)[0])
        for count in (0,2):
            x=torch.ones(count,dtype=torch.float64,requires_grad=True);a=native(x)
            other=fr.ones(count,dtype='float64',requires_grad=True);torch_other=torch.ones_like(x,requires_grad=True)
            output,reference=F.celu(a,1e100),torch.nn.functional.celu(x,1e100)
            self.same(output,reference)
            self.assertIsNone(fr.autograd.grad(output,other,fr.ones_like(output),allow_unused=True)[0])
            self.assertIsNone(torch.autograd.grad(reference,torch_other,torch.ones_like(reference),allow_unused=True)[0])
            self.same(fr.autograd.grad(output,output,fr.ones_like(output))[0],torch.ones_like(reference))
            with self.assertRaisesRegex(ValueError,'float32'):
                fr.autograd.grad(output,a,fr.ones_like(output))
            with self.assertRaisesRegex(RuntimeError,'float'):
                torch.autograd.grad(reference,x,torch.ones_like(reference))

    def test_scalar_categories_precision_overflow_and_module_options(self):
        import numpy as np
        class Floatable:
            def __float__(self):return .7
        for name in ('elu','celu'):
            for value in (True,1,.7,2**63+2**39+1,np.float32(.7),np.int64(2),np.bool_(True)):
                for dtype in (torch.float32,torch.float64):
                    x=torch.tensor([-1.,-0.,0.,1.],dtype=dtype)
                    self.same(getattr(F,name)(native(x),value),getattr(torch.nn.functional,name)(x,value))
            x=fr.tensor([-1.,0.,1.]);y=torch.tensor([-1.,0.,1.])
            self.same(getattr(F,name)(x,fr.tensor(.7)),getattr(torch.nn.functional,name)(y,torch.tensor(.7)))
            for bad in (Fraction(1,2),Decimal('.7'),Floatable(),fr.tensor([.7]),fr.tensor(.7,requires_grad=True)):
                with self.assertRaises((TypeError,ValueError,OverflowError)):getattr(F,name)(x,bad)
            with self.assertRaises((ValueError,OverflowError)):getattr(F,name)(fr.zeros(0),1e100)
            with self.assertRaises((ValueError,OverflowError)):getattr(F,name)(x,alpha=1e100)
        for name in ('ELU','CELU','SELU'):
            cls,reference=getattr(fr.nn,name),getattr(torch.nn,name)
            opts={} if name=='SELU' else {'alpha':.7}
            layer,other=cls(**opts),reference(**opts)
            self.same(layer(input=fr.tensor([-.2,0.,.3])),other(input=torch.tensor([-.2,0.,.3])))
            self.assertFalse(layer.inplace)
            with self.assertRaises(NotImplementedError):cls(inplace=True)

    def test_storage_versions_empty_shapes_and_explicit_ad_boundaries(self):
        for name in ('elu','celu','selu'):
            for dtype in ('float32','float64'):
                x=fr.tensor([-.2,.3],dtype=dtype,requires_grad=True);out=getattr(F,name)(x)
                before=fr.autograd.grad(out,x,fr.ones_like(out))[0]
                out.detach().fill_(3.)
                self.same(fr.autograd.grad(out,x,fr.ones_like(out))[0],torch.tensor(before.tolist(),dtype=getattr(torch,dtype)))
                x.detach().fill_(2.)
                with self.assertRaisesRegex(ValueError,'modified'):fr.autograd.grad(out,x,fr.ones_like(out))
                empty=fr.zeros(2,0,3,dtype=dtype,requires_grad=True);output=getattr(F,name)(empty)
                self.assertEqual(output.shape,[2,0,3]);self.assertEqual(fr.autograd.grad(output,empty,fr.ones_like(output))[0].shape,[2,0,3])
            x=fr.tensor([-.2,.3],requires_grad=True)
            with self.assertRaises(ValueError):fr.autograd.grad(getattr(F,name)(x),x,fr.ones_like(x),create_graph=True)
            with self.assertRaises(ValueError):fr.capture(lambda:getattr(F,name)(x))
            a=fr.tensor([-.2,.3]);v=fr.tensor([.7,-.4])
            out,tangent=fr.func.jvp(getattr(F,name),(a,),(v,))
            p,t=torch.func.jvp(getattr(torch.nn.functional,name),(torch.tensor(a.tolist()),),(torch.tensor(v.tolist()),))
            self.same(out,p);self.same(tangent,t)

    def test_complete_checkpoint_preserves_config_optimizer_and_rng(self):
        class Model(fr.nn.Module):
            def __init__(self,alpha=.7):
                super().__init__()
                self.linear=fr.nn.Linear(2,2)
                self.elu=fr.nn.ELU(alpha)
                self.celu=fr.nn.CELU(3.7)
                self.selu=fr.nn.SELU()
            def forward(self,input):return self.selu(self.celu(self.elu(self.linear(input))))
        model=Model();opt=fr.optim.AdamW(model.parameters(),lr=.01)
        generator=fr.Generator(47);x=fr.tensor([[-.7,.2],[.4,-.8]]);target=fr.tensor([[.1,.3],[-.2,.4]])
        def step():
            opt.zero_grad();loss=(model(x)-target).square().sum();loss.backward();opt.step();opt.zero_grad();return loss.item()
        step();snapshot=checkpoint.snapshot_training(model,optimizers={'main':opt},generator=generator,step=1)
        losses=[step() for _ in range(3)];expected=checkpoint.snapshot_training(model,optimizers={'main':opt},generator=generator,step=4)
        snapshot.restore(model,optimizers={'main':opt},generator=generator);self.assertEqual([step() for _ in range(3)],losses)
        actual=checkpoint.snapshot_training(model,optimizers={'main':opt},generator=generator,step=4)
        self.assertEqual(set(actual.tensors()),set(expected.tensors()))
        for name,value in actual.tensors().items():self.assertEqual(value.tolist(),expected.tensors()[name].tolist(),name)
        bad=Model(.8);bad_opt=fr.optim.AdamW(bad.parameters(),lr=.01)
        with self.assertRaises(ValueError):snapshot.restore(bad,optimizers={'main':bad_opt},generator=fr.Generator(1))


if __name__=='__main__':unittest.main()
