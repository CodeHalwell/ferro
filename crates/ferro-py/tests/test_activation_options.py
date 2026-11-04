"""Actual Torch activation options, typed softplus and explicit mutation limits."""
import inspect
import math
from fractions import Fraction
from decimal import Decimal
import sys
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    value=value.detach()
    result=fr.tensor(value.tolist(),dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


class ActivationOptionsTests(unittest.TestCase):
    def close(self, actual, expected, exact=False):
        self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.dtype,str(expected.dtype).removeprefix('torch.'))
        value=torch.tensor(actual.tolist(),dtype=expected.dtype).reshape(expected.shape)
        tolerance=0. if exact else 2e-12 if expected.dtype==torch.float64 else 3e-6
        torch.testing.assert_close(value,expected.detach(),atol=tolerance,rtol=tolerance*10,equal_nan=True)
        zero=expected.detach()==0
        self.assertEqual(torch.signbit(value[zero]).tolist(),torch.signbit(expected.detach()[zero]).tolist())

    def test_softplus_real_options_typed_strides_and_gradients(self):
        for dtype in (torch.float32,torch.float64):
            for beta in (-2.,0.,.5,1.,3.):
                for threshold in (-1.,0.,1.,20.,math.inf):
                    with self.subTest(dtype=dtype,beta=beta,threshold=threshold):
                        original=torch.tensor([[-2.1,-.3,.2],[.7,1.9,3.2]],dtype=dtype,requires_grad=True)
                        value=native(original,True)
                        expected=torch.nn.functional.softplus(original.T,beta=beta,threshold=threshold)
                        actual=F.softplus(input=value.T,beta=beta,threshold=threshold)
                        self.close(actual,expected)
                        seed=torch.tensor([[.2,-.5],[1.3,.7],[-.9,.4]],dtype=dtype)
                        actual.backward_with(native(seed));expected.backward(seed)
                        self.close(value.grad,original.grad)
                        self.assertEqual(actual.stride(),expected.stride())

    def test_nonfinite_underflow_and_staged_overflow_adjoint(self):
        for dtype in (torch.float32,torch.float64):
            data=torch.tensor([-math.inf,-100.,-2.,-0.,0.,2.,100.,math.inf,math.nan],dtype=dtype)
            for beta,threshold in ((0.,20.),(0.,-1.),(-1.,0.),(math.inf,20.),(-math.inf,20.),(math.nan,20.),(1.,math.nan),(1.,math.inf),(1e-100,20.)):
                with self.subTest(dtype=dtype,beta=beta,threshold=threshold):
                    x=data.clone().requires_grad_();value=native(data,True)
                    actual=F.softplus(value,beta,threshold);expected=torch.nn.functional.softplus(x,beta,threshold)
                    self.close(actual,expected)
                    actual.sum().backward();expected.sum().backward();self.close(value.grad,x.grad)
            x=torch.tensor([2.,-100.],dtype=dtype,requires_grad=True);value=native(x,True)
            actual=F.softplus(value,threshold=math.inf);expected=torch.nn.functional.softplus(x,threshold=math.inf)
            seed=torch.tensor([1e38 if dtype==torch.float32 else 1e308,1.],dtype=dtype)
            actual.backward_with(native(seed));expected.backward(seed)
            self.close(value.grad,x.grad,True)
            self.assertEqual(value.grad.tolist()[0],math.inf)
        value=fr.tensor([-100.],requires_grad=True)
        output=F.softplus(value);output.sum().backward()
        self.assertEqual(output.item(),3.783505853677006e-44)
        self.assertEqual(value.grad.item(),3.783505853677006e-44)

    def test_softplus_scalar_types_precision_and_invalid_arguments(self):
        for key in ('beta','threshold'):
            for scalar in (True,0,2,2**63,2**64-1,.123456789012345,math.inf,math.nan):
                for dtype in (torch.float32,torch.float64):
                    kwargs={key:scalar};x=torch.tensor([-.3,.7],dtype=dtype)
                    self.close(F.softplus(native(x),**kwargs),torch.nn.functional.softplus(x,**kwargs))
            for scalar in (fr.tensor(.5),fr.tensor(.5,dtype='float64'),fr.tensor(2,dtype='int64'),fr.tensor(True,dtype='bool')):
                reference=torch.tensor(scalar.item(),dtype=getattr(torch,scalar.dtype))
                self.close(F.softplus(fr.ones(2),**{key:scalar}),torch.nn.functional.softplus(torch.ones(2),**{key:reference}))
            for value in (None,'1',Fraction(1,2),Decimal('.5'),fr.tensor([.5]),fr.tensor(.5,requires_grad=True)):
                with self.assertRaises(TypeError):F.softplus(fr.ones(2),**{key:value})
            with self.assertRaises(OverflowError):F.softplus(fr.ones(2),**{key:2**64})
            for shape in ((2,),(0,)):
                with self.assertRaises(ValueError):F.softplus(fr.zeros(shape),**{key:1e100})
                with self.assertRaises(RuntimeError):torch.nn.functional.softplus(torch.zeros(shape),**{key:1e100})
        x=torch.tensor([.25,.5],dtype=torch.float64)
        self.close(F.softplus(native(x),beta=1.0000000000000002),torch.nn.functional.softplus(x,beta=1.0000000000000002))

    def test_softplus_modes_empty_versions_capacity_and_native_jvp(self):
        for dtype in ('float32','float64'):
            x=fr.tensor([.2,.8],dtype=dtype,requires_grad=True)
            with fr.no_grad():self.assertFalse(F.softplus(x,.75,1.).requires_grad)
            with self.assertRaises(ValueError):fr.capture(lambda:F.softplus(x))
            output=F.softplus(x,.75,1.)
            with self.assertRaises(ValueError):fr.autograd.grad(output.sum(),[x],create_graph=True)
            with fr.no_grad():output.detach().fill_(9.)
            gradient=fr.autograd.grad(output.sum(),[x])[0]
            expected=torch.tensor([.2,.8],dtype=getattr(torch,dtype),requires_grad=True)
            torch.nn.functional.softplus(expected,.75,1.).sum().backward();self.close(gradient,expected.grad)
            with fr.no_grad():x.fill_(4.)
            with self.assertRaises(ValueError):fr.autograd.grad(output.sum(),[x])
            base=fr.zeros(0,8,dtype=dtype,requires_grad=True);output=F.softplus(base.narrow(1,4,4),math.nan,math.nan)
            self.assertEqual(output.shape,[0,4]);output.sum().backward();self.assertEqual(base.grad.shape,[0,8])
        for dtype in ('int64','bool'):
            with self.assertRaises((TypeError,ValueError)):F.softplus(fr.ones(2,dtype=dtype))
        huge=fr.ones(1).expand(sys.maxsize//4+1)
        with self.assertRaises(ValueError):F.softplus(huge)
        x=torch.tensor([-.7,.2,1.9]);v=torch.tensor([.3,-.2,.8])
        actual=fr.func.jvp(lambda value:F.softplus(value,.75,1.),(native(x),),(native(v),))
        expected=torch.func.jvp(lambda value:torch.nn.functional.softplus(value,.75,1.),(x,),(v,))
        for value,target in zip(actual,expected):self.close(value,target)
        with self.assertRaises(ValueError):fr.func.jvp(lambda value:F.softplus(value,.75,1.),(native(x,True),),(native(v),))

    def test_canonical_functional_and_module_input_signatures(self):
        options=(('relu','ReLU',{}),('sigmoid','Sigmoid',{}),('tanh','Tanh',{}),('gelu','GELU',{}),
                 ('silu','SiLU',{}),('mish','Mish',{}),('elu','ELU',{'alpha':.7}),('celu','CELU',{'alpha':.7}),
                 ('selu','SELU',{}),('leaky_relu','LeakyReLU',{'negative_slope':.3}),('relu6','ReLU6',{}),
                 ('hardtanh','Hardtanh',{'min_val':-.5,'max_val':.8}),('hardsigmoid','Hardsigmoid',{}),('hardswish','Hardswish',{}),
                 ('hardshrink','Hardshrink',{'lambd':.3}),('softshrink','Softshrink',{'lambd':.3}),('tanhshrink','Tanhshrink',{}),
                 ('softsign','Softsign',{}),('logsigmoid','LogSigmoid',{}),('threshold','Threshold',{'threshold':.3,'value':-.7}),
                 ('softmax','Softmax',{'dim':1}),('log_softmax','LogSoftmax',{'dim':1}),('softmin','Softmin',{'dim':1}))
        for name,cls,kwargs in options:
            with self.subTest(name=name):
                x=torch.tensor([[-.7,.2,1.3],[.9,-.4,.6]],requires_grad=True);value=native(x,True)
                actual=getattr(F,name)(input=value,**kwargs);expected=getattr(torch.nn.functional,name)(input=x,**kwargs)
                self.close(actual,expected)
                actual.sum().backward();expected.sum().backward();self.close(value.grad,x.grad)
                self.close(getattr(fr.nn,cls)(**kwargs)(input=value),getattr(torch.nn,cls)(**kwargs)(input=x))
                self.assertIn('input',inspect.signature(getattr(F,name)).parameters)

    def test_inplace_false_options_and_explicit_rejections_preserve_inputs(self):
        options=(('relu','ReLU',{}),('silu','SiLU',{}),('mish','Mish',{}),('elu','ELU',{}),('celu','CELU',{}),('selu','SELU',{}),
                 ('leaky_relu','LeakyReLU',{}),('relu6','ReLU6',{}),('hardtanh','Hardtanh',{}),('hardsigmoid','Hardsigmoid',{}),
                 ('hardswish','Hardswish',{}),('threshold','Threshold',{'threshold':.3,'value':0.}))
        for name,cls,kwargs in options:
            value=fr.tensor([-.5,.7],requires_grad=True);before=value.tolist()
            model=getattr(fr.nn,cls)(inplace=False,**kwargs)
            self.assertFalse(model.inplace);self.assertNotIn('inplace',model.__dict__)
            with self.assertRaises(AttributeError):model.inplace=True
            with self.assertRaises(NotImplementedError):getattr(fr.nn,cls)(inplace=True,**kwargs)
            with self.assertRaises(NotImplementedError):getattr(F,name)(input=value,inplace=True,**kwargs)
            with self.assertRaises(TypeError):getattr(F,name)(input=value,inplace=1,**kwargs)
            self.close(getattr(F,name)(input=value,inplace=False,**kwargs),getattr(torch.nn.functional,name)(torch.tensor(before),inplace=False,**kwargs))
            self.assertEqual(value.tolist(),before);self.assertIsNone(value.grad)
        for cls in ('Sigmoid','Tanh','Tanhshrink','Softsign','LogSigmoid'):
            with self.assertRaises(TypeError):getattr(fr.nn,cls)(inplace=False)

    def test_softplus_module_config_state_and_exact_training_restart(self):
        def make(beta=.75,threshold=1.):
            model=fr.nn.Sequential(fr.nn.Linear(2,3),fr.nn.Softplus(beta,threshold),fr.nn.Linear(3,1))
            return model,fr.optim.AdamW(model.parameters(),lr=.01),fr.Generator(19)
        model,optimizer,generator=make();x=fr.tensor([[.2,-.4],[.7,.8]])
        def step(model,optimizer):
            optimizer.zero_grad();model(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
        step(model,optimizer)
        snapshot=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=1)
        other,second,rng=make()
        snapshot.restore(other,optimizers={'main':second},generator=rng)
        for _ in range(3):step(model,optimizer);step(other,second)
        left=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=4).tensors()
        right=fr.checkpoint.snapshot_training(other,optimizers={'main':second},generator=rng,step=4).tensors()
        self.assertEqual(set(left),set(right))
        for key in left:self.assertEqual(left[key].tolist(),right[key].tolist(),key)
        bad,opt,bad_rng=make(threshold=2.)
        before=[parameter.tolist() for parameter in bad.parameters()]
        with self.assertRaises(ValueError):snapshot.restore(bad,optimizers={'main':opt},generator=bad_rng)
        self.assertEqual([parameter.tolist() for parameter in bad.parameters()],before)
        module=fr.nn.Softplus(0.,-1.);self.close(module(input=fr.tensor([-2.,2.])),torch.nn.Softplus(0.,-1.)(torch.tensor([-2.,2.])))
        self.assertIs(type(fr.nn.Softplus(2,20).beta),float)
        self.assertIs(type(fr.nn.Softplus(2**63+1,20).beta),int)
        self.assertEqual(fr.nn.Softplus(2**63+1,20).beta,2**63+1)


if __name__=='__main__':unittest.main(verbosity=2)
