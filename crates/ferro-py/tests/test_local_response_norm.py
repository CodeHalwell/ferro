"""Actual Torch channel-window normalization, native gradients and state."""
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import Module, Linear, LocalResponseNorm
from ferro.nn.functional import local_response_norm
import ferro.nn._local_response_norm as implementation
from test_layer_norm import native, close


class Network(Module):
    def __init__(self, size=4):
        self._initialize()
        self.linear = Linear(3,5)
        self.norm = LocalResponseNorm(size,alpha=.15,beta=.8,k=.9)

    def forward(self, value):
        return self.norm(self.linear(value).transpose(1,2))


class LocalResponseNormTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(3007)

    def compare(self, shape, size, dtype=torch.float32, strided=False, **kwargs):
        base=torch.randn(shape,dtype=dtype)
        value=base.transpose(0,-1).contiguous().transpose(0,-1).detach().requires_grad_() if strided else base.requires_grad_()
        x=native(base.transpose(0,-1).contiguous()).transpose(0,len(shape)-1).requires_grad_(True) if strided else native(value)
        actual=local_response_norm(x,size,**kwargs)
        expected=torch.nn.functional.local_response_norm(value,size,**kwargs)
        close(actual,expected)
        seed=torch.randn(shape,dtype=dtype)
        (actual*native(seed)).sum().backward();(expected*seed).sum().backward()
        close(x.grad,value.grad)

    def test_window_sizes_ranks_dtypes_strides_and_all_input_adjoints(self):
        for dtype in (torch.float32,torch.float64):
            for shape in ((2,5,4),(2,5,3,4),(2,5,2,3,4),(2,3,2,2,2,2)):
                for size in (1,2,3,4,9):
                    with self.subTest(dtype=dtype,shape=shape,size=size):
                        self.compare(shape,size,dtype,len(shape)<=4,alpha=.13,beta=.71,k=1.2)

    def test_strided_higher_rank_extension_matches_contiguous_torch_algebra(self):
        for dtype in (torch.float32,torch.float64):
            for shape in ((2,5,2,3,4),(2,3,2,2,2,2)):
                value=torch.randn(shape,dtype=dtype).transpose(0,-1).contiguous().transpose(0,-1).detach().requires_grad_()
                x=native(value.transpose(0,-1).contiguous()).transpose(0,len(shape)-1).requires_grad_(True)
                with self.assertRaisesRegex(RuntimeError,'view size is not compatible'):
                    torch.nn.functional.local_response_norm(value,4)
                actual=local_response_norm(x,4)
                expected=torch.nn.functional.local_response_norm(value.contiguous(),4)
                close(actual,expected)
                seed=torch.randn(shape,dtype=dtype)
                (actual*native(seed)).sum().backward();(expected*seed).sum().backward();close(x.grad,value.grad)

    def test_asymmetric_even_windows_and_full_padded_divisor(self):
        for size in (2,4,8):
            x=fr.tensor([1.,2.,3.,4.]).reshape([1,4,1])
            y=local_response_norm(x,size,alpha=1.,beta=1.,k=0.)
            manual=[(i+1)/(sum((j+1)**2 for j in range(max(0,i-size//2),min(4,i+(size-1)//2+1)))/size) for i in range(4)]
            torch.testing.assert_close(torch.tensor(y.tolist()).flatten(),torch.tensor(manual))
            close(y,torch.nn.functional.local_response_norm(torch.arange(1.,5.).reshape(1,4,1),size,alpha=1.,beta=1.,k=0.))

    def test_empty_identity_zero_gradients_and_virtual_shapes(self):
        for dtype in (torch.float32,torch.float64):
            for shape in ((0,3,4),(2,0,4),(2,3,0),(0,3,0,2)):
                value=torch.empty(shape,dtype=dtype,requires_grad=True);x=native(value)
                actual=local_response_norm(x,4);expected=torch.nn.functional.local_response_norm(value,4)
                self.assertIs(actual,x);self.assertIs(expected,value)
                actual.sum().backward();expected.sum().backward();close(x.grad,value.grad)
        x=fr.ones([1,1,1]).expand(sys.maxsize,0,1)
        self.assertIs(local_response_norm(x,3),x)

    def test_module_signature_modes_configuration_and_checkpoint(self):
        self.assertEqual(list(inspect.signature(LocalResponseNorm).parameters),list(inspect.signature(torch.nn.LocalResponseNorm).parameters))
        module=LocalResponseNorm(4,.15,.8,.9);oracle=torch.nn.LocalResponseNorm(4,.15,.8,.9)
        self.assertEqual(list(module.parameters()),[])
        for mode in (False,True):
            module.train(mode);oracle.train(mode)
            value=torch.randn(2,5,3,requires_grad=True);x=native(value)
            close(module(x),oracle(value))
        source=Network();optimizer=fr.optim.Adam(source.parameters(),lr=.004);rng=fr.Generator(86)
        for _ in range(2):
            source(fr.Tensor.randn([2,4,3],generator=rng)).square().mean().backward();optimizer.step();optimizer.zero_grad()
        source.eval()
        saved=fr.checkpoint.snapshot_training(source,optimizers={'main':optimizer},generator=rng,step=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lrn.safetensors';saved.save(str(path))
            saved=fr.checkpoint.load(str(path))
        target=Network();other=fr.optim.Adam(target.parameters(),lr=.004);other_rng=fr.Generator(31)
        saved.restore(target,optimizers={'main':other},generator=other_rng)
        self.assertFalse(target.training);self.assertFalse(target.norm.training)
        for _ in range(3):
            a=source(fr.Tensor.randn([2,4,3],generator=rng));b=target(fr.Tensor.randn([2,4,3],generator=other_rng))
            self.assertEqual(a.tolist(),b.tolist())
            a.square().mean().backward();b.square().mean().backward();optimizer.step();other.step();optimizer.zero_grad();other.zero_grad()
        def state(m,o,g):return {k:(v.dtype,v.shape,v.tolist()) for k,v in fr.checkpoint.snapshot_training(m,optimizers={'main':o},generator=g).tensors().items()}
        self.assertEqual(state(source,optimizer,rng),state(target,other,other_rng))
        wrong=Network(3);before=[p.tensor().tolist() for p in wrong.parameters()]
        with self.assertRaises(ValueError):saved.restore(wrong,optimizers={'main':fr.optim.Adam(wrong.parameters(),lr=.004)},generator=fr.Generator(1))
        self.assertEqual(before,[p.tensor().tolist() for p in wrong.parameters()])

    def test_zero_and_nonfinite_scalar_algebra(self):
        for dtype in (torch.float32,torch.float64):
            for options in ({'alpha':0.},{'beta':0.},{'k':0.},{'alpha':-.2,'beta':2.},
                            {'alpha':float('inf')},{'alpha':float('nan')},{'beta':float('inf')},{'k':float('inf')}):
                with self.subTest(dtype=dtype,options=options):self.compare((2,4,3),3,dtype,**options)
            for values,options in (([0.]*4,{'alpha':.1,'k':0.,'beta':0.}),
                                   ([float('inf'),float('nan'),-float('inf'),0.],{}),([0.]*4,{'k':0.})):
                value=torch.tensor(values,dtype=dtype).reshape(1,4,1).requires_grad_();x=native(value)
                actual=local_response_norm(x,3,**options);expected=torch.nn.functional.local_response_norm(value,3,**options)
                close(actual,expected);actual.sum().backward();expected.sum().backward();close(x.grad,value.grad)

    def test_shape_options_dtype_and_preallocation_failures(self):
        for size in (0,-1,True,2.5,sys.maxsize+1):
            with self.assertRaises(ValueError):LocalResponseNorm(size)
        for name in ('alpha','beta','k'):
            with self.assertRaises(TypeError):LocalResponseNorm(3,**{name:'bad'})
        for shape in ([],[3],[2,3]):
            with self.assertRaises(ValueError):local_response_norm(fr.ones(shape),3)
        for dtype in ('int64','bool'):
            with self.assertRaises(NotImplementedError):local_response_norm(fr.ones([2,3,4],dtype=dtype),3)
        huge=fr.ones([1,1,1]).expand(1,sys.maxsize//4,2)
        with patch.object(implementation,'cat',side_effect=AssertionError('allocation reached')):
            with self.assertRaises(ValueError):local_response_norm(huge,3)
        with self.assertRaises(NotImplementedError):fr.capture(lambda:local_response_norm(fr.ones([2,3,4]),3))

    def test_double_directional_derivative_uses_native_graph(self):
        value=torch.randn(1,3,2,requires_grad=True);x=native(value)
        expected=torch.nn.functional.local_response_norm(value,2,alpha=.2,beta=.7,k=1.1)
        actual=local_response_norm(x,2,alpha=.2,beta=.7,k=1.1)
        gradient=torch.autograd.grad(expected.sum(),value,create_graph=True)[0]
        native_gradient=fr.autograd.grad(actual.sum(),(x,),create_graph=True)[0]
        close(native_gradient,gradient)
        direction=torch.randn_like(value)
        second=torch.autograd.grad((gradient*direction).sum(),value)[0]
        native_second=fr.autograd.grad((native_gradient*native(direction)).sum(),(x,))[0]
        close(native_second,second)


if __name__=='__main__':unittest.main(verbosity=2)
