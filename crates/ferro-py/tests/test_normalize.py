"""General p-normalization and canonical functional normalization keywords."""
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    result = fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


class NormalizeTests(unittest.TestCase):
    def close(self, value, expected):
        self.assertEqual(value.shape, list(expected.shape))
        self.assertEqual(value.dtype, str(expected.dtype).removeprefix('torch.'))
        actual = torch.tensor(value.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 2e-12 if expected.dtype == torch.float64 else 4e-6
        torch.testing.assert_close(actual, expected.detach(), atol=tolerance, rtol=tolerance * 10, equal_nan=True)

    def compare(self, data, p, dim, eps=1e-12, strided=False):
        expected_input = data.detach().clone().requires_grad_()
        actual_input = native(data, True)
        x, y = actual_input, expected_input
        if strided:
            x, y = x.transpose(0, 1), y.transpose(0, 1)
        actual = F.normalize(input=x, p=p, dim=dim, eps=eps)
        expected = torch.nn.functional.normalize(input=y, p=p, dim=dim, eps=eps)
        self.close(actual, expected)
        seed = torch.linspace(-.8, 1.3, data.numel(), dtype=data.dtype).reshape(y.shape)
        actual.backward_with(native(seed)); expected.backward(seed)
        self.close(actual_input.grad, expected_input.grad)

    def test_real_orders_tuple_axes_typed_strides_and_input_gradients(self):
        for dtype in (torch.float32, torch.float64):
            data = torch.tensor([-.3,.7,1.2,-2.,.2,.6,1.5,-.9,2.3,-1.1,.4,.8], dtype=dtype).reshape(2,3,2)
            for p in (-math.inf,-2.,-.5,0.,.5,1.,1.5,2.,3.,math.inf):
                for dim in (0,-1,(0,2),(),None):
                    with self.subTest(dtype=dtype,p=p,dim=dim):self.compare(data,p,dim,strided=True)

    def test_zero_rows_tied_extrema_and_clamp_equality(self):
        for dtype in (torch.float32,torch.float64):
            data=torch.tensor([[0.,0.,0.],[.5,0.,0.],[2.,-2.,1.]],dtype=dtype)
            for p in (-math.inf,-2.,0.,.5,1.,2.,3.,math.inf):
                for eps in ((.75,1e-12) if p == .5 else (.5,1e-12)):
                    with self.subTest(dtype=dtype,p=p,eps=eps):self.compare(data,p,1,eps)
        # Stable native accumulation gives .5 exactly; Torch's float32 sqrt
        # then square produces .49999997 and selects the other clamp branch.
        data=torch.tensor([[.5,0.,0.]])
        reference=torch.nn.functional.normalize(data.requires_grad_(),p=.5,dim=1,eps=.5)
        value=native(data,True);actual=F.normalize(value,p=.5,dim=1,eps=.5)
        self.close(actual,reference)
        actual.sum().backward();reference.sum().backward()
        self.assertEqual(value.grad.tolist(),[[0.,2.,2.]])
        self.assertEqual(data.grad.tolist(),[[2.,2.,2.]])

    def test_nonfinite_zero_negative_and_tiny_epsilon(self):
        for dtype in (torch.float32,torch.float64):
            for eps in (0.,-1.,math.inf,-math.inf,math.nan,1e-100):
                data=torch.tensor([[0.,0.],[3.,4.]],dtype=dtype)
                with self.subTest(dtype=dtype,eps=eps):self.compare(data,2.,1,eps)
            for p in (0.,1.,2.,3.,math.inf,-math.inf):
                data=torch.tensor([[math.nan,1.],[math.inf,1.],[-math.inf,2.]],dtype=dtype)
                with self.subTest(dtype=dtype,p=p):self.compare(data,p,(1,),.5)

    def test_scalar_and_empty_reductions(self):
        for dtype in (torch.float32,torch.float64):
            for value in (0.,-2.,3.):
                for p in (0.,.5,2.,-2.,math.inf):self.compare(torch.tensor(value,dtype=dtype),p,0)
            for shape,dim in (((2,0),1),((0,3),1),((2,0,3),(0,2))):
                for p in (0.,.5,1.,2.,3.):self.compare(torch.empty(shape,dtype=dtype),p,dim)
            for p in (-1.,math.inf,-math.inf):
                with self.assertRaises(ValueError):F.normalize(fr.empty(2,0,dtype=str(dtype).removeprefix('torch.')),p=p,dim=1)
                with self.assertRaises(RuntimeError):torch.nn.functional.normalize(torch.empty(2,0,dtype=dtype),p=p,dim=1)

    def test_validation_and_capacity_precede_materialization(self):
        for dtype in ('bool','int64'):
            with self.assertRaises(ValueError):F.normalize(fr.ones(2,3,dtype=dtype))
        for p in (None,'fro',math.nan):
            with self.assertRaises((TypeError,ValueError)):F.normalize(fr.ones(2,3),p=p)
        for dim in (2,-3,(1,1),[0,True],'1'):
            with self.assertRaises((TypeError,ValueError)):F.normalize(fr.ones(2,3,dtype='float64'),dim=dim)
        with self.assertRaises(OverflowError):F.normalize(fr.empty(0,3),eps=1e100)
        with self.assertRaises(NotImplementedError):F.normalize(fr.ones(2,3),out=fr.zeros(2,3))
        huge=fr.ones(1,dtype='float64').expand(sys.maxsize//8+1)
        import ferro.nn._normalize as implementation
        with patch.object(implementation,'vector_norm',side_effect=AssertionError('read oversized input')):
            with self.assertRaises(ValueError):F.normalize(huge,p=1.,dim=0)
        empty=fr.zeros(1,0,dtype='float64').expand(2**60,0)
        self.assertEqual(F.normalize(empty,p=1.,dim=0).shape,[2**60,0])

    def test_native_single_axis_nonfinite_rows(self):
        self.compare(torch.tensor([[math.nan,1.],[math.inf,1.],[-math.inf,0.],[3.,4.]]),2.,1)
        program="import ferro as f; x=f.zeros(1,0).expand(2**60,0).requires_grad_(True); y=f.nn.functional.normalize(x,dim=1); assert y.shape==[2**60,0]; y.sum().backward(); assert x.grad.shape==x.shape"
        result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_modes_saved_input_versions_and_explicit_derivative_limits(self):
        for dtype,p,dim in (('float32',2.,1),('float32',1.,(0,1)),('float64',3.,1)):
            x=fr.tensor([[.2,.4],[.6,.8]],dtype=dtype,requires_grad=True)
            with fr.no_grad():self.assertFalse(F.normalize(x,p,dim).requires_grad)
            with self.assertRaises((ValueError,NotImplementedError)):fr.capture(lambda:F.normalize(x,p,dim))
            loss=F.normalize(x,p,dim).sum()
            with self.assertRaises(ValueError):fr.autograd.grad(loss,[x],create_graph=True)
            with fr.no_grad():x.fill_(2.)
            with self.assertRaises(ValueError):fr.autograd.grad(loss,[x])
            self.assertIsNone(x.grad)
        for p in (1.,2.,3.):
            x=torch.tensor([[.2,.4],[.6,.8]]);v=torch.tensor([[.1,.3],[-.4,.7]])
            actual=fr.func.jvp(lambda value:F.normalize(value,p=p,dim=(1,)),(native(x),),(native(v),))
            expected=torch.func.jvp(lambda value:torch.nn.functional.normalize(value,p=p,dim=(1,)),(x,),(v,))
            for value,reference in zip(actual,expected):self.close(value,reference)
            with self.assertRaises(ValueError):
                fr.func.jvp(lambda value:F.normalize(value,p=p,dim=(1,)),(native(x,True),),(native(v),))
        self.assertTrue(fr.is_grad_enabled())

    def test_canonical_input_keywords_for_functionals_and_modules(self):
        x=torch.tensor([[.2,-.4],[.6,.9]],requires_grad=True);a=native(x,True)
        for name,kwargs in (('dropout',{'p':0.}),('layer_norm',{'normalized_shape':(2,)}),
                            ('rms_norm',{'normalized_shape':(2,)}),('group_norm',{'num_groups':1})):
            self.close(getattr(F,name)(input=a,**kwargs),getattr(torch.nn.functional,name)(input=x,**kwargs))
        self.assertIs(fr.nn.Dropout(0.)(input=a),a)
        batch=fr.nn.BatchNorm1d(2);reference=torch.nn.BatchNorm1d(2)
        self.close(batch(input=a),reference(input=x))


if __name__=='__main__':unittest.main(verbosity=2)
