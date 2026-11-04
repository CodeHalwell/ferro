"""Typed arithmetic names, fused alpha semantics and real native adjoints."""
import inspect
import subprocess
import sys
import unittest

import torch
import ferro as fr


def native(value, grad=False):
    result=fr.tensor(value.detach().tolist(),dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


class ArithmeticMethodsTests(unittest.TestCase):
    def same(self, actual, expected, exact=True):
        self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.dtype,str(expected.dtype).removeprefix('torch.'))
        value=torch.tensor(actual.tolist(),dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value,expected.detach(),atol=0 if exact else 3e-6,rtol=0 if exact else 3e-5,equal_nan=True)
        if expected.dtype.is_floating_point:
            zero=expected.detach()==0
            self.assertEqual(torch.signbit(value[zero]).tolist(),torch.signbit(expected.detach()[zero]).tolist())

    def test_all_dtype_pairs_broadcast_alpha_categories_and_named_aliases(self):
        dtypes=('bool','int64','float32','float64')
        for left in dtypes:
            for right in dtypes:
                x=torch.tensor([[1,0],[2,3]],dtype=getattr(torch,left));y=torch.tensor([[2],[1]],dtype=getattr(torch,right))
                a,b=native(x),native(y)
                for operation in ('add','sub'):
                    for alpha in (False,True,0,2,-3,2.,.25):
                        with self.subTest(left=left,right=right,operation=operation,alpha=alpha):
                            try:expected=getattr(torch,operation)(x,y,alpha=alpha)
                            except (RuntimeError,TypeError):
                                with self.assertRaises((ValueError,TypeError)):getattr(fr,operation)(a,b,alpha=alpha)
                            else:
                                self.same(getattr(fr,operation)(a,b,alpha=alpha),expected)
                                self.same(getattr(a,operation)(b,alpha=alpha),expected)
                self.same(a.multiply(b),torch.multiply(x,y))
                self.same(fr.mul(a,b),torch.mul(x,y))
                self.same(a.true_divide(b),torch.true_divide(x,y))
                self.same(fr.true_divide(a,b),torch.true_divide(x,y))
        self.same(fr.subtract(fr.ones(2),fr.ones(2),alpha=2),torch.subtract(torch.ones(2),torch.ones(2),alpha=2))
        self.same(fr.ones(2).subtract(fr.ones(2)),torch.ones(2).subtract(torch.ones(2)))

    def test_fused_primal_overflow_cancellation_and_promote_before_alpha(self):
        for dtype in (torch.float32,torch.float64):
            for count in (1,2,33):
                for x,y,alpha in ((-float('inf'),1e30,1e10),(float('inf'),-1e30,1e10),(-1e30,1e30,1.0000001192092896),(1.,1e-30,1e30)):
                    a,b=torch.full((count,),x,dtype=dtype),torch.full((count,),y,dtype=dtype)
                    self.same(fr.add(native(a),native(b),alpha=alpha),torch.add(a,b,alpha=alpha))
                    self.same(fr.sub(native(a),native(b),alpha=alpha),torch.sub(a,b,alpha=alpha))
        a=torch.tensor([1.],dtype=torch.float64);b=torch.tensor([1.0000001192092896])
        for alpha in (1.0000000000000002,1e100):
            self.same(fr.add(native(a),native(b),alpha=alpha),torch.add(a,b,alpha=alpha))

    def test_python_numbers_weak_precision_and_exact_integer_scalar_rounding(self):
        for operation in ('add','sub','mul','true_divide'):
            for x,y in ((1,2),(True,False),(1.5,2),(2,1.5)):
                try:expected=getattr(torch,operation)(x,y)
                except (RuntimeError,TypeError):
                    with self.assertRaises((ValueError,TypeError)):getattr(fr,operation)(x,y)
                else:self.same(getattr(fr,operation)(x,y),expected)
            a=torch.tensor([.5,1.5],dtype=torch.float64)
            self.same(getattr(fr,operation)(1e100,native(a)),getattr(torch,operation)(1e100,a))
        for integer in (2**62+2**38+1,-(2**62+2**38+1),2**64-1):
            a=torch.tensor([.5,1.5]);b=native(a)
            self.same(b*integer,a*integer)
            self.same(fr.multiply(b,integer),torch.multiply(a,integer))
            self.same(fr.add(b,integer),torch.add(a,integer))
        self.assertEqual(list(inspect.signature(fr.add).parameters),list(inspect.signature(fr.sub).parameters))
        self.assertIn('alpha=1',str(inspect.signature(fr.Tensor.add)))
        for dtype in (torch.int64,torch.float32,torch.float64):
            alpha=torch.tensor(2,dtype=dtype)
            self.same(fr.add(fr.ones(2),fr.ones(2),alpha=native(alpha)),torch.add(torch.ones(2),torch.ones(2),alpha=alpha))

    def test_live_gradients_higher_derivatives_and_typed_seed_provenance(self):
        for operation in ('add','sub'):
            for alpha in (0.,.3,-2.):
                x=torch.tensor([[.2,-.4],[.7,.9]],requires_grad=True);y=torch.tensor([[.3],[.6]],requires_grad=True)
                a,b=native(x,True),native(y,True)
                loss=getattr(fr,operation)(a,b,alpha=alpha).tanh().square().sum()
                expected=getattr(torch,operation)(x,y,alpha=alpha).tanh().square().sum()
                actual=fr.autograd.grad(loss,[a,b],create_graph=True);reference=torch.autograd.grad(expected,[x,y],create_graph=True)
                for value,target in zip(actual,reference):self.same(value,target,False);self.assertEqual(value.requires_grad,target.requires_grad)
                second=fr.autograd.grad(actual[0].sum()+actual[1].sum(),[a,b])
                oracle=torch.autograd.grad(reference[0].sum()+reference[1].sum(),[x,y])
                for value,target in zip(second,oracle):self.same(value,target,False)
        for dtype in ('float32','float64'):
            a=fr.ones(2,dtype=dtype,requires_grad=True);b=fr.ones(2,dtype=dtype,requires_grad=True);seed=fr.ones(2,dtype=dtype,requires_grad=True)
            result=fr.add(a,b,alpha=1)
            ga,gb=fr.autograd.vjp(result,[a,b],seed)
            self.same(ga,torch.ones(2,dtype=getattr(torch,dtype)));self.same(gb,torch.ones(2,dtype=getattr(torch,dtype)))
            self.assertFalse(ga.requires_grad);self.assertFalse(gb.requires_grad)
            if dtype=='float32':
                _,pullback=fr.func.vjp(lambda x,y:fr.add(x,y,alpha=1),a,b)
                ga,gb=pullback(seed,create_graph=False)
                self.assertEqual(ga._id(),seed._id());self.assertEqual(gb._id(),seed._id())

    def test_forward_alpha_rounding_and_original_f64_tangent_precision(self):
        for operation in ('add','sub'):
            for alpha in (1.0000000000000002,1.0000001192092896,-.3,0.):
                x=torch.tensor([.2,.4],requires_grad=True);y=torch.tensor([.7,.9],requires_grad=True)
                for dtype in (torch.float32,torch.float64):
                    vx=torch.tensor([1.,2.],dtype=dtype,requires_grad=True);vy=torch.tensor([3.,4.],dtype=dtype,requires_grad=True)
                    actual=fr.func.jvp(lambda a,b:getattr(fr,operation)(a,b,alpha=alpha),(native(x,True),native(y,True)),(native(vx,True),native(vy,True)))
                    expected=torch.func.jvp(lambda a,b:getattr(torch,operation)(a,b,alpha=alpha),(x,y),(vx,vy))
                    for value,target in zip(actual,expected):self.same(value,target);self.assertEqual(value.requires_grad,target.requires_grad)

    def test_integer_limit_primal_and_reference_ad_conventions(self):
        minimum=-(2**63)
        for alpha in (minimum,2**63,2**64-1):
            for dtype in (torch.int64,torch.float32,torch.float64):
                x=torch.tensor([1,2],dtype=dtype);y=torch.tensor([2,3],dtype=dtype)
                self.same(fr.sub(native(x),native(y),alpha=alpha),torch.sub(x,y,alpha=alpha))
        for dtype in (torch.float32,torch.float64):
            x=torch.ones(2,dtype=dtype,requires_grad=True);y=torch.ones(2,dtype=dtype,requires_grad=True)
            a,b=native(x,True),native(y,True)
            fr.sub(a,b,alpha=minimum).sum().backward();torch.sub(x,y,alpha=minimum).sum().backward()
            self.same(a.grad,x.grad);self.same(b.grad,y.grad)
            for value in (2**63,2**64-1):
                result=fr.sub(a,b,alpha=value);reference=torch.sub(x,y,alpha=value)
                self.same(result,reference)
                self.same(fr.autograd.grad(result.sum(),[a])[0],torch.autograd.grad(reference.sum(),[x],retain_graph=True)[0])
                with self.assertRaises(ValueError):fr.autograd.grad(result.sum(),[b])
                with self.assertRaises((RuntimeError,OverflowError)):torch.autograd.grad(reference.sum(),[y])

    def test_saved_values_invalid_alpha_and_explicit_out_capture_boundaries(self):
        a=fr.ones(2,requires_grad=True);b=fr.ones(2,requires_grad=True)
        result=fr.add(a,b,alpha=2)
        with fr.no_grad():a.fill_(3.);b.fill_(4.)
        result.sum().backward();self.assertEqual(a.grad.tolist(),[1.,1.]);self.assertEqual(b.grad.tolist(),[2.,2.])
        for alpha in (None,'2',fr.tensor([2.]),fr.tensor(2.,requires_grad=True),[2],2**64):
            with self.assertRaises((TypeError,ValueError,OverflowError)):fr.add(a,b,alpha=alpha)
        with self.assertRaises(ValueError):fr.add(a,b,alpha=1e100)
        with self.assertRaises(NotImplementedError):fr.add(a,b,out=fr.zeros(2))
        with self.assertRaises(ValueError):fr.capture(lambda:fr.add(a,b))
        self.assertTrue(fr.is_grad_enabled())

    def test_empty_scalar_strided_and_huge_virtual_empty_paths(self):
        for dtype in (torch.float32,torch.float64):
            for shape in ((),(2,0,3),(0,3)):
                x=torch.zeros(shape,dtype=dtype);y=torch.ones(shape,dtype=dtype)
                self.same(fr.add(native(x),native(y),alpha=.3),torch.add(x,y,alpha=.3))
            x=torch.arange(12,dtype=dtype).reshape(3,4).T
            self.same(fr.sub(native(x.T).T,native(x.T).T,alpha=.3),torch.sub(x,x,alpha=.3))
        program="import ferro as f; a=f.ones(1,1).expand(2**60,1); b=f.zeros(1,0,dtype='float64'); r=f.add(a,b,alpha=2); assert r.shape==[2**60,0] and r.numel()==0"
        result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
