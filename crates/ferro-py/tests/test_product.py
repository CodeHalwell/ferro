"""Typed product reductions and explicit transform policies versus Torch."""
import itertools
import math
import subprocess
import sys
import unittest

import torch
import ferro as fr


def native(x, grad=False):
    return fr.tensor(x.detach().reshape(-1).tolist(), dtype=str(x.dtype).removeprefix('torch.')).reshape(list(x.shape)).requires_grad_(grad)


class ProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def close(self, actual, expected, exact=False):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value=torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        atol,rtol=(0.,0.) if exact or not expected.dtype.is_floating_point else ((2e-6,2e-5) if expected.dtype==torch.float32 else (2e-12,2e-11))
        torch.testing.assert_close(value, expected.detach(), atol=atol, rtol=rtol, equal_nan=True)

    def test_typed_options_shapes_strides_and_wrapping(self):
        for source,target in itertools.product((torch.bool,torch.int64,torch.float32,torch.float64),(None,torch.bool,torch.int64,torch.float32,torch.float64)):
            for shape in ((),(2,3),(2,0,3),(0,3)):
                x=torch.arange(math.prod(shape)).remainder(3).reshape(shape).to(source)
                a=native(x)
                if len(shape)>1:x=x.transpose(0,-1);a=a.transpose(0,-1)
                options={} if target is None else {'dtype':str(target).removeprefix('torch.')}
                reference={} if target is None else {'dtype':target}
                with self.subTest(source=source,target=target,shape=shape):
                    self.close(a.prod(**options),x.prod(**reference));self.close(fr.prod(a,**options),torch.prod(x,**reference))
                    for dim in range(max(x.ndim,1)):
                        for keep in (False,True):self.close(a.prod(dim,keep,**options),x.prod(dim,keep,**reference))
        integer=torch.tensor([2**63-1,2,-1],dtype=torch.int64)
        self.close(fr.prod(native(integer)),integer.prod(),exact=True)
        for values in ([1e30,1e30,1e-30],[1e-30,1e-30,1e30],[1e30,1e30,0.,1e-30]):
            x=torch.tensor(values);self.close(native(x).prod(),x.prod(),exact=True)

    def test_ordinary_mixed_dtype_zero_and_nonfinite_cotangents(self):
        for source,target in itertools.product((torch.float32,torch.float64),repeat=2):
            for values in itertools.product((0.,2.,math.inf,math.nan),repeat=3):
                for dim,sv in itertools.product((None,0),(1.,math.inf)):
                    x=torch.tensor(values,dtype=source,requires_grad=True);a=native(x,True)
                    y=x.prod(dtype=target) if dim is None else x.prod(dim,dtype=target)
                    actual=a.prod(dtype=str(target).removeprefix('torch.')) if dim is None else a.prod(dim,dtype=str(target).removeprefix('torch.'))
                    seed=torch.full_like(y,sv)
                    with self.subTest(source=source,target=target,values=values,dim=dim,seed=sv):
                        self.close(actual,y);self.close(fr.autograd.grad(actual,a,native(seed))[0],torch.autograd.grad(y,x,seed)[0])
                        self.assertIsNone(a.grad)
        x=torch.tensor([1e-50,2.],dtype=torch.float64,requires_grad=True);a=native(x,True)
        self.close(fr.autograd.grad(a.prod(dtype=fr.float32),a)[0],torch.autograd.grad(x.prod(dtype=torch.float32),x)[0])

    def test_dim_batched_strided_gradients_and_live_higher(self):
        for dtype,dim,zero in itertools.product((torch.float32,torch.float64),(0,1),(False,True)):
            x=torch.tensor([[.5,1.1,-.7],[.3,-.2,.9]],dtype=dtype)
            if zero:x[0,0]=0
            x.requires_grad_();a=native(x,True)
            y=x.T.prod(dim);z=a.T.prod(dim)
            seed=torch.arange(1,y.numel()+1,dtype=dtype).reshape(y.shape)
            self.close(z,y);self.close(fr.autograd.grad(z,a,native(seed))[0],torch.autograd.grad(y,x,seed)[0])
        for values in ([2.,3.,4.],[2.,0.,4.],[0.,0.,4.]):
            x=torch.tensor(values,requires_grad=True);a=native(x,True)
            v=torch.tensor(3.,requires_grad=True);b=native(v,True)
            first=fr.autograd.grad(a.prod(),a,b,create_graph=True)[0]
            expected=torch.autograd.grad(x.prod(),x,v,create_graph=True)[0]
            self.close(first,expected)
            second=fr.autograd.grad(first.sum(),(a,b),create_graph=True)
            reference=torch.autograd.grad(expected.sum(),(x,v),create_graph=True)
            for lhs,rhs in zip(second,reference):self.close(lhs,rhs)
            self.close(fr.autograd.grad(second[0].sum(),a)[0],torch.autograd.grad(reference[0].sum(),x)[0])

    def test_functional_reverse_policy_and_returned_primal_all_four_apis(self):
        for values in ([math.inf],[math.inf,2.],[0.,math.inf,0.],[],[2.,3.]):
            for family in ('grad_value','vjp','jvp','aux'):
                x=torch.tensor(values,requires_grad=True);a=native(x,True)
                if family=='grad_value':
                    expected,y=torch.func.grad_and_value(lambda z:z.prod())(x)
                    actual,b=fr.func.grad_and_value(lambda z:z.prod())(a)
                    self.close(actual,expected)
                elif family=='vjp':
                    y,pb=torch.func.vjp(lambda z:z.prod(),x);b,fb=fr.func.vjp(lambda z:z.prod(),a)
                    direction=torch.tensor(.7,requires_grad=True);v=native(direction,True)
                    actual=fb(v)[0];expected=pb(direction)[0]
                    self.close(actual,expected);self.assertEqual(actual.requires_grad,expected.requires_grad)
                    if expected.requires_grad:
                        for lhs,rhs in zip(fr.autograd.grad(actual.sum(),(a,v),allow_unused=True),torch.autograd.grad(expected.sum(),(x,direction),allow_unused=True,retain_graph=True)):
                            self.close(lhs,rhs)
                elif family=='jvp':
                    y,expected=torch.func.jvp(lambda z:z.prod(),(x,),(torch.ones_like(x),))
                    b,actual=fr.func.jvp(lambda z:z.prod(),(a,),(fr.ones(a.shape),))
                    self.close(actual,expected)
                else:
                    expected,y=torch.func.grad(lambda z:(z.sum(),z.prod()),has_aux=True)(x)
                    actual,b=fr.func.grad(lambda z:(z.sum(),z.prod()),has_aux=True)(a)
                    self.close(actual,expected)
                with self.subTest(values=values,family=family):
                    self.close(b,y)
                    self.close(fr.autograd.grad(b,a)[0],torch.autograd.grad(y,x)[0])
        for enabled,tracked in itertools.product((False,True),repeat=2):
            x=torch.tensor([math.inf],requires_grad=tracked);a=native(x,tracked)
            with torch.set_grad_enabled(enabled), fr.enable_grad() if enabled else fr.no_grad():
                for name in ('grad','jacrev'):
                    expected=getattr(torch.func,name)(lambda z:z.prod())(x)
                    actual=getattr(fr.func,name)(lambda z:z.prod())(a)
                    self.close(actual,expected);self.assertEqual(actual.requires_grad,expected.requires_grad)
        # A precomputed captured product retains its original ordinary tape.
        w=torch.tensor([math.inf],requires_grad=True);b=native(w,True)
        held=w.prod();saved=b.prod()
        x=torch.tensor(2.,requires_grad=True);a=native(x,True)
        expected=torch.func.grad(lambda z:z*held)(x);actual=fr.func.grad(lambda z:z*saved)(a)
        self.close(fr.autograd.grad(actual,b)[0],torch.autograd.grad(expected,w)[0])

    def test_functional_jvp_mixed_dtype_and_reverse_through_tangent(self):
        for values,sdtype in itertools.product(([2.,3.],[0.,2.],[0.,0.],[math.inf],[]),(torch.float32,torch.float64)):
            x=torch.tensor(values,requires_grad=True);a=native(x,True)
            seed=torch.ones_like(x,dtype=sdtype,requires_grad=True);v=native(seed,True)
            y,dy=torch.func.jvp(lambda z:z.prod(),(x,),(seed,));b,db=fr.func.jvp(lambda z:z.prod(),(a,),(v,))
            self.close(b,y);self.close(db,dy)
            if dy.requires_grad:
                expected=torch.autograd.grad(dy,(x,seed),allow_unused=True)
                actual=fr.autograd.grad(db,(a,v),allow_unused=True)
                for lhs,rhs in zip(actual,expected):self.close(lhs,rhs)

        # Functional forward operands require the composite-safe local rule,
        # including jacrev even though its standalone pullback is ordinary.
        def tree_close(actual, expected):
            if isinstance(expected, torch.Tensor):
                self.close(actual,expected)
                self.assertEqual(actual.requires_grad,expected.requires_grad)
            else:
                self.assertEqual(len(actual),len(expected))
                for lhs,rhs in zip(actual,expected):tree_close(lhs,rhs)
        for values,family in itertools.product(([math.inf],[2.,3.],[2.,math.inf],[0.,math.inf,0.],[]),('grad','grad_and_value','jacrev')):
            with self.subTest(composition=family,values=values):
                x=torch.tensor(values,requires_grad=True);a=native(x,True)
                expected=torch.func.jvp(getattr(torch.func,family)(lambda z:z.prod()),(x,),(torch.ones_like(x),))
                actual=fr.func.jvp(getattr(fr.func,family)(lambda z:z.prod()),(a,),(fr.ones(a.shape),))
                tree_close(actual,expected)
        for values,sdtype in itertools.product(([2.,3.],[0.,2.],[math.inf]),(torch.float32,torch.float64)):
            x=torch.tensor(values,requires_grad=True);a=native(x,True)
            seed=torch.ones_like(x,dtype=sdtype,requires_grad=True);v=native(seed,True)
            expected=torch.func.jvp(lambda z:z.prod(dtype=torch.float64),(x,),(seed,))
            actual=fr.func.jvp(lambda z:z.prod(dtype=fr.float64),(a,),(v,))
            tree_close(actual,expected)

    def test_saved_versions_and_functional_direction_dependencies(self):
        for shape,which in itertools.product(((),(0,),(1,),(3,)),('input','output')):
            x=torch.ones(shape,requires_grad=True);a=native(x,True);y=x.prod();b=a.prod()
            with torch.no_grad(),fr.no_grad():
                (x if which=='input' else y).detach().fill_(2);(a if which=='input' else b).detach().fill_(2)
            with self.assertRaises(RuntimeError):torch.autograd.grad(y,x)
            with self.assertRaises((ValueError,RuntimeError)):fr.autograd.grad(b,a)
            self.assertIsNone(a.grad)
        for changed in ('input','output','seed','tangent'):
            x=torch.tensor([2.,3.],requires_grad=True);a=native(x,True)
            seed=torch.ones_like(x,requires_grad=True);v=native(seed,True)
            y,dy=torch.func.jvp(lambda z:z.prod(),(x,),(seed,));b,db=fr.func.jvp(lambda z:z.prod(),(a,),(v,))
            with torch.no_grad(),fr.no_grad():
                dict(input=x,output=y,seed=seed,tangent=dy)[changed].detach().fill_(9)
                dict(input=a,output=b,seed=v,tangent=db)[changed].detach().fill_(9)
            if changed=='seed':
                with self.assertRaises(RuntimeError):torch.autograd.grad(dy,(x,seed))
                with self.assertRaises((ValueError,RuntimeError)):fr.autograd.grad(db,(a,v))
            else:
                for lhs,rhs in zip(fr.autograd.grad(db,(a,v)),torch.autograd.grad(dy,(x,seed))):self.close(lhs,rhs)

    def test_standard_functional_api_remains_ordinary_and_error_restores_policy(self):
        x=torch.tensor([math.inf],requires_grad=True);a=native(x,True)
        self.close(fr.autograd.functional.jacobian(lambda z:z.prod(),a),torch.autograd.functional.jacobian(lambda z:z.prod(),x))
        for _ in range(2):
            with self.assertRaises((ValueError,RuntimeError)):
                fr.func.vjp(lambda z:z.prod(),a)[1](fr.ones([2]))
            self.close(fr.autograd.grad(a.prod(),a)[0],torch.autograd.grad(x.prod(),x)[0])
            self.close(fr.func.grad(lambda z:z.prod())(a),torch.func.grad(lambda z:z.prod())(x))

    def test_pullback_recording_mode_is_separate_from_needed_graph(self):
        for values,enabled,tracked in itertools.product(([math.inf],[0.,math.inf,0.],[],[2.,3.]),(False,True),(False,True)):
            x=torch.tensor(values,requires_grad=tracked);a=native(x,tracked)
            with self.subTest(values=values,enabled=enabled,tracked=tracked):
                with torch.set_grad_enabled(enabled),fr.enable_grad() if enabled else fr.no_grad():
                    for chunk in (None,1,2):
                        expected=torch.func.jacrev(lambda z:z.prod(),chunk_size=chunk)(x)
                        actual=fr.func.jacrev(lambda z:z.prod(),chunk_size=chunk)(a)
                        self.close(actual,expected);self.assertEqual(actual.requires_grad,expected.requires_grad)
                    _,expected_pb=torch.func.vjp(lambda z:z.prod(),x)
                    _,actual_pb=fr.func.vjp(lambda z:z.prod(),a)
                for caller,override in itertools.product((False,True),(None,False,True)):
                    with torch.set_grad_enabled(caller),fr.enable_grad() if caller else fr.no_grad():
                        expected=expected_pb(torch.tensor(.7),create_graph=override)[0]
                        actual=actual_pb(fr.tensor(.7),create_graph=override)[0]
                    self.close(actual,expected);self.assertEqual(actual.requires_grad,expected.requires_grad)
                # Query policy cannot leak into ordinary differentiation afterward.
                ordinary=torch.tensor(values,requires_grad=True);fresh=native(ordinary,True)
                self.close(fr.autograd.grad(fresh.prod(),fresh)[0],torch.autograd.grad(ordinary.prod(),ordinary)[0])

    def test_validation_and_bounded_empty_output_capacities(self):
        a=fr.ones([2,3]);x=torch.ones(2,3)
        for options in ({'dim':None},{'keepdim':True},{'dim':True},{'dim':(0,)},{'dim':3}):
            with self.assertRaises((TypeError,IndexError,RuntimeError)):x.prod(**options)
            with self.assertRaises((TypeError,ValueError,RuntimeError)):a.prod(**options)
        code='''import ferro as fr,sys
x=fr.zeros([sys.maxsize//8+1,0])
try:x.prod(1)
except (ValueError,RuntimeError):pass
else:raise RuntimeError('oversized reduction output accepted')
if x.prod(0).shape != [0] or x.prod().item()!=1:raise RuntimeError('empty product mismatch')
y=fr.tensor([1.]).expand([sys.maxsize//8+1])
try:y.prod()
except (ValueError,RuntimeError):pass
else:raise RuntimeError('oversized input accepted')
'''
        subprocess.run([sys.executable,'-c',code],check=True,timeout=10)


if __name__=='__main__':unittest.main()
