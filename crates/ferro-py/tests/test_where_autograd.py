"""Conditional selection AD against actual Torch; typed branch promotion and conditional gradients."""
import subprocess
import sys
import unittest

import torch
import ferro as fr
from ferro import func


def native(value):
    return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(value.requires_grad)


class WhereAutogradTests(unittest.TestCase):
    def same(self,actual,expected,flags=True):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertIsNotNone(actual)
        self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.dtype,str(expected.dtype).removeprefix('torch.'))
        value=torch.tensor(actual.tolist(),dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value,expected.detach(),atol=3e-6,rtol=3e-5,equal_nan=True)
        if flags:
            self.assertEqual(actual.requires_grad,expected.requires_grad)
            self.assertEqual(actual.is_leaf,expected.is_leaf)
        zeros=expected.detach()==0
        self.assertEqual(torch.signbit(value[zeros]).tolist(),torch.signbit(expected.detach()[zeros]).tolist())

    def test_broadcast_strides_typed_values_first_gradients_and_numeric_mask_extension(self):
        condition=torch.tensor([[True,False],[False,True],[True,False]]).T
        for dtype in (torch.float32,torch.float64):
            x=torch.tensor([[.2,-.0],[.7,.9],[1.1,-.3]],dtype=dtype).T.detach().requires_grad_()
            y=torch.tensor([[-.5],[.8]],dtype=dtype,requires_grad=True)
            a=native(x.T).T;b=native(y);c=native(condition.T).T
            out=fr.where(c,a,b);wanted=torch.where(condition,x,y)
            self.same(out,wanted)
            seed=torch.tensor([[.3,-.8,.7],[1.,-.2,.5]],dtype=dtype)
            grads=fr.autograd.grad(out,(a,b),native(seed));references=torch.autograd.grad(wanted,(x,y),seed)
            for result,reference in zip(grads,references):self.same(result,reference)
        for dtype,values in ((torch.int64,[2**60+3,-2**60+1]),(torch.bool,[True,False])):
            a=torch.tensor(values,dtype=dtype);b=torch.flip(a,[0]);c=torch.tensor([False,True])
            self.same(fr.where(native(c),native(a),native(b)),torch.where(c,a,b))
        a=fr.tensor([1.,2.],requires_grad=True);b=fr.tensor([3.,4.],requires_grad=True)
        condition=fr.tensor([float('nan'),0.],requires_grad=True)
        out=fr.where(condition,a,b)
        self.assertEqual(out.tolist(),[1.,4.])
        self.assertEqual(fr.autograd.grad(out,(condition,a,b),fr.ones(2),allow_unused=True)[0],None)
        self.assertFalse(fr.where(condition,a.detach(),b.detach()).requires_grad)
        promoted=fr.where(fr.tensor([True,False],dtype='bool'),a,b.to(dtype='float64'))
        ta=torch.tensor([1.,2.],requires_grad=True);tb=torch.tensor([3.,4.],dtype=torch.float64,requires_grad=True)
        wanted=torch.where(torch.tensor([True,False]),ta,tb)
        self.same(promoted,wanted)
        for actual,expected in zip(fr.autograd.grad(promoted.sum(),(a,b)),torch.autograd.grad(wanted.sum(),(ta,tb))):
            # The original right input is F32; its explicit cast has a real adjoint.
            self.same(actual,expected.to(torch.float32))

    def test_higher_broadcast_live_seed_optional_dependencies_and_nonfinite_masking(self):
        condition=torch.tensor([[True,False,True],[False,True,False]])
        a=torch.tensor([[.2],[.4]],requires_grad=True);b=torch.tensor([[.5,.7,.9]],requires_grad=True)
        seed=torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
        x,y,g,c=map(native,(a,b,seed,condition))
        out,wanted=fr.where(c,x,y),torch.where(condition,a,b)
        gradients=fr.autograd.grad(out,(x,y),g,create_graph=True)
        references=torch.autograd.grad(wanted,(a,b),seed,create_graph=True)
        for result,reference in zip(gradients,references):self.same(result,reference)
        actual=fr.autograd.grad(gradients[0].sum(),(x,y,g),allow_unused=True)
        expected=torch.autograd.grad(references[0].sum(),(a,b,seed),allow_unused=True)
        for result,reference in zip(actual,expected):self.same(result,reference)
        for value in (float('nan'),float('inf'),-float('inf'),-0.):
            a=torch.tensor([1.,2.],requires_grad=True);b=torch.tensor([3.,4.],requires_grad=True)
            seed=torch.full((2,),value,requires_grad=True);condition=torch.tensor([True,False])
            x,y,g,c=map(native,(a,b,seed,condition))
            result=fr.autograd.grad(fr.where(c,x,y),(x,y),g,create_graph=True)
            reference=torch.autograd.grad(torch.where(condition,a,b),(a,b),seed,create_graph=True)
            for actual,expected in zip(result,reference):self.same(actual,expected)

    def test_data_output_writes_are_allowed_but_condition_versions_are_saved(self):
        for graph in (False,True):
            for changed in ('left','right','condition','output'):
                a=torch.tensor([1.,2.],requires_grad=True);b=torch.tensor([3.,4.],requires_grad=True)
                c=torch.tensor([True,False]);x,y,mask=map(native,(a,b,c))
                output,reference=fr.where(mask,x,y),torch.where(c,a,b)
                actual_fields=dict(left=x,right=y,condition=mask,output=output)
                expected_fields=dict(left=a,right=b,condition=c,output=reference)
                with fr.no_grad(),torch.no_grad():
                    actual_fields[changed].detach().fill_(0);expected_fields[changed].detach().fill_(0)
                if changed=='condition':
                    with self.assertRaisesRegex(ValueError,'modified inplace'):fr.autograd.grad(output,(x,y),fr.ones(2),create_graph=graph)
                    with self.assertRaisesRegex(RuntimeError,'modified by an inplace'):torch.autograd.grad(reference,(a,b),torch.ones(2),create_graph=graph)
                else:
                    actual=fr.autograd.grad(output,(x,y),fr.ones(2),create_graph=graph)
                    expected=torch.autograd.grad(reference,(a,b),torch.ones(2),create_graph=graph)
                    for result,wanted in zip(actual,expected):self.same(result,wanted)
        c=fr.tensor([True,False],dtype='bool');x=fr.ones(2,requires_grad=True);y=fr.ones(2,requires_grad=True);g=fr.ones(2,requires_grad=True)
        derivative=fr.autograd.grad(fr.where(c,x,y),x,g,create_graph=True)[0]
        c.fill_(False)
        with self.assertRaisesRegex(ValueError,'modified inplace'):fr.autograd.grad(derivative.sum(),g)

    def test_direct_jvp_promotes_tangents_by_scalar_rank_and_preserves_reverse_flags(self):
        for left_shape,right_shape in (((),(3,)),((3,),()),((1,),(3,))):
            for which in ('both','left','right'):
                for left_dtype,right_dtype in ((torch.float32,torch.float32),(torch.float64,torch.float32),(torch.float32,torch.float64),(torch.float64,torch.float64)):
                    a=torch.ones(left_shape,requires_grad=True);b=torch.ones(right_shape,requires_grad=True)
                    va=torch.full(left_shape,2.,dtype=left_dtype,requires_grad=True);vb=torch.full(right_shape,3.,dtype=right_dtype,requires_grad=True)
                    condition=torch.tensor([True,False,True]);x,y,v,w,c=map(native,(a,b,va,vb,condition))
                    if which=='both':
                        actual=func.jvp(lambda left,right:fr.where(c,left,right),(x,y),(v,w))
                        expected=torch.func.jvp(lambda left,right:torch.where(condition,left,right),(a,b),(va,vb))
                    elif which=='left':
                        actual=func.jvp(lambda left:fr.where(c,left,y),(x,),(v,))
                        expected=torch.func.jvp(lambda left:torch.where(condition,left,b),(a,),(va,))
                    else:
                        actual=func.jvp(lambda right:fr.where(c,x,right),(y,),(w,))
                        expected=torch.func.jvp(lambda right:torch.where(condition,a,right),(b,),(vb,))
                    for value,reference in zip(actual,expected):self.same(value,reference)
                    gradients=fr.autograd.grad(actual[1].sum(),(x,y,v,w),allow_unused=True)
                    references=torch.autograd.grad(expected[1].sum(),(a,b,va,vb),allow_unused=True)
                    for value,reference in zip(gradients,references):self.same(value,reference)
        for enabled in (False,True):
            for live in (False,True):
                x=torch.tensor([1.,2.],requires_grad=True);v=torch.tensor([.3,.7],requires_grad=live);c=torch.tensor([True,False])
                a,g,mask=map(native,(x,v,c))
                with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                    actual=func.jvp(lambda value:fr.where(mask,value,fr.ones(2)),(a,),(g,))
                    expected=torch.func.jvp(lambda value:torch.where(c,value,torch.ones(2)),(x,),(v,))
                for value,reference in zip(actual,expected):self.same(value,reference)

    def test_functional_vjp_forward_over_reverse_and_inactive_branch_real_zeros(self):
        condition=torch.tensor([True,False,True]);mask=native(condition)
        for enabled in (False,True):
            a=torch.tensor([.2,.4,.6],requires_grad=True);seed=torch.tensor([.3,.7,.9],requires_grad=True)
            x,g=map(native,(a,seed))
            with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                actual,pullback=func.vjp(lambda value:fr.where(mask,value.square(),value.tanh()),x)
                expected,reference=torch.func.vjp(lambda value:torch.where(condition,value.square(),value.tanh()),a)
            self.same(actual,expected)
            gradients=pullback(g);references=reference(seed)
            self.same(gradients[0],references[0])
            for value,wanted in zip(fr.autograd.grad(gradients[0].sum(),(x,g),allow_unused=True),torch.autograd.grad(references[0].sum(),(a,seed),allow_unused=True)):
                self.same(value,wanted)
        a=torch.tensor([.2,.4,.6],requires_grad=True);seed=torch.tensor([.3,.7,.9],requires_grad=True);x,g=map(native,(a,seed))
        objective=lambda value:fr.where(mask,value.square(),value.tanh()).sum()
        reference=lambda value:torch.where(condition,value.square(),value.tanh()).sum()
        actual=func.jvp(func.grad(objective),(x,),(g,));expected=torch.func.jvp(torch.func.grad(reference),(a,),(seed,))
        for value,wanted in zip(actual,expected):self.same(value,wanted)
        for value,wanted in zip(fr.autograd.grad(actual[1].sum(),(x,g),allow_unused=True),torch.autograd.grad(expected[1].sum(),(a,seed),allow_unused=True)):
            self.same(value,wanted)
        a=torch.tensor([0.,2.],requires_grad=True);x=native(a);c=torch.tensor([False,True]);mask=native(c)
        actual=fr.where(mask,fr.ones(2)/x,x).sum();expected=torch.where(c,torch.ones(2)/a,a).sum()
        self.same(fr.autograd.grad(actual,x)[0],torch.autograd.grad(expected,a)[0])

    def test_jvp_reverse_uses_saved_mask_but_not_mutable_primal_values(self):
        for changed in ('left','right','output','condition'):
            a=torch.tensor([1.,2.],requires_grad=True);b=torch.tensor([3.,4.],requires_grad=True);v=torch.tensor([.3,.7],requires_grad=True);c=torch.tensor([True,False])
            x,y,g,mask=map(native,(a,b,v,c))
            actual=func.jvp(lambda value:fr.where(mask,value,y),(x,),(g,))
            expected=torch.func.jvp(lambda value:torch.where(c,value,b),(a,),(v,))
            left=dict(left=x,right=y,output=actual[0],condition=mask)[changed]
            right=dict(left=a,right=b,output=expected[0],condition=c)[changed]
            with fr.no_grad(),torch.no_grad():left.detach().fill_(0);right.detach().fill_(0)
            if changed=='condition':
                with self.assertRaisesRegex(ValueError,'modified inplace'):fr.autograd.grad(actual[1].sum(),g)
                with self.assertRaisesRegex(RuntimeError,'modified by an inplace'):torch.autograd.grad(expected[1].sum(),v)
            else:self.same(fr.autograd.grad(actual[1].sum(),g)[0],torch.autograd.grad(expected[1].sum(),v)[0])

    def test_huge_empty_paths_and_capacity_rejection_are_bounded(self):
        script='''
import sys
import ferro as fr
from ferro import func
size=10**12
for dtype in ('float32','float64'):
 c=fr.zeros(1,0,dtype='bool').expand([size,0])
 a=fr.tensor(2.,dtype=dtype).expand([size,1]).requires_grad_()
 b=fr.zeros(1,0,dtype=dtype,requires_grad=True)
 y=fr.where(c,a,b)
 ga,gb=fr.autograd.grad(y,(a,b),fr.zeros(size,0,dtype=dtype))
 if ga.shape != [size,1] or ga.stride() != (0,1) or gb.shape != [1,0]:raise RuntimeError('wrong empty broadcast adjoint')
 a=fr.zeros(size,0,requires_grad=True);v=fr.zeros(size,0,dtype='float64',requires_grad=True)
 y,dy=func.jvp(lambda value:fr.where(c,value,fr.zeros(1,0)),(a,),(v,))
 if dy.shape != [size,0] or dy.dtype != 'float64':raise RuntimeError('wrong typed empty JVP')
huge=fr.ones(1).expand([sys.maxsize//4+1])
try:fr.where(fr.tensor(True,dtype='bool'),huge,huge)
except ValueError:pass
else:raise RuntimeError('missing selection capacity check')
print('where-huge-empty-ok')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('where-huge-empty-ok',result.stdout)


if __name__=='__main__':unittest.main()
