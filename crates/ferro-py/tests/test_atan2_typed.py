"""Actual Torch atan2 values, ordinary adjoints, layouts and typed boundaries."""
import itertools
import subprocess
import sys
import unittest

import torch
import ferro as fr


def native(tensor, grad=False):
    return fr.tensor(tensor.detach().reshape(-1).tolist(),dtype=str(tensor.dtype).split('.')[-1],requires_grad=grad).reshape(list(tensor.shape))


class Atan2TypedTests(unittest.TestCase):
    def close(self,a,b):
        actual=torch.tensor(a.tolist(),dtype=getattr(torch,a.dtype)).reshape(a.shape)
        self.assertEqual(actual.dtype,b.dtype)
        tolerance=3e-6 if a.dtype=='float32' else 5e-15
        torch.testing.assert_close(actual,b.detach(),rtol=tolerance,atol=tolerance,equal_nan=True)
        zero=actual==0
        self.assertEqual(torch.signbit(actual[zero]).tolist(),torch.signbit(b.detach()[zero]).tolist())

    def test_mixed_dtype_rank_broadcast_and_strided_value_all_gradients(self):
        for ashape,bshape,ad,bd in itertools.product([(),(2,1)], [(),(1,3)], [torch.float32,torch.float64],[torch.float32,torch.float64]):
            a=torch.full(ashape,.7,dtype=ad,requires_grad=True);b=torch.full(bshape,-.3,dtype=bd,requires_grad=True)
            aa,bb=native(a,True),native(b,True);out=aa.atan2(bb);wanted=torch.atan2(a,b);self.close(out,wanted)
            for seed_dtype in [torch.float32,torch.float64]:
                seed=torch.linspace(.2,.8,wanted.numel(),dtype=seed_dtype).reshape(wanted.shape)
                actual=fr.autograd.grad(out,(aa,bb),native(seed),retain_graph=True)
                expect=torch.autograd.grad(wanted,(a,b),seed,retain_graph=True)
                for left,right in zip(actual,expect):self.close(left,right)
        for dtype in [torch.float32,torch.float64]:
            av=torch.arange(12,dtype=dtype).reshape(3,4)/7-.6;bv=torch.linspace(-.9,.8,12,dtype=dtype).reshape(3,4)
            for transposed in [False,True]:
                aa=native(av,True).T;bb=native(bv.T,True) if not transposed else native(bv,True).T
                a=av.T.detach().requires_grad_();b=(bv.T.contiguous() if not transposed else bv.T).detach().requires_grad_()
                out,wanted=aa.atan2(bb),torch.atan2(a,b);self.close(out,wanted);self.assertEqual(out.stride(),wanted.stride())
                for seed_dtype in (torch.float32,torch.float64):
                    seed=torch.ones(3,4,dtype=seed_dtype).T
                    actual=fr.autograd.grad(out,(aa,bb),native(seed.T).T,retain_graph=True)
                    expected=torch.autograd.grad(wanted,(a,b),seed,retain_graph=True)
                    for x,y in zip(actual,expected):self.close(x,y);self.assertEqual(x.stride(),y.stride())

    def test_972_special_cases_match_zero_radius_underflow_overflow_and_seed_staging(self):
        for dtype in (torch.float32,torch.float64):
            tiny,large=(1e-30,1e30) if dtype==torch.float32 else (1e-200,1e200)
            values=[0.,-0.,1.,-1.,float('inf'),-float('inf'),float('nan'),tiny,large]
            rows=list(itertools.product(values,values,[0.,-0.,1.,-1.,float('inf'),float('nan')]))
            a=torch.tensor([r[0]for r in rows],dtype=dtype,requires_grad=True);b=torch.tensor([r[1]for r in rows],dtype=dtype,requires_grad=True)
            g=torch.tensor([r[2]for r in rows],dtype=dtype);aa,bb=native(a,True),native(b,True)
            out,wanted=aa.atan2(bb),torch.atan2(a,b);self.close(out,wanted)
            actual=fr.autograd.grad(out,(aa,bb),native(g));expect=torch.autograd.grad(wanted,(a,b),g)
            for left,right in zip(actual,expect):self.close(left,right)

    def test_both_primal_mutations_reject_and_output_mutation_is_allowed(self):
        for dtype in (torch.float32,torch.float64):
            for changed in range(3):
                a=torch.tensor([.3,.7],dtype=dtype,requires_grad=True);b=torch.tensor([.8,-.9],dtype=dtype,requires_grad=True)
                aa,bb=native(a,True),native(b,True);out,wanted=aa.atan2(bb),torch.atan2(a,b)
                with fr.no_grad(),torch.no_grad():[aa,bb,out][changed].detach().fill_(2);[a,b,wanted][changed].fill_(2)
                if changed<2:
                    with self.assertRaisesRegex(ValueError,'modified'):fr.autograd.grad(out,(aa,bb),fr.ones(2,dtype=str(dtype).split('.')[-1]))
                    with self.assertRaises(RuntimeError):torch.autograd.grad(wanted,(a,b),torch.ones_like(wanted))
                else:
                    actual=fr.autograd.grad(out,(aa,bb),fr.ones(2,dtype=str(dtype).split('.')[-1]));expect=torch.autograd.grad(wanted,(a,b),torch.ones_like(wanted))
                    for left,right in zip(actual,expect):self.close(left,right)

    def test_eager_flags_first_order_boundary_and_existing_float32_jvp(self):
        for dtype in (torch.float32,torch.float64):
            a=torch.tensor([.3,.7],dtype=dtype,requires_grad=True);b=torch.tensor([.8,-.9],dtype=dtype,requires_grad=True)
            aa,bb=native(a,True),native(b,True)
            with fr.no_grad(),torch.no_grad():self.assertFalse(aa.atan2(bb).requires_grad);self.assertFalse(torch.atan2(a,b).requires_grad)
            with self.assertRaises((ValueError,NotImplementedError)):fr.autograd.grad(aa.atan2(bb).sum(),aa,create_graph=True)
        a=fr.tensor([.3,.7]);b=fr.tensor([.8,-.9]);g=fr.tensor([.2,.4])
        out,direction=fr.func.jvp(lambda x:x.atan2(b),(a,),(g,))
        expected,tangent=torch.func.jvp(lambda x:torch.atan2(x,torch.tensor([.8,-.9])),(torch.tensor([.3,.7]),),(torch.tensor([.2,.4]),))
        self.close(out,expected);self.close(direction,tangent)
        for dtype in ('int64','bool'):
            with self.assertRaises((ValueError,TypeError)):fr.ones(2,dtype=dtype).atan2(fr.ones(2))
        with self.assertRaises((ValueError,NotImplementedError)):fr.capture(lambda:fr.ones(2,dtype='float64').atan2(fr.ones(2,dtype='float64')))

    def test_mixed_extreme_rounding_and_subnormal_forward_is_not_narrowed(self):
        for ad,bd in itertools.product((torch.float32,torch.float64),repeat=2):
            for ashape,bshape in [((),(3,)),((3,),()),((3,),(3,))]:
                a=torch.full(ashape,1e20,dtype=ad,requires_grad=True);b=torch.full(bshape,1e-20,dtype=bd,requires_grad=True)
                aa,bb=native(a,True),native(b,True);out,wanted=aa.atan2(bb),torch.atan2(a,b);self.close(out,wanted)
                actual=fr.autograd.grad(out,(aa,bb),fr.ones(*out.shape,dtype=out.dtype));expected=torch.autograd.grad(wanted,(a,b),torch.ones_like(wanted))
                for x,y in zip(actual,expected):self.close(x,y)
        a=fr.tensor([1e-200],dtype='float64');out=a.atan2(fr.tensor(1.));self.assertEqual(out.tolist(),[1e-200])

    def test_huge_empty_broadcast_public_grad_and_capacity_are_bounded(self):
        script='''
import ferro as f,sys
size=10**12
a=f.tensor(.3,dtype='float64',requires_grad=True).expand([size,1]);b=f.zeros(1,0,dtype='float64',requires_grad=True).expand([size,0])
y=a.atan2(b)
ga,gb=f.autograd.grad(y,(a,b),f.zeros(size,0,dtype='float64'))
if y.shape != [size,0] or ga.shape != [size,1] or gb.shape != [size,0] or ga.stride()[0]!=0:raise RuntimeError('empty result changed')
try:f.tensor(1.,dtype='float64').expand([sys.maxsize//8+1,1]).atan2(f.ones(1,2,dtype='float64'))
except ValueError:pass
else:raise RuntimeError('capacity not checked')
print('bounded-empty-atan2')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('bounded-empty-atan2',result.stdout)


if __name__=='__main__':unittest.main()
