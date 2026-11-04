"""Actual Torch CPU acos typed values, ordinary reverse staging and ownership."""
import itertools
import subprocess
import sys
import unittest
import torch
import ferro as fr
import test_atan2_typed as helpers
native=helpers.native

class AcosTypedTests(unittest.TestCase):
    close=helpers.Atan2TypedTests.close

    def test_typed_strided_forward_seed_cast_layout_and_ordinary_gradients(self):
        for dtype,seed_dtype in itertools.product((torch.float32,torch.float64),repeat=2):
            raw=torch.linspace(-.9,.9,12,dtype=dtype).reshape(3,4)
            x=raw.T.detach().requires_grad_();a=native(raw,True).T
            seed=torch.linspace(-.7,.9,12,dtype=seed_dtype).reshape(3,4).T
            out,wanted=a.acos(),torch.acos(x);self.close(out,wanted);self.assertEqual(out.stride(),wanted.stride())
            actual,=fr.autograd.grad(out,a,native(seed.T).T);expected,=torch.autograd.grad(wanted,x,seed)
            self.close(actual,expected);self.assertEqual(actual.stride(),expected.stride())

    def test_260_endpoint_nonfinite_subnormal_and_extreme_seed_cases(self):
        for dtype in (torch.float32,torch.float64):
            tiny=torch.nextafter(torch.tensor(0.,dtype=dtype),torch.tensor(1.,dtype=dtype)).item();large=torch.finfo(dtype).max
            rows=list(itertools.product([-float('inf'),-2.,-1.,-.99,-.7,-0.,0.,.7,.99,1.,2.,float('inf'),float('nan')],[0.,-0.,1.,-1.,tiny,-tiny,large,-large,float('inf'),float('nan')]))
            x=torch.tensor([r[0]for r in rows],dtype=dtype,requires_grad=True);g=torch.tensor([r[1]for r in rows],dtype=dtype)
            a=native(x,True);out,wanted=a.acos(),torch.acos(x);self.close(out,wanted)
            actual,=fr.autograd.grad(out,a,native(g));expected,=torch.autograd.grad(wanted,x,g);self.close(actual,expected)

    def test_version_flags_capture_and_existing_opaque_higher_order_boundary(self):
        for dtype in (torch.float32,torch.float64):
            for change_output in (False,True):
                a=fr.tensor([-.7,.3],dtype=str(dtype).split('.')[-1],requires_grad=True);x=torch.tensor([-.7,.3],dtype=dtype,requires_grad=True)
                out,wanted=a.acos(),torch.acos(x)
                with fr.no_grad(),torch.no_grad():(out if change_output else a).detach().fill_(.4);(wanted if change_output else x).fill_(.4)
                if change_output:
                    actual,=fr.autograd.grad(out,a,fr.ones(2,dtype=out.dtype));expected,=torch.autograd.grad(wanted,x,torch.ones_like(wanted));self.close(actual,expected)
                else:
                    with self.assertRaisesRegex(ValueError,'modified'):fr.autograd.grad(out,a,fr.ones(2,dtype=out.dtype))
                    with self.assertRaises(RuntimeError):torch.autograd.grad(wanted,x,torch.ones_like(wanted))
            a=fr.tensor([.3],dtype=str(dtype).split('.')[-1],requires_grad=True)
            with fr.no_grad():self.assertFalse(a.acos().requires_grad)
            with self.assertRaises((ValueError,NotImplementedError)):fr.autograd.grad(a.acos().sum(),a,create_graph=True)
        with self.assertRaises((ValueError,TypeError)):fr.ones(2,dtype='int64').acos()
        with self.assertRaises((ValueError,NotImplementedError)):fr.capture(lambda:fr.ones(2,dtype='float64').acos())

    def test_huge_empty_typed_layout_and_gradient_are_bounded(self):
        script='''
import ferro as f,sys
x=f.zeros(1,0,dtype='float64',requires_grad=True).expand([10**12,0]);y=x.acos()
g,=f.autograd.grad(y,x,f.zeros(10**12,0,dtype='float64'))
if y.shape != x.shape or g.shape != x.shape:raise RuntimeError('wrong empty shape')
try:f.tensor(.3,dtype='float64').expand([sys.maxsize//8+1]).acos()
except ValueError:pass
else:raise RuntimeError('capacity not checked')
print('bounded-empty-acos')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('bounded-empty-acos',result.stdout)

if __name__=='__main__':unittest.main()
