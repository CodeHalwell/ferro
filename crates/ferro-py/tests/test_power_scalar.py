"""Scalar powers: half-domain semantics and staged floating adjoints."""
import math
import unittest
import torch
import ferro as fr


def native(value, live=False):
    return fr.tensor(value.detach().tolist(),dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).clone().requires_grad_(live)


def same(test,actual,expected,exact=False,working_dtype=None):
    test.assertEqual(actual.shape,list(expected.shape));test.assertEqual(actual.dtype,str(expected.dtype).split('.')[-1])
    test.assertEqual(actual.stride(),expected.stride())
    value=torch.tensor(actual.tolist(),dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(value,expected.detach(),rtol=0 if exact else 4*torch.finfo(working_dtype or expected.dtype).eps,atol=0,equal_nan=True)
    mask=expected.detach()==0
    test.assertEqual(torch.signbit(value[mask]).tolist(),torch.signbit(expected.detach()[mask]).tolist())


class PowerScalarTests(unittest.TestCase):
    def test_half_primal_domains_signed_zero_layout_and_owning_results(self):
        for dtype in (torch.float32,torch.float64):
            data=torch.tensor([[-float('inf'),-4.,-0.],[0.,4.,float('inf')]],dtype=dtype)
            for power in (0.,1.,.5,-.5):
                for transpose in (False,True):
                    x,t=native(data),data
                    if transpose:x,t=x.T,t.T
                    output=x.pow(power);same(self,output,t.pow(power),exact=True)
                    self.assertFalse(output._shares_storage(x))
                    same(self,x**power,t**power,exact=True)

    def test_staged_ordinary_and_graph_adjoints_and_non_passthrough_power_one(self):
        values=[-float('inf'),-1.,-0.,0.,1e-20,.2,1.,2.,1e20,float('inf'),float('nan')]
        tiny=torch.nextafter(torch.tensor(0.),torch.tensor(1.)).item()
        for power in (-3.,-2.,-1.,-.5,0.,.5,1.,2.,3.):
            for amount in (tiny,.7,2e38):
                for graph in (False,True):
                    with self.subTest(power=power,seed=amount,graph=graph):
                        tx=torch.tensor(values,requires_grad=True);x=native(tx,True)
                        ts=torch.full_like(tx,amount,requires_grad=True);seed=native(ts,True)
                        expected=torch.autograd.grad(tx.pow(power),tx,ts,create_graph=graph)[0]
                        actual=fr.autograd.grad(x.pow(power),x,grad_outputs=seed,create_graph=graph)[0]
                        same(self,actual,expected)
                        self.assertEqual(actual.requires_grad,expected.requires_grad)
                        if power==1.:
                            self.assertIsNot(actual,seed)
                            self.assertEqual(actual._shares_storage(seed),torch._C._is_alias_of(expected,ts))
                        self.assertIsNone(x.grad);self.assertIsNone(seed.grad)

    def test_direct_jvp_retains_compatible_wide_direction_and_primal_slope_order(self):
        for power in (-.5,0.,.5,1.,3.):
            for dtype in (torch.float32,torch.float64):
                for enabled in (False,True):
                    with self.subTest(power=power,dtype=dtype,enabled=enabled):
                        tx=torch.tensor([.2,.7,2.],requires_grad=True);tv=torch.tensor([.3,.7,1.1],dtype=dtype,requires_grad=True)
                        x,v=native(tx,True),native(tv,True)
                        with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jvp(lambda a:a.pow(power),(x,),(v,))
                        with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jvp(lambda a:a.pow(power),(tx,),(tv,))
                        for index,(a,b) in enumerate(zip(actual,expected)):
                            same(self,a,b,working_dtype=torch.float32 if index==1 else None)
                            self.assertEqual(a.requires_grad,b.requires_grad)
                        if dtype==torch.float64 and power!=0.:
                            # The slope is computed in primal F32 before multiplying the wide direction.
                            slope=(x.pow(power-1.)*power).detach()
                            own=torch.tensor(slope.tolist(),dtype=torch.float32).double()*tv.detach()
                            same(self,actual[1],own,exact=True)
                        self.assertIsNone(x.grad);self.assertIsNone(v.grad)

    def test_power_zero_one_and_half_save_original_inputs_not_outputs(self):
        for power in (0.,.5,1.,-.5):
            tx=torch.tensor([.2,.7],requires_grad=True);x=native(tx,True)
            y,ty=x.pow(power),tx.pow(power)
            y.detach().fill_(0.);ty.detach().fill_(0.)
            same(self,fr.autograd.grad(y,x,grad_outputs=fr.ones(2),retain_graph=True)[0],torch.autograd.grad(ty,tx,torch.ones(2),retain_graph=True)[0])
            with fr.no_grad(),torch.no_grad():x.fill_(3.);tx.fill_(3.)
            with self.assertRaises(ValueError):fr.autograd.grad(y,x,grad_outputs=fr.ones(2))
            with self.assertRaises(RuntimeError):torch.autograd.grad(ty,tx,torch.ones(2))
            self.assertIsNone(x.grad)


    def test_original_exponent_scalar_domains_do_not_take_a_narrowed_special_branch(self):
        powers=(math.nextafter(.5,0.),math.nextafter(.5,1.),math.nextafter(-.5,-1.),math.nextafter(-.5,0.),math.nextafter(1.,2.),math.nextafter(2.,3.),math.nextafter(0.,1.),1.7,3.3)
        for dtype in (torch.float32,torch.float64):
            for power in powers:
                with self.subTest(dtype=dtype,power=power):
                    tx=torch.tensor([-float('inf'),-4.,-0.,0.,.2],dtype=dtype,requires_grad=True);x=native(tx,True)
                    y,ty=x.pow(power),tx.pow(power);same(self,y,ty)
                    seed,tseed=fr.ones(5,dtype=str(dtype).split('.')[-1]),torch.ones_like(tx)
                    dx=fr.autograd.grad(y,x,grad_outputs=seed)[0];tdx=torch.autograd.grad(ty,tx,tseed)[0]
                    same(self,dx,tdx);self.assertFalse(dx.requires_grad);self.assertIsNone(x.grad)

    def test_original_scalar_history_survives_repeated_derivatives(self):
        for power in (math.nextafter(.5,1.),math.nextafter(1.,2.),2.**-52,1.7,3.3,1.):
            with self.subTest(power=power):
                tx=torch.tensor([.2,.7,2.],requires_grad=True);x=native(tx,True)
                tv=torch.full_like(tx,.7,requires_grad=True);v=native(tv,True)
                first=fr.autograd.grad(x.pow(power),x,grad_outputs=v,create_graph=True)[0]
                tfirst=torch.autograd.grad(tx.pow(power),tx,tv,create_graph=True)[0]
                same(self,first,tfirst);self.assertEqual(first.requires_grad,tfirst.requires_grad)
                second=fr.autograd.grad(first.sum(),x,create_graph=True)[0]
                tsecond=torch.autograd.grad(tfirst.sum(),tx,create_graph=True)[0]
                same(self,second,tsecond);self.assertEqual(second.requires_grad,tsecond.requires_grad)
                if tsecond.requires_grad:
                    third=fr.autograd.grad(second.sum(),x,create_graph=True)[0]
                    tthird=torch.autograd.grad(tsecond.sum(),tx,create_graph=True)[0]
                    same(self,third,tthird);self.assertEqual(third.requires_grad,tthird.requires_grad)
                self.assertIsNone(x.grad);self.assertIsNone(v.grad)

    def test_portable_original_scalar_policy_is_shape_independent_and_replayable(self):
        # Torch's CPU vector kernel may narrow its exponent in ISA-specific
        # blocks. This portable path intentionally preserves scalar domains at
        # every element; the scalar Torch call supplies the reference here.
        values=(-4.,-0.,0.,4.,-float('inf'))
        for power in (math.nextafter(.5,1.),math.nextafter(1.,2.),math.nextafter(0.,1.)):
            scalar=[torch.tensor(value).pow(power).item() for value in values]
            for count in (1,5,32,33,1000):
                data=(list(values)*(count//5+1))[:count]
                reference=torch.tensor((scalar*(count//5+1))[:count])
                same(self,fr.tensor(data).pow(power),reference,exact=True)
        source=fr.tensor([[.2,.7,1.3],[2.,3.,4.]])
        for power in (1.7,math.nextafter(.5,1.),.5):
            chain=fr.capture(lambda:source.T.pow(power)).compile_fused()
            source.fill_(2.);result=chain.replay()
            same(self,result,torch.full((2,3),2.).T.pow(power))
        self.assertEqual(fr.empty(0,2**60).pow(1.7).shape,[0,2**60])
        with self.assertRaises(ValueError):fr.tensor(1.).expand(2**62).pow(1.7)

    def test_original_exponent_mixed_direction_uses_primal_precision_slope(self):
        for power in (1.7,math.nextafter(1.,2.),2.**-52):
            for enabled in (False,True):
                with self.subTest(power=power,enabled=enabled):
                    tx=torch.tensor([.2,.7,2.],requires_grad=True);tv=torch.tensor([.3,.7,1.1],dtype=torch.float64,requires_grad=True)
                    x,v=native(tx,True),native(tv,True)
                    with fr.enable_grad() if enabled else fr.no_grad():result=fr.func.jvp(lambda a:a.pow(power),(x,),(v,))
                    with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jvp(lambda a:a.pow(power),(tx,),(tv,))
                    for a,b in zip(result,expected):
                        same(self,a,b,working_dtype=torch.float32);self.assertEqual(a.requires_grad,b.requires_grad)
                    own=torch.tensor((x.pow(power-1.)*power).detach().tolist(),dtype=torch.float32).double()*tv.detach()
                    same(self,result[1],own,exact=True)
                    if enabled:
                        dv=fr.autograd.grad(result[1].sum(),v)[0]
                        coefficient=torch.tensor((x.pow(power-1.)*power).detach().tolist(),dtype=torch.float32).double()
                        same(self,dv,coefficient,exact=True)


if __name__=='__main__':unittest.main()
