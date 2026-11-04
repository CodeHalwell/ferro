"""Saved activation output derivatives against actual Torch and exact staged kernels."""
import unittest
import torch
import ferro as fr


def tensor(value, live=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).clone().requires_grad_(live)


def observed(value):
    return torch.tensor(value.tolist(), dtype=getattr(torch, value.dtype)).reshape(value.shape)


def exact(test, actual, expected):
    value=observed(actual)
    torch.testing.assert_close(value, expected.detach(), rtol=0, atol=0, equal_nan=True)
    zero=expected.detach()==0
    test.assertEqual(torch.signbit(value[zero]).tolist(), torch.signbit(expected.detach()[zero]).tolist())


def staged(name, seed, output):
    if name=='exp':return seed*output
    if name=='sigmoid':return (seed*(1-output))*output
    return seed*torch.addcmul(torch.ones_like(output),output,output,value=-1)


class ActivationOutputDerivatives(unittest.TestCase):
    def test_extreme_cotangents_use_exact_saved_output_stages_both_precisions(self):
        for dtype in (torch.float32,torch.float64):
            tiny=torch.nextafter(torch.tensor(0.,dtype=dtype),torch.tensor(1.,dtype=dtype)).item()
            reference=torch.tensor([-float('inf'),-30.,-.7,-0.,.2,2.,30.,float('inf'),float('nan')],dtype=dtype)
            for name in ('exp','sigmoid','tanh'):
                x=tensor(reference,True);y=getattr(x,name)()
                for coefficient in (0.,-0.,tiny,-tiny,.7,float('inf'),-float('inf'),float('nan')):
                    seed=torch.full_like(reference,coefficient)
                    for graph in ((False,True) if dtype==torch.float32 else (False,)):
                        with self.subTest(dtype=dtype,name=name,seed=coefficient,graph=graph):
                            gradient=fr.autograd.grad(y,x,grad_outputs=tensor(seed),create_graph=graph)[0]
                            exact(self,gradient,staged(name,seed,observed(y)))
                self.assertIsNone(x.grad)

    def test_original_input_mutation_allows_saved_output_first_second_and_third_derivatives(self):
        for name in ('exp','sigmoid','tanh'):
            x=fr.tensor([.2,-.7],requires_grad=True);tx=torch.tensor([.2,-.7],requires_grad=True)
            y=getattr(x,name)();ty=getattr(tx,name)()
            with fr.no_grad(),torch.no_grad():x.fill_(8.);tx.fill_(8.)
            seed=fr.tensor([.7,-.3],requires_grad=True);ts=torch.tensor([.7,-.3],requires_grad=True)
            left,right=y,ty
            for order in (1,2,3):
                left=fr.autograd.grad(left,x,grad_outputs=seed if order==1 else fr.ones_like(left),create_graph=True)[0]
                right=torch.autograd.grad(right,tx,grad_outputs=ts if order==1 else torch.ones_like(right),create_graph=True)[0]
                torch.testing.assert_close(observed(left),right.detach(),rtol=3e-6,atol=2e-7)
            self.assertTrue(left.requires_grad);self.assertIsNone(x.grad)
            dseed=fr.autograd.grad(left.sum(),seed)[0];tdseed=torch.autograd.grad(right.sum(),ts)[0]
            torch.testing.assert_close(observed(dseed),tdseed,rtol=4e-6,atol=2e-7)

    def test_output_and_backward_saved_operands_reject_mutation_before_publication(self):
        for name in ('exp','sigmoid','tanh'):
            for dtype in (torch.float32,torch.float64):
                x=tensor(torch.tensor([.2,-.7],dtype=dtype),True);y=getattr(x,name)()
                old=x.square().sum();old.backward();previous=x.grad
                y.detach().zero_()
                with self.assertRaisesRegex(ValueError,'modified'):fr.autograd.grad(y,x,grad_outputs=fr.ones_like(y))
                self.assertEqual(x.grad._id(),previous._id())
            for target in ('seed','output'):
                x=fr.tensor([.2,-.7],requires_grad=True);y=getattr(x,name)();seed=fr.tensor([.7,-.3],requires_grad=True)
                gradient=fr.autograd.grad(y,x,grad_outputs=seed,create_graph=True)[0]
                (seed if target=='seed' else y).detach().zero_()
                with self.assertRaisesRegex(ValueError,'modified'):fr.autograd.grad(gradient.sum(),x)
                self.assertIsNone(x.grad)

    def test_direct_jvp_mixed_directions_and_no_grad_use_real_backward_rules(self):
        for name in ('exp','sigmoid','tanh'):
            for dtype in (torch.float32,torch.float64):
                for enabled in (False,True):
                    tx=torch.tensor([.2,-.7,30.],requires_grad=True)
                    tv=torch.tensor([.7,-.3,1e-30],dtype=dtype,requires_grad=True)
                    x=tensor(tx,True);v=tensor(tv,True)
                    with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                        y,dy=fr.func.jvp(lambda a:getattr(a,name)(),(x,),(v,))
                        ty,tdy=torch.func.jvp(lambda a:getattr(a,name)(),(tx,),(tv,))
                    exact(self,dy,staged(name,tv.detach(),observed(y).to(dtype)))
                    self.assertEqual(dy.requires_grad,tdy.requires_grad)
                    self.assertEqual(y.requires_grad,ty.requires_grad)
                    if enabled:
                        gradients=fr.autograd.grad(dy.sum(),(x,v));wanted=torch.autograd.grad(tdy.sum(),(tx,tv))
                        for actual,expected in zip(gradients,wanted):torch.testing.assert_close(observed(actual),expected,rtol=4e-6,atol=2e-7)

    def test_frozen_pullbacks_use_original_outputs_and_keep_cotangent_history_only(self):
        for name in ('exp','sigmoid','tanh'):
            x=fr.tensor([.2,-.7],requires_grad=True);tx=torch.tensor([.2,-.7],requires_grad=True)
            with fr.no_grad(),torch.no_grad():
                y,pullback=fr.func.vjp(lambda a:getattr(a,name)(),x)
                ty,tpullback=torch.func.vjp(lambda a:getattr(a,name)(),tx)
            seed=fr.tensor([.7,-.3],requires_grad=True);tseed=torch.tensor([.7,-.3],requires_grad=True)
            with fr.no_grad(),torch.no_grad():x.fill_(8.);tx.fill_(8.)
            gradient=pullback(seed,create_graph=True)[0];expected=tpullback(tseed,create_graph=True)[0]
            torch.testing.assert_close(observed(gradient),expected.detach(),rtol=3e-6,atol=1e-7)
            self.assertEqual(gradient.requires_grad,expected.requires_grad)
            actual=fr.autograd.grad(gradient.sum(),(x,seed),allow_unused=True)
            wanted=torch.autograd.grad(expected.sum(),(tx,tseed),allow_unused=True)
            self.assertIsNone(actual[0]);self.assertIsNone(wanted[0])
            torch.testing.assert_close(observed(actual[1]),wanted[1],rtol=3e-6,atol=1e-7)

    def test_strided_empty_and_functional_slot_isolation(self):
        for dtype in (torch.float32,torch.float64):
            for name in ('exp','sigmoid','tanh'):
                tx=torch.tensor([[.2,-.4,.3],[.7,-.1,.6]],dtype=dtype,requires_grad=True)
                x=tensor(tx,True);y=getattr(x.T,name)();ty=getattr(tx.T,name)()
                seed=torch.tensor([[.7,-.2],[.1,.3],[-.4,.5]],dtype=dtype)
                actual=fr.autograd.grad(y,x,grad_outputs=tensor(seed))[0]
                expected=torch.autograd.grad(ty,tx,seed)[0]
                torch.testing.assert_close(observed(actual),expected,rtol=3e-6,atol=1e-7)
                self.assertIsNone(x.grad)
                empty=fr.zeros(0,8,dtype=str(dtype).split('.')[-1],requires_grad=True)
                getattr(empty.narrow(1,4,4),name)().sum().backward()
                self.assertEqual(empty.grad.shape,[0,8])


if __name__=='__main__':unittest.main()
