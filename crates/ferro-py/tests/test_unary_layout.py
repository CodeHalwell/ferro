"""Unary physical iteration, saved-output layout and immutable-zero copies."""
import unittest
import torch
import ferro as fr


def native(value, live=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).clone().requires_grad_(live)


def observed(value):
    return torch.tensor(value.tolist(), dtype=getattr(torch, value.dtype)).reshape(value.shape)


class UnaryLayoutTests(unittest.TestCase):
    def same(self, actual, expected, *, exact=False):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), tuple(expected.stride()))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        epsilon = torch.finfo(expected.dtype).eps if expected.dtype.is_floating_point else 0
        value = observed(actual)
        torch.testing.assert_close(value, expected.detach(), rtol=0 if exact else 4*epsilon, atol=0, equal_nan=True)
        if expected.dtype.is_floating_point:
            mask = expected.detach() == 0
            self.assertEqual(torch.signbit(value[mask]).tolist(), torch.signbit(expected.detach()[mask]).tolist())

    def test_raw_unary_layouts_and_owned_values_across_precisions(self):
        for dtype in (torch.float32, torch.float64):
            values = torch.arange(1, 25, dtype=dtype).reshape(2, 3, 4)/32
            base = native(values)
            for layout in (lambda x:x, lambda x:x.permute(2,0,1), lambda x:x[:,:,::2], lambda x:x[:1,:,:].expand(5,3,4)):
                a, t = layout(base), layout(values)
                for name in ('neg','negative','relu','exp','sigmoid','tanh','sqrt','sign'):
                    with self.subTest(dtype=dtype, stride=t.stride(), name=name):
                        actual, expected = getattr(a,name)(), getattr(t,name)()
                        self.same(actual, expected)
                        self.assertFalse(actual._shares_storage(a))
                        self.assertEqual(actual.storage_offset(), 0)
            output = base.permute(2,0,1).exp()
            before = observed(output)
            base.zero_()
            torch.testing.assert_close(observed(output), before, rtol=0, atol=0)
        values = torch.tensor([[-(1<<63),(1<<63)-1,0],[-1,1,(1<<53)+1]])
        for name in ('neg','negative','sign'):
            self.same(getattr(native(values).T,name)(), getattr(values.T,name)(), exact=True)

        for dtype in (torch.float32, torch.float64):
            for name in ('neg', 'negative'):
                data=torch.tensor([-0.,.2,-.7],dtype=dtype)
                x=native(data,True);tx=data.clone().requires_grad_()
                y,ty=getattr(x,name)(),getattr(tx,name)()
                self.same(y,ty,exact=True)
                self.assertTrue(y.requires_grad);self.assertFalse(y.is_leaf)
                self.same(fr.autograd.grad(y.sum(),x)[0],torch.autograd.grad(ty.sum(),tx)[0],exact=True)
                with fr.no_grad(),torch.no_grad():
                    y,ty=getattr(x,name)(),getattr(tx,name)()
                self.same(y,ty,exact=True);self.assertFalse(y.requires_grad)
        for name in ('neg','negative'):
            with self.assertRaises((ValueError,TypeError)):
                getattr(fr.tensor([True,False]),name)()
            with self.assertRaises(RuntimeError):
                getattr(torch.tensor([True,False]),name)()

    def test_saved_output_vjps_preserve_seed_layout_and_prior_gradient_slots(self):
        for dtype in (torch.float32, torch.float64):
            for name in ('exp','sigmoid','tanh','sqrt'):
                for graph in ((False, True) if dtype==torch.float32 else (False,)):
                    for seed_kind in ('expanded','contiguous','transposed'):
                        with self.subTest(dtype=dtype,name=name,graph=graph,seed=seed_kind):
                            values = torch.tensor([[.2,.4,.3],[.7,.1,.6]],dtype=dtype)
                            tx = values.T.detach().requires_grad_()
                            x = native(values).T.detach().requires_grad_(True)
                            x.square().sum().backward(); tx.square().sum().backward()
                            held = x.grad; held_value = observed(held).clone()
                            y, ty = getattr(x,name)(), getattr(tx,name)()
                            if seed_kind=='expanded':
                                ts = torch.tensor(.7,dtype=dtype).expand(3,2)
                                seed = fr.tensor(.7,dtype=str(dtype).split('.')[-1]).expand(3,2)
                            elif seed_kind=='transposed':
                                ts = torch.full((2,3),.7,dtype=dtype).T
                                seed = fr.full((2,3),.7,dtype=str(dtype).split('.')[-1]).T
                            else:
                                ts = torch.full((3,2),.7,dtype=dtype)
                                seed = fr.full((3,2),.7,dtype=str(dtype).split('.')[-1])
                            seed.requires_grad_(True);ts.requires_grad_(True)
                            gradient = fr.autograd.grad(y,x,grad_outputs=seed,create_graph=graph)[0]
                            expected = torch.autograd.grad(ty,tx,ts,create_graph=graph)[0]
                            self.same(gradient, expected)
                            self.assertEqual(gradient.requires_grad, expected.requires_grad)
                            self.assertEqual(x.grad._id(), held._id())
                            torch.testing.assert_close(observed(held),held_value,rtol=0,atol=0)
                            if graph:
                                twice=fr.autograd.grad(gradient.sum(),(x,seed))
                                wanted=torch.autograd.grad(expected.sum(),(tx,ts))
                                for left,right in zip(twice,wanted):
                                    self.assertEqual(left.stride(),tuple(right.stride()))
                                    torch.testing.assert_close(observed(left),right,rtol=4e-6,atol=2e-7)

    def test_live_mixed_jvp_and_no_grad_preserve_computed_output_layouts(self):
        for name in ('exp','sigmoid','tanh','sqrt'):
            for dtype in (torch.float32,torch.float64):
                for enabled in (False,True):
                    tx = torch.tensor([[.2,.4,.3],[.7,.1,.6]]).T.detach().requires_grad_()
                    x = native(tx.T).T.detach().requires_grad_(True)
                    tv = torch.tensor(.7,dtype=dtype,requires_grad=True).expand(3,2)
                    v = fr.tensor(.7,dtype=str(dtype).split('.')[-1],requires_grad=True).expand(3,2)
                    with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                        y,dy = fr.func.jvp(lambda a:getattr(a,name)(),(x,),(v,))
                        ty,tdy = torch.func.jvp(lambda a:getattr(a,name)(),(tx,),(tv,))
                    self.same(y,ty);self.same(dy,tdy)
                    self.assertEqual(dy.requires_grad,tdy.requires_grad)
                    self.assertEqual(y.requires_grad,ty.requires_grad)
                    if enabled:
                        actual = fr.autograd.grad(dy.sum(),(x,v))
                        expected = torch.autograd.grad(tdy.sum(),(tx,tv))
                        self.same(actual[0],expected[0])
                        # The expanded seed is copied to the F32 primal layout/dtype.
                        # Its derivative is computed in F32 before widening back.
                        def coefficient(saved):
                            if name=='exp':return saved
                            if name=='sigmoid':return torch.ops.aten.sigmoid_backward(torch.ones_like(saved),saved)
                            if name=='tanh':return torch.ops.aten.tanh_backward(torch.ones_like(saved),saved)
                            return 1/(2*saved)
                        left,right=actual[1],expected[1]
                        self.assertEqual(left.shape,list(right.shape))
                        self.assertEqual(left.stride(),tuple(right.stride()))
                        self.assertEqual(left.dtype,str(right.dtype).split('.')[-1])
                        self.assertEqual(left.requires_grad,right.requires_grad)
                        own=coefficient(observed(y)).to(dtype=dtype)
                        reference=coefficient(ty.detach()).to(dtype=dtype)
                        torch.testing.assert_close(observed(left),own,rtol=0,atol=0)
                        torch.testing.assert_close(right,reference,rtol=0,atol=0)
                        torch.testing.assert_close(observed(left),right,rtol=4*torch.finfo(torch.float32).eps,atol=0)

    def test_symbolic_zero_unaries_and_allocating_casts_are_contiguous_but_clone_preserves(self):
        x=fr.ones(2,3,requires_grad=True);c=fr.tensor(2.,requires_grad=True)
        _,dy=fr.func.jvp(lambda a:a*c,(x,),(fr.ones_like(x),))
        z=fr.autograd.grad(dy.sum(),x)[0]
        self.assertTrue(z._is_structural_zero())
        for dtype in (torch.float32,torch.float64):
            value=z if dtype==torch.float32 else z*fr.ones_like(z,dtype='float64')
            value=value.T
            reference=torch._efficientzerotensor((2,3),dtype=dtype).T
            for name in ('neg','negative','relu','exp','sigmoid','tanh','sqrt','sign'):
                actual,expected=getattr(value,name)(),getattr(reference,name)()
                self.same(actual,expected)
                self.assertFalse(actual._is_structural_zero())
            for target in (torch.float32,torch.float64):
                actual=value.to(dtype=str(target).split('.')[-1],copy=True)
                expected=reference.to(dtype=target,copy=True)
                self.same(actual,expected,exact=True)
                self.assertFalse(actual._is_structural_zero())
            self.same(value.clone(),reference.clone(),exact=True)
            self.assertIs(value.to(dtype=str(dtype).split('.')[-1]),value)
            self.assertTrue(value._is_structural_zero())

    def test_ordinary_backward_of_live_derivatives_publishes_detached_gradients(self):
        for name in ('exp','sigmoid','tanh','sqrt'):
            for transpose in (False,True):
                for enabled in (False,True):
                    values=torch.tensor([[.2,.4,.3],[.7,.1,.6]])
                    x=native(values);tx=values.clone()
                    if transpose:x=x.T;tx=tx.T
                    x=x.detach().requires_grad_(True);tx=tx.detach().requires_grad_()
                    g=fr.autograd.grad(getattr(x,name)().sum(),x,create_graph=True)[0]
                    tg=torch.autograd.grad(getattr(tx,name)().sum(),tx,create_graph=True)[0]
                    loss=g.sum();tloss=tg.sum()
                    with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                        loss.backward();tloss.backward()
                        self.assertEqual(fr.is_grad_enabled(),enabled)
                    self.assertFalse(x.grad.requires_grad);self.assertTrue(x.grad.is_leaf)
                    self.assertEqual(x.grad.stride(),tuple(tx.grad.stride()))
                    torch.testing.assert_close(observed(x.grad),tx.grad,rtol=4e-6,atol=2e-7)
        self.assertTrue(fr.is_grad_enabled())

    def test_empty_offsets_scalar_capacity_and_capture_replay(self):
        for dtype in (torch.float32,torch.float64,torch.int64):
            name=str(dtype).split('.')[-1]
            a=fr.zeros(0,8,dtype=name).narrow(1,4,4).T
            t=torch.zeros(0,8,dtype=dtype).narrow(1,4,4).T
            self.same(a.sign(),t.sign(),exact=True)
            self.assertEqual(a.storage_offset(),4)
            self.same(fr.tensor(1,dtype=name).neg(),torch.tensor(1,dtype=dtype).neg(),exact=True)
            width=4 if dtype==torch.float32 else 8
            huge=fr.ones(1,dtype=name).expand(((1<<63)-1)//width+1)
            with self.assertRaises(ValueError):huge.sign()
            # Do not convert this zero-size shape to nested Python lists.
            actual=fr.zeros(1<<45,0,dtype=name).T.sign()
            expected=torch.zeros(1<<45,0,dtype=dtype).T.sign()
            self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.stride(),tuple(expected.stride()))
            self.assertEqual(actual.numel(),0)
        value=fr.tensor([[.2,.4,.3],[.7,.1,.6]])
        captured=fr.capture(lambda:value.T.exp()).compile_fused()
        value.fill_(1.)
        self.same(captured.replay(),torch.ones(2,3).T.exp())


if __name__=='__main__':unittest.main()
