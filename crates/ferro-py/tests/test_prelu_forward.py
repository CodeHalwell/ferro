"""Actual Torch controls for direct PReLU JVP on the existing Op engine."""
import math
import unittest
import ferro as fr
import torch


def native(value):
    return fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).split('.')[-1], requires_grad=value.requires_grad).reshape(list(value.shape))


class PReLUForwardTests(unittest.TestCase):
    def same(self, actual, expected, flags=True):
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        self.assertEqual(actual.shape, list(expected.shape))
        if flags: self.assertEqual(actual.requires_grad, expected.requires_grad)
        values=actual.detach().reshape([-1]).tolist()
        for a,b in zip(values,expected.detach().clone().reshape(-1).tolist()):
            if math.isnan(b): self.assertTrue(math.isnan(a))
            else:
                self.assertEqual(a,b)
                if a==b==0.: self.assertEqual(math.copysign(1.,a),math.copysign(1.,b))

    def test_actual_missing_tangents_mixed_precision_flags_and_reverse_dependencies(self):
        for dtype in (torch.float32,torch.float64):
            for mode in ('input','weight','both'):
                for live in (False,True):
                    for enabled in (False,True):
                        with self.subTest(dtype=dtype,mode=mode,live=live,enabled=enabled):
                            x=torch.tensor([-math.inf,-2.,-0.,0.,2.,math.inf,math.nan],requires_grad=live)
                            w=torch.tensor([.3],requires_grad=live)
                            dx=torch.full(x.shape,.7,dtype=dtype,requires_grad=live);dw=torch.full(w.shape,.2,dtype=dtype,requires_grad=live)
                            xx,ww,vv,uu=map(native,(x,w,dx,dw))
                            fn=(lambda value:fr.nn.functional.prelu(value,ww)) if mode=='input' else (lambda value:fr.nn.functional.prelu(xx,value)) if mode=='weight' else fr.nn.functional.prelu
                            ref=(lambda value:torch.nn.functional.prelu(value,w)) if mode=='input' else (lambda value:torch.nn.functional.prelu(x,value)) if mode=='weight' else torch.nn.functional.prelu
                            args,dirs=((xx,),(vv,)) if mode=='input' else ((ww,),(uu,)) if mode=='weight' else ((xx,ww),(vv,uu))
                            ta,tv=((x,),(dx,)) if mode=='input' else ((w,),(dw,)) if mode=='weight' else ((x,w),(dx,dw))
                            with fr.enable_grad() if enabled else fr.no_grad(): actual,tangent=fr.func.jvp(fn,args,dirs)
                            with torch.enable_grad() if enabled else torch.no_grad(): expected,direction=torch.func.jvp(ref,ta,tv)
                            self.same(actual,expected);self.same(tangent,direction)
                            if enabled and live:
                                graph=dtype==torch.float32
                                got=fr.autograd.grad(tangent,(xx,ww,vv,uu),fr.ones_like(tangent),allow_unused=True,create_graph=graph,retain_graph=True)
                                want=torch.autograd.grad(direction,(x,w,dx,dw),torch.ones_like(direction),allow_unused=True,create_graph=graph,retain_graph=True)
                                for a,b in zip(got,want):
                                    self.assertEqual(a is None,b is None)
                                    if a is not None:self.same(a,b)

    def test_channel_strides_live_directions_and_jacfwd_zero_boundary(self):
        for transpose in (False,True):
            x=torch.linspace(-2.,2.,24).reshape(4,3,2).permute(2,1,0) if transpose else torch.linspace(-2.,2.,24).reshape(2,3,4)
            x=x.detach().requires_grad_();w=torch.tensor([.2,.3,.4],requires_grad=True)
            dx=torch.ones_like(x,requires_grad=True);dw=torch.tensor([.7,.8,.9],requires_grad=True)
            xx,ww,vv,uu=map(native,(x,w,dx,dw))
            # Preserve the tested physical input and direction permutations.
            if transpose:
                xx=native(x.permute(2,1,0)).permute(2,1,0)
                vv=native(dx.permute(2,1,0)).permute(2,1,0)
            out,derivative=fr.func.jvp(fr.nn.functional.prelu,(xx,ww),(vv,uu))
            expected,wanted=torch.func.jvp(torch.nn.functional.prelu,(x,w),(dx,dw))
            self.same(out,expected);self.same(derivative,wanted);self.assertEqual(derivative.stride(),wanted.stride())
            got=fr.autograd.grad(derivative,(xx,ww,vv,uu),fr.ones_like(derivative),create_graph=True)
            refs=torch.autograd.grad(wanted,(x,w,dx,dw),torch.ones_like(wanted),create_graph=True)
            for a,b in zip(got,refs):self.same(a,b)
        x=fr.tensor([-2.,-0.,0.,2.]);tx=torch.tensor([-2.,-0.,0.,2.])
        actual=fr.func.jacfwd(lambda value:fr.nn.functional.prelu(value,fr.tensor([.3])))(x)
        expected=torch.func.jacfwd(lambda value:torch.nn.functional.prelu(value,torch.tensor([.3])))(tx)
        self.same(actual,expected)

    def test_scalar_empty_modes_and_remaining_reverse_capture_boundaries(self):
        for shape,weights in (((),()),((0,3,2),(3,)),((2,0,4),(0,))):
            x=torch.zeros(shape,requires_grad=True);w=torch.full(weights,.3,requires_grad=True)
            xx,ww=native(x),native(w)
            out,tangent=fr.func.jvp(fr.nn.functional.prelu,(xx,ww),(fr.ones_like(xx),fr.ones_like(ww)))
            expected,desired=torch.func.jvp(torch.nn.functional.prelu,(x,w),(torch.ones_like(x),torch.ones_like(w)))
            self.same(out,expected);self.same(tangent,desired)
        x=torch.tensor([-2.,0.,3.],requires_grad=True);dx=torch.tensor([.3,.7,-.2],requires_grad=True)
        xx,vv=native(x),native(dx);w=fr.tensor([.12345678912345678],dtype='float64');tw=torch.tensor([.12345678912345678],dtype=torch.float64)
        actual,tangent=fr.func.jvp(lambda value:fr.nn.functional.prelu(value.double(),w),(xx,),(vv,))
        expected,direction=torch.func.jvp(lambda value:torch.nn.functional.prelu(value.double(),tw),(x,),(dx,))
        self.same(actual,expected);self.same(tangent,direction)
        ga=fr.autograd.grad(tangent,(xx,vv),fr.ones_like(tangent),allow_unused=True)
        gb=torch.autograd.grad(direction,(x,dx),torch.ones_like(direction),allow_unused=True)
        for a,b in zip(ga,gb):
            self.assertEqual(a is None,b is None)
            if a is not None:self.same(a,b)
        for changed in range(6):
            x=torch.tensor([-.2,.3],requires_grad=True);w=torch.tensor([.4],requires_grad=True)
            dx=torch.tensor([.7,.8],requires_grad=True);dw=torch.tensor([.2],requires_grad=True)
            xx,ww,vv,uu=map(native,(x,w,dx,dw))
            out,tangent=fr.func.jvp(fr.nn.functional.prelu,(xx,ww),(vv,uu))
            expected,direction=torch.func.jvp(torch.nn.functional.prelu,(x,w),(dx,dw))
            with fr.no_grad():(xx,ww,vv,uu,out,tangent)[changed].detach().fill_(1.)
            with torch.no_grad():(x,w,dx,dw,expected,direction)[changed].detach().fill_(1.)
            if changed<4:
                with self.assertRaises(ValueError):fr.autograd.grad(tangent,(xx,ww,vv,uu),fr.ones_like(tangent))
                with self.assertRaises(RuntimeError):torch.autograd.grad(direction,(x,w,dx,dw),torch.ones_like(direction))
            else:
                ga=fr.autograd.grad(tangent,(xx,ww,vv,uu),fr.ones_like(tangent))
                gb=torch.autograd.grad(direction,(x,w,dx,dw),torch.ones_like(direction))
                for a,b in zip(ga,gb):self.same(a,b)
        x=fr.tensor([-.2,.3],requires_grad=True);w=fr.tensor([.4],requires_grad=True)
        self.assertEqual(fr.autograd.grad(fr.nn.functional.prelu(x,w),x,fr.ones_like(x),create_graph=True)[0].shape,x.shape)
        with self.assertRaises((ValueError,NotImplementedError)):fr.capture(lambda:fr.nn.functional.prelu(x,w))
        self.assertEqual(fr.nn.functional.prelu(x,w).shape,x.shape)

if __name__=='__main__':unittest.main()
