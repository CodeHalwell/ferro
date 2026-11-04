"""Saved LayerNorm statistics and the real power history of Square JVPs."""
import itertools
import unittest
import torch
import ferro as fr


def actual(value):
    return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)


def native(value):
    return fr.tensor(value.detach().tolist(),dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).requires_grad_(value.requires_grad)


class SavedStatisticsTests(unittest.TestCase):
    def same(self,left,right):
        self.assertEqual(left.shape,list(right.shape))
        self.assertEqual(left.requires_grad,right.requires_grad)
        self.assertEqual(left.dtype,str(right.dtype).split('.')[-1])
        self.assertEqual(left._is_structural_zero(),torch._is_zerotensor(right))
        torch.testing.assert_close(actual(left),right.detach().clone(),rtol=1e-4,atol=5e-5,equal_nan=True)

    def test_square_jacobian_retains_actual_power_history_through_empty_and_scalar_outputs(self):
        for shape in ((),(2,),(2,3),(0,3)):
            for enabled,live in itertools.product((False,True),repeat=2):
                with self.subTest(shape=shape,enabled=enabled,live=live):
                    tx=torch.full(shape,2.,requires_grad=live);x=native(tx)
                    calls=[]
                    def operation(a):calls.append(1);return a.square()
                    with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                        y=fr.func.jacfwd(operation)(x);ty=torch.func.jacfwd(lambda a:a.square())(tx)
                    self.assertEqual(calls,[1]);self.same(y,ty);self.assertEqual(y.stride(),tuple(ty.stride()))
                    if ty.requires_grad:
                        second=fr.autograd.grad(y.sum(),x,create_graph=True)[0]
                        tsecond=torch.autograd.grad(ty.sum(),tx,create_graph=True)[0]
                        self.same(second,tsecond)
                        third=fr.autograd.grad(second.sum(),x,create_graph=True)[0]
                        tthird=torch.autograd.grad(tsecond.sum(),tx,create_graph=True)[0]
                        self.same(third,tthird)
                    self.assertIsNone(x.grad)

    def test_normal_and_physical_layer_norm_keep_saved_statistics_and_live_affine_paths(self):
        for physical,mask,enabled,live in itertools.product((False,True),range(1,8),(False,True),(False,True)):
            with self.subTest(physical=physical,mask=mask,enabled=enabled,live=live):
                shapes=((2,3),(3,),(3,))
                tv=[(torch.arange(torch.tensor(s).prod().item(),dtype=torch.float32).reshape(s)*.17+.2+i*.1).requires_grad_(live) for i,s in enumerate(shapes)]
                values=[native(t) for t in tv]
                selected=tuple(i for i in range(3) if mask>>i&1)
                def function(library,base,args):
                    operands=list(base)
                    for i,a in zip(selected,args):operands[i]=a
                    return library.nn.functional.layer_norm(operands[0],(3,),operands[1],operands[2],1e-5)
                f=lambda *args:function(fr,values,args)
                t=lambda *args:function(torch,tv,args)
                with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                    if physical:
                        result=fr.func.jacfwd(f,argnums=tuple(range(len(selected))))(*[values[i] for i in selected])
                        reference=torch.func.jacfwd(t,argnums=tuple(range(len(selected))))(*[tv[i] for i in selected])
                    else:
                        directions=[torch.full_like(tv[i],.7) for i in selected]
                        result=(fr.func.jvp(f,tuple(values[i] for i in selected),tuple(native(d) for d in directions))[1],)
                        reference=(torch.func.jvp(t,tuple(tv[i] for i in selected),tuple(directions))[1],)
                for left,right in zip(result,reference):
                    self.same(left,right)
                    if right.requires_grad:
                        gradients=fr.autograd.grad(left.sum(),values,create_graph=True,allow_unused=True)
                        expected=torch.autograd.grad(right.sum(),tv,create_graph=True,allow_unused=True,retain_graph=True)
                        for a,b in zip(gradients,expected):
                            self.assertEqual(a is None,b is None)
                            if b is not None:self.same(a,b)
                self.assertTrue(all(value.grad is None for value in values))

    def test_layer_norm_tangent_uses_saved_statistics_without_saving_primal_output(self):
        tx=torch.tensor([[.2,.7,-.4],[1.,.3,.8]])
        x=native(tx);tv=torch.full_like(tx,.7,requires_grad=True);v=native(tv)
        y,dy=fr.func.jvp(lambda a:fr.nn.functional.layer_norm(a,(3,)),(x,),(v,))
        ty,tdy=torch.func.jvp(lambda a:torch.nn.functional.layer_norm(a,(3,)),(tx,),(tv,))
        with fr.no_grad(),torch.no_grad():
            y.fill_(3.);ty.fill_(3.);x.fill_(2.);tx.fill_(2.)
        self.same(fr.autograd.grad(dy.sum(),v)[0],torch.autograd.grad(tdy.sum(),tv)[0])


if __name__=='__main__':unittest.main()
