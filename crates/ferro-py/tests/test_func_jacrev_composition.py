"""Actual Torch JVP of jacrev over the shared native reverse frame."""
import itertools
import unittest
import torch
import ferro as fr
from test_func_forward_composition import tree_same, gradients_match
from test_func_forward_ad import tensor

class JacrevCompositionTests(unittest.TestCase):
    def test_selected_trees_shared_identity_and_multi_output_columns(self):
        for enabled,xg,vg in itertools.product((False,True),repeat=3):
            with self.subTest(enabled=enabled,xg=xg,vg=vg):
                tx=torch.tensor([.2,-.4],requires_grad=xg)
                tv=torch.tensor([.3,.7],requires_grad=vg)
                tu=torch.tensor([.5,-.2],requires_grad=vg)
                x,v,u=map(tensor,(tx,tv,tu)); calls=[]
                def objective(library,a,b):
                    calls.append(library.__name__)
                    return {'product':(a['x']*b[0]).tanh(),'linear':a['x']*2,'constant':library.tensor([3.])}
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual=fr.func.jvp(fr.func.jacrev(lambda a,b:objective(fr,a,b),argnums=(1,0)),({'x':x},[x]),({'x':v},[u]))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected=torch.func.jvp(torch.func.jacrev(lambda a,b:objective(torch,a,b),argnums=(1,0)),({'x':tx},[tx]),({'x':tv},[tu]))
                tree_same(actual,expected)
                self.assertEqual(calls,['ferro','torch'])
                gradients_match(self,actual[1]['product'][0][0],expected[1]['product'][0][0],[(x,tx),(v,tv),(u,tu)],second=True)
                for value in (x,v,u):self.assertIsNone(value.grad)

    def test_auxiliary_actual_flags_identities_and_strides(self):
        for enabled,xg,vg,strided in itertools.product((False,True),repeat=4):
            with self.subTest(enabled=enabled,xg=xg,vg=vg,strided=strided):
                tx=torch.tensor([.2,-.1,-.4,.3],requires_grad=xg)
                tv=torch.tensor([.3,-.4,.7,.2],requires_grad=vg)
                x,v=map(tensor,(tx,tv))
                if strided:tx,tv,x,v=tx[::2],tv[::2],x[::2],v[::2]
                tw=torch.tensor(.5,requires_grad=True);w=tensor(tw)
                def objective(library,a,weight):
                    base=(a*a).tanh()*weight
                    with library.no_grad():view=base.reshape(base.shape)
                    return base,{'input':a,'view':view,'base':base,'capture':weight}
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual=fr.func.jvp(fr.func.jacrev(lambda a:objective(fr,a,w),has_aux=True),(x,),(v,),has_aux=True)
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected=torch.func.jvp(torch.func.jacrev(lambda a:objective(torch,a,tw),has_aux=True),(tx,),(tv,),has_aux=True)
                tree_same(actual,expected,leaves=True)
                self.assertIs(actual[2]['capture'],w)
                gradients_match(self,actual[1],expected[1],[(x,tx),(v,tv),(w,tw)],second=True)

    def test_empty_connected_linear_and_independent_strict_rows(self):
        for enabled,live,strict,kind in itertools.product((False,True),(False,True),(False,True),('empty_input','empty_output','empty_linear','constant_empty')):
            with self.subTest(enabled=enabled,live=live,strict=strict,kind=kind):
                tx=torch.empty(0,requires_grad=live) if kind in ('empty_input','empty_linear') else torch.tensor([.2,-.4],requires_grad=live)
                tv=torch.ones_like(tx,requires_grad=live);x,v=map(tensor,(tx,tv))
                def objective(library,a):
                    if kind=='empty_input':return a*a
                    if kind=='empty_linear':return a
                    if kind=='constant_empty':return library.zeros(0)
                    return (a*a)[:0]
                expected_error=None
                with torch.enable_grad() if enabled else torch.no_grad():
                    try:expected=torch.func.jvp(torch.func.jacrev(lambda a:objective(torch,a)),(tx,),(tv,),strict=strict)
                    except RuntimeError as error:expected_error=error
                with fr.enable_grad() if enabled else fr.no_grad():
                    if expected_error is not None:
                        with self.assertRaises((RuntimeError,ValueError)):fr.func.jvp(fr.func.jacrev(lambda a:objective(fr,a)),(x,),(v,),strict=strict)
                    else:
                        actual=fr.func.jvp(fr.func.jacrev(lambda a:objective(fr,a)),(x,),(v,),strict=strict)
                        tree_same(actual,expected)
                self.assertTrue(fr.is_grad_enabled())

    def test_nonlinear_captured_parameters_and_no_grad_islands_32_cases(self):
        for xg, vg, wg, enabled, island in itertools.product((False, True), repeat=5):
            with self.subTest(xg=xg, vg=vg, wg=wg, enabled=enabled, island=island):
                tx = torch.tensor([.2, -.4], requires_grad=xg)
                tv = torch.tensor([.3, .7], requires_grad=vg)
                tw = torch.tensor(.5, requires_grad=wg)
                x, v, w = map(tensor, (tx, tv, tw))
                calls = []
                def objective(library, a, weight):
                    calls.append(library.__name__)
                    if island:
                        with library.no_grad(): coefficient = (a * a).tanh()
                        return a * coefficient * weight
                    return (a * a).tanh() * weight
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = fr.func.jvp(fr.func.jacrev(lambda a: objective(fr, a, w)), (x,), (v,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected = torch.func.jvp(torch.func.jacrev(lambda a: objective(torch, a, tw)), (tx,), (tv,))
                tree_same(actual, expected)
                gradients_match(self, actual[1], expected[1], [(x, tx), (v, tv), (w, tw)], second=True)
                self.assertEqual(calls, ['ferro', 'torch'])
                self.assertTrue(all(value.grad is None for value in (x, v, w)))

if __name__=='__main__':unittest.main()
