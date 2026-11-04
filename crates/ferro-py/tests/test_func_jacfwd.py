"""Actual Torch Jacobians from one multicolumn native forward evaluation."""
import itertools
import unittest
from unittest.mock import patch

import torch
import ferro as fr
from test_func_forward_ad import tensor, same
from test_func_forward_composition import tree_same, gradients_match


def flatten(value):
    if isinstance(value, (fr.Tensor, torch.Tensor)):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in flatten(item)]
    return [leaf for item in value for leaf in flatten(item)]


class JacfwdTests(unittest.TestCase):
    def test_values_graph_flags_and_mixed_derivatives_32_cases(self):
        for enabled, xg, wg, kind in itertools.product((False, True), (False, True), (False, True), ('linear','square','tanh','island')):
            with self.subTest(enabled=enabled, xg=xg, wg=wg, kind=kind):
                tx=torch.tensor([.2,-.4],requires_grad=xg);tw=torch.tensor(.5,requires_grad=wg)
                x,w=map(tensor,(tx,tw));calls=[]
                def objective(library,a,weight):
                    calls.append(library.__name__)
                    if kind=='island':
                        with library.no_grad():coefficient=(a*a).tanh()
                        value=a*coefficient*weight
                    else:value=(a*2 if kind=='linear' else a*a if kind=='square' else (a*a).tanh())*weight
                    return {'value':value,'constant':library.tensor([3.])}
                with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(lambda a:objective(fr,a,w))(x)
                with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jacfwd(lambda a:objective(torch,a,tw))(tx)
                tree_same(actual,expected,leaves=True)
                gradients_match(self,actual['value'],expected['value'],[(x,tx),(w,tw)],second=True)
                self.assertEqual(calls,['ferro','torch'])
                self.assertIsNone(x.grad);self.assertIsNone(w.grad)

    def test_argument_trees_duplicates_strides_and_shared_input_block_flags(self):
        for enabled,xg,wg in itertools.product((False,True),repeat=3):
            with self.subTest(enabled=enabled,xg=xg,wg=wg):
                tx=torch.tensor([.2,-.1,-.4,.3],requires_grad=xg)[::2];tw=torch.tensor(.5,requires_grad=wg)
                x,w=tensor(tx),tensor(tw)
                backing=fr.tensor([.2,-.1,-.4,.3],requires_grad=xg);x=backing[::2]
                calls=[]
                def objective(library,a,b,weight):
                    calls.append(library.__name__)
                    return {'nonlinear':(a['x']*b[0]).tanh()*weight,'linear':a['x']*2,'constant':library.ones(1)}
                with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(lambda a,b:objective(fr,a,b,w),argnums=(1,0))({'x':x},[x])
                with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jacfwd(lambda a,b:objective(torch,a,b,tw),argnums=(1,0))({'x':tx},[tx])
                tree_same(actual,expected,leaves=True)
                gradients_match(self,actual['nonlinear'][0][0],expected['nonlinear'][0][0],[(x,tx),(w,tw)],second=True)
                self.assertEqual(calls,['ferro','torch'])

    def test_empty_input_output_and_unused_blocks_retain_real_empty_graphs(self):
        for enabled,live,kind in itertools.product((False,True),(False,True),('identity','square','sum','independent','two_empty','one_empty','empty_output')):
            with self.subTest(enabled=enabled,live=live,kind=kind):
                tx=torch.empty(0,requires_grad=live);ty=torch.tensor([.2,-.4],requires_grad=live)
                x,y=map(tensor,(tx,ty));calls=[]
                def objective(library,a,b):
                    calls.append(library.__name__)
                    if kind=='identity':return a
                    if kind=='square':return a*a
                    if kind=='sum':return a.sum()
                    if kind=='independent':return library.ones(2)
                    if kind=='two_empty':return a*a,a*2
                    if kind=='one_empty':return b*2
                    return (b*b)[:0]
                all_args=kind in ('two_empty','one_empty','empty_output')
                with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(lambda a,b:objective(fr,a,b),argnums=(0,1) if all_args else 0)(x,y)
                with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jacfwd(lambda a,b:objective(torch,a,b),argnums=(0,1) if all_args else 0)(tx,ty)
                tree_same(actual,expected,leaves=True)
                self.assertEqual(calls,['ferro','torch'])

    def test_auxiliary_wrappers_flags_storage_and_single_function_call(self):
        for enabled,live in itertools.product((False,True),repeat=2):
            tx=torch.tensor([.2,-.4],requires_grad=live);tw=torch.tensor(.5,requires_grad=True)
            x,w=map(tensor,(tx,tw));calls=[]
            def objective(library,a,weight):
                calls.append(library.__name__)
                base=a*a
                with library.no_grad():view=base.reshape(base.shape)
                return base,{'input':a,'computed':base,'view':view,'capture':weight}
            with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(lambda a:objective(fr,a,w),has_aux=True)(x)
            with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jacfwd(lambda a:objective(torch,a,tw),has_aux=True)(tx)
            tree_same(actual,expected,leaves=True)
            self.assertIsNot(actual[1]['input'],x);self.assertIsNot(actual[1]['capture'],w)
            self.assertTrue(actual[1]['input']._shares_storage(x))
            self.assertEqual(calls,['ferro','torch'])

    def test_forward_over_reverse_hessian_includes_no_grad_coefficients(self):
        for enabled,xg,wg,island in itertools.product((False,True),repeat=4):
            with self.subTest(enabled=enabled,xg=xg,wg=wg,island=island):
                tx=torch.tensor([.2,-.4],requires_grad=xg);tw=torch.tensor(.5,requires_grad=wg)
                x,w=map(tensor,(tx,tw));calls=[]
                def objective(library,a,weight):
                    calls.append(library.__name__)
                    if island:
                        with library.no_grad():coefficient=(a*a).tanh()
                        return a*coefficient*weight
                    return (a*a).tanh()*weight
                with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(fr.func.jacrev(lambda a:objective(fr,a,w)))(x)
                with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.hessian(lambda a:objective(torch,a,tw))(tx)
                tree_same(actual,expected,leaves=True)
                gradients_match(self,actual,expected,[(x,tx),(w,tw)],second=True)
                self.assertEqual(calls,['ferro','torch'])

    def test_random_error_policy_before_draw_and_same_mask_columns(self):
        x=fr.tensor([.2,-.4,.6],requires_grad=True)
        tx=torch.tensor([.2,-.4,.6],requires_grad=True)
        for p in (0.,1.):
            actual=fr.func.jacfwd(lambda a:fr.nn.functional.dropout(a,p,True))(x)
            expected=torch.func.jacfwd(lambda a:torch.nn.functional.dropout(a,p,True))(tx)
            tree_same(actual,expected,leaves=True)
        for shape,p,live in itertools.product(((0,),(2,0)),(0.,.5,1.),(False,True)):
            a=fr.empty(shape,requires_grad=live);ta=torch.empty(shape,requires_grad=live)
            with patch('ferro.nn.functional.os.urandom',side_effect=AssertionError('empty dropout drew a seed')):
                actual=fr.func.jacfwd(lambda value:fr.nn.functional.dropout(value,p,True))(a)
            expected=torch.func.jacfwd(lambda value:torch.nn.functional.dropout(value,p,True))(ta)
            tree_same(actual,expected,leaves=True)
        for draw in (fr.rand,fr.randn):
            with self.assertRaises((ValueError,RuntimeError)):
                fr.func.jacfwd(lambda a:a+draw(0).sum())(x)
        gen=fr.Generator(7)
        with self.assertRaises((ValueError,RuntimeError)):
            fr.func.jacfwd(lambda a:a*fr.rand(3,generator=gen))(x)
        self.assertEqual(fr.rand(3,generator=gen).tolist(),fr.rand(3,generator=fr.Generator(7)).tolist())
        gen.manual_seed(7)
        with patch('ferro.nn.functional.os.urandom',side_effect=AssertionError('draw before guard')):
            with self.assertRaises((ValueError,RuntimeError)):
                fr.func.jacfwd(lambda a:fr.nn.functional.dropout(a,.5,True))(x)
        calls=[];masks=[]
        def objective(a):
            calls.append(1)
            mask=fr.rand(3,generator=gen)
            masks.append(mask)
            return a*mask
        actual=fr.func.jacfwd(objective,randomness='same')(x)
        mask=torch.tensor(masks[0].tolist())
        expected=torch.func.jacfwd(lambda a:a*mask,randomness='same')(tx)
        tree_same(actual,expected,leaves=True)
        self.assertEqual(calls,[1])
        control=fr.Generator(7);fr.rand(3,generator=control)
        self.assertEqual(fr.rand(3,generator=gen).tolist(),fr.rand(3,generator=control).tolist())
        native_mask=fr.nn.functional.dropout(fr.ones(3),.5,True,seed=17)
        actual=fr.func.jacfwd(lambda a:fr.nn.functional.dropout(a,.5,True,seed=17),randomness='same')(x)
        expected=torch.func.jacfwd(lambda a:a*torch.tensor(native_mask.tolist()),randomness='same')(tx)
        tree_same(actual,expected,leaves=True)
        gradients_match(self,actual,expected,[(x,tx)])
        module=fr.nn.Dropout(.5)
        before=module.rng_state.tolist()
        with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):
            fr.func.jacfwd(module,randomness='same')(x)
        self.assertEqual(module.rng_state.tolist(),before)
        tree_same(fr.func.jacfwd(module)(fr.empty(0)),torch.func.jacfwd(torch.nn.Dropout(.5))(torch.empty(0)))

    def test_invalid_capacity_and_nested_level_reject_before_callable(self):
        calls=[]
        function=lambda a:(calls.append(1) or a*a)
        for kwargs in ({'argnums':()}, {'argnums':(0,0)}, {'argnums':2}, {'randomness':'different'}):
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError,RuntimeError,NotImplementedError)):
                fr.func.jacfwd(function,**kwargs)(fr.ones(2))
        huge=fr.ones(1).expand([1<<31])
        with self.assertRaises((ValueError,OverflowError)):
            fr.func.jacfwd(function)(huge)
        self.assertEqual(calls,[])
        with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):
            fr.func.jvp(fr.func.jacfwd(function),(fr.ones(2),),(fr.ones(2),))
        self.assertEqual(calls,[])
        def huge_output(a):
            calls.append(1)
            return a[:1].expand([1 << 60])
        with self.assertRaises((ValueError, OverflowError)):
            fr.func.jacfwd(huge_output)(fr.ones(2))
        self.assertEqual(calls,[1])
        self.assertTrue(fr.is_grad_enabled())
        same(fr.func.jacfwd(lambda a:a*a)(fr.tensor([.2])),torch.func.jacfwd(lambda a:a*a)(torch.tensor([.2])))

    def test_seeded_shared_parameter_network_input_and_parameter_derivatives(self):
        for seed in (7,11,43):
            generator=torch.Generator().manual_seed(seed)
            tx=torch.randn(2,3,generator=generator).requires_grad_()
            tw=torch.randn(3,2,generator=generator).requires_grad_()
            tb=torch.randn(2,generator=generator).requires_grad_()
            x,w,b=map(tensor,(tx,tw,tb));calls=[]
            def objective(library,a,weight,bias):
                calls.append(library.__name__)
                hidden=(a@weight+bias).tanh()
                return (hidden*hidden+hidden).sum(0)
            actual=fr.func.jacfwd(lambda a,weight,bias:objective(fr,a,weight,bias),argnums=(0,1,2))(x,w,b)
            expected=torch.func.jacfwd(lambda a,weight,bias:objective(torch,a,weight,bias),argnums=(0,1,2))(tx,tw,tb)
            tree_same(actual,expected)
            for av,tv in zip(actual,expected):gradients_match(self,av,tv,[(x,tx),(w,tw),(b,tb)],second=True)
            self.assertEqual(calls,['ferro','torch'])
            self.assertTrue(all(value.grad is None for value in (x,w,b)))


if __name__=='__main__':unittest.main()
