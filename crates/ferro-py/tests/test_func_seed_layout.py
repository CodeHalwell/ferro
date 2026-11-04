"""Actual Torch initial dual directions: metadata-copy, dtype, aliases and gradients."""
import itertools
import unittest
import torch
import ferro as fr


def make(library, kind, wide=False, live=True):
    dtype=library.float64 if wide else library.float32
    count=20 if kind.startswith('extra') else 12
    data=[.7+i*.123456789123 for i in range(count)]
    base=library.tensor(data,dtype=dtype)
    if kind=='contiguous':value=base[:6].clone().reshape(3,2)
    elif kind=='transpose':value=base[:6].clone().reshape(2,3).transpose(0,1)
    elif kind=='gap':value=base.reshape(3,4)[:,:2]
    elif kind=='expanded':value=library.tensor(.7,dtype=dtype).expand(3,2)
    elif kind=='row_expand':value=base[:2].clone().reshape(1,2).expand(3,2)
    elif kind=='offset_contiguous':value=base[:8][2:].reshape(3,2)
    elif kind=='offset_transpose':value=base[:8][2:].reshape(2,3).transpose(0,1)
    elif kind=='extra_contiguous':value=base[:6].reshape(3,2)
    elif kind=='extra_transpose':value=base[:6].reshape(2,3).transpose(0,1)
    else:raise AssertionError(kind)
    return value.detach().requires_grad_(live)


def same(actual, expected):
    if actual.shape!=list(expected.shape) or actual.stride()!=expected.stride() or actual.storage_offset()!=expected.storage_offset():
        raise AssertionError(('layout',actual.shape,actual.stride(),actual.storage_offset(),expected.shape,expected.stride(),expected.storage_offset()))
    if actual.dtype!=str(expected.dtype).split('.')[-1] or actual.requires_grad!=expected.requires_grad or actual.is_leaf!=expected.is_leaf:
        raise AssertionError(('metadata',actual.dtype,actual.requires_grad,actual.is_leaf,expected.dtype,expected.requires_grad,expected.is_leaf))
    observed=torch.tensor(actual.reshape(-1).tolist(),dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(observed,expected.detach().clone(),rtol=2e-6,atol=1e-7,equal_nan=True)


class SeedLayoutTests(unittest.TestCase):
    def test_normalized_layout_dtype_and_identity_162_cases(self):
        layouts=('contiguous','transpose','gap','expanded','row_expand','offset_contiguous','offset_transpose','extra_contiguous','extra_transpose')
        for left,right,wide in itertools.product(layouts,layouts,(False,True)):
            with self.subTest(left=left,right=right,wide=wide):
                x,v=make(fr,left),make(fr,right,wide);tx,tv=make(torch,left),make(torch,right,wide)
                try:expected=torch.func.jvp(lambda a:a,(tx,),(tv,))[1]
                except RuntimeError:
                    with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):fr.func.jvp(lambda a:a,(x,),(v,))
                    self.assertTrue(fr.is_grad_enabled());continue
                actual=fr.func.jvp(lambda a:a,(x,),(v,))[1]
                same(actual,expected)
                self.assertEqual(actual._shares_storage(v),torch._C._is_alias_of(expected,tv))
                self.assertEqual(actual is v,expected is tv)
                ga,=fr.autograd.grad(actual.sum(),v)
                gt,=torch.autograd.grad(expected.sum(),tv)
                same(ga,gt)
                self.assertIsNone(x.grad);self.assertIsNone(v.grad)

    def test_no_grad_copy_is_detached_but_compatible_direction_identity_is_live(self):
        for enabled,wide,matched in itertools.product((False,True),repeat=3):
            x=make(fr,'transpose');tx=make(torch,'transpose')
            v=make(fr,'transpose' if matched else 'contiguous',wide)
            tv=make(torch,'transpose' if matched else 'contiguous',wide)
            with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jvp(lambda a:a,(x,),(v,))[1]
            with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jvp(lambda a:a,(tx,),(tv,))[1]
            same(actual,expected);self.assertEqual(actual is v,expected is tv)
            self.assertTrue(v.requires_grad);self.assertIsNone(v.grad)

    def test_empty_offset_and_storage_span_controls(self):
        for wide,kind in itertools.product((False,True),('equal_empty','different_span','scalar_offset','singleton')):
            def pair(library):
                dtype=library.float64 if wide else library.float32
                if kind=='equal_empty':return library.zeros(0,8).narrow(1,4,4).transpose(0,1),library.zeros(4,0,dtype=dtype)
                if kind=='different_span':return library.ones(3)[2:2].reshape(0,1),library.zeros(0,1,dtype=dtype)
                if kind=='scalar_offset':return library.tensor([.1,.2,.3])[1],library.tensor(.7,dtype=dtype)
                return library.tensor([[.1],[.2]]).transpose(0,1),library.tensor([[.7,.8]],dtype=dtype)
            x,v=pair(fr);tx,tv=pair(torch)
            x.requires_grad_(True);v.requires_grad_(True);tx.requires_grad_();tv.requires_grad_()
            actual=fr.func.jvp(lambda a:a,(x,),(v,))[1];expected=torch.func.jvp(lambda a:a,(tx,),(tv,))[1]
            same(actual,expected);self.assertEqual(actual._shares_storage(v),torch._C._is_alias_of(expected,tv))

    def test_copy_preserves_live_cotangent_higher_order_and_ignores_source_values(self):
        x=make(fr,'transpose');tx=make(torch,'transpose')
        v=make(fr,'contiguous');tv=make(torch,'contiguous')
        actual=fr.func.jvp(lambda a:a,(x,),(v,))[1];expected=torch.func.jvp(lambda a:a,(tx,),(tv,))[1]
        g=make(fr,'contiguous');tg=make(torch,'contiguous')
        ga,=fr.autograd.grad(actual,v,grad_outputs=g,create_graph=True)
        gt,=torch.autograd.grad(expected,tv,grad_outputs=tg,create_graph=True)
        same(ga,gt);self.assertEqual(ga is g,gt is tg)
        with fr.no_grad():v.fill_(2.)
        with torch.no_grad():tv.fill_(2.)
        same(actual,expected)
        ga,=fr.autograd.grad(actual.sum(),v);gt,=torch.autograd.grad(expected.sum(),tv)
        same(ga,gt)

    def test_broadcast_direction_unary_reverse_gradients_and_modes(self):
        for name,wide,enabled in itertools.product(('exp','sigmoid','tanh','sqrt'),(False,True),(False,True)):
            x=make(fr,'transpose');tx=make(torch,'transpose')
            v=make(fr,'expanded',wide);tv=make(torch,'expanded',wide)
            with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jvp(lambda a:getattr(a,name)(),(x,),(v,))
            with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jvp(lambda a:getattr(a,name)(),(tx,),(tv,))
            for a,b in zip(actual,expected):same(a,b)
            if enabled:
                ga=fr.autograd.grad(actual[1].sum(),(x,v));gt=torch.autograd.grad(expected[1].sum(),(tx,tv))
                for a,b in zip(ga,gt):same(a,b)

    def test_copy_boundary_keeps_validation_and_future_level_atomic(self):
        x=make(fr,'expanded');v=make(fr,'contiguous')
        with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):fr.func.jvp(lambda a:a,(x,),(v,))
        self.assertIsNone(x.grad);self.assertIsNone(v.grad)
        x=make(fr,'transpose');v=make(fr,'contiguous',True)
        direction=fr.func.jvp(lambda a:a,(x,),(v,))[1]
        with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):fr.autograd.grad(direction.sum(),v,create_graph=True)
        self.assertTrue(fr.is_grad_enabled())
        expected=torch.func.jvp(lambda a:a,(make(torch,'transpose'),),(make(torch,'contiguous',True),))[1]
        same(direction,expected)


if __name__=='__main__':unittest.main()
