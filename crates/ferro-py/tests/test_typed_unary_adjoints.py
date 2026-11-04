"""Typed unary iteration and staged adjoints, with real Torch references."""
import unittest
import torch
import ferro as fr


def native(value):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).clone()


def observed(value):
    return torch.tensor(value.tolist(), dtype=getattr(torch, value.dtype)).reshape(value.shape)


def unary(value, name, native_side=False):
    if name.startswith('pow'):
        return value.pow(float(name[3:]))
    if name == 'clamp':
        return value.clamp(.2, .6)
    if name == 'gelu':
        return value.gelu() if native_side else torch.nn.functional.gelu(value, approximate='tanh')
    if name == 'silu':
        return value.silu() if native_side else torch.nn.functional.silu(value)
    return getattr(value, name)()


OPS = ('abs','log','log1p','log10','log2','sin','cos','tan','sinh','cosh','expm1','exp2','reciprocal','square','pow0','pow1','pow3.3','clamp','gelu','silu')


class TypedUnaryAdjoints(unittest.TestCase):
    def same(self, actual, expected, *, exact=False, layout=True):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        if layout:
            self.assertEqual(actual.stride(), expected.stride())
        got = observed(actual)
        torch.testing.assert_close(got, expected.detach(), rtol=0 if exact else 2e-12, atol=0 if exact else 1e-14, equal_nan=True)
        zeros = expected.detach() == 0
        self.assertEqual(torch.signbit(got[zeros]).tolist(), torch.signbit(expected.detach()[zeros]).tolist())

    def test_float64_input_output_and_cotangent_layouts_match_real_iteration(self):
        data = torch.tensor([[.13,.31,.47],[.27,.53,.71]], dtype=torch.float64)
        for name in OPS:
            for seed_kind in ('expanded','transpose','contiguous'):
                with self.subTest(name=name, seed=seed_kind):
                    x, tx = native(data).T.detach().requires_grad_(True), data.T.detach().requires_grad_()
                    if seed_kind == 'expanded':
                        seed, ts = fr.tensor(.7,dtype='float64').expand(3,2), torch.tensor(.7,dtype=torch.float64).expand(3,2)
                    elif seed_kind == 'transpose':
                        seed, ts = fr.full((2,3),.7,dtype='float64').T, torch.full((2,3),.7,dtype=torch.float64).T
                    else:
                        seed, ts = fr.full((3,2),.7,dtype='float64'), torch.full((3,2),.7,dtype=torch.float64)
                    y, ty = unary(x,name,True), unary(tx,name)
                    self.same(y,ty)
                    g = fr.autograd.grad(y,x,grad_outputs=seed)[0]
                    tg = torch.autograd.grad(ty,tx,ts)[0]
                    self.same(g,tg)
                    self.assertFalse(g.requires_grad)
                    self.assertFalse(y._shares_storage(x))
                    self.assertIsNone(x.grad)

    def test_saved_output_four_operations_allow_input_edits_and_reject_output_writes(self):
        for dtype in (torch.float32,torch.float64):
            for name in ('expm1','exp2','reciprocal','tan'):
                with self.subTest(dtype=dtype,name=name):
                    data = torch.tensor([[.2,.7,.3],[.4,1.1,.6]],dtype=dtype)
                    x,tx = native(data).T.detach().requires_grad_(True), data.T.detach().requires_grad_()
                    y,ty = unary(x,name,True), unary(tx,name)
                    seed = fr.tensor(.3,dtype=str(dtype).split('.')[-1]).expand(y.shape)
                    self.assertEqual(y.shape,list(ty.shape));self.assertEqual(y.stride(),ty.stride())
                    self.assertEqual(y.storage_offset(),ty.storage_offset());self.assertEqual(y.dtype,str(dtype).split('.')[-1])
                    self.assertFalse(y._shares_storage(x))
                    saved = torch.empty_strided(ty.shape,ty.stride(),dtype=dtype).copy_(observed(y))
                    s = torch.tensor(.3,dtype=dtype).expand(saved.shape)
                    expected = {'expm1':lambda:s*(saved+1),'exp2':lambda:(s*saved)*0.6931471805599453094,'reciprocal':lambda:(-s)*(saved*saved),'tan':lambda:s*(1+saved*saved)}[name]()
                    with fr.no_grad(),torch.no_grad():
                        x.T.fill_(9.)
                        tx.T.fill_(9.)
                    gradient = fr.autograd.grad(y,x,grad_outputs=seed,retain_graph=True)[0]
                    self.same(gradient,expected,exact=True)
                    expected_torch = torch.autograd.grad(ty,tx,torch.tensor(.3,dtype=dtype).expand(ty.shape),retain_graph=True)[0]
                    self.assertEqual(gradient.stride(),expected_torch.stride())
                    torch.testing.assert_close(observed(gradient),expected_torch,rtol=4*torch.finfo(dtype).eps,atol=0)
                    y.T.detach().zero_();ty.T.detach().zero_()
                    with self.assertRaises(ValueError):
                        fr.autograd.grad(y,x,grad_outputs=seed)
                    with self.assertRaises(RuntimeError):
                        torch.autograd.grad(ty,tx,torch.ones_like(ty))
                    self.assertIsNone(x.grad)

    def test_float64_input_saved_rules_allow_output_edits_but_guard_original_data(self):
        for name in OPS:
            if name in ('expm1','exp2','reciprocal','tan'):
                continue
            with self.subTest(name=name):
                data = torch.tensor([.2,.4],dtype=torch.float64)
                x,tx = native(data).requires_grad_(True), data.clone().requires_grad_()
                y,ty = unary(x,name,True),unary(tx,name)
                y.detach().zero_();ty.detach().zero_()
                self.same(fr.autograd.grad(y,x,grad_outputs=fr.ones(2,dtype='float64'),retain_graph=True)[0],torch.autograd.grad(ty,tx,torch.ones_like(ty),retain_graph=True)[0])
                with fr.no_grad(),torch.no_grad():
                    x.fill_(3.);tx.fill_(3.)
                with self.assertRaises(ValueError):
                    fr.autograd.grad(y,x,grad_outputs=fr.ones(2,dtype='float64'))
                with self.assertRaises(RuntimeError):
                    torch.autograd.grad(ty,tx,torch.ones_like(ty))

    def test_cancellation_half_powers_and_large_seed_use_exact_staging(self):
        for dtype in (torch.float32,torch.float64):
            data = torch.tensor([-50.,-40.],dtype=dtype)
            x,tx = native(data).requires_grad_(True),data.requires_grad_()
            self.same(fr.autograd.grad(x.expm1().sum(),x)[0],torch.autograd.grad(tx.expm1().sum(),tx)[0],exact=True)
        data = torch.tensor([-float('inf'),-0.,0.,float('inf'),float('nan')],dtype=torch.float64)
        for power in (.5,-.5):
            self.same(native(data).pow(power),data.pow(power),exact=True)
        x = fr.tensor([1e-150],dtype='float64',requires_grad=True)
        tx = torch.tensor([1e-150],dtype=torch.float64,requires_grad=True)
        seed,ts = fr.tensor([1e308],dtype='float64'),torch.tensor([1e308],dtype=torch.float64)
        actual = fr.autograd.grad(x.pow(3.),x,grad_outputs=seed)[0]
        expected = torch.autograd.grad(tx.pow(3.),tx,ts)[0]
        self.same(actual,expected,exact=True)

    def test_empty_transposed_views_keep_metadata_and_zero_derivative_shapes(self):
        for name in OPS:
            with self.subTest(name=name):
                x = fr.zeros((2,0),dtype='float64').T.detach().requires_grad_(True)
                tx = torch.zeros((2,0),dtype=torch.float64).T.detach().requires_grad_()
                y,ty = unary(x,name,True),unary(tx,name)
                self.same(y,ty)
                seed,ts = fr.zeros((2,0),dtype='float64').T,torch.zeros((2,0),dtype=torch.float64).T
                self.same(fr.autograd.grad(y,x,grad_outputs=seed)[0],torch.autograd.grad(ty,tx,ts)[0])
        huge = fr.zeros((1<<45,0),dtype='float64').T.expm1()
        self.assertEqual(huge.shape,[0,1<<45]);self.assertEqual(huge.numel(),0)

    def test_efficient_zero_cotangents_use_real_primitive_semantics_at_nonfinite_values(self):
        for dtype in ('float32','float64'):
            source = fr.ones((2,3),requires_grad=True)
            tsource = torch.ones((2,3),requires_grad=True)
            _, direction = fr.func.jvp(lambda a:a*2.,(source,),(fr.ones((2,3)),))
            _, tdirection = torch.func.jvp(lambda a:a*2.,(tsource,),(torch.ones((2,3)),))
            zero = fr.autograd.grad(direction.sum(),source)[0]
            tzero = torch.autograd.grad(tdirection.sum(),tsource)[0]
            if dtype == 'float64':
                zero = zero * fr.ones((2,3),dtype=dtype)
                tzero = tzero * torch.ones((2,3),dtype=torch.float64)
            zero,tzero = zero.T,tzero.T
            self.assertTrue(zero._is_structural_zero());self.assertTrue(torch._is_zerotensor(tzero))
            names = ('sqrt','expm1','exp2','tan','reciprocal') if dtype=='float32' else OPS+('sqrt',)
            for name in names:
                with self.subTest(dtype=dtype,name=name):
                    values = torch.tensor([[-1.,-0.,0.],[1.,float('inf'),float('nan')]],dtype=getattr(torch,dtype))
                    x,tx = native(values).T.detach().requires_grad_(True),values.T.detach().requires_grad_()
                    actual = fr.autograd.grad(unary(x,name,True),x,grad_outputs=zero)[0]
                    expected = torch.autograd.grad(unary(tx,name),tx,tzero)[0]
                    self.assertEqual(actual._is_structural_zero(),torch._is_zerotensor(expected))
                    self.same(actual,expected.clone(),exact=True)
                    self.assertIsNone(x.grad)

    def test_float64_ordinary_backward_keeps_gradients_detached_and_mode_restored(self):
        for name in OPS:
            data = torch.tensor([[.2,.4,.3],[.7,.1,.6]],dtype=torch.float64)
            x,tx = native(data).T.detach().requires_grad_(True),data.T.detach().requires_grad_()
            unary(x,name,True).sum().backward();unary(tx,name).sum().backward()
            self.same(x.grad,tx.grad)
            self.assertFalse(x.grad.requires_grad)
            self.assertTrue(fr.is_grad_enabled())


if __name__ == '__main__':
    unittest.main()
