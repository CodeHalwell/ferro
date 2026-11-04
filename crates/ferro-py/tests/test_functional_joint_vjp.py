"""Joint transform roots retain real cotangent semantics on the native Op engine."""
import unittest
import torch
import ferro as fr


def native(value):
    return fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(value.requires_grad)


def shared_square(x):
    value = x * x
    return value, value


class FunctionalJointVjpTests(unittest.TestCase):
    def same(self, actual, expected, flags=True):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertIsNotNone(actual)
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        if flags:
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape), expected.detach(), atol=4e-6, rtol=4e-5)

    def test_functional_vjp_finite_shared_roots_cancel_before_overflow(self):
        for dtype in (torch.float32, torch.float64):
            for graph in ((False, True) if dtype == torch.float32 else (False,)):
                for strict in (False, True):
                    x = torch.tensor([1e10 if dtype == torch.float32 else 1e100], dtype=dtype, requires_grad=True)
                    p = torch.tensor([1e30 if dtype == torch.float32 else 1e250], dtype=dtype, requires_grad=True)
                    n = (-p.detach()).requires_grad_(True)
                    actual = fr.autograd.functional.vjp(shared_square, native(x), (native(p), native(n)), create_graph=graph, strict=strict)
                    expected = torch.autograd.functional.vjp(shared_square, x, (p,n), create_graph=graph, strict=strict)
                    for value, reference in zip(actual[0], expected[0]): self.same(value, reference)
                    self.same(actual[1], expected[1]); self.assertEqual(actual[1].tolist(), [0.])
                    self.assertTrue(torch.isfinite(expected[0][0]).all())

    def test_pytree_pullbacks_share_ancestor_roots_and_live_seed_gradients(self):
        for graph in (False, True):
            x = torch.tensor([1e10], requires_grad=True); a = native(x)
            p = torch.tensor([1e30], requires_grad=True); n = torch.tensor([-1e30], requires_grad=True)
            g,k = native(p),native(n)
            def model(value):
                h = value * value
                return {'first':h,'nested':(h,value.sum())}
            _, closure = fr.func.vjp(model,a); _, reference = torch.func.vjp(model,x)
            result, = closure({'first':g,'nested':(k,fr.tensor(1.))},create_graph=graph)
            expected, = reference({'first':p,'nested':(n,torch.tensor(1.))},create_graph=graph)
            self.same(result,expected);self.assertEqual(result.tolist(),[1.])
            if graph:
                derivatives=fr.autograd.grad(result.sum(),(a,g,k),allow_unused=True)
                wanted=torch.autograd.grad(expected.sum(),(x,p,n),allow_unused=True)
                for actual,target in zip(derivatives,wanted):self.same(actual,target)
            self.assertIsNone(a.grad);self.assertIsNone(g.grad);self.assertIsNone(k.grad)

    def test_unused_output_cannot_destroy_exact_passthrough_seed_identity(self):
        for graph in (False,True):
            for reverse in (False,True):
                x=torch.tensor(2.,requires_grad=True);y=torch.tensor(3.,requires_grad=True)
                a,b=native(x),native(y);seed=torch.tensor(7.,requires_grad=True);v=native(seed)
                output=lambda value,other: (other,value) if reverse else (value,other)
                _,closure=fr.func.vjp(lambda value:output(value,b),a)
                _,reference=torch.func.vjp(lambda value:output(value,y),x)
                seeds=(fr.tensor(9.),v) if reverse else (v,fr.tensor(9.))
                references=(torch.tensor(9.),seed) if reverse else (seed,torch.tensor(9.))
                got,=closure(seeds,create_graph=graph);expected,=reference(references,create_graph=graph)
                self.same(got,expected);self.assertIs(got,v);self.assertIs(expected,seed)

    def test_frozen_joint_forward_retains_captures_and_correct_mixed_partials(self):
        x=torch.tensor(2.,requires_grad=True);w=torch.tensor(3.,requires_grad=True)
        a,weight=native(x),native(w)
        def model(value,scale):
            h=value*value
            out=h*h*scale
            return out,out
        with fr.no_grad():_,closure=fr.func.vjp(lambda value:model(value,weight),a)
        with torch.no_grad():_,reference=torch.func.vjp(lambda value:model(value,w),x)
        p=torch.tensor(3.,requires_grad=True);n=torch.tensor(5.,requires_grad=True)
        v,z=native(p),native(n)
        result,=closure((v,z),create_graph=True);expected,=reference((p,n),create_graph=True)
        self.same(result,expected)
        for actual,target in zip(fr.autograd.grad(result,(a,weight,v,z),allow_unused=True),torch.autograd.grad(expected,(x,w,p,n),allow_unused=True)):
            self.same(actual,target)
        self.assertTrue(all(value.grad is None for value in (a,weight,v,z)))

    def test_private_reverse_frame_joint_roots_propagate_real_forward_tangents(self):
        x=torch.tensor([1e10,2e10],requires_grad=True);direction=torch.tensor([3.,5.])
        p=torch.tensor([1e30,1e30]);n=torch.tensor([-1e30,-1e30]);v,z=native(p),native(n)
        def actual_function(value):
            with fr._native._ForwardReverse(), fr.enable_grad():
                _, values, excluded, _, _, _, _ = fr.func._prepare((value,), (0,), True)
                h=values[0]*values[0]
                result=fr._native._forward_ad_vjp_many([h,h],list(values),[v,z])[0]
                return fr.func._post_dual(result,excluded,True)
        def expected_function(value):
            _,pullback=torch.func.vjp(shared_square,value)
            return pullback((p,n))[0]
        actual=fr.func.jvp(actual_function,(native(x),),(native(direction),))
        expected=torch.func.jvp(expected_function,(x,),(direction,))
        for value,target in zip(actual,expected):self.same(value,target)
        self.same(fr.func.jacfwd(actual_function)(native(x)),torch.func.jacfwd(expected_function)(x))
        with self.assertRaises(ValueError):fr._native._forward_ad_vjp_many([v],[v],[z])

    def test_constant_outputs_strict_modes_and_late_errors_preserve_slots(self):
        for graph in (False,True):
            x=torch.tensor([.2,.4],requires_grad=True);a=native(x)
            actual=fr.autograd.functional.vjp(lambda v:(v*v,fr.ones(2)),a,(fr.ones(2),fr.ones(2)),create_graph=graph)
            expected=torch.autograd.functional.vjp(lambda v:(v*v,torch.ones(2)),x,(torch.ones(2),torch.ones(2)),create_graph=graph)
            self.same(actual[1],expected[1])
            with self.assertRaises(RuntimeError):fr.autograd.functional.vjp(lambda v:(v*v,fr.ones(2)),a,(fr.ones(2),fr.ones(2)),strict=True)
            with self.assertRaises(RuntimeError):torch.autograd.functional.vjp(lambda v:(v*v,torch.ones(2)),x,(torch.ones(2),torch.ones(2)),strict=True)
        a=fr.tensor([.2,.4],requires_grad=True);a.sum().backward();before=(a.grad._id(),a.grad.tolist(),a.tolist())
        def model(value):
            h=value.exp()
            return h,h
        outputs,closure=fr.func.vjp(model,a)
        with self.assertRaises(RuntimeError):closure((fr.ones(2),fr.ones(3)))
        self.assertEqual((a.grad._id(),a.grad.tolist(),a.tolist()),before)
        outputs[0].detach().fill_(9.)
        with self.assertRaises(ValueError):closure((fr.ones(2),fr.ones(2)))
        self.assertEqual((a.grad._id(),a.grad.tolist(),a.tolist()),before)
        self.assertTrue(fr.is_grad_enabled())


if __name__ == '__main__':unittest.main()
