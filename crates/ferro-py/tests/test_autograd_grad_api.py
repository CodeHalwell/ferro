"""Canonical joint autograd.grad against actual Torch; native passthrough identities."""
import unittest
import warnings
import torch
import ferro as fr


def native(value, requires_grad=None):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.'),
                     requires_grad=value.requires_grad if requires_grad is None else requires_grad)


class AutogradGradApiTests(unittest.TestCase):
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
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=5e-6, rtol=4e-5, equal_nan=True)

    def test_joint_duplicate_and_ancestor_outputs_are_one_reverse_traversal(self):
        for graph in (False, True):
            x = torch.tensor([1e20], requires_grad=True); a = native(x)
            out, target = a.square(), x.square()
            seeds = [torch.tensor([1e20], requires_grad=True), torch.tensor([-1e20], requires_grad=True)]
            actual = fr.autograd.grad((out, out), a, [native(v) for v in seeds], create_graph=graph)[0]
            expected = torch.autograd.grad((target, target), x, seeds, create_graph=graph, retain_graph=True)[0]
            self.same(actual, expected)
            self.assertEqual(actual.tolist(), [0.])
            separate = [torch.autograd.grad(target, x, v, retain_graph=True)[0] for v in seeds]
            self.assertTrue(torch.isnan(separate[0] + separate[1]).all())
            x = torch.tensor(2., requires_grad=True); a = native(x)
            seed = torch.tensor(5., requires_grad=True); g = native(seed)
            out, target = a.square(), x.square()
            result = fr.autograd.grad((a, out), (a, out, a), (fr.tensor(3.), g), create_graph=graph)
            expected = torch.autograd.grad((x, target), (x, target, x), (torch.tensor(3.), seed), create_graph=graph, retain_graph=True)
            for v, e in zip(result, expected): self.same(v, e)
            self.assertIs(result[0], result[2]); self.assertIs(result[1], g)
            self.assertIsNone(a.grad); self.assertIsNone(g.grad)

    def test_live_identity_cotangent_cast_follows_outer_grad_mode(self):
        for graph in (False, True):
            for enabled in (False, True):
                for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
                    with self.subTest(graph=graph, enabled=enabled, dtype=dtype):
                        x = torch.tensor([.2, .4], requires_grad=True); a = native(x)
                        seed = torch.tensor([1, 2], dtype=dtype, requires_grad=dtype in (torch.float32, torch.float64))
                        g = native(seed)
                        with torch.set_grad_enabled(enabled), (fr.enable_grad() if enabled else fr.no_grad()):
                            result = fr.autograd.grad(a, a, g, create_graph=graph)[0]
                            expected = torch.autograd.grad(x, x, seed, create_graph=graph)[0]
                        self.same(result, expected)
                        self.assertEqual(result is g, expected is seed)
                        self.assertIsNone(g.grad)

    def test_sum_views_and_computed_gradients_preserve_the_correct_seed_graph(self):
        for graph in (False, True):
            for operation in ('sum', 'add', 'multiply'):
                x = torch.tensor([.2, -.4], requires_grad=True); a = native(x)
                if operation == 'sum':
                    out, target = a.sum(), x.sum(); seed = torch.tensor(.7, requires_grad=True)
                elif operation == 'add':
                    out, target = a+a, x+x; seed = torch.tensor([.3,.4], requires_grad=True)
                else:
                    out, target = a*a, x*x; seed = torch.tensor([.3,.4], requires_grad=True)
                g = native(seed)
                result = fr.autograd.grad(out, a, g, create_graph=graph)[0]
                expected = torch.autograd.grad(target, x, seed, create_graph=graph, retain_graph=True)[0]
                self.same(result, expected)
                if graph:
                    actual2 = fr.autograd.grad(result.sum(), (a, g), allow_unused=True)
                    expected2 = torch.autograd.grad(expected.sum(), (x, seed), allow_unused=True)
                    for value, wanted in zip(actual2, expected2): self.same(value, wanted)

    def test_unused_none_materialized_and_connected_zero_have_distinct_semantics(self):
        for graph in (False, True):
            for enabled in (False, True):
                for materialize in (False, True):
                    x = torch.tensor([2.,3.], requires_grad=True); y = torch.tensor([7.,8.], requires_grad=True)
                    a, b = native(x), native(y)
                    actual_out, expected_out = (a*0).sum(), (x*0).sum()
                    with (fr.enable_grad() if enabled else fr.no_grad()), torch.set_grad_enabled(enabled):
                        result = fr.autograd.grad(actual_out, (a,b), create_graph=graph, allow_unused=True, materialize_grads=materialize)
                        expected = torch.autograd.grad(expected_out, (x,y), create_graph=graph, allow_unused=True, materialize_grads=materialize)
                    for value, wanted in zip(result, expected): self.same(value, wanted)
                    self.assertIsNotNone(result[0]); self.assertIsNone(a.grad); self.assertIsNone(b.grad)
        a = fr.tensor(2., requires_grad=True); b = fr.tensor(3., requires_grad=True)
        with self.assertRaises(RuntimeError): fr.autograd.grad(a.square(), (a,b))
        with self.assertRaises(ValueError): fr.autograd.grad(a.square(), a, allow_unused=False, materialize_grads=True)
        with self.assertRaises(RuntimeError): fr.autograd.grad(a.detach(), a, allow_unused=True)
        with self.assertRaises(RuntimeError): fr.autograd.grad(a.square(), b.detach(), allow_unused=True)
        self.assertEqual(fr.autograd.grad((), (a,), allow_unused=True), (None,))
        with self.assertRaises(RuntimeError): fr.autograd.grad((), (a,))

    def test_multiple_outputs_live_seeds_mixed_derivatives_and_dict_inputs(self):
        x = torch.tensor([[-.7,.2,.9],[.4,-.3,.6]], requires_grad=True)
        y = torch.tensor([[.8],[-.2]], requires_grad=True)
        a, b = native(x), native(y)
        h, th = (a*b).tanh(), (x*y).tanh()
        out, target = (h*h+a).sum(), (th*th+x).sum()
        seeds = (torch.full_like(th,.3,requires_grad=True), torch.tensor(-.4,requires_grad=True))
        g, k = [native(v) for v in seeds]
        result = fr.autograd.grad((h,out), {'left':a,'right':b}, (g,k), create_graph=True)
        expected = torch.autograd.grad((th,target), {'left':x,'right':y}, seeds, create_graph=True, retain_graph=True)
        self.assertEqual(list(result),list(expected))
        for key in result: self.same(result[key], expected[key])
        total = result['left'].sum()+result['right'].sum()
        wanted = expected['left'].sum()+expected['right'].sum()
        for value, reference in zip(fr.autograd.grad(total,(a,b,g,k)), torch.autograd.grad(wanted,(x,y,*seeds))):
            self.same(value,reference)

    def test_float64_and_mixed_outputs_inputs_match_torch(self):
        x = torch.tensor([.2,-.4],dtype=torch.float32,requires_grad=True)
        y = torch.tensor([.3,.7],dtype=torch.float64,requires_grad=True)
        a,b = native(x),native(y)
        outputs, expected = (a,(a+b).square()),(x,(x+y).square())
        seeds = (torch.tensor([.2,.3]),torch.tensor([.4,-.5],dtype=torch.float64))
        for v,e in zip(fr.autograd.grad(outputs,(a,b),[native(t) for t in seeds]),torch.autograd.grad(expected,(x,y),seeds)):
            self.same(v,e)
        with self.assertRaises(ValueError): fr.autograd.grad(outputs,(a,b),[native(t) for t in seeds],create_graph=True)

    def test_implicit_none_empty_one_element_and_sequence_forms(self):
        for shape in ((),(1,),(1,1)):
            x = torch.ones(shape,requires_grad=True); a = native(x)
            self.same(fr.autograd.grad([a.square()],a)[0],torch.autograd.grad([x.square()],x)[0])
        x = torch.tensor([.2,.4],requires_grad=True); a = native(x)
        result = fr.autograd.grad((a.sum(),a), (v for v in (a,)), (None,fr.ones(2)))
        expected = torch.autograd.grad((x.sum(),x),(x,),(None,torch.ones(2)))
        self.same(result[0],expected[0])
        a=fr.empty(0,2,requires_grad=True);x=torch.empty(0,2,requires_grad=True)
        self.same(fr.autograd.grad(a,a,fr.empty(0,2))[0],torch.autograd.grad(x,x,torch.empty(0,2))[0])
        with self.assertRaises(RuntimeError):fr.autograd.grad(a,a)
        with self.assertRaises(ValueError):fr.autograd.grad(fr.tensor(1.,requires_grad=True),())

    def test_preflight_and_late_version_failures_leave_grad_slots_and_modes_unchanged(self):
        a=fr.tensor([.2,.4],requires_grad=True);b=fr.tensor([.3,.6],requires_grad=True)
        a.sum().backward();b.sum().backward();before=[(v.grad._id(),v.grad.tolist(),v.tolist()) for v in (a,b)]
        outputs=(a.square(),b.square())
        for args in ((outputs,(a,b),(fr.ones(2),fr.ones(3))), (outputs,(a,b),(fr.ones(2),)), (outputs,(a,b),(fr.ones(2),3))):
            with self.assertRaises((ValueError,TypeError)):fr.autograd.grad(*args)
            self.assertEqual([(v.grad._id(),v.grad.tolist(),v.tolist()) for v in (a,b)],before)
        with fr.no_grad():b.fill_(9.)
        stale=[(v.grad._id(),v.grad.tolist(),v.tolist()) for v in (a,b)]
        with self.assertRaises(ValueError):fr.autograd.grad(outputs,(a,b),(fr.ones(2),fr.ones(2)))
        self.assertEqual([(v.grad._id(),v.grad.tolist(),v.tolist()) for v in (a,b)],stale)
        self.assertTrue(fr.is_grad_enabled())

    def test_graph_retention_batched_flags_and_legacy_permissive_paths_are_explicit(self):
        a=fr.tensor(2.,requires_grad=True);b=fr.tensor(3.,requires_grad=True);out=a.square()
        for retain in (None,True):self.assertEqual(fr.autograd.grad(out,a,retain_graph=retain)[0].item(),4.)
        with self.assertRaises(NotImplementedError):fr.autograd.grad(out,a,retain_graph=False)
        with self.assertRaises(NotImplementedError):fr.autograd.grad(out,a,is_grads_batched=True)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always');fr.autograd.grad(out,a,only_inputs=False)
            self.assertTrue(any(issubclass(v.category,FutureWarning) for v in seen))
        self.assertEqual(out.grad_wrt([b])[0].item(),0.)
        self.assertEqual(fr.autograd.vjp(out,[b],fr.tensor(1.))[0].item(),0.)
        with self.assertRaises(ValueError):fr.func.jvp(lambda x:fr.autograd.grad(x.sum(),x)[0],(fr.ones(2),),(fr.ones(2),))


if __name__ == '__main__':unittest.main()
