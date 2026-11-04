"""Actual Torch controls for PReLU's shared two-output reverse operation."""
import itertools
import unittest
import torch
import ferro as fr


def native(value):
    return fr.tensor(value.detach().reshape(-1).tolist(), dtype=str(value.dtype).split('.')[-1], requires_grad=value.requires_grad).reshape(list(value.shape))


def gradients(engine, outputs, inputs, seeds=None, graph=True):
    return engine.autograd.grad(outputs, inputs, seeds, create_graph=graph, retain_graph=True, allow_unused=True)


def first(engine, x, w, g):
    y = engine.nn.functional.prelu(x, w)
    return y, gradients(engine, y, (x, w), g)


class PReLUHigherTests(unittest.TestCase):
    def same(self, actual, expected, *, layout=False, flags=True):
        self.assertEqual(actual is None, expected is None)
        if actual is None:
            return
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        if flags:
            self.assertEqual(actual.requires_grad, expected.requires_grad)
        if layout:
            self.assertEqual(tuple(actual.stride()), expected.stride())
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=2e-5, atol=2e-6, equal_nan=True)
        zero = expected.detach() == 0
        self.assertTrue(torch.equal(torch.signbit(values)[zero], torch.signbit(expected.detach())[zero]))

    def sequence(self, actual, expected, **kwargs):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.same(a, b, **kwargs)

    def test_channel_joint_first_second_third_and_optional_sibling_seeds(self):
        for transposed, branch, live in itertools.product((False, True), (0, 1, 2), (False, True)):
            with self.subTest(transposed=transposed, branch=branch, live=live):
                x = torch.linspace(-1.7, 1.9, 24).reshape(2, 3, 4).requires_grad_()
                g = torch.linspace(.2, 1.1, 24).reshape(2, 3, 4).requires_grad_()
                xx, gg = native(x), native(g)
                if transposed:
                    x, g = x.transpose(0, 2), g.transpose(0, 2)
                    xx, gg = xx.transpose(0, 2), gg.transpose(0, 2)
                w = torch.tensor([.2, -.3, .6], requires_grad=True)
                ww = native(w)
                _, actual = first(fr, xx, ww, gg)
                _, expected = first(torch, x, w, g)
                self.sequence(actual, expected, layout=True)
                a = torch.linspace(-.4, .8, x.numel()).reshape(x.shape).requires_grad_(live)
                b = torch.tensor([.3, -.7, .2], requires_grad=live)
                aa, bb = native(a), native(b)
                ids = (0,) if branch == 0 else (1,) if branch == 1 else (0, 1)
                second = gradients(fr, tuple(actual[i] for i in ids), (xx, ww, gg), tuple((aa, bb)[i] for i in ids))
                desired = gradients(torch, tuple(expected[i] for i in ids), (x, w, g), tuple((a, b)[i] for i in ids))
                self.sequence(second, desired)
                left = [value.square().sum() for value in second if value is not None and value.requires_grad]
                right = [value.square().sum() for value in desired if value is not None and value.requires_grad]
                self.assertTrue(left and right)
                selected_a = (xx, ww, gg) + ((aa, bb) if live else ())
                selected_b = (x, w, g) + ((a, b) if live else ())
                self.sequence(gradients(fr, sum(left[1:], left[0]), selected_a), gradients(torch, sum(right[1:], right[0]), selected_b))
                self.assertTrue(all(value.grad is None for value in (xx, ww, gg, aa, bb)))

    def test_absence_nonfinite_zero_boundaries_and_joint_signed_zero(self):
        cases = [(-float('inf'), float('inf'), 1., 0.), (2., 3., -0., 0.), (-0., .3, 1., 0.), (0., .3, 1., 0.), (float('nan'), .4, 1., -.2)]
        for xv, wv, av, bv in cases:
            for selected in ((0,), (1,), (0, 1)):
                with self.subTest(x=xv, w=wv, outputs=selected):
                    x = torch.tensor(xv, requires_grad=True)
                    w = torch.tensor(wv, requires_grad=True)
                    g = torch.tensor(1.2, requires_grad=True)
                    xx, ww, gg = map(native, (x, w, g))
                    _, actual = first(fr, xx, ww, gg)
                    _, expected = first(torch, x, w, g)
                    self.sequence(actual, expected)
                    seeds = (torch.tensor(av), torch.tensor(bv))
                    directions = tuple(native(seeds[i]) for i in selected)
                    got = gradients(fr, tuple(actual[i] for i in selected), (xx, ww, gg), directions)
                    want = gradients(torch, tuple(expected[i] for i in selected), (x, w, g), tuple(seeds[i] for i in selected))
                    self.sequence(got, want)

    def test_jvp_of_grad_preserves_real_flags_modes_and_captured_parameters(self):
        for x_live, v_live, w_live, enabled in itertools.product((False, True), repeat=4):
            with self.subTest(x=x_live, v=v_live, w=w_live, mode=enabled):
                x = torch.tensor([-1.2, -0., 0., .7], requires_grad=x_live)
                v = torch.tensor([.4, -.3, .2, .8], requires_grad=v_live)
                w = torch.tensor([.3], requires_grad=w_live)
                xx, vv, ww = map(native, (x, v, w))
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual, tangent = fr.func.jvp(fr.func.grad(lambda value: fr.nn.functional.prelu(value, ww).sum()), (xx,), (vv,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected, direction = torch.func.jvp(torch.func.grad(lambda value: torch.nn.functional.prelu(value, w).sum()), (x,), (v,))
                self.same(actual, expected)
                self.same(tangent, direction)
                left = tuple(value for value in (xx, vv, ww) if value.requires_grad)
                right = tuple(value for value in (x, v, w) if value.requires_grad)
                if tangent.requires_grad and left:
                    self.sequence(gradients(fr, tangent.sum(), left), gradients(torch, direction.sum(), right))

    def test_original_primal_and_seed_versions_guard_both_outputs(self):
        for changed in range(6):
            x = torch.tensor([-1., .7], requires_grad=True)
            w = torch.tensor([.3], requires_grad=True)
            g = torch.tensor([.2, .4], requires_grad=True)
            xx, ww, gg = map(native, (x, w, g))
            y, actual = first(fr, xx, ww, gg)
            reference, expected = first(torch, x, w, g)
            with fr.no_grad():
                (xx, ww, gg, y, *actual)[changed].detach().fill_(1.)
            with torch.no_grad():
                (x, w, g, reference, *expected)[changed].detach().fill_(1.)
            seeds_a = tuple(fr.ones_like(value) for value in actual)
            seeds_b = tuple(torch.ones_like(value) for value in expected)
            if changed < 3:
                with self.assertRaises(ValueError):
                    gradients(fr, actual, (xx, ww, gg), seeds_a)
                with self.assertRaises(RuntimeError):
                    gradients(torch, expected, (x, w, g), seeds_b)
            else:
                self.sequence(gradients(fr, actual, (xx, ww, gg), seeds_a), gradients(torch, expected, (x, w, g), seeds_b))

    def test_absent_tangents_materialize_ordinary_preserve_format_zeros(self):
        for dtype in (torch.float32, torch.float64):
            base=torch.full((3,4),float('nan'),dtype=dtype,requires_grad=True)
            nb=native(base)
            cases=((nb.T,base.T),(nb[:,::2].T,base[:,::2].T),
                   (nb[:1].expand(3,4),base[:1].expand(3,4)),(nb[:0].T,base[:0].T))
            for actual,expected in cases:
                _,zero=fr.func.jvp(lambda ignored:actual,(fr.tensor(1.),),(fr.tensor(.3),))
                _,reference=torch.func.jvp(lambda ignored:expected,(torch.tensor(1.),),(torch.tensor(.3),))
                self.same(zero,reference,layout=True)
                self.assertFalse(zero.requires_grad)
                self.same(zero*float('inf'),reference*float('inf'))
        x=torch.tensor([[-1.,2.,-.3],[.7,-.9,1.2]],requires_grad=True).T
        xx=native(x.T).T
        w=torch.tensor([.2,.3],requires_grad=True);ww=native(w)
        for select_weight in (False,True):
            function=lambda a,b:fr.nn.functional.prelu(a,b).sum()
            reference=lambda a,b:torch.nn.functional.prelu(a,b).sum()
            fg=fr.func.grad(function,argnums=(0,1));tg=torch.func.grad(reference,argnums=(0,1))
            if select_weight:
                actual,directions=fr.func.jvp(lambda q:fg(xx,q),(ww,),(fr.ones_like(ww),))
                expected,tdirections=torch.func.jvp(lambda q:tg(x,q),(w,),(torch.ones_like(w),))
            else:
                actual,directions=fr.func.jvp(lambda q:fg(q,ww),(xx,),(fr.ones_like(xx),))
                expected,tdirections=torch.func.jvp(lambda q:tg(q,w),(x,),(torch.ones_like(x),))
            self.sequence(actual,expected,layout=True)
            self.sequence(directions,tdirections,layout=True)
            with self.assertRaises(RuntimeError):
                fr.func.jvp((lambda q:fg(xx,q)) if select_weight else (lambda q:fg(q,ww)),
                            (ww,) if select_weight else (xx,),
                            (fr.ones_like(ww),) if select_weight else (fr.ones_like(xx),),strict=True)
            with self.assertRaises(RuntimeError):
                torch.func.jvp((lambda q:tg(x,q)) if select_weight else (lambda q:tg(q,w)),
                               (w,) if select_weight else (x,),
                               (torch.ones_like(w),) if select_weight else (torch.ones_like(x),),strict=True)

    def test_empty_shapes_and_remaining_typed_capture_limits(self):
        for shape in ((0, 3, 2), (2, 0, 3), (2, 3, 0), (0,)):
            x = torch.empty(shape, requires_grad=True)
            w = torch.ones(shape[1] if len(shape) > 1 else 1, requires_grad=True)
            g = torch.ones_like(x, requires_grad=True)
            xx, ww, gg = map(native, (x, w, g))
            _, actual = first(fr, xx, ww, gg)
            _, expected = first(torch, x, w, g)
            self.sequence(actual, expected)
            self.sequence(gradients(fr, actual, (xx, ww, gg), tuple(fr.ones_like(v) for v in actual)), gradients(torch, expected, (x, w, g), tuple(torch.ones_like(v) for v in expected)))
        x = fr.tensor([-.2, .7], dtype='float64', requires_grad=True)
        w = fr.tensor([.3], dtype='float64', requires_grad=True)
        with self.assertRaises(ValueError):
            gradients(fr, fr.nn.functional.prelu(x, w), (x, w), fr.ones_like(x))
        with self.assertRaises((ValueError, NotImplementedError)):
            fr.capture(lambda: fr.nn.functional.prelu(x.float(), w.float()))


if __name__ == '__main__':
    unittest.main()
