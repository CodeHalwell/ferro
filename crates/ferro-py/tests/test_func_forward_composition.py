"""Actual Torch forward-over-reverse transforms, including private alias handling."""
import itertools
import unittest

import ferro as fr
from ferro import func
import torch

from test_func_forward_ad import same, tensor


def tree_same(actual, expected, *, leaves=False):
    if isinstance(expected, torch.Tensor):
        same(actual, expected)
        if leaves and actual.is_leaf != expected.is_leaf:
            raise AssertionError(('is_leaf', actual.is_leaf, expected.is_leaf))
    elif isinstance(expected, dict):
        if list(actual) != list(expected): raise AssertionError('dictionary structure')
        for key in expected: tree_same(actual[key], expected[key], leaves=leaves)
    else:
        if type(actual) is not type(expected) or len(actual) != len(expected): raise AssertionError('sequence structure')
        for a, b in zip(actual, expected): tree_same(a, b, leaves=leaves)


def gradients_match(test, actual, expected, pairs, *, second=False):
    selected = [(a, b) for a, b in pairs if b.requires_grad]
    if not expected.requires_grad or not selected: return
    gradients = fr.autograd.grad(actual.sum(), [a for a, _ in selected], create_graph=True, allow_unused=True)
    references = torch.autograd.grad(expected.sum(), [b for _, b in selected], create_graph=True, allow_unused=True)
    for gradient, reference in zip(gradients, references):
        if reference is None:
            test.assertIsNone(gradient)
        else:
            same(gradient, reference, symbolic=True)
            if second and reference.requires_grad:
                gradients_match(test, gradient, reference, selected)


class ForwardCompositionTests(unittest.TestCase):
    def test_nonlinear_shared_capture_and_no_grad_islands_32_torch_cases(self):
        for xg, vg, wg, enabled, island in itertools.product((False, True), repeat=5):
            with self.subTest(xg=xg, vg=vg, wg=wg, enabled=enabled, island=island):
                tx = torch.tensor([.2, -.4], requires_grad=xg)
                tv = torch.tensor([.3, .7], requires_grad=vg)
                tw = torch.tensor(.5, requires_grad=wg)
                x, v, w = tensor(tx), tensor(tv), tensor(tw)
                calls = []
                def objective(library, a, weight):
                    if island:
                        with library.no_grad(): coefficient = (a * a).tanh()
                        return (a * coefficient * weight).sum()
                    return ((a * a).tanh() * weight).sum()
                def actual(a):
                    calls.append(1)
                    return objective(fr, a, w)
                with fr.enable_grad() if enabled else fr.no_grad():
                    primal, tangent = func.jvp(func.grad(actual), (x,), (v,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected, expected_tangent = torch.func.jvp(torch.func.grad(lambda a: objective(torch, a, tw)), (tx,), (tv,))
                same(primal, expected)
                same(tangent, expected_tangent)
                gradients_match(self, tangent, expected_tangent, [(x, tx), (v, tv), (w, tw)], second=True)
                self.assertEqual(calls, [1])
                for value in (x, v, w): self.assertIsNone(value.grad)
                self.assertTrue(fr.is_grad_enabled())

    def test_linear_product_and_quadratic_provenance_24_torch_cases(self):
        for enabled, xg, vg, kind in itertools.product((False, True), (False, True), (False, True), ('mul', 'linear', 'square')):
            with self.subTest(enabled=enabled, xg=xg, vg=vg, kind=kind):
                tx = torch.tensor(.2, requires_grad=xg)
                tz = torch.tensor(.7, requires_grad=xg)
                tv = torch.tensor(.3, requires_grad=vg)
                tu = torch.tensor(.4, requires_grad=vg)
                x, z, v, u = map(tensor, (tx, tz, tv, tu))
                def objective(a, b): return a * b if kind == 'mul' else a + b if kind == 'linear' else a * a
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = func.jvp(func.grad(objective, argnums=0), (x, z), (v, u))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected = torch.func.jvp(torch.func.grad(objective, argnums=0), (tx, tz), (tv, tu))
                tree_same(actual, expected, leaves=True)
                gradients_match(self, actual[1], expected[1], list(zip((x, z, v, u), (tx, tz, tv, tu))))

    def test_grad_and_value_auxiliary_views_trees_and_duplicate_arguments(self):
        for enabled, xg, vg, kind in itertools.product((False, True), (False, True), (False, True), ('identity', 'view', 'mul')):
            with self.subTest(enabled=enabled, xg=xg, vg=vg, kind=kind):
                tx = torch.tensor(.2, requires_grad=xg)
                tv = torch.tensor(.3, requires_grad=vg)
                x, v = tensor(tx), tensor(tv)
                calls = []
                def objective(a):
                    calls.append(1)
                    value = a if kind == 'identity' else a.reshape(()) if kind == 'view' else a * 2
                    return value, {'identity': a, 'view': a.reshape(()), 'math': a * 2}
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = func.jvp(func.grad_and_value(objective, has_aux=True), (x,), (v,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected = torch.func.jvp(torch.func.grad_and_value(objective, has_aux=True), (tx,), (tv,))
                tree_same(actual, expected, leaves=True)
                self.assertEqual(calls, [1, 1])
                self.assertIs(actual[1][1][1]['identity'], v)
        tx = torch.tensor([.2, -.4], requires_grad=True)
        tv, tu = torch.tensor([.3, .7], requires_grad=True), torch.tensor([.5, -.2])
        x, v, u = map(tensor, (tx, tv, tu))
        def objective(a, b): return (a['x'] * b[0]).tanh().sum()
        actual = func.jvp(func.grad(objective, argnums=(1, 0)), ({'x': x}, [x]), ({'x': v}, [u]))
        expected = torch.func.jvp(torch.func.grad(objective, argnums=(1, 0)), ({'x': tx}, [tx]), ({'x': tv}, [tu]))
        tree_same(actual, expected)
        gradients_match(self, actual[1][0][0], expected[1][0][0], [(x, tx), (v, tv)])
        captured, torch_captured = fr.tensor(.5, requires_grad=True), torch.tensor(.5, requires_grad=True)
        for enabled in (False, True):
            with fr.enable_grad() if enabled else fr.no_grad():
                actual = func.jvp(func.grad(lambda a: ((a * a).sum(), {'capture': captured, 'arg': a}), has_aux=True),
                                  (x,), (v,), has_aux=True)
            with torch.enable_grad() if enabled else torch.no_grad():
                expected = torch.func.jvp(torch.func.grad(lambda a: ((a * a).sum(), {'capture': torch_captured, 'arg': a}), has_aux=True),
                                         (tx,), (tv,), has_aux=True)
            tree_same(actual, expected, leaves=True)
            self.assertIs(actual[2]['capture'], captured)
        for enabled, xg, kind in itertools.product((False, True), (False, True), ('plain', 'enable', 'disable_enable')):
            with self.subTest(temporary_base=True, enabled=enabled, xg=xg, kind=kind):
                tx = torch.tensor([.2, -.4], requires_grad=xg)
                tv = torch.tensor([.3, .7], requires_grad=True)
                x, v = tensor(tx), tensor(tv)
                held = []
                def objective(library, a):
                    if kind == 'plain': base = a * a
                    elif kind == 'enable':
                        with library.enable_grad(): base = a * a
                    else:
                        with library.no_grad():
                            with library.enable_grad(): base = a * a
                    with library.no_grad(): view = base.reshape(2)
                    held.append((base, view))
                    return base.sum(), {'base': base, 'view': view, 'input': a}
                with fr.enable_grad() if enabled else fr.no_grad():
                    actual = func.jvp(func.grad_and_value(lambda a: objective(fr, a), has_aux=True), (x,), (v,))
                with torch.enable_grad() if enabled else torch.no_grad():
                    expected = torch.func.jvp(torch.func.grad_and_value(lambda a: objective(torch, a), has_aux=True), (tx,), (tv,))
                tree_same(actual, expected, leaves=True)
                for a, b in zip(held[0], held[1]):
                    self.assertEqual(a.requires_grad, b.requires_grad)
                    self.assertEqual(a.is_leaf, b.is_leaf)
                gradients_match(self, actual[0][1][1]['view'], expected[0][1][1]['view'], [(x, tx)])
                gradients_match(self, actual[1][1][1]['view'], expected[1][1][1]['view'], [(x, tx), (v, tv)])

    def test_unused_strict_factory_detach_and_remaining_public_query_boundaries(self):
        for enabled, live, strict, kind in itertools.product((False, True), (False, True), (False, True), ('detached', 'island', 'constant', 'linear', 'independent')):
            with self.subTest(enabled=enabled, live=live, strict=strict, kind=kind):
                tx = torch.tensor([.2, -.4], requires_grad=live)
                tv = torch.tensor([.3, .7], requires_grad=live)
                x, v = tensor(tx), tensor(tv)
                def objective(library, a):
                    if kind == 'detached': return (a * a).detach().sum()
                    if kind == 'island':
                        with library.no_grad(): return (a * a).sum()
                    if kind == 'constant': return (a * 0).sum()
                    if kind == 'linear': return a.sum()
                    return library.tensor(3.)
                with torch.enable_grad() if enabled else torch.no_grad():
                    if strict:
                        with self.assertRaises(RuntimeError):
                            torch.func.jvp(torch.func.grad(lambda a: objective(torch, a)), (tx,), (tv,), strict=True)
                    else:
                        expected = torch.func.jvp(torch.func.grad(lambda a: objective(torch, a)), (tx,), (tv,))
                with fr.enable_grad() if enabled else fr.no_grad():
                    if strict:
                        with self.assertRaises(RuntimeError): func.jvp(func.grad(lambda a: objective(fr, a)), (x,), (v,), strict=True)
                    else:
                        tree_same(func.jvp(func.grad(lambda a: objective(fr, a)), (x,), (v,)), expected)
        x = fr.tensor([.2, -.4], requires_grad=True)
        v = fr.tensor([.3, .7], requires_grad=True)
        held = []
        def factory_objective(a):
            weight = fr.tensor(.5, requires_grad=True)
            held.append(weight)
            return (a * weight).sum()
        primal, tangent = func.jvp(func.grad(factory_objective), (x,), (v,))
        self.assertEqual(primal.tolist(), [.5, .5])
        self.assertEqual(tangent.tolist(), [0., 0.])
        self.assertFalse(primal.requires_grad or tangent.requires_grad)
        self.assertTrue(held[0].requires_grad)
        calls = []
        def inner(a):
            calls.append('inner')
            return (a * a).sum()
        with self.assertRaisesRegex(ValueError, 'nested reverse'):
            func.jvp(func.grad(lambda a: func.grad(inner)(a).sum()), (x,), (v,))
        self.assertEqual(calls, [])
        for vector_grad in (False, True):
            tx = torch.tensor([.2, -.4], requires_grad=True)
            tv = torch.tensor([.3, .7], requires_grad=vector_grad)
            base, direction = tensor(tx), tensor(tv)
            with fr.no_grad(): blocked = base.reshape(2)
            with torch.no_grad(): torch_blocked = tx.reshape(2)
            actual = func.jvp(func.grad(lambda a: (a * a).sum()), (blocked,), (direction,))
            expected = torch.func.jvp(torch.func.grad(lambda a: (a * a).sum()), (torch_blocked,), (tv,))
            tree_same(actual, expected, leaves=True)
            gradients_match(self, actual[1], expected[1], [(base, tx), (blocked, torch_blocked), (direction, tv)])
        with self.assertRaisesRegex(ValueError, 'active forward AD'):
            func.jvp(lambda a: fr.autograd.grad((a * a).sum(), [a])[0], (x,), (v,))
        with self.assertRaises(ValueError): func.jvp(func.grad(lambda a: a.gelu().sum()), (x,), (v,))
        with self.assertRaisesRegex(RuntimeError, 'composition failure'):
            func.jvp(func.grad(lambda a: (_ for _ in ()).throw(RuntimeError('composition failure'))), (x,), (v,))
        same(func.jvp(func.grad(lambda a: (a * a).sum()), (x,), (v,))[1],
             torch.func.jvp(torch.func.grad(lambda a: (a * a).sum()),
                            (torch.tensor([.2, -.4], requires_grad=True),),
                            (torch.tensor([.3, .7], requires_grad=True),))[1])
        self.assertIsNone(x.grad)
        self.assertIsNone(v.grad)


if __name__ == '__main__':
    unittest.main()
