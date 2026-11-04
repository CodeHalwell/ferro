"""Actual Torch forward-mode values, reverse provenance, and scope boundaries."""
import unittest

import ferro as fr
from ferro import func
import torch


def tensor(value):
    data = value.detach().clone().reshape(-1).tolist()
    result = fr.Tensor.from_f64(data, list(value.shape)) if value.dtype == torch.float64 else fr.Tensor(data, list(value.shape))
    return result.requires_grad_(value.requires_grad)


def same(actual, expected, *, symbolic=False):
    if actual.shape != list(expected.shape) or actual.dtype != str(expected.dtype).split('.')[-1]:
        raise AssertionError((actual.shape, actual.dtype, expected.shape, expected.dtype))
    torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape),
                               expected.detach().clone(), atol=1e-6, rtol=1e-5, equal_nan=True)
    if actual.requires_grad != expected.requires_grad:
        raise AssertionError(('requires_grad', actual.requires_grad, expected.requires_grad))
    if symbolic and actual._is_structural_zero() != torch._is_zerotensor(expected):
        raise AssertionError(('structural zero', actual._is_structural_zero(), torch._is_zerotensor(expected)))


class ForwardADTests(unittest.TestCase):
    def test_auxiliary_no_grad_views_keep_leaf_edges_base_flags_and_storage(self):
        for enabled in (False, True):
            for input_grad in (False, True):
                for captured_grad in (False, True):
                    for transform in ('direct', 'grad'):
                        with self.subTest(enabled=enabled, input_grad=input_grad,
                                          captured_grad=captured_grad, transform=transform):
                            tx = torch.tensor([.2, -.4], requires_grad=input_grad)
                            tv = torch.tensor([.3, .7], requires_grad=True)
                            tw = torch.tensor(.5, requires_grad=captured_grad)
                            x, v, w = map(tensor, (tx, tv, tw))
                            held, torch_held = [], []
                            def objective(library, a, weight, saved):
                                base = (a * a) * weight
                                with library.no_grad(): view = base.reshape(base.shape)
                                saved.append(view)
                                return (base if transform == 'direct' else base.sum()), view
                            actual_fn = lambda a: objective(fr, a, w, held)
                            expected_fn = lambda a: objective(torch, a, tw, torch_held)
                            if transform == 'grad':
                                actual_fn = func.grad(actual_fn, has_aux=True)
                                expected_fn = torch.func.grad(expected_fn, has_aux=True)
                            with fr.enable_grad() if enabled else fr.no_grad():
                                actual = func.jvp(actual_fn, (x,), (v,), has_aux=True)
                            with torch.enable_grad() if enabled else torch.no_grad():
                                expected = torch.func.jvp(expected_fn, (tx,), (tv,), has_aux=True)
                            for value, reference in zip(actual, expected): same(value, reference)
                            self.assertEqual(actual[2].is_leaf, expected[2].is_leaf)
                            self.assertTrue(actual[2]._shares_storage(held[0]))
                            self.assertTrue(held[0].is_leaf)
                            self.assertEqual(len(held), 1)
                            if expected[2].requires_grad:
                                with self.assertRaises(ValueError):
                                    fr.autograd.vjp(actual[2], [actual[2]], fr.ones(2))
                                with self.assertRaises(RuntimeError):
                                    torch.autograd.grad(expected[2], expected[2], grad_outputs=torch.ones(2))
                            for value in (x, v, w): self.assertIsNone(value.grad)
        tx = torch.tensor([.2, -.4], requires_grad=True)
        tv = torch.tensor([.3, .7], requires_grad=True)
        x, v = tensor(tx), tensor(tv)
        with fr.no_grad(): captured = x.reshape(2)
        with torch.no_grad(): torch_captured = tx.reshape(2)
        self.assertIs(func.jvp(lambda a: (a * a, captured), (x,), (v,), has_aux=True)[2], captured)
        def objective(library, a):
            with library.no_grad(): view = a.reshape(2)
            return a * a, view
        auxiliary = func.jvp(lambda a: objective(fr, a), (x,), (v,), has_aux=True)[2]
        reference = torch.func.jvp(lambda a: objective(torch, a), (tx,), (tv,), has_aux=True)[2]
        for flag in (True, False, True):
            x.requires_grad_(flag)
            tx.requires_grad_(flag)
            same(auxiliary, reference)
            self.assertEqual(auxiliary.is_leaf, reference.is_leaf)

    def test_factory_leaves_and_primal_aliases_match_functional_dual_unwrapping(self):
        for input_grad in (False, True):
            for vector_grad in (False, True):
                for enabled in (False, True):
                    with self.subTest(input_grad=input_grad, vector_grad=vector_grad, enabled=enabled):
                        tx = torch.tensor([.2, -.4], requires_grad=input_grad)
                        tv = torch.tensor([.3, .7], requires_grad=vector_grad)
                        tw = torch.tensor([.5, .5], requires_grad=True)
                        x, v, w = tensor(tx), tensor(tv), tensor(tw)
                        held, reference_held = [], []
                        def run(library, a, captured, saved):
                            factory = library.tensor([.5, .5], requires_grad=True)
                            saved.append(factory)
                            output = {'identity': a, 'captured': captured, 'factory': factory,
                                      'factory_view': factory.reshape(2), 'product': a * factory.reshape(2)}
                            auxiliary = {'identity': a, 'captured': captured, 'factory': factory,
                                         'factory_view': factory.reshape(2), 'math': a * a}
                            return output, auxiliary
                        with fr.enable_grad() if enabled else fr.no_grad():
                            actual, direction, auxiliary = func.jvp(lambda a: run(fr, a, w, held), (x,), (v,), has_aux=True)
                        with torch.enable_grad() if enabled else torch.no_grad():
                            expected, expected_direction, expected_aux = torch.func.jvp(
                                lambda a: run(torch, a, tw, reference_held), (tx,), (tv,), has_aux=True)
                        for result, reference in ((actual, expected), (direction, expected_direction), (auxiliary, expected_aux)):
                            for key in reference:
                                same(result[key], reference[key])
                                self.assertEqual(result[key].is_leaf, reference[key].is_leaf)
                        self.assertIs(direction['identity'], v)
                        self.assertIs(auxiliary['captured'], w)
                        self.assertTrue(held[0].requires_grad)
                        self.assertTrue(reference_held[0].requires_grad)
                        if direction['product'].requires_grad:
                            pairs = [(a, b) for a, b in ((x, tx), (v, tv), (held[0], reference_held[0])) if b.requires_grad]
                            gradients = fr.autograd.grad(direction['product'].sum(), [a for a, _ in pairs], create_graph=True, allow_unused=True)
                            references = torch.autograd.grad(expected_direction['product'].sum(), [b for _, b in pairs],
                                                             create_graph=True, allow_unused=True)
                            for gradient, reference in zip(gradients, references):
                                if reference is None:
                                    self.assertIsNone(gradient)
                                else:
                                    same(gradient, reference, symbolic=True)
                        if input_grad and not enabled:
                            with self.assertRaises(ValueError):
                                fr.autograd.vjp(actual['identity'], [x], fr.ones([2]))
                            with self.assertRaises(RuntimeError):
                                torch.autograd.grad(expected['identity'], [tx], grad_outputs=torch.ones(2))
                        for value in (x, v, w, held[0]): self.assertIsNone(value.grad)

    def test_product_quotient_and_linear_provenance_174_torch_controls(self):
        count = 0
        configurations = []
        for operation in ('mul', 'div', 'add', 'sub', 'neg'):
            for form in ('number', 'scalar', 'vector'):
                if operation == 'neg' and form != 'number':
                    continue
                for coefficient_grad in (False, True):
                    if form == 'number' and coefficient_grad:
                        continue
                    for input_grad in (False, True):
                        for vector_grad in (False, True):
                            for nonfinite in (False, True):
                                configurations.append((operation, form, coefficient_grad, input_grad, vector_grad, nonfinite, 2.))
        configurations.extend((operation, 'scalar', True, True, True, False, value)
                              for operation in ('mul', 'div') for value in (0., float('inf'), float('nan')))
        for operation, form, coefficient_grad, input_grad, vector_grad, nonfinite, coefficient in configurations:
            with self.subTest(operation=operation, form=form, coefficient_grad=coefficient_grad,
                              input_grad=input_grad, vector_grad=vector_grad, nonfinite=nonfinite, coefficient=coefficient):
                tx = torch.tensor([float('inf'), float('nan')] if nonfinite else [.2, -.4], requires_grad=input_grad)
                tv = torch.tensor([.3, -.7], requires_grad=vector_grad)
                tc = coefficient if form == 'number' else torch.tensor(coefficient if form == 'scalar' else [coefficient, coefficient], requires_grad=coefficient_grad)
                x, v = tensor(tx), tensor(tv)
                c = tc if form == 'number' else tensor(tc)
                def operation_fn(a, b):
                    if operation == 'mul': return a * b
                    if operation == 'div': return a / b
                    if operation == 'add': return a + b
                    if operation == 'sub': return a - b
                    return -a
                primal, tangent = func.jvp(lambda a: operation_fn(a, c), (x,), (v,))
                expected_primal, expected = torch.func.jvp(lambda a: operation_fn(a, tc), (tx,), (tv,))
                same(primal, expected_primal)
                same(tangent, expected, symbolic=True)
                pairs = [(a, b) for a, b in ((x, tx), (v, tv), (c, tc)) if isinstance(b, torch.Tensor) and b.requires_grad]
                if expected.requires_grad and pairs:
                    actual_grad = fr.autograd.grad(tangent.sum(), [a for a, _ in pairs], create_graph=True, allow_unused=True)
                    reference_grad = torch.autograd.grad(expected.sum(), [b for _, b in pairs], allow_unused=True, create_graph=True)
                    for actual, reference in zip(actual_grad, reference_grad):
                        if reference is None:
                            self.assertIsNone(actual)
                        else:
                            same(actual, reference, symbolic=True)
                self.assertIsNone(x.grad)
                self.assertIsNone(v.grad)
                if isinstance(c, fr.Tensor): self.assertIsNone(c.grad)
                count += 1
        self.assertEqual(count, 174)

    def test_no_grad_islands_detach_and_explicit_enable_grad_preserve_forward_flow(self):
        for input_grad in (False, True):
            for vector_grad in (False, True):
                for outer_grad in (False, True):
                    for kind in ('no_grad', 'detach', 'enable_grad'):
                        with self.subTest(input_grad=input_grad, vector_grad=vector_grad, outer_grad=outer_grad, kind=kind):
                            tx = torch.tensor([.2, -.4], requires_grad=input_grad)
                            tv = torch.tensor([.3, .7], requires_grad=vector_grad)
                            x, v = tensor(tx), tensor(tv)
                            def run(library, a):
                                if kind == 'detach': return (a * a).detach().tanh().exp()
                                with library.no_grad() if kind == 'no_grad' else library.enable_grad():
                                    return (a * a).tanh().exp()
                            with fr.enable_grad() if outer_grad else fr.no_grad():
                                actual = func.jvp(lambda a: run(fr, a), (x,), (v,))
                                self.assertEqual(fr.is_grad_enabled(), outer_grad)
                            with torch.enable_grad() if outer_grad else torch.no_grad():
                                expected = torch.func.jvp(lambda a: run(torch, a), (tx,), (tv,))
                            for a, b in zip(actual, expected): same(a, b)
        self.assertTrue(fr.is_grad_enabled())

    def test_views_shared_arguments_mixed_cat_and_one_evaluation(self):
        tx = torch.tensor([[.2, .4], [-.3, .5]], requires_grad=True)
        tv = torch.tensor([[.3, .7], [.1, -.2]], dtype=torch.float64, requires_grad=True)
        x, v = tensor(tx), tensor(tv)
        calls = []
        def actual(a, b):
            calls.append(1)
            return {'cat': fr.cat([a, b.detach()], 0), 'view': a.transpose(0, 1).unsqueeze(0).expand(3, 2, 2), 'empty': a[:0]}, {'identity': a}
        def reference(a, b):
            return {'cat': torch.cat([a, b.detach()], 0), 'view': a.transpose(0, 1).unsqueeze(0).expand(3, 2, 2), 'empty': a[:0]}, {'identity': a}
        output, tangent, aux = func.jvp(actual, (x, x), (v, fr.Tensor.ones([2, 2])), has_aux=True)
        expected, expected_tangent, expected_aux = torch.func.jvp(reference, (tx, tx), (tv, torch.ones(2, 2)), has_aux=True)
        for key in output:
            same(output[key], expected[key])
            same(tangent[key], expected_tangent[key])
        same(aux['identity'], expected_aux['identity'])
        self.assertEqual(aux['identity'].is_leaf, expected_aux['identity'].is_leaf)
        self.assertEqual(calls, [1])
        same(fr.autograd.grad(tangent['cat'].sum(), [v])[0], torch.autograd.grad(expected_tangent['cat'].sum(), [tv])[0])
        _, duplicate = func.jvp(lambda a, b: a * b, (x, x), (fr.Tensor.ones([2, 2]), fr.full([2, 2], 2.)))
        _, duplicate_reference = torch.func.jvp(lambda a, b: a * b, (tx, tx), (torch.ones(2, 2), torch.full((2, 2), 2.)))
        same(duplicate, duplicate_reference)

    def test_unary_matmul_reverse_derivatives_and_saved_output_mutations(self):
        operations = ('pow0', 'pow1', 'pow2', 'tanh', 'exp', 'sigmoid', 'sqrt', 'log', 'abs', 'sign', 'relu', 'softmax', 'matmul', 'sin', 'cos')
        for name in operations:
            with self.subTest(operation=name):
                tx = torch.tensor([[.3, .5], [.7, .2]], requires_grad=True)
                tv = torch.tensor([[.2, -.4], [.1, .3]], requires_grad=True)
                x, v = tensor(tx), tensor(tv)
                def run(library, a):
                    if name.startswith('pow'): return a ** int(name[-1])
                    if name == 'matmul': return a @ a.transpose(0, 1)
                    if name == 'softmax': return a.softmax(1)
                    return getattr(a, name)()
                primal, tangent = func.jvp(lambda a: run(fr, a), (x,), (v,))
                expected_primal, expected = torch.func.jvp(lambda a: run(torch, a), (tx,), (tv,))
                same(primal, expected_primal)
                same(tangent, expected)
                if expected.requires_grad:
                    actual_grads = fr.autograd.grad(tangent.sum(), [x, v], create_graph=True, allow_unused=True)
                    reference_grads = torch.autograd.grad(expected.sum(), [tx, tv], create_graph=True, allow_unused=True)
                    for actual, reference in zip(actual_grads, reference_grads):
                        if reference is not None: same(actual, reference)
                        else: self.assertIsNone(actual)
        x = fr.Tensor([.3], [1]).requires_grad_(True)
        v = fr.Tensor([.2], [1]).requires_grad_(True)
        primal, tangent = func.jvp(lambda a: a.tanh(), (x,), (v,))
        primal.detach().copy_(fr.Tensor.zeros([1]))
        with self.assertRaises(ValueError): fr.autograd.grad(tangent.sum(), [x, v])
        with self.assertRaises(ValueError): fr.autograd.grad(primal.sum(), [x])

    def test_opaque_local_fallback_and_explicit_graph_boundary(self):
        tx = torch.tensor([.2, -.4])
        tv = torch.tensor([.3, .7], requires_grad=True)
        x, v = tensor(tx), tensor(tv)
        actual = func.jvp(lambda a: fr.nn.functional.gelu(a), (x,), (v,))
        expected = torch.func.jvp(lambda a: torch.nn.functional.gelu(a), (tx,), (tv,))
        for a, b in zip(actual, expected): same(a, b)

        same(fr.autograd.grad(actual[1].sum(), [v])[0], torch.autograd.grad(expected[1].sum(), [tv])[0])
        x.requires_grad_(True)
        with self.assertRaises(ValueError): func.jvp(lambda a: fr.nn.functional.gelu(a), (x,), (v,))
        tx.requires_grad_(True)
        with fr.no_grad(): actual = func.jvp(lambda a: fr.nn.functional.gelu(a), (x,), (v,))
        with torch.no_grad(): expected = torch.func.jvp(lambda a: torch.nn.functional.gelu(a), (tx,), (tv,))
        for a, b in zip(actual, expected): same(a, b)
        tx, tv = torch.empty(0, 2), torch.empty(0, 2, requires_grad=True)
        x, v = tensor(tx), tensor(tv)
        actual = func.jvp(lambda a: fr.nn.functional.gelu(a), (x,), (v,))
        expected = torch.func.jvp(lambda a: torch.nn.functional.gelu(a), (tx,), (tv,))
        for a, b in zip(actual, expected): same(a, b)
        same(fr.autograd.grad(actual[1].sum(), [v])[0], torch.autograd.grad(expected[1].sum(), [tv])[0])

    def test_derivative_graphs_save_only_the_values_used_by_each_local_rule(self):
        for name in ('tanh', 'sqrt', 'sin', 'cos', 'silu', 'sigmoid', 'softmax', 'log', 'gelu', 'layer_norm'):
            for target in ('input', 'output'):
                with self.subTest(operation=name, mutation=target):
                    tx = torch.tensor([[.3, .5, .7], [.2, .4, .6]])
                    tv = torch.tensor([[.2, -.4, .1], [.3, .5, -.2]], requires_grad=True)
                    x, v = tensor(tx), tensor(tv)
                    def run(library, a):
                        if name == 'softmax': return a.softmax(1)
                        if name == 'gelu': return library.nn.functional.gelu(a)
                        if name == 'silu': return library.nn.functional.silu(a)
                        if name == 'layer_norm': return library.nn.functional.layer_norm(a, [3])
                        return getattr(a, name)()
                    primal, tangent = func.jvp(lambda a: run(fr, a), (x,), (v,))
                    expected_primal, expected = torch.func.jvp(lambda a: run(torch, a), (tx,), (tv,))
                    (primal if target == 'output' else x).copy_(fr.Tensor.zeros([2, 3]))
                    with torch.no_grad(): (expected_primal if target == 'output' else tx).zero_()
                    try:
                        reference = torch.autograd.grad(expected.sum(), [tv])[0]
                    except RuntimeError:
                        with self.assertRaises(ValueError): fr.autograd.grad(tangent.sum(), [v])
                    else:
                        same(fr.autograd.grad(tangent.sum(), [v])[0], reference)

    def test_strict_scope_errors_mutation_and_symbolic_zero_aliases(self):
        x = fr.Tensor([.2, -.4], [2]).requires_grad_(True)
        v = fr.Tensor([.3, .7], [2])
        before = x.tolist()
        with self.assertRaises(RuntimeError): func.jvp(lambda a: a.detach(), (x,), (v,), strict=True)
        _, zero = func.jvp(lambda a: a.sign(), (x,), (v,), strict=True)
        self.assertEqual(zero.tolist(), [0., 0.])
        with self.assertRaises(ValueError): func.jvp(lambda a: func.jvp(lambda b: b * b, (a,), (v,))[0], (x,), (v,))
        with self.assertRaises(ValueError): fr.capture(lambda: func.jvp(lambda a: a * a, (x,), (v,)))
        with self.assertRaises(ValueError): func.jvp(lambda a: fr.capture(lambda: a * a), (x,), (v,))
        for enabled in (False, True):
            primal = x.detach().requires_grad_(enabled)
            composition = func.jvp(func.grad(lambda a: (a * a).sum()), (primal,), (v,))
            reference = torch.func.jvp(torch.func.grad(lambda a: (a * a).sum()),
                                       (torch.tensor([.2, -.4], requires_grad=enabled),), (torch.tensor([.3, .7]),))
            for actual_part, expected_part in zip(composition, reference): same(actual_part, expected_part)
            actual = func.grad(lambda a: func.jvp(lambda z: (z * z).sum(), (a,), (v,))[1])(primal)
            tx = torch.tensor([.2, -.4], requires_grad=enabled)
            tv = torch.tensor([.3, .7])
            expected = torch.func.grad(lambda a: torch.func.jvp(lambda z: (z * z).sum(), (a,), (tv,))[1])(tx)
            same(actual, expected)
        with self.assertRaises(ValueError): func.jvp(lambda a: a.sum().backward() or a, (x,), (v,))
        with self.assertRaisesRegex(RuntimeError, 'user failure'):
            def broken(a): raise RuntimeError('user failure')
            func.jvp(broken, (x,), (v,))
        with self.assertRaises(ValueError):
            with fr.no_grad(): func.jvp(lambda a: a.fill_(2.), (x,), (v,))
        self.assertEqual(x.tolist(), before)
        primal, tangent = func.jvp(lambda a: a * 2., (x,), (v,))
        gradient = fr.autograd.grad(tangent.sum(), [x], create_graph=True)[0]
        self.assertTrue(gradient._is_structural_zero())
        for alias in (gradient, gradient.detach(), gradient.reshape([1, 2])):
            with self.assertRaises(ValueError): alias.fill_(1.)
        copied = gradient.clone()
        self.assertFalse(copied._is_structural_zero())
        copied.fill_(1.)
        self.assertEqual(gradient.tolist(), [0., 0.])
        same(tangent, torch.tensor([.6, 1.4], requires_grad=True))
        self.assertTrue(fr.is_grad_enabled())
        self.assertIsNone(x.grad)
        self.assertIsNone(v.grad)


if __name__ == '__main__':
    unittest.main()
