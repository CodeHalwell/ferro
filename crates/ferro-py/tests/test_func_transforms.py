"""Public function transforms: Tensor trees, graph flags and actual Torch derivatives."""
import unittest

import ferro as fr
import ferro.func as ff
import torch


def tensor(value):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(value.requires_grad)


def same(actual, expected, flags=True):
    if expected is None:
        if actual is not None: raise AssertionError("unused derivative must be None")
        return
    if isinstance(expected, (tuple, list, dict)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError('transform tree structure differs')
        if isinstance(expected, dict):
            if list(actual) != list(expected):
                raise AssertionError('transform dictionary keys differ')
            for key in expected:
                same(actual[key], expected[key], flags)
        else:
            for a, b in zip(actual, expected):
                same(a, b, flags)
        return
    if actual.shape != list(expected.shape) or actual.dtype != 'float32':
        raise AssertionError('transform shape/dtype differs')
    torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(expected.shape), expected.detach(), atol=1e-5, rtol=1e-4)
    if flags and actual.requires_grad != expected.requires_grad:
        raise AssertionError(f'requires_grad differs: {actual.requires_grad} vs {expected.requires_grad}')


def total(value):
    if isinstance(value, dict):
        value = tuple(value.values())
    if isinstance(value, (tuple, list)):
        result = None
        for child in value:
            part = total(child)
            result = part if result is None else result + part
        return result
    return value.sum()


class FuncTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_grad_value_flags_captured_parameters_and_nested_transforms(self):
        for input_grad in (False, True):
            for captured_grad in (False, True):
                with self.subTest(input_grad=input_grad, captured_grad=captured_grad):
                    tx = torch.tensor([0.2, -0.4], requires_grad=input_grad)
                    tw = torch.tensor([0.7, 0.5], requires_grad=captured_grad)
                    x, w = tensor(tx), tensor(tw)
                    def fn(a):
                        z = (a * w).tanh()
                        return (z * z).sum()
                    def reference(a):
                        z = (a * tw).tanh()
                        return (z * z).sum()
                    actual = ff.grad_and_value(fn)(x)
                    expected = torch.func.grad_and_value(reference)(tx)
                    same(actual, expected)
                    same(ff.grad(fn)(x), torch.func.grad(reference)(tx))
                    if captured_grad:
                        same(fr.autograd.grad(actual[0].sum(), [w]), torch.autograd.grad(expected[0].sum(), [tw]))
        for requires_grad in (False, True):
            tx = torch.tensor(0.3, requires_grad=requires_grad)
            x = tensor(tx)
            same(ff.grad(ff.grad(lambda a: a.exp()))(x), torch.func.grad(torch.func.grad(lambda a: a.exp()))(tx))

    def test_argnums_order_negative_indices_pytree_arguments_and_aux(self):
        tx = torch.tensor([0.2, 0.4], requires_grad=True)
        tb = torch.tensor(0.6, requires_grad=False)
        x, b = tensor(tx), tensor(tb)
        def fn(a, bias, scale=2.):
            return ((a * bias).tanh() * scale).sum(), {'argument': a, 'computed': [a * bias], 'empty': ()}
        for argnums in (0, -1, (1, 0), (0,)):
            for name in ('grad', 'grad_and_value', 'jacrev'):
                with self.subTest(argnums=argnums, operation=name):
                    same(getattr(ff, name)(fn, argnums=argnums, has_aux=True)(x, b),
                         getattr(torch.func, name)(fn, argnums=argnums, has_aux=True)(tx, tb))
        def tree_fn(values):
            return (values['x'] * values['b']).tanh().sum()
        same(ff.grad(tree_fn)({'x': x, 'b': b}), torch.func.grad(tree_fn)({'x': tx, 'b': tb}))

    def test_jacobian_and_vector_output_hessian_trees_call_function_once(self):
        tx = torch.tensor([0.2, -0.3], requires_grad=True)
        tb = torch.tensor(0.4, requires_grad=True)
        x, b = tensor(tx), tensor(tb)
        for name in ('jacrev', 'hessian'):
            for argnums in (0, (1, 0)):
                calls, torch_calls = [], []
                def actual(a, bias):
                    calls.append(1)
                    z = (a * bias).tanh()
                    return {'vector': z, 'nested': (z.sum(), [a * a])}
                def reference(a, bias):
                    torch_calls.append(1)
                    z = (a * bias).tanh()
                    return {'vector': z, 'nested': (z.sum(), [a * a])}
                with self.subTest(operation=name, argnums=argnums):
                    result = getattr(ff, name)(actual, argnums=argnums)(x, b)
                    expected = getattr(torch.func, name)(reference, argnums=argnums)(tx, tb)
                    same(result, expected)
                    self.assertEqual((len(calls), len(torch_calls)), (1, 1))
                    same(fr.autograd.grad(total(result), [x, b], allow_unused=True), torch.autograd.grad(total(expected), [tx, tb], allow_unused=True)
                         if name == 'hessian' or argnums != 0 else torch.autograd.grad(total(expected), [tx, tb]))

    def test_jvp_aux_tree_live_vectors_and_strict_unused_inputs(self):
        for requires_grad in (False, True):
            tx = torch.tensor([0.3, 0.5], requires_grad=requires_grad)
            tb = torch.tensor(0.7, requires_grad=requires_grad)
            tv = torch.tensor([0.2, -0.4], requires_grad=True)
            x, b, v = tensor(tx), tensor(tb), tensor(tv)
            def fn(a, bias):
                result = (a * bias).tanh()
                return {'vector': result, 'scalar': result.sum()}, {'aux': a}
            same(ff.jvp(fn, (x, b), (v, fr.Tensor([0.1], [])), has_aux=True),
                 torch.func.jvp(fn, (tx, tb), (tv, torch.tensor(0.1)), has_aux=True))
            same(ff.jvp(lambda a, bias: a * a, (x, b), (v, fr.Tensor([0.1], [])), strict=True),
                 torch.func.jvp(lambda a, bias: a * a, (tx, tb), (tv, torch.tensor(0.1)), strict=True))

    def test_constant_and_unused_blocks_retain_shared_derivative_graphs(self):
        tx = torch.tensor([0.2, -0.4], requires_grad=True)
        tb = torch.tensor([0.7, 0.3], requires_grad=True)
        x, b = tensor(tx), tensor(tb)
        for mode in ('unused', 'quadratic', 'linear'):
            def fn(a, bias):
                nonlinear = a.exp() if mode == 'unused' else a * a if mode == 'quadratic' else a * 2.
                return {'first': nonlinear, 'second': bias * bias if mode == 'quadratic' else a * 3.}
            for name in ('jacrev', 'hessian'):
                with self.subTest(mode=mode, operation=name):
                    actual = getattr(ff, name)(fn, argnums=(0, 1))(x, b)
                    expected = getattr(torch.func, name)(fn, argnums=(0, 1))(tx, tb)
                    same(actual, expected)
                    if total(expected).requires_grad:
                        references = torch.autograd.grad(total(expected), (tx, tb), allow_unused=True)
                        same(fr.autograd.grad(total(actual), (x, b), allow_unused=True), references)

    def test_vjp_closure_modes_live_cotangents_tree_and_aux(self):
        tx = torch.tensor([0.2, -0.5], requires_grad=True)
        x = tensor(tx)
        def fn(a):
            return {'value': a.tanh(), 'sum': (a * a).sum()}, {'input': a}
        value, pullback, aux = ff.vjp(fn, x, has_aux=True)
        expected, reference, torch_aux = torch.func.vjp(fn, tx, has_aux=True)
        same((value, aux), (expected, torch_aux))
        for enabled in (False, True):
            for create_graph in (None, False, True):
                tv = torch.tensor([0.4, -0.7], requires_grad=True)
                v = tensor(tv)
                with fr.enable_grad() if enabled else fr.no_grad():
                    result = pullback({'value': v, 'sum': fr.Tensor([0.3], [])}, create_graph=create_graph)
                    self.assertEqual(fr.is_grad_enabled(), enabled)
                with torch.enable_grad() if enabled else torch.no_grad():
                    wanted = reference({'value': tv, 'sum': torch.tensor(0.3)}, create_graph=create_graph)
                same(result, wanted)

    def test_empty_tensors_preserve_real_derivative_paths(self):
        for requires_grad in (False, True):
            tx = torch.zeros([0, 2], requires_grad=requires_grad)
            x = tensor(tx)
            for nonlinear in (False, True):
                def fn(a):
                    return a * a if nonlinear else a * 2.
                for name in ('jacrev', 'hessian'):
                    with self.subTest(requires_grad=requires_grad, nonlinear=nonlinear, operation=name):
                        actual, expected = getattr(ff, name)(fn)(x), getattr(torch.func, name)(fn)(tx)
                        same(actual, expected)
                        if expected.requires_grad:
                            same(fr.autograd.grad(actual.sum(), (x,)), torch.autograd.grad(expected.sum(), (tx,)))
                actual_jvp = ff.jvp(fn, (x,), (fr.Tensor.zeros([0, 2]),))
                expected_jvp = torch.func.jvp(fn, (tx,), (torch.zeros(0, 2),))
                same(actual_jvp, expected_jvp)
                if expected_jvp[1].requires_grad:
                    same(fr.autograd.grad(actual_jvp[1].sum(), (x,)), torch.autograd.grad(expected_jvp[1].sum(), (tx,)))
                actual, pullback = ff.vjp(fn, x)
                expected, reference = torch.func.vjp(fn, tx)
                same(actual, expected)
                same(pullback(fr.Tensor.zeros([0, 2])), reference(torch.zeros(0, 2)))

    def test_explicit_limits_errors_and_no_input_gradient_mutation(self):
        x = fr.Tensor([0.2, 0.4], [2]).requires_grad_(True)
        (x * x).sum().backward()
        held = x.grad.tolist()
        ff.grad(lambda a: (a * a).sum())(x)
        self.assertEqual(x.grad.tolist(), held)
        for argnums in ((), (0, 0), [0], 2):
            with self.assertRaises((TypeError, RuntimeError)):
                ff.grad(lambda a: a.sum(), argnums)(x)
        with self.assertRaises(RuntimeError):
            ff.grad(lambda a: a.sum().reshape([1]))(x)
        with self.assertRaises(TypeError):
            ff.grad(lambda a: (a.sum(), {'bad': 1}), has_aux=True)(x)
        for empty in ((), [], {}):
            for operation in (lambda: ff.jacrev(lambda a: empty)(x), lambda: ff.hessian(lambda a: empty)(x),
                              lambda: ff.jvp(lambda a: empty, (x,), (fr.Tensor.ones([2]),)),
                              lambda: ff.vjp(lambda a: empty, x)):
                with self.assertRaises(RuntimeError):
                    operation()
        with fr.no_grad():
            self.assertFalse(ff.grad(lambda a: a.sum())(x).requires_grad)
            self.assertFalse(fr.is_grad_enabled())
        with self.assertRaises(NotImplementedError):
            ff.grad(lambda a: a.sum())(fr.Tensor.from_f64([0.3], [1]))
        _, closure = ff.vjp(lambda a: a * a, x)
        with self.assertRaises(NotImplementedError):
            closure(fr.Tensor.ones([2]), retain_graph=False)
        with self.assertRaises(RuntimeError):
            ff.jvp(lambda a: fr.Tensor.ones([2]), (x,), (fr.Tensor.ones([2]),), strict=True)
        with self.assertRaises(ValueError):
            fr.capture(lambda: ff.grad(lambda a: a.sum())(x))

    def test_no_grad_entry_values_auxiliary_identities_and_mode_restoration(self):
        for input_grad in (False, True):
            for capture_grad in (False, True):
                tx = torch.tensor(0.3, requires_grad=input_grad)
                tw = torch.tensor(0.7, requires_grad=capture_grad)
                x, w = tensor(tx), tensor(tw)
                old, torch_old = w * 2., tw * 2.
                def actual(a):
                    return a, {'arg': a, 'capture': w, 'old_graph': old, 'computed': (a * w).tanh(),
                               'detached': a.detach(), 'factory': fr.Tensor([0.8], []).requires_grad_(True)}
                def reference(a):
                    return a, {'arg': a, 'capture': tw, 'old_graph': torch_old, 'computed': (a * tw).tanh(),
                               'detached': a.detach(), 'factory': torch.tensor(0.8, requires_grad=True)}
                for name in ('grad', 'grad_and_value', 'jacrev'):
                    with self.subTest(input_grad=input_grad, capture_grad=capture_grad, operation=name):
                        with fr.no_grad():
                            got = getattr(ff, name)(actual, has_aux=True)(x)
                            self.assertFalse(fr.is_grad_enabled())
                        with torch.no_grad():
                            expected = getattr(torch.func, name)(reference, has_aux=True)(tx)
                        same(got, expected)
                with fr.no_grad():
                    got = ff.hessian(lambda a: (a * w).tanh())(x)
                    jvp = ff.jvp(actual, (x,), (fr.Tensor([0.4], []),), has_aux=True)
                    value, pullback, auxiliary = ff.vjp(actual, x, has_aux=True)
                with torch.no_grad():
                    expected = torch.func.hessian(lambda a: (a * tw).tanh())(tx)
                    torch_jvp = torch.func.jvp(reference, (tx,), (torch.tensor(0.4),), has_aux=True)
                    torch_value, torch_pullback, torch_auxiliary = torch.func.vjp(reference, tx, has_aux=True)
                same(got, expected)
                same(jvp, torch_jvp)
                same((value, auxiliary), (torch_value, torch_auxiliary))
                same(pullback(fr.Tensor([0.6], [])), torch_pullback(torch.tensor(0.6)))
        with fr.no_grad(), self.assertRaisesRegex(RuntimeError, 'injected'):
            ff.grad(lambda a: (_ for _ in ()).throw(RuntimeError('injected')))(x)
        self.assertTrue(fr.is_grad_enabled())
        with fr.no_grad():
            value, closure = ff.vjp(lambda a: (a * w).tanh(), x)
        with torch.no_grad():
            expected_value, reference = torch.func.vjp(lambda a: (a * tw).tanh(), tx)
        same(value, expected_value)
        v, tv = fr.Tensor([0.4], []).requires_grad_(True), torch.tensor(0.4, requires_grad=True)
        actual, expected = closure(v), reference(tv)
        same(actual, expected)
        reference_gradients = torch.autograd.grad(total(expected), (tx, tw, tv), allow_unused=True)
        same(fr.autograd.grad(total(actual), (x, w, v), allow_unused=True), reference_gradients)

    def test_vjp_and_jvp_preserve_live_identity_cotangents_without_recording_computation(self):
        x = fr.Tensor([0.2, 0.4], [2]).requires_grad_(True)
        tx = torch.tensor([0.2, 0.4], requires_grad=True)
        for kind in ('identity', 'addition', 'multiply', 'duplicate'):
            def fn(a):
                return a if kind == 'identity' else a + 2. if kind == 'addition' else a * 2. if kind == 'multiply' else (a, a)
            _, closure = ff.vjp(fn, x)
            _, reference = torch.func.vjp(fn, tx)
            for enabled in (False, True):
                for create_graph in (None, False, True):
                    v = fr.Tensor([0.3, -0.6], [2]).requires_grad_(True)
                    tv = torch.tensor([0.3, -0.6], requires_grad=True)
                    inputs, torch_inputs = ((v, v), (tv, tv)) if kind == 'duplicate' else (v, tv)
                    with fr.enable_grad() if enabled else fr.no_grad():
                        actual = closure(inputs, create_graph=create_graph)
                    with torch.enable_grad() if enabled else torch.no_grad():
                        expected = reference(torch_inputs, create_graph=create_graph)
                    same(actual, expected)
                    if kind in ('identity', 'addition'):
                        self.assertIs(actual[0], v)
                        self.assertIs(expected[0], tv)
                        same(fr.autograd.grad(actual[0].sum(), (v,)), torch.autograd.grad(expected[0].sum(), (tv,)))
            if kind == 'identity':
                with fr.no_grad():
                    actual = ff.jvp(fn, (x,), (v,))
                with torch.no_grad():
                    expected = torch.func.jvp(fn, (tx,), (tv,))
                same(actual, expected)
                self.assertIs(actual[1], v)
                self.assertIs(expected[1], tv)

    def test_vjp_no_grad_views_retain_leaf_flags_and_true_cotangent_dependencies(self):
        tx = torch.tensor([[0.2, -0.4, 0.3], [0.7, -0.2, 0.6]], requires_grad=True)
        x = tensor(tx)
        for kind in ('reshape', 'transpose', 'unsqueeze', 'clone'):
            def fn(a):
                if kind == 'reshape':
                    return a.reshape([3, 2])
                if kind == 'transpose':
                    return a.transpose(0, 1)
                return a.unsqueeze(0) if kind == 'unsqueeze' else a.clone()
            value, closure = ff.vjp(fn, x)
            torch_value, reference = torch.func.vjp(fn, tx)
            same(value, torch_value)
            for create_graph in (False, True):
                with self.subTest(operation=kind, create_graph=create_graph):
                    # Match tensor(tv)'s owning storage; a leaf view of a
                    # non-requiring base has different Torch no-grad flags.
                    tv = torch.linspace(-0.3, 0.4, 6).reshape(torch_value.shape).clone().requires_grad_()
                    v = tensor(tv)
                    with fr.no_grad():
                        actual = closure(v, create_graph=create_graph)[0]
                    with torch.no_grad():
                        expected = reference(tv, create_graph=create_graph)[0]
                    same(actual, expected)
                    reference_grad = torch.autograd.grad(expected.sum(), tv, allow_unused=True)[0]
                    same(fr.autograd.grad(actual.sum(), (v,), allow_unused=True)[0], reference_grad)
                    if kind == 'clone':
                        self.assertIs(actual, v)
                        self.assertIs(expected, tv)

    def test_reduction_vjp_preserves_live_expanded_views_without_creating_edges(self):
        tx = torch.linspace(-0.4, 0.8, 24).reshape(2, 3, 4).clone().requires_grad_()
        x = tensor(tx)
        for dim, keepdim in ((None, False), (0, False), (1, False), (1, True), ((0, 2), False)):
            def fn(a):
                return a.sum() if dim is None else a.sum(dim=dim, keepdim=keepdim)
            value, closure = ff.vjp(fn, x)
            expected_value, reference = torch.func.vjp(fn, tx)
            same(value, expected_value)
            for live in (False, True):
                for enabled in (False, True):
                    with self.subTest(dim=dim, keepdim=keepdim, live=live, enabled=enabled):
                        tv = torch.linspace(0.2, 0.7, expected_value.numel()).reshape(expected_value.shape).clone().requires_grad_(live)
                        v = tensor(tv)
                        if dim == 1:
                            # Preserve a genuinely noncontiguous cotangent with
                            # a requiring-grad root on both sides.
                            base = torch.arange(8, dtype=torch.float32).reshape(4, 2).clone().requires_grad_(live)
                            fv = tensor(base)
                            tv, v = base.transpose(0, 1), fv.transpose(0, 1)
                            if keepdim:
                                tv, v = tv.unsqueeze(1), v.unsqueeze(1)
                        with fr.enable_grad() if enabled else fr.no_grad():
                            actual = closure(v, create_graph=False)[0]
                        with torch.enable_grad() if enabled else torch.no_grad():
                            expected = reference(tv, create_graph=False)[0]
                        same(actual, expected)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertEqual(actual.is_leaf, expected.is_leaf)
                        if live:
                            derivative = torch.autograd.grad(expected.sum(), tv, allow_unused=True)[0]
                            self.assertIsNone(derivative)
                            self.assertIsNone(fr.autograd.grad(actual.sum(), (v,), allow_unused=True)[0])
                        self.assertIsNone(v.grad)
                        self.assertIsNone(x.grad)

    def test_cat_stack_and_singleton_pullbacks_preserve_noncontiguous_cotangent_views(self):
        for kind in ('cat', 'stack', 'squeeze', 'unsqueeze'):
            for live in (False, True):
                for create_graph in (False, True):
                    with self.subTest(operation=kind, live=live, create_graph=create_graph):
                        shape = [2, 1, 2] if kind == 'squeeze' else [2, 2]
                        tx = torch.arange(4, dtype=torch.float32).reshape(shape).clone().requires_grad_()
                        ty = (tx.detach() + 1).requires_grad_()
                        x, y = tensor(tx), tensor(ty)
                        def fn(a, b, lib):
                            if kind == 'cat':
                                return lib.cat((a, b), dim=1)
                            if kind == 'stack':
                                return lib.stack((a, b), dim=1)
                            return a.squeeze(1) if kind == 'squeeze' else a.unsqueeze(1)
                        value, closure = ff.vjp(lambda a, b: fn(a, b, fr), x, y)
                        expected_value, reference = torch.func.vjp(lambda a, b: fn(a, b, torch), tx, ty)
                        base_shape = list(expected_value.shape)
                        base_shape[0], base_shape[-1] = base_shape[-1], base_shape[0]
                        tb = torch.linspace(0.2, 0.9, expected_value.numel()).reshape(base_shape).clone().requires_grad_(live)
                        b = tensor(tb)
                        v, tv = b.transpose(0, -1), tb.transpose(0, -1)
                        actual, expected = closure(v, create_graph=create_graph), reference(tv, create_graph=create_graph)
                        same(actual, expected)
                        for got, wanted in zip(actual, expected):
                            self.assertEqual(got.stride(), wanted.stride())
                            self.assertEqual(got.storage_offset(), wanted.storage_offset())
                            self.assertEqual(got.is_leaf, wanted.is_leaf)
                        if expected[0].requires_grad:
                            derivative = torch.autograd.grad(expected[0].sum(), tb, allow_unused=True)[0]
                            same(fr.autograd.grad(actual[0].sum(), (b,), allow_unused=True)[0], derivative)
                        with fr.no_grad(), torch.no_grad():
                            b.copy_(fr.full(b.shape, 2.))
                            tb.copy_(torch.full_like(tb, 2.))
                        same(actual, expected)
                        self.assertIsNone(x.grad)
                        self.assertIsNone(y.grad)
                        self.assertIsNone(b.grad)

    def test_mixed_direction_dtypes_preserve_invocation_mode_cast_graphs(self):
        def typed(value):
            return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).requires_grad_(value.requires_grad)
        def compare(actual, expected):
            if expected is None:
                self.assertIsNone(actual)
                return
            self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
            self.assertEqual(actual.shape, list(expected.shape))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype), expected.detach(), atol=1e-5, rtol=1e-4)
        for operation in ('jvp', 'vjp'):
            for dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
                if operation == 'jvp' and not dtype.is_floating_point:
                    continue
                for created_enabled in ((False, True) if operation == 'vjp' else (True,)):
                    for enabled in (False, True):
                        for kind in ('identity', 'square', 'tanh'):
                            for graph in ((None, False, True) if operation == 'vjp' else (None,)):
                                with self.subTest(operation=operation, dtype=dtype, created=created_enabled, enabled=enabled, kind=kind, graph=graph):
                                    tx = torch.tensor([0.2, -0.4], requires_grad=True)
                                    tv = torch.tensor([1, 0], dtype=dtype, requires_grad=dtype.is_floating_point)
                                    x, v = typed(tx), typed(tv)
                                    fn = lambda a: a if kind == 'identity' else a * a if kind == 'square' else a.tanh()
                                    if operation == 'vjp':
                                        with fr.enable_grad() if created_enabled else fr.no_grad():
                                            _, closure = ff.vjp(fn, x)
                                        with torch.enable_grad() if created_enabled else torch.no_grad():
                                            _, reference = torch.func.vjp(fn, tx)
                                        with fr.enable_grad() if enabled else fr.no_grad():
                                            actual = closure(v, create_graph=graph)[0]
                                        with torch.enable_grad() if enabled else torch.no_grad():
                                            expected = reference(tv, create_graph=graph)[0]
                                    else:
                                        with fr.enable_grad() if enabled else fr.no_grad():
                                            _, actual = ff.jvp(fn, (x,), (v,))
                                        with torch.enable_grad() if enabled else torch.no_grad():
                                            _, expected = torch.func.jvp(fn, (tx,), (tv,))
                                    compare(actual, expected)
                                    if expected.requires_grad:
                                        inputs = (x, v) if tv.requires_grad else (x,)
                                        torch_inputs = (tx, tv) if tv.requires_grad else (tx,)
                                        gradients = fr.autograd.grad(actual.sum(), inputs, allow_unused=True)
                                        targets = torch.autograd.grad(expected.sum(), torch_inputs, allow_unused=True)
                                        for got, wanted, original in zip(gradients, targets, torch_inputs):
                                            compare(got, wanted)
                                    self.assertIsNone(x.grad)
                                    self.assertIsNone(v.grad)
        for dtype in ('int64', 'bool'):
            with self.assertRaises(ValueError):
                ff.jvp(lambda a: a * a, (fr.ones(2),), (fr.ones(2, dtype=dtype),))

    def test_vjp_late_enabled_saved_coefficient_builds_a_new_derivative_graph(self):
        for created_enabled in (False, True):
            for input_enabled in (False, True):
                with self.subTest(created=created_enabled, input_enabled=input_enabled):
                    tx = torch.tensor([0.3, -0.2], requires_grad=input_enabled)
                    tw = torch.tensor([0.7, 0.5])
                    x, w = tensor(tx), tensor(tw)
                    with fr.enable_grad() if created_enabled else fr.no_grad():
                        value, closure = ff.vjp(lambda a: a * w, x)
                    with torch.enable_grad() if created_enabled else torch.no_grad():
                        expected_value, reference = torch.func.vjp(lambda a: a * tw, tx)
                    w.requires_grad_(True)
                    tw.requires_grad_(True)
                    same(value, expected_value)
                    tv = torch.tensor([0.4, -0.6])
                    actual, expected = closure(tensor(tv))[0], reference(tv)[0]
                    same(actual, expected)
                    same(fr.autograd.grad(actual.sum(), (w,)), torch.autograd.grad(expected.sum(), (tw,)))
                    self.assertIsNone(w.grad)
                    self.assertIsNone(x.grad)

    def test_frozen_vjp_distinguishes_original_primals_intermediates_and_saved_outputs(self):
        kinds = ('square', 'cube', 'pow_cube', 'view_square', 'tanh_square', 'log', 'abs',
                 'sqrt', 'pow_half', 'exp', 'tanh', 'sigmoid', 'relu', 'softmax')
        for kind in kinds:
            for input_grad in (False, True):
                for live_seed in (False, True):
                    with self.subTest(operation=kind, input_grad=input_grad, live_seed=live_seed):
                        tx = torch.tensor([0.2, 0.4] if kind in ('log', 'sqrt', 'pow_half') else [-0.2, 0.4], requires_grad=input_grad)
                        tv = torch.tensor([0.3, 0.7], requires_grad=live_seed)
                        x, v = tensor(tx), tensor(tv)
                        def fn(a):
                            if kind == 'square':
                                return a * a
                            if kind == 'cube':
                                return (a * a) * a
                            if kind == 'pow_cube':
                                return a.pow(3.)
                            if kind == 'view_square':
                                return a.reshape([2]) * a.reshape([2])
                            if kind == 'tanh_square':
                                return (a * a).tanh()
                            if kind == 'pow_half':
                                return a.pow(0.5)
                            return a.softmax(0) if kind == 'softmax' else getattr(a, kind)()
                        with fr.no_grad():
                            value, closure = ff.vjp(fn, x)
                        with torch.no_grad():
                            expected_value, reference = torch.func.vjp(fn, tx)
                        same(value, expected_value)
                        actual, expected = closure(v, create_graph=True)[0], reference(tv, create_graph=True)[0]
                        same(actual, expected)
                        if expected.requires_grad:
                            inputs = tuple(a for a, enabled in ((x, input_grad), (v, live_seed)) if enabled)
                            torch_inputs = tuple(a for a in (tx, tv) if a.requires_grad)
                            gradients = fr.autograd.grad(actual.sum(), inputs, allow_unused=True)
                            torch_gradients = torch.autograd.grad(expected.sum(), torch_inputs, allow_unused=True)
                            for got, wanted, original in zip(gradients, torch_gradients, torch_inputs):
                                same(got, wanted)
                        self.assertEqual(x.requires_grad, input_grad)
                        self.assertIsNone(x.grad)
                        self.assertIsNone(v.grad)

    def test_frozen_saved_outputs_reject_mutation_in_later_derivative_queries(self):
        for kind in ('tanh', 'relu', 'softmax'):
            with self.subTest(operation=kind):
                tx = torch.tensor([-0.2, 0.4], requires_grad=True)
                tv = torch.tensor([0.3, 0.7], requires_grad=True)
                x, v = tensor(tx), tensor(tv)
                fn = lambda a: a.softmax(0) if kind == 'softmax' else getattr(a, kind)()
                with fr.no_grad():
                    value, closure = ff.vjp(fn, x)
                with torch.no_grad():
                    expected_value, reference = torch.func.vjp(fn, tx)
                actual, expected = closure(v, create_graph=True)[0], reference(tv, create_graph=True)[0]
                same(actual, expected)
                with fr.no_grad():
                    value.copy_(fr.ones(value.shape))
                with torch.no_grad():
                    expected_value.copy_(torch.ones_like(expected_value))
                with self.assertRaises(RuntimeError):
                    torch.autograd.grad(expected.sum(), tv)
                with self.assertRaises((ValueError, RuntimeError)):
                    fr.autograd.grad(actual.sum(), (v,))
                self.assertIsNone(x.grad)
                self.assertIsNone(v.grad)

    def test_frozen_view_flags_distinguish_synthetic_primals_and_captured_storage(self):
        for kind in ('direct_view', 'computed_view', 'captured_view'):
            for input_grad in (False, True):
                with self.subTest(operation=kind, input_grad=input_grad):
                    tx = torch.tensor([0.2, 0.4], requires_grad=input_grad)
                    tw = torch.tensor([0.7, 0.5], requires_grad=True)
                    tv = torch.tensor([0.3, 0.7])
                    x, w, v = tensor(tx), tensor(tw), tensor(tv)
                    def fn(a, capture):
                        if kind == 'captured_view':
                            return a * capture.reshape([2])
                        z = a.reshape([2]) if kind == 'direct_view' else (a * a).reshape([2])
                        return z * z + capture * 0.
                    with fr.no_grad():
                        value, closure = ff.vjp(lambda a: fn(a, w), x)
                    with torch.no_grad():
                        expected_value, reference = torch.func.vjp(lambda a: fn(a, tw), tx)
                    same(value, expected_value)
                    actual = closure(v, create_graph=True)[0]
                    expected = reference(tv, create_graph=True)[0]
                    same(actual, expected)
                    if expected.requires_grad:
                        inputs = (x, w) if input_grad else (w,)
                        torch_inputs = (tx, tw) if input_grad else (tw,)
                        gradients = fr.autograd.grad(actual.sum(), inputs, allow_unused=True)
                        expected_gradients = torch.autograd.grad(expected.sum(), torch_inputs, allow_unused=True)
                        for got, wanted, original in zip(gradients, expected_gradients, torch_inputs):
                            same(got, wanted)
                    self.assertIsNone(x.grad)
                    self.assertIsNone(w.grad)
                    self.assertEqual(x.requires_grad, input_grad)

    def test_vjp_creation_mode_freezes_forward_operands_but_preserves_capture_and_seed_provenance(self):
        for created_enabled in (False, True):
            for invoked_enabled in (False, True):
                for create_graph in (None, False, True):
                    for duplicate, shared_capture in ((False, False), (True, True)):
                        with self.subTest(created=created_enabled, invoked=invoked_enabled,
                                          create_graph=create_graph, duplicate=duplicate, shared_capture=shared_capture):
                            tx = torch.tensor([0.3, -0.2], requires_grad=True)
                            tw = tx if shared_capture else torch.tensor([0.7, 0.5], requires_grad=True)
                            x, w = tensor(tx), None
                            w = x if shared_capture else tensor(tw)
                            actual_calls, reference_calls = [], []
                            def actual_fn(a):
                                actual_calls.append(1)
                                z = (a * w).tanh()
                                return (z, z) if duplicate else z
                            def reference_fn(a):
                                reference_calls.append(1)
                                z = (a * tw).tanh()
                                return (z, z) if duplicate else z
                            with fr.enable_grad() if created_enabled else fr.no_grad():
                                value, closure = ff.vjp(actual_fn, x)
                            with torch.enable_grad() if created_enabled else torch.no_grad():
                                expected_value, reference = torch.func.vjp(reference_fn, tx)
                            same(value, expected_value)
                            tv = torch.tensor([0.4, -0.3], requires_grad=True)
                            v = tensor(tv)
                            seeds, torch_seeds = ((v, v * 0.3), (tv, tv * 0.3)) if duplicate else (v, tv)
                            with fr.enable_grad() if invoked_enabled else fr.no_grad():
                                actual = closure(seeds, create_graph=create_graph)
                            with torch.enable_grad() if invoked_enabled else torch.no_grad():
                                expected = reference(torch_seeds, create_graph=create_graph)
                            same(actual, expected)
                            self.assertEqual((len(actual_calls), len(reference_calls)), (1, 1))
                            if total(expected).requires_grad:
                                derivatives = torch.autograd.grad(total(expected), (tx, tw, tv), allow_unused=True)
                                same(fr.autograd.grad(total(actual), (x, w, v), allow_unused=True), derivatives)
                            self.assertIsNone(x.grad)
                            self.assertIsNone(w.grad)
                            self.assertIsNone(v.grad)


if __name__ == '__main__':
    unittest.main()
