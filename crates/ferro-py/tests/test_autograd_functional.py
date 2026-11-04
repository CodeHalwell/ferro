"""Functional Jacobian/Hessian/JVP/VJP values and derivative graphs against Torch."""
import unittest

import ferro as fr
from ferro.autograd import functional as af
import torch
from torch.autograd import functional as tf


def tensor(value, requires_grad=True):
    constructor = fr.Tensor.from_f64 if value.dtype == torch.float64 else fr.Tensor
    return constructor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def close(actual, expected, mixed=False):
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise AssertionError('functional tuple structure differs')
        for a, b in zip(actual, expected):
            close(a, b, mixed)
        return
    if actual.shape != list(expected.shape) or actual.dtype != str(expected.dtype).removeprefix('torch.'):
        raise AssertionError('functional shape/dtype differs')
    got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(got, expected.detach(), atol=1e-5 if mixed or expected.dtype == torch.float32 else 1e-12,
                               rtol=1e-4 if mixed or expected.dtype == torch.float32 else 1e-12)


def total(value):
    if isinstance(value, tuple):
        result = None
        for item in value:
            term = total(item)
            result = term if result is None else result + term
        return result
    return value.sum()


class FunctionalAutogradTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_jacobian_block_shapes_multiinput_multioutput_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            tx = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=dtype, requires_grad=True)
            tb = torch.tensor([0.1, 0.5], dtype=dtype, requires_grad=True)
            x, b = tensor(tx), tensor(tb)
            for input_tuple, output_tuple in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(dtype=dtype, inputs=input_tuple, outputs=output_tuple):
                    def fn(a, bias=None):
                        z = a.tanh() if bias is None else (a * bias + a).tanh()
                        return (z, z.sum()) if output_tuple else z
                    actual_inputs, expected_inputs = ((x, b), (tx, tb)) if input_tuple else (x, tx)
                    result, expected = af.jacobian(fn, actual_inputs), tf.jacobian(fn, expected_inputs)
                    close(result, expected)
                    self.assertFalse(total(result).requires_grad)

    def test_jvp_and_vjp_nonuniform_vectors_and_default_scalar_directions(self):
        for dtype in (torch.float32, torch.float64):
            tx = torch.tensor([0.3, -0.2], dtype=dtype, requires_grad=True)
            tb = torch.tensor([0.1, 0.7], dtype=dtype, requires_grad=True)
            vx, vb = torch.tensor([0.2, -0.4], dtype=dtype), torch.tensor([-0.3, 0.8], dtype=dtype)
            x, b = tensor(tx), tensor(tb)
            def fn(a, bias):
                z = (a * bias + a.exp()).tanh()
                return z, z.sum()
            close(af.jvp(fn, (x, b), (tensor(vx), tensor(vb))), tf.jvp(fn, (tx, tb), (vx, vb)))
            vs = torch.tensor(0.6, dtype=dtype)
            close(af.vjp(fn, (x, b), (tensor(vx), tensor(vs))), tf.vjp(fn, (tx, tb), (vx, vs)))
            scalar = torch.tensor(0.4, dtype=dtype)
            for name in ('jvp', 'vjp'):
                close(getattr(af, name)(lambda a: a.exp(), tensor(scalar)), getattr(tf, name)(lambda a: a.exp(), scalar))

    def test_hessian_mixed_blocks_nonlinear_matrix_network(self):
        tx = torch.tensor([[0.2, -0.3], [0.4, 0.6]], requires_grad=True)
        tw = torch.tensor([[0.5, -0.2], [0.1, 0.4]], requires_grad=True)
        x, w = tensor(tx), tensor(tw)
        def fn(a, weight):
            z = (a @ weight).tanh()
            return (z * z).sum() + (a * a).sum() * 0.1
        close(af.hessian(fn, (x, w)), tf.hessian(fn, (tx, tw)))
        close(af.hessian(lambda a: (a.exp() * a).sum(), x), tf.hessian(lambda a: (a.exp() * a).sum(), tx))

    def test_mixed_float32_float64_inputs_outputs_and_directions(self):
        tx = torch.tensor([0.25, -0.5], requires_grad=True)
        tb = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)
        x, b = tensor(tx), tensor(tb)
        def actual(a, bias):
            z = (a.to(dtype='float64') * bias).tanh()
            return z.to(dtype='float32'), z.sum()
        def reference(a, bias):
            z = (a.to(dtype=torch.float64) * bias).tanh()
            return z.to(dtype=torch.float32), z.sum()
        close(af.jacobian(actual, (x, b)), tf.jacobian(reference, (tx, tb)), mixed=True)
        vx, vb = torch.tensor([0.2, -0.4]), torch.tensor([0.6, 0.3], dtype=torch.float64)
        close(af.jvp(actual, (x, b), (tensor(vx), tensor(vb))),
              tf.jvp(reference, (tx, tb), (vx, vb)), mixed=True)
        scalar = torch.tensor(0.3, dtype=torch.float64)
        close(af.vjp(actual, (x, b), (tensor(vx), tensor(scalar))),
              tf.vjp(reference, (tx, tb), (vx, scalar)), mixed=True)
        for name in ('jacobian', 'hessian'):
            with self.assertRaises(NotImplementedError):
                getattr(af, name)(actual, (x, b), create_graph=True)
        close(af.jvp(actual, (x, b), (tensor(vx), tensor(vx))),
              tf.jvp(reference, (tx, tb), (vx, vx)), mixed=True)
        self.assertIsNone(x.grad)
        self.assertIsNone(b.grad)

    def test_create_graph_reaches_inputs_captured_parameters_and_live_vectors(self):
        for name in ('jacobian', 'hessian', 'jvp', 'vjp'):
            with self.subTest(operation=name):
                tx = torch.tensor([0.2, 0.4], requires_grad=True)
                tw = torch.tensor([0.7, -0.3], requires_grad=True)
                tv = torch.tensor([0.3, -0.6], requires_grad=True)
                x, w, v = tensor(tx), tensor(tw), tensor(tv)
                def actual(a):
                    z = (a * w).tanh()
                    return (z * z).sum() if name == 'hessian' else z
                def reference(a):
                    z = (a * tw).tanh()
                    return (z * z).sum() if name == 'hessian' else z
                kwargs, torch_kwargs = ({'v': v}, {'v': tv}) if name in ('jvp', 'vjp') else ({}, {})
                result = getattr(af, name)(actual, x, create_graph=True, **kwargs)
                expected = getattr(tf, name)(reference, tx, create_graph=True, **torch_kwargs)
                close(result, expected)
                inputs, references = [x, w], [tx, tw]
                if name in ('jvp', 'vjp'):
                    inputs.append(v)
                    references.append(tv)
                derivatives = fr.autograd.grad(total(result), inputs)
                wanted = torch.autograd.grad(total(expected), references)
                close(derivatives, wanted)

    def test_duplicate_inputs_are_distinct_arguments_with_shared_upstream_identity(self):
        tx = torch.tensor([0.2, -0.7], requires_grad=True)
        x = tensor(tx)
        def fn(a, b):
            return a * b + a
        actual = af.jacobian(fn, (x, x), create_graph=True)
        expected = tf.jacobian(fn, (tx, tx), create_graph=True)
        close(actual, expected)
        close(fr.autograd.grad(total(actual), [x]), torch.autograd.grad(total(expected), [tx]))

    def test_empty_unused_and_constant_outputs_without_mutating_inputs(self):
        for shape in ([0, 2], [2]):
            tx = torch.zeros(shape, requires_grad=True)
            x = tensor(tx)
            # Torch's legacy functional.jacobian cannot stack zero output rows.
            expected = torch.func.jacrev(lambda a: a * a)(tx) if not tx.numel() else tf.jacobian(lambda a: a * a, tx)
            close(af.jacobian(lambda a: a * a, x), expected)
            close(af.jvp(lambda a: a * a, x, fr.Tensor.ones(shape)), tf.jvp(lambda a: a * a, tx, torch.ones(shape)))
        x, tx = fr.Tensor([0.2, 0.4], [2]).requires_grad_(True), torch.tensor([0.2, 0.4], requires_grad=True)
        y, ty = fr.Tensor([0.6], [1]).requires_grad_(True), torch.tensor([0.6], requires_grad=True)
        close(af.jacobian(lambda a, b: a * a, (x, y)), tf.jacobian(lambda a, b: a * a, (tx, ty)))
        close(af.hessian(lambda a: (a * 2).sum(), x), tf.hessian(lambda a: (a * 2).sum(), tx))
        close(af.jacobian(lambda a: fr.Tensor.ones([3]), x), tf.jacobian(lambda a: torch.ones(3), tx))
        (x * x).sum().backward()
        before = x.grad.tolist()
        with fr.no_grad():
            result = af.jacobian(lambda a: a.exp(), x, create_graph=True)
            self.assertTrue(result.requires_grad)
            self.assertFalse(fr.is_grad_enabled())
        self.assertEqual(x.grad.tolist(), before)
        self.assertTrue(x.requires_grad)
        for operation in (lambda: af.jacobian(lambda a: (_ for _ in ()).throw(RuntimeError('injected')), x),
                          lambda: af.hessian(lambda a: a, x)):
            with fr.no_grad(), self.assertRaises(RuntimeError):
                operation()
            self.assertTrue(fr.is_grad_enabled())

    def test_strict_structural_dependencies_zero_values_and_linear_derivatives(self):
        self.assertTrue(hasattr(fr.Tensor, '_depends_on'), 'requires candidate native graph reachability helper')
        x = fr.Tensor([0.2, -0.4], [2]).requires_grad_(True)
        tx = torch.tensor([0.2, -0.4], requires_grad=True)
        for name in ('jacobian', 'jvp', 'vjp'):
            extra = {'v': fr.Tensor.ones([2])} if name != 'jacobian' else {}
            torch_extra = {'v': torch.ones(2)} if name != 'jacobian' else {}
            with self.subTest(name=name):
                close(getattr(af, name)(lambda a: a * 0., x, strict=True, **extra),
                      getattr(tf, name)(lambda a: a * 0., tx, strict=True, **torch_extra))
                with self.assertRaises(RuntimeError):
                    getattr(af, name)(lambda a: fr.Tensor.ones([2]), x, strict=True, **extra)
                with self.assertRaises(RuntimeError):
                    getattr(af, name)(lambda a: a * 2., x, strict=True, create_graph=True, **extra)
        with self.assertRaises(RuntimeError):
            af.jacobian(lambda a, b: a * a, (x, x), strict=True)
        with self.assertRaises(RuntimeError):
            af.hessian(lambda a: a.sum(), x, strict=True)
        def disjoint(a, b):
            return a * a, b * b
        for name in ('jvp', 'vjp'):
            close(getattr(af, name)(disjoint, (x, x), (fr.Tensor.ones([2]), fr.Tensor.ones([2])), strict=True, create_graph=True),
                  getattr(tf, name)(disjoint, (tx, tx), (torch.ones(2), torch.ones(2)), strict=True, create_graph=True))
        direction = fr.Tensor.ones([2]).requires_grad_(True)
        close(af.jvp(lambda a: a * 2., x, direction, strict=True, create_graph=True),
              tf.jvp(lambda a: a * 2., tx, torch.ones(2, requires_grad=True), strict=True, create_graph=True))
        captured = fr.Tensor([0.4, 0.7], [2]).requires_grad_(True)
        torch_captured = torch.tensor([0.4, 0.7], requires_grad=True)
        for create_graph in (False, True):
            with self.assertRaises(RuntimeError):
                af.jvp(lambda a: (a * a, captured * captured), x, direction, strict=True, create_graph=create_graph)
            close(af.vjp(lambda a: (a * a, captured * captured), x, (direction, direction), strict=True, create_graph=create_graph),
                  tf.vjp(lambda a: (a * a, torch_captured * torch_captured), tx,
                         (torch.ones(2, requires_grad=True), torch.ones(2, requires_grad=True)), strict=True, create_graph=create_graph))

    def test_errors_structure_shapes_dtype_modes_and_explicit_limits(self):
        x = fr.Tensor.ones([2])
        for operation in (lambda: af.jacobian(lambda a: a, [x]), lambda: af.jacobian(lambda a: [a], x),
                          lambda: af.jacobian(None, x)):
            with self.assertRaises(TypeError):
                operation()
        for name in ('jvp', 'vjp'):
            for v in (None, fr.Tensor.ones([3]), (fr.Tensor.ones([2]), fr.Tensor.ones([2]))):
                with self.assertRaises(RuntimeError):
                    getattr(af, name)(lambda a: a, x, v)
            close(getattr(af, name)(lambda a: a, x, (fr.Tensor.ones([2]),)),
                  getattr(tf, name)(lambda a: a, torch.ones(2), (torch.ones(2),)))
        with self.assertRaises(RuntimeError):
            af.hessian(lambda a: (a.sum(),), x)
        for options in ({'vectorize': True}, {'strategy': 'forward-mode'}):
            with self.assertRaises(NotImplementedError):
                af.jacobian(lambda a: a, x, **options)
        double = fr.Tensor.from_f64([0.2], [1])
        with self.assertRaises(NotImplementedError):
            af.jacobian(lambda a: a * a, double, create_graph=True)
        with self.assertRaises(NotImplementedError):
            af.hessian(lambda a: a.sum(), double)
        with self.assertRaises(ValueError):
            fr.capture(lambda: af.jacobian(lambda a: a, x))
        with self.assertRaises(ValueError):
            af.jacobian(lambda a: a, fr.Tensor.from_i64([1], [1]))

    def test_direction_dtypes_cast_to_paired_tensor_and_keep_live_float_gradients(self):
        for name in ('jvp', 'vjp'):
            for dtype in (torch.float32, torch.float64):
                for direction_dtype in (torch.float32, torch.float64, torch.int64, torch.bool):
                    for create_graph in ((False, True) if dtype == torch.float32 else (False,)):
                        with self.subTest(operation=name, dtype=dtype, direction=direction_dtype, graph=create_graph):
                            tx = torch.tensor([0.2, -0.4], dtype=dtype, requires_grad=True)
                            tv = torch.tensor([1, 0], dtype=direction_dtype, requires_grad=direction_dtype.is_floating_point)
                            x = tensor(tx)
                            v = fr.tensor(tv.detach().tolist(), dtype=str(direction_dtype).removeprefix('torch.')).requires_grad_(tv.requires_grad)
                            actual = getattr(af, name)(lambda a: a.tanh(), x, v, create_graph=create_graph)
                            expected = getattr(tf, name)(lambda a: a.tanh(), tx, tv, create_graph=create_graph)
                            close(actual, expected)
                            self.assertEqual(actual[1].requires_grad, expected[1].requires_grad)
                            if create_graph:
                                inputs = (x, v) if tv.requires_grad else (x,)
                                torch_inputs = (tx, tv) if tv.requires_grad else (tx,)
                                close(fr.autograd.grad(actual[1].sum(), inputs), torch.autograd.grad(expected[1].sum(), torch_inputs), mixed=True)
                            self.assertEqual(v.requires_grad, tv.requires_grad)
                            self.assertEqual(v.dtype, str(direction_dtype).removeprefix('torch.'))
                            self.assertIsNone(x.grad)
                            self.assertIsNone(v.grad)
        tx = torch.tensor([0.2, -0.4], requires_grad=True)
        tv = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        close(af.jvp(lambda a: a * 2., tensor(tx), tensor(tv), create_graph=True, strict=True),
              tf.jvp(lambda a: a * 2., tx, tv, create_graph=True, strict=True))

    def test_strict_jvp_distinguishes_seed_independent_and_seed_dependent_zeros(self):
        tx = torch.tensor([-0.4, 0.3], requires_grad=True)
        x = tensor(tx)
        for create_graph in (False, True):
            for live_direction in (False, True):
                tv = torch.tensor([0.6, 0.2], requires_grad=live_direction)
                v = tensor(tv, live_direction)
                for kind in ('sign', 'zero', 'linear', 'square'):
                    def function(a):
                        if kind == 'sign':
                            return a.sign()
                        return a * (0. if kind == 'zero' else 2.) if kind != 'square' else a * a
                    with self.subTest(create_graph=create_graph, live=live_direction, operation=kind):
                        reject = kind == 'sign' or create_graph and not live_direction and kind in ('zero', 'linear')
                        if reject:
                            with self.assertRaises(RuntimeError):
                                tf.jvp(function, tx, tv, strict=True, create_graph=create_graph)
                            with self.assertRaises(RuntimeError):
                                af.jvp(function, x, v, strict=True, create_graph=create_graph)
                        else:
                            actual = af.jvp(function, x, v, strict=True, create_graph=create_graph)
                            expected = tf.jvp(function, tx, tv, strict=True, create_graph=create_graph)
                            close(actual, expected)
                            self.assertEqual(actual[1].requires_grad, expected[1].requires_grad)
        with self.assertRaises(NotImplementedError):
            af.jvp(lambda a: a * a, fr.Tensor.from_f64([0.3], [1]), fr.Tensor.from_f64([1.], [1]), strict=True)
        self.assertIsNone(x.grad)

    def test_strict_unused_adjoint_differs_from_present_numerical_zero(self):
        import ferro.linalg as fl
        x = fr.Tensor([0.2, 0.4], [2]).requires_grad_(True)
        tx = torch.tensor([0.2, 0.4], requires_grad=True)
        (x * x).sum().backward()
        saved = x.grad.tolist()
        for name in ('jacobian', 'vjp', 'jvp', 'hessian'):
            extras = {'v': fr.Tensor.ones([2])} if name == 'jvp' else {}
            torch_extras = {'v': torch.ones(2)} if name == 'jvp' else {}
            for create_graph in (False, True):
                with self.subTest(operation=name, create_graph=create_graph):
                    with self.assertRaises(RuntimeError):
                        getattr(af, name)(lambda a: fl.vector_norm(a, 0), x,
                                          strict=True, create_graph=create_graph, **extras)
                    with self.assertRaises(RuntimeError):
                        getattr(tf, name)(lambda a: torch.linalg.vector_norm(a, 0), tx,
                                          strict=True, create_graph=create_graph, **torch_extras)
                    close(getattr(af, name)(lambda a: fl.vector_norm(a, 0), x, create_graph=create_graph, **extras),
                          getattr(tf, name)(lambda a: torch.linalg.vector_norm(a, 0), tx, create_graph=create_graph, **torch_extras))
            if name != 'hessian':
                close(getattr(af, name)(lambda a: (a * 0.).sum(), x, strict=True, **extras),
                      getattr(tf, name)(lambda a: (a * 0.).sum(), tx, strict=True, **torch_extras))
        def actual(a):
            return fl.vector_norm(a, 0), a * a
        def reference(a):
            return torch.linalg.vector_norm(a, 0), a * a
        close(af.vjp(actual, x, (fr.Tensor([1.], []), fr.Tensor.ones([2])), strict=True),
              tf.vjp(reference, tx, (torch.tensor(1.), torch.ones(2)), strict=True))
        with self.assertRaises(RuntimeError):
            af.jvp(actual, x, fr.Tensor.ones([2]), strict=True)
        self.assertEqual(x.grad.tolist(), saved)


if __name__ == '__main__':
    unittest.main()
