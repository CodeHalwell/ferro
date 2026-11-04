import math
import unittest

import ferro as fr
import torch


class InplaceApi(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_broadcast_scalars_alpha_and_aliases_return_original_object(self):
        a = fr.tensor([[1., 2.], [3., 4.]])
        x = torch.tensor([[1., 2.], [3., 4.]])
        alias, reference = a.transpose(0, 1), x.T
        identity = a._id()
        for name, rhs, expected_rhs, options in [
                ('add_', fr.tensor([.5, 1.5]), torch.tensor([.5, 1.5]), {'alpha': 2}),
                ('mul_', -2., -2., {}), ('add_', True, True, {}),
                ('add_', a, x, {'alpha': -.5}), ('mul_', a, x, {})]:
            self.assertIs(getattr(a, name)(rhs, **options), a)
            self.assertIs(getattr(x, name)(expected_rhs, **options), x)
            self.assertEqual(a._id(), identity)
            self.same(a, x); self.same(alias, reference)

    def test_no_grad_leaf_updates_keep_gradient_slots_and_invalidate_old_graphs(self):
        a = fr.tensor([2., 3.], requires_grad=True)
        x = torch.tensor([2., 3.], requires_grad=True)
        a.square().sum().backward(); x.square().sum().backward()
        old, reference = a.square().sum(), x.square().sum()
        identity, gradient = a._id(), a.grad._id()
        with fr.no_grad(), torch.no_grad():
            self.assertIs(a.add_(.5), a); x.add_(.5)
            self.assertIs(a.mul_(2.), a); x.mul_(2.)
        self.assertEqual(a._id(), identity)
        self.assertEqual(a.grad._id(), gradient)
        self.assertTrue(a.is_leaf and a.requires_grad)
        self.same(a, x); self.same(a.grad, x.grad)
        with self.assertRaises(ValueError): old.backward()
        with self.assertRaises(RuntimeError): reference.backward()
        a.square().sum().backward(); x.square().sum().backward()
        self.same(a.grad, x.grad)
        with fr.no_grad(), torch.no_grad():
            self.assertIs(a.zero_(), a); x.zero_()
            self.assertIs(a.fill_(7.), a); x.fill_(7.)
        self.same(a, x)
        self.assertEqual(a._id(), identity)

    def test_zero_fill_nonfinite_scalars_signed_zero_and_empty_offsets(self):
        for value in [0., -0., .25, float('inf'), float('-inf'), float('nan'), True]:
            with self.subTest(value=value):
                a, x = fr.ones(3), torch.ones(3)
                self.assertIs(a.fill_(value), a); x.fill_(value)
                self.same(a, x)
                if value == 0:
                    self.assertEqual(math.copysign(1., a.tolist()[0]), math.copysign(1., x.tolist()[0]))
                self.assertIs(a.zero_(), a); x.zero_()
                self.same(a, x)
        for name in ['add_', 'mul_']:
            a, x = fr.ones(2), torch.ones(2)
            getattr(a, name)(1e40); getattr(x, name)(1e40)
            self.same(a, x)
        source = fr.zeros(0, 8).narrow(1, 4, 4)
        destination = fr.zeros(0, 4)
        self.assertIs(destination.add_(source), destination)
        self.assertEqual(destination.shape, [0, 4])
        self.assertEqual(source.storage_offset(), 4)
        huge = 1 << 50
        empty = fr.zeros(0, huge)
        empty.add_(fr.ones(1).expand(1, huge), alpha=2.)
        self.assertEqual(empty.shape, [0, huge])
        self.assertEqual(empty.numel(), 0)

    def test_rejections_leave_values_and_saved_versions_unchanged(self):
        a = fr.tensor([[2., 3.], [4., 5.]], requires_grad=True)
        saved = a.square().sum()
        with self.assertRaises(ValueError): a.add_(1.)
        with self.assertRaises(ValueError): a.mul_(2.)
        with self.assertRaises(ValueError): a.zero_()
        with self.assertRaises(ValueError): a.fill_(7.)
        alias = a.detach()
        for action in [lambda: alias.add_(fr.ones(3)), lambda: alias.mul_(fr.ones(3, 2, 2)),
                       lambda: alias.add_(1., alpha=1e40), lambda: alias.fill_(1e40),
                       lambda: alias.transpose(0, 1).zero_(), lambda: alias.narrow(0, 0, 1).add_(2.)]:
            with self.assertRaises(ValueError): action()
        with self.assertRaises(TypeError): alias.add_(fr.ones(2, 2, dtype='float64'))
        for dtype in ['float64', 'int64', 'bool']:
            actual = fr.ones(2, dtype=dtype)
            self.assertIs(actual.zero_(), actual)
            self.assertEqual(actual.tolist(), [False, False] if dtype == 'bool' else [0, 0])
        saved.backward()
        self.assertEqual(a.tolist(), [[2., 3.], [4., 5.]])
        self.assertEqual(a.grad.tolist(), [[4., 6.], [8., 10.]])

    def test_differentiable_sources_require_explicit_no_grad(self):
        source = fr.tensor([2., 3.], requires_grad=True)
        for name in ['copy_', 'add_', 'mul_']:
            with self.subTest(name=name):
                target = fr.ones(2)
                with self.assertRaisesRegex(ValueError, 'disabled grad recording'):
                    getattr(target, name)(source)
                self.assertEqual(target.tolist(), [1., 1.])
                with fr.no_grad(): getattr(target, name)(source)
                self.assertFalse(target.requires_grad)
                self.assertIsNone(source.grad)
        history = source.square()
        with fr.no_grad():
            with self.assertRaises(ValueError): history.add_(1.)
        self.assertEqual(history.tolist(), [4., 9.])

    def test_capture_rejects_mutation_before_writes_and_recovers(self):
        for name, arguments in [('copy_', (fr.full((2,), 7.),)), ('add_', (3.,)),
                                ('mul_', (2.,)), ('zero_', ()), ('fill_', (9.,))]:
            with self.subTest(name=name):
                target = fr.tensor([2., 3.])
                with self.assertRaisesRegex(ValueError, 'inference capture'):
                    fr.capture(lambda: getattr(target, name)(*arguments) * 2)
                self.assertEqual(target.tolist(), [2., 3.])
                graph = fr.capture(lambda: target * 2).compile_fused()
                self.assertEqual(graph.replay().tolist(), [4., 6.])


if __name__ == '__main__':
    unittest.main()
