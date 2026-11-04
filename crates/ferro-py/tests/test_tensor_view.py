"""Shape-only views preserve real typed aliases or reject incompatible strides."""
import sys
import unittest

import ferro as fr
import torch
import test_tensor_view_arguments as view_helpers


class TensorViewTests(unittest.TestCase):
    same = view_helpers.TensorViewArguments.same

    def test_dense_strided_expanded_and_singleton_chunks_match_torch_all_dtypes(self):
        cases = [
            (lambda a: a, (6, 4)),
            (lambda a: a[:, :, ::2], (3, 4)),
            (lambda a: a.transpose(0, 1), (3, 2, 2, 2)),
            (lambda a: a[:1, :1, :1].expand(2, 3, 4), (24,)),
            (lambda a: a[:1].expand(3, 3, 4), (3, 12)),
            (lambda a: a[:1, :, :1].transpose(0, 1), (3, 1, 1)),
        ]
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            reference = torch.arange(24).reshape(2, 3, 4).to(getattr(torch, dtype))
            if dtype == 'int64': reference[0, 0, 0] = 2**63 - 1
            if dtype == 'float64': reference[0, 0, 0] = 1e-200; reference[1, 2, 3] = 1e200
            base = fr.tensor(reference.tolist(), dtype=dtype)
            aliases = []
            for layout, shape in cases:
                for args in ((shape,), (list(shape),), shape):
                    with self.subTest(dtype=dtype, shape=shape, args=args):
                        source, expected_source = layout(base), layout(reference)
                        actual, expected = source.view(*args), expected_source.view(*args)
                        self.same(actual, expected)
                        self.assertTrue(actual._shares_storage(base))
                        self.assertNotEqual(actual._id(), source._id())
                        aliases.append((actual, expected))
            base.copy_(fr.zeros(base.shape, dtype=dtype)); reference.zero_()
            for actual, expected in aliases: self.same(actual, expected)

    def test_empty_offsets_scalars_inferred_shape_and_view_as(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            x = fr.zeros(0, 8, dtype=dtype).narrow(1, 4, 4)
            expected = torch.zeros(0, 8, dtype=getattr(torch, dtype)).narrow(1, 4, 4)
            for shape in ((0, 4), (2, 0, 2), (0,), (-1,), (4, -1)):
                self.same(x.view(shape), expected.view(shape))
                self.assertTrue(x.view(shape)._shares_storage(x))
            scalar, reference = fr.tensor(1, dtype=dtype), torch.tensor(1, dtype=getattr(torch, dtype))
            for shape in ((), (1,), (1, 1, 1), (-1,)):
                self.same(scalar.view(shape), reference.view(shape))
            x, reference = fr.arange(6).to(dtype=dtype), torch.arange(6).to(getattr(torch, dtype))
            self.same(x.view(2, -1), reference.view(2, -1))
            self.same(x.view(size=(3, -1)), reference.view(size=(3, -1)))
            self.same(x.view_as(fr.zeros(3, 2)), reference.view_as(torch.zeros(3, 2)))
        huge = fr.tensor(1.).expand(sys.maxsize // 4 + 1)
        view = huge.view(1, -1)
        self.assertEqual(view.stride(), (0, 0)); self.assertTrue(view._shares_storage(huge))

    def test_incompatible_views_and_bad_shapes_reject_without_copy_or_state_change(self):
        x, reference = fr.arange(12.).reshape(2, 6), torch.arange(12.).reshape(2, 6)
        for layout in (lambda a: a.T, lambda a: a[:, :2], lambda a: a[:1].expand(3, 6)):
            source, expected = layout(x), layout(reference)
            with self.assertRaises(ValueError): source.view(-1)
            with self.assertRaises(RuntimeError): expected.view(-1)
            self.assertEqual(source.tolist(), expected.tolist())
            self.assertTrue(source._shares_storage(x))
        for args in ((), ((-1, -1),), ((-2, 6),), ((5,),), ((True, 12),), ((2., 6),), ('float64',)):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError, OverflowError)): x.view(*args)
                with self.assertRaises((TypeError, RuntimeError)): reference.view(*args)
        with self.assertRaises((TypeError, ValueError)): x.view_as([3, 4])
        with self.assertRaises(TypeError): x.view(2, 6, size=(2, 6))
        with self.assertRaises(ValueError): fr.zeros(0).view(0, -1)
        self.assertEqual(x.tolist(), reference.tolist())

    def test_gradients_flags_versions_and_shape_only_replay(self):
        for dtype in ('float32', 'float64'):
            reference = torch.arange(12., dtype=getattr(torch, dtype)).reshape(2, 6).clone().requires_grad_()
            x = fr.tensor(reference.tolist(), dtype=dtype, requires_grad=True)
            actual, expected = x[:, 1::2].view(3, 2), reference[:, 1::2].view(3, 2)
            self.same(actual, expected)
            actual.square().sum().backward(); expected.square().sum().backward()
            self.same(x.grad, reference.grad)
            with fr.no_grad(), torch.no_grad():
                self.same(actual.view(6), expected.view(6))
            stale = actual.square().sum()
            with fr.no_grad(): x.zero_()
            with self.assertRaises(ValueError): fr.autograd.grad(stale, [x])
            self.assertEqual(fr.autograd.grad(actual.sum(), [x])[0].tolist(), [[0., 1., 0., 1., 0., 1.]] * 2)
            source = fr.ones(2, 6, dtype=dtype)
            captured = fr.capture(lambda: source[:, 1::2].view(3, 2)).compile_fused()
            source.zero_(); result = captured.replay()
            self.same(result, torch.zeros(2, 6, dtype=getattr(torch, dtype))[:, 1::2].view(3, 2))
            self.assertTrue(result._shares_storage(source))

    def test_higher_derivative_and_noncontiguous_forward_tangent(self):
        x = fr.tensor([[.2, -.3, .7], [.8, .5, -.2]], requires_grad=True)
        reference = torch.tensor(x.tolist(), requires_grad=True)
        actual, expected = x.view(6), reference.view(6)
        gradient = fr.autograd.grad((actual * actual).sum(), [x], create_graph=True)[0]
        target = torch.autograd.grad((expected * expected).sum(), reference, create_graph=True)[0]
        self.same(gradient, target)
        self.same(fr.autograd.grad(gradient.sum(), [x])[0], torch.autograd.grad(target.sum(), reference)[0])
        direction = fr.arange(6.).reshape(3, 2).T
        target_direction = torch.arange(6.).reshape(3, 2).T
        primal, tangent = fr.func.jvp(lambda a: a.view(-1), (x,), (direction,))
        expected, expected_tangent = torch.func.jvp(lambda a: a.view(-1), (reference,), (target_direction,))
        self.same(primal, expected); self.same(tangent, expected_tangent)


if __name__ == '__main__': unittest.main()
