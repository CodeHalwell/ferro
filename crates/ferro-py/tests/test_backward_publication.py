"""Ordinary publication retains Torch slot identity without retaining callback graphs."""
import unittest
import warnings

import ferro as fr
import torch


class BackwardPublication(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual((actual.requires_grad, actual.is_leaf), (expected.requires_grad, expected.is_leaf))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), atol=0, rtol=0)

    def test_initial_leaf_publication_owns_seed_and_keeps_held_gradient_on_repeat(self):
        for dtype in ('float32', 'float64'):
            for shape in ((), (0,), (2, 3), (0, 3)):
                for transpose in (False, True):
                    if transpose and len(shape) != 2:
                        continue
                    with self.subTest(dtype=dtype, shape=shape, transpose=transpose):
                        a = fr.full(shape, 2., dtype=dtype)
                        x = torch.full(shape, 2., dtype=getattr(torch, dtype))
                        if transpose:
                            a, x = a.T, x.T
                        a.requires_grad_(True); x.requires_grad_()
                        seed = fr.full(a.shape, 3., dtype=dtype, requires_grad=True)
                        reference = torch.full(x.shape, 3., dtype=x.dtype, requires_grad=True)
                        a.backward_with(seed); x.backward(reference)
                        held, expected = a.grad, x.grad
                        self.same(held, expected)
                        self.assertFalse(held._shares_storage(seed))
                        self.assertFalse(held._shares_storage(a))
                        a.backward_with(seed); x.backward(reference)
                        self.assertEqual(a.grad._id(), held._id()); self.assertIs(x.grad, expected)
                        self.same(held, expected)
                        self.assertIsNone(seed.grad); self.assertIsNone(reference.grad)

    def test_manual_slots_retain_history_flags_layout_offset_and_identity(self):
        for dtype in ('float32', 'float64'):
            for kind in ('plain', 'leaf', 'history', 'transpose', 'gapped'):
                with self.subTest(dtype=dtype, kind=kind), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    a = fr.ones(2, 3, dtype=dtype, requires_grad=True)
                    x = torch.ones(2, 3, dtype=getattr(torch, dtype), requires_grad=True)
                    owner = fr.full((2, 3), 3., dtype=dtype, requires_grad=True)
                    base = torch.full((2, 3), 3., dtype=x.dtype, requires_grad=True)
                    if kind == 'history':
                        gradient, expected = owner * 2, base * 2
                    elif kind == 'transpose':
                        gradient = fr.full((3, 2), 7., dtype=dtype).T
                        expected = torch.full((3, 2), 7., dtype=x.dtype).T
                    elif kind == 'gapped':
                        gradient = fr.full((2, 8), 7., dtype=dtype)[:, 1:7:2]
                        expected = torch.full((2, 8), 7., dtype=x.dtype)[:, 1:7:2]
                    else:
                        gradient = fr.full((2, 3), 7., dtype=dtype, requires_grad=kind == 'leaf')
                        expected = torch.full((2, 3), 7., dtype=x.dtype, requires_grad=kind == 'leaf')
                    a.grad = gradient; x.grad = expected
                    for _ in range(2):
                        a.square().sum().backward(); x.square().sum().backward()
                        self.assertEqual(a.grad._id(), gradient._id()); self.assertIs(x.grad, expected)
                        self.same(gradient, expected)
                    if kind == 'history':
                        gradient.sum().backward(); expected.sum().backward()
                        self.same(owner.grad, base.grad)

    def test_shared_manual_gradients_observe_sequential_in_place_updates(self):
        for dtype in ('float32', 'float64'):
            a = fr.ones(2, 3, dtype=dtype, requires_grad=True)
            b = fr.ones(3, 2, dtype=dtype, requires_grad=True)
            x = torch.ones(2, 3, dtype=getattr(torch, dtype), requires_grad=True)
            y = torch.ones(3, 2, dtype=x.dtype, requires_grad=True)
            gradient = fr.full((2, 3), 3., dtype=dtype, requires_grad=True) * 2
            expected = torch.full((2, 3), 3., dtype=x.dtype, requires_grad=True) * 2
            view, target_view = gradient.T, expected.T
            a.grad = gradient; b.grad = view; x.grad = expected; y.grad = target_view
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                (a.sum() * 2 + b.sum() * 3).backward()
                (x.sum() * 2 + y.sum() * 3).backward()
            self.same(gradient, expected); self.same(view, target_view)
            self.assertEqual(a.grad._id(), gradient._id()); self.assertEqual(b.grad._id(), view._id())
            self.assertIs(x.grad, expected); self.assertIs(y.grad, target_view)

    def test_retained_manual_history_still_enforces_its_saved_output_version(self):
        for dtype in ('float32', 'float64'):
            a = fr.ones(2, dtype=dtype, requires_grad=True)
            x = torch.ones(2, dtype=getattr(torch, dtype), requires_grad=True)
            owner = fr.zeros(2, dtype=dtype, requires_grad=True)
            reference = torch.zeros(2, dtype=x.dtype, requires_grad=True)
            gradient, expected = owner.exp(), reference.exp()
            a.grad = gradient; x.grad = expected
            a.sum().backward(); x.sum().backward()
            self.assertEqual(a.grad._id(), gradient._id()); self.assertIs(x.grad, expected)
            self.same(gradient, expected)
            with self.assertRaisesRegex(ValueError, 'modified|version'):
                gradient.sum().backward()
            with self.assertRaisesRegex(RuntimeError, 'modified|version'):
                expected.sum().backward()
            self.assertIsNone(owner.grad); self.assertIsNone(reference.grad)
            self.assertTrue(fr.is_grad_enabled())

    def test_late_invalid_manual_layout_rejects_without_partial_publication(self):
        a = fr.ones(2, 2, requires_grad=True); b = fr.ones(2, 2, requires_grad=True)
        held = fr.full((2, 2), 7.); invalid = fr.ones(1).expand(2, 2)
        a.grad = held; b.grad = invalid
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            (a + b).sum().backward()
        self.assertEqual(held.tolist(), [[7., 7.], [7., 7.]])
        self.assertEqual(a.grad._id(), held._id()); self.assertEqual(b.grad._id(), invalid._id())
        x = torch.ones(2, 2, requires_grad=True); x.grad = torch.ones(1).expand(2, 2)
        with self.assertRaisesRegex(RuntimeError, 'single memory|overlap'):
            x.sum().backward()
        b.grad = fr.zeros(2, 2)
        (a + b).sum().backward()
        self.assertEqual(held.tolist(), [[8., 8.], [8., 8.]])
        self.assertTrue(fr.is_grad_enabled())

    def test_functional_live_passthrough_and_manual_setter_are_unchanged(self):
        a = fr.ones(2, requires_grad=True); x = torch.ones(2, requires_grad=True)
        owner = fr.full((2,), 3., requires_grad=True); base = torch.full((2,), 3., requires_grad=True)
        seed, reference = owner * 2, base * 2
        held, expected = fr.full((2,), 7., requires_grad=True), torch.full((2,), 7., requires_grad=True)
        a.grad = held; x.grad = expected
        with fr.no_grad(), torch.no_grad():
            actual = fr.autograd.grad(a, [a], grad_outputs=seed, create_graph=True)[0]
            target = torch.autograd.grad(x, [x], grad_outputs=reference, create_graph=True)[0]
            self.assertEqual(actual._id(), seed._id()); self.assertIs(target, reference)
            self.assertEqual(a.grad._id(), held._id()); self.assertIs(x.grad, expected)
            a.backward_with(seed); x.backward(reference)
            self.assertFalse(fr.is_grad_enabled())
        self.same(held, expected)
        self.assertTrue(held.requires_grad)
        self.assertTrue(fr.is_grad_enabled())


if __name__ == '__main__':
    unittest.main()
