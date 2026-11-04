import unittest

import ferro as fr
import torch


class LeafGradAssignment(unittest.TestCase):
    def test_typed_handles_and_independent_gradient_history_are_preserved(self):
        for dtype in ['float32', 'float64']:
            a = fr.ones(2, dtype=dtype, requires_grad=True)
            x = torch.ones(2, dtype=getattr(torch, dtype), requires_grad=True)
            source = fr.tensor([.25, .75], dtype=dtype, requires_grad=True)
            reference = torch.tensor([.25, .75], dtype=x.dtype, requires_grad=True)
            gradient, expected = source * 2, reference * 2
            identity = a._id()
            a._set_leaf_grad(gradient); x.grad = expected
            self.assertEqual(a.grad._id(), gradient._id())
            self.assertEqual(a._id(), identity)
            self.assertEqual(a.grad.requires_grad, x.grad.requires_grad)
            torch.testing.assert_close(torch.tensor(a.grad.tolist(), dtype=x.dtype), x.grad)
            self.assertIsNone(source.grad)
            a.grad.sum().backward(); x.grad.sum().backward()
            torch.testing.assert_close(torch.tensor(source.grad.tolist(), dtype=x.dtype), reference.grad)
            a._set_leaf_grad(); x.grad = None
            self.assertIsNone(a.grad)
            self.assertEqual(a._id(), identity)

    def test_assignment_aliases_gradient_data_without_copying(self):
        a = fr.ones(2, dtype='float64', requires_grad=True)
        gradient = fr.tensor([1. + 2. ** -40, 1e100], dtype='float64')
        a._set_leaf_grad(gradient)
        self.assertEqual(a.grad._id(), gradient._id())
        self.assertEqual(a.grad.tolist(), gradient.tolist())
        gradient.copy_(fr.tensor([3., 5.], dtype='float64'))
        self.assertEqual(a.grad.tolist(), [3., 5.])
        self.assertEqual(a.tolist(), [1., 1.])

    def test_invalid_metadata_and_reference_cycles_are_atomic(self):
        a = fr.ones(2, requires_grad=True)
        prior = fr.full((2,), 3.)
        a._set_leaf_grad(prior)
        for invalid in [fr.ones(1, 2), fr.ones(2, dtype='float64'), fr.ones(2, dtype='int64'), a, a.square()]:
            with self.assertRaises(ValueError): a._set_leaf_grad(invalid)
            self.assertEqual(a.grad._id(), prior._id())
        with self.assertRaises(ValueError): a.square()._set_leaf_grad(prior)
        with self.assertRaises(ValueError): fr.ones(2, dtype='int64')._set_leaf_grad()
        other = fr.full((2,), 5., requires_grad=True)
        a._set_leaf_grad(other)
        with self.assertRaises(ValueError): other._set_leaf_grad(a)
        self.assertEqual(a.grad._id(), other._id())
        self.assertIsNone(other.grad)
        a._set_leaf_grad()

    def test_capture_rejection_preserves_slots_and_restores_mode(self):
        a = fr.ones(2, requires_grad=True)
        gradient = fr.full((2,), 3.)
        a._set_leaf_grad(gradient)
        for replacement in [None, fr.zeros(2)]:
            with self.assertRaisesRegex(ValueError, 'inference capture'):
                fr.capture(lambda: (a._set_leaf_grad(replacement), a * 2)[1])
            self.assertEqual(a.grad._id(), gradient._id())
            self.assertTrue(fr.is_grad_enabled())
        constant = fr.ones(2)
        replay = fr.capture(lambda: constant * 2).compile_fused()
        self.assertEqual(replay.replay().tolist(), [2., 2.])


if __name__ == '__main__':
    unittest.main()
