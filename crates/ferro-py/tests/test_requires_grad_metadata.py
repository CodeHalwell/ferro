import unittest

import ferro as fr
import torch


class RequiresGradMetadata(unittest.TestCase):
    def pair(self, dtype, requires_grad=False):
        a = fr.tensor([2., 3.], dtype=dtype, requires_grad=requires_grad)
        x = torch.tensor([2., 3.], dtype=getattr(torch, dtype), requires_grad=requires_grad)
        return a, x

    def same(self, actual, expected):
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=0, atol=0)

    def test_leaf_toggles_preserve_identity_slots_and_preexisting_graphs(self):
        for dtype in ('float32', 'float64'):
            a, x = self.pair(dtype, True)
            identity = a._id()
            loss, reference = a.square().sum(), x.square().sum()
            loss.backward(); reference.backward(retain_graph=True)
            saved = a.grad._id()
            self.assertIs(a.requires_grad_(False), a)
            self.assertIs(x.requires_grad_(False), x)
            self.assertEqual(a._id(), identity)
            self.assertEqual(a.grad._id(), saved)
            loss.backward(); reference.backward(retain_graph=True)
            self.same(a.grad, x.grad)
            self.assertEqual(a.grad._id(), saved)
            self.assertIs(a.requires_grad_(), a)
            self.assertIs(x.requires_grad_(), x)
            self.assertEqual(a._id(), identity)
            loss.backward(); reference.backward()
            self.same(a.grad, x.grad)

    def test_late_enabled_operand_has_no_old_edge_but_keeps_saved_value_checks(self):
        for dtype in ('float32', 'float64'):
            a, x = self.pair(dtype)
            b, y = self.pair(dtype, True)
            loss, reference = (a * b).sum(), (x * y).sum()
            aid = a._id()
            a.requires_grad_(True); x.requires_grad_(True)
            self.assertEqual(a._id(), aid)
            self.assertFalse(loss._depends_on(a))
            self.assertTrue(loss._depends_on(b))
            self.assertFalse(loss._has_grad_leaf_except([b]))
            self.assertTrue(loss._has_current_grad_leaf_except([b]))
            expected = torch.autograd.grad(reference, (x, y), allow_unused=True, retain_graph=True)
            actual = loss.grad_wrt([a, b])
            self.assertIsNone(expected[0])
            self.assertEqual(actual[0].tolist(), [0., 0.])
            self.same(actual[1], expected[1])
            loss.backward(); reference.backward(retain_graph=True)
            self.assertIsNone(a.grad); self.assertIsNone(x.grad)
            self.same(b.grad, y.grad)
            with fr.no_grad(), torch.no_grad():
                a.copy_(fr.zeros(2, dtype=dtype)); x.zero_()
            with self.assertRaises(ValueError): loss.backward()
            with self.assertRaises(RuntimeError): reference.backward()

    def test_late_base_flags_do_not_invent_edges_for_preexisting_views(self):
        for dtype in ('float32', 'float64'):
            for enabled in (False, True):
                a, x = self.pair(dtype)
                with (fr.enable_grad() if enabled else fr.no_grad()), torch.set_grad_enabled(enabled):
                    view, expected = a.reshape(1, 2), x.reshape(1, 2)
                vid = view._id()
                for flag in (True, False, True):
                    a.requires_grad_(flag); x.requires_grad_(flag)
                    self.same(view, expected)
                    self.assertEqual(view._id(), vid)
                    loss, reference = view.square().sum(), expected.square().sum()
                    self.same(loss, reference)
                    if flag:
                        self.assertEqual(torch.autograd.grad(reference, (expected, x), allow_unused=True, retain_graph=True), (None, None))
                        self.assertEqual(loss.grad_wrt([view, a])[0].tolist(), [[0., 0.]])
                        loss.backward(); reference.backward()
                        self.assertIsNone(view.grad); self.assertIsNone(a.grad)
                        self.assertIsNone(expected.grad); self.assertIsNone(x.grad)
                with self.assertRaises(ValueError): view.backward_with(fr.ones(view.shape, dtype=dtype))
                with self.assertRaises(RuntimeError): expected.backward(torch.ones_like(expected))

    def test_old_view_edges_survive_disable_but_new_edges_do_not_appear(self):
        for dtype in ('float32', 'float64'):
            a, x = self.pair(dtype)
            view, expected = a.reshape(1, 2), x.reshape(1, 2)
            identity = view._id()
            view.requires_grad_(True); expected.requires_grad_()
            old, reference = view.square().sum(), expected.square().sum()
            a.requires_grad_(True); x.requires_grad_()
            view.requires_grad_(False); expected.requires_grad_(False)
            self.assertEqual(view._id(), identity)
            self.same(view, expected)
            self.assertTrue(old._depends_on(view))
            self.same(old.grad_wrt([view])[0], torch.autograd.grad(reference, expected, retain_graph=True)[0])
            old.backward(); reference.backward()
            self.same(view.grad, expected.grad)
            self.assertIsNone(a.grad); self.assertIsNone(x.grad)
            new, tnew = view.square().sum(), expected.square().sum()
            self.assertFalse(new._depends_on(view))
            new.backward(); tnew.backward()
            self.same(view.grad, expected.grad)
            with self.assertRaises(ValueError): view.backward_with(fr.ones(view.shape, dtype=dtype))
            with self.assertRaises(RuntimeError): expected.backward(torch.ones_like(expected))

    def test_no_grad_children_follow_ultimate_base_and_copies_reset_provenance(self):
        for dtype in ('float32', 'float64'):
            a, x = self.pair(dtype)
            view, expected = a.reshape(1, 2), x.reshape(1, 2)
            view.requires_grad_(True); expected.requires_grad_()
            recorded, trecorded = view.T, expected.T
            with fr.no_grad(), torch.no_grad():
                child, tchild = view.T, expected.T
                copied, tcopied = recorded.clone(), trecorded.clone()
            detached, tdetached = recorded.detach(), trecorded.detach()
            self.same(child, tchild)
            self.assertFalse(child.requires_grad)
            view.requires_grad_(False); expected.requires_grad_(False)
            self.assertTrue(recorded._has_grad_leaf_except([]))
            self.assertFalse(recorded._has_current_grad_leaf_except([]))
            for flag in (True, False, True):
                a.requires_grad_(flag); x.requires_grad_(flag)
                self.same(child, tchild)
                self.same(copied, tcopied)
                self.same(detached, tdetached)
                self.same(recorded, trecorded)

    def test_functional_cotangent_view_uses_original_base_provenance(self):
        for owning in (False, True):
            a = fr.ones((2, 3), requires_grad=True)
            x = torch.ones((2, 3), requires_grad=True)
            _, pullback = fr.func.vjp(lambda value: value.T, a)
            _, reference = torch.func.vjp(lambda value: value.T, x)
            base = fr.tensor([.1, .2, .3, .4, .5, .6])
            tb = torch.tensor([.1, .2, .3, .4, .5, .6])
            seed, ts = base.reshape(3, 2), tb.reshape(3, 2)
            if owning: seed, ts = seed.clone(), ts.clone()
            seed.requires_grad_(True); ts.requires_grad_()
            with fr.no_grad(), torch.no_grad():
                actual, expected = pullback(seed, create_graph=False)[0], reference(ts, create_graph=False)[0]
            self.same(actual, expected)
            self.assertEqual(actual.requires_grad, owning)

    def test_invalid_flags_and_nonrequiring_backward_are_atomic(self):
        for dtype in ('bool', 'int64'):
            a = fr.ones(2, dtype=dtype)
            identity = a._id()
            with self.assertRaises(ValueError): a.requires_grad_(True)
            self.assertEqual(a._id(), identity)
            self.assertFalse(a.requires_grad)
        a, x = self.pair('float64', True)
        out, expected = a.square().sum(), x.square().sum()
        self.assertIs(out.requires_grad_(True), out)
        with self.assertRaises(ValueError): out.requires_grad_(False)
        with self.assertRaises(RuntimeError): expected.requires_grad_(False)
        self.same(out, expected)
        for initially_enabled in (False, True):
            constant = fr.tensor(2., requires_grad=initially_enabled)
            reference = torch.tensor(2., requires_grad=initially_enabled)
            if initially_enabled:
                constant.requires_grad_(False); reference.requires_grad_(False)
            with self.assertRaises(ValueError): constant.backward()
            with self.assertRaises(RuntimeError): reference.backward()
            self.assertIsNone(constant.grad)

    def test_keyword_default_alias_and_inference_replay(self):
        value = fr.tensor([2., 3.])
        identity = value._id()
        self.assertIs(value.requires_grad_(requires_grad=True), value)
        self.assertEqual(value._id(), identity)
        value.requires_grad_(req=False)
        with self.assertRaises(TypeError): value.requires_grad_(True, req=False)
        self.assertFalse(value.requires_grad)
        replay = fr.capture(lambda: value.reshape(1, 2)).compile_fused()
        value.requires_grad_()
        result = replay.replay()
        self.assertEqual(result.tolist(), [[2., 3.]])
        self.assertFalse(result.requires_grad)
        self.assertIsNone(value.grad)


if __name__ == '__main__':
    unittest.main()
