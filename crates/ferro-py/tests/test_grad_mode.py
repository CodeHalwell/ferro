"""Real native recording controls and sanctioned leaf updates (CPU only)."""
import threading
import unittest
import ferro


class GradModeTests(unittest.TestCase):
    def test_nested_exception_restore_and_real_recording(self):
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        self.assertTrue(ferro.is_grad_enabled())
        with ferro.no_grad():
            self.assertFalse(ferro.is_grad_enabled())
            self.assertFalse((x * x).requires_grad)
            self.assertTrue(x.requires_grad)
            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                with ferro.enable_grad():
                    self.assertTrue((x * x).requires_grad)
                    raise RuntimeError('sentinel')
            self.assertFalse(ferro.is_grad_enabled())
        self.assertTrue(ferro.is_grad_enabled())
        (x * x).backward()
        self.assertEqual(x.grad.tolist(), [4.0])

    def test_leaf_copy_preserves_identity_grad_and_saved_version(self):
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        y = (x * x).sum()
        with self.assertRaises(ValueError):
            x.copy_(ferro.Tensor([3.0], [1]))
        with ferro.no_grad():
            self.assertIs(x.copy_(ferro.Tensor([3.0], [1])), x)
        self.assertTrue(x.requires_grad)
        self.assertEqual(x.tolist(), [3.0])
        with self.assertRaisesRegex(ValueError, 'version|modified|in.place'):
            y.backward()
        (x * x).sum().backward()
        self.assertEqual(x.grad.tolist(), [6.0])

    def test_thread_local_and_entry_time_restoration(self):
        context = ferro.no_grad()
        observations = []
        with ferro.no_grad():
            def worker():
                observations.append(ferro.is_grad_enabled())
                with ferro.no_grad():
                    observations.append(ferro.is_grad_enabled())
                observations.append(ferro.is_grad_enabled())
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            with context:
                self.assertFalse(ferro.is_grad_enabled())
            self.assertFalse(ferro.is_grad_enabled())
        self.assertEqual(observations, [True, False, True])
        self.assertTrue(ferro.is_grad_enabled())

    def test_setter_returns_previous_mode(self):
        previous = ferro.is_grad_enabled()
        try:
            self.assertEqual(ferro.set_grad_enabled(False), previous)
            self.assertFalse(ferro.set_grad_enabled(True))
            self.assertTrue(ferro.is_grad_enabled())
        finally:
            ferro.set_grad_enabled(previous)

    def test_no_grad_preserves_nonleaf_and_layout_mutation_guards(self):
        x = ferro.Tensor.ones([2, 2]).requires_grad_(True)
        nonleaf = x * x
        transposed = x.transpose(0, 1)
        with ferro.no_grad():
            for unsafe in (nonleaf, transposed):
                with self.assertRaises(ValueError):
                    unsafe.copy_(ferro.Tensor.zeros([2, 2]))
            with self.assertRaises(ValueError):
                x.copy_(ferro.Tensor.zeros([4]))
        self.assertEqual(x.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_leaf_copy_keeps_existing_grad_slot_and_host_alias(self):
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        with ferro.no_grad():
            alias = x.reshape([1])
        x.sum().backward()
        with ferro.no_grad():
            x.copy_(ferro.Tensor([4.0], [1]))
        self.assertEqual(x.grad.tolist(), [1.0])
        self.assertEqual(alias.tolist(), [4.0])
        self.assertTrue(x.requires_grad)

    def test_higher_derivatives_inside_no_grad_restore_mode(self):
        from ferro.autograd import grad
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        weight = ferro.Tensor([3.0], [1]).requires_grad_(True)
        y = (weight * x * x).sum()
        with ferro.no_grad():
            dx, = grad(y, [x], create_graph=True)
            self.assertTrue(dx.requires_grad)
            self.assertFalse(ferro.is_grad_enabled())
            dw, = grad(dx, [weight])
            self.assertEqual(dw.tolist(), [4.0])
            self.assertFalse(ferro.is_grad_enabled())


if __name__ == '__main__':
    unittest.main(verbosity=2)
