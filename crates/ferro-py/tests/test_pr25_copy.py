"""Copy transfer semantics and mutation safety on the real native backend."""
import os
import unittest
import ferro


class CopyReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = ['cpu']
        if ferro.cuda_is_available() and ferro.cuda_init(0):
            cls.devices.append('cuda')
        elif os.environ.get('FERRO_REQUIRE_CUDA'):
            raise AssertionError('CUDA required, not available')

    def tensor(self, values, device, grad=False):
        return ferro.Tensor(values, [len(values)]).to(device).requires_grad_(grad)

    def test_cross_device_non_grad_copy_in_both_modes(self):
        for source in self.devices:
            for destination in self.devices:
                for enabled in (True, False):
                    with self.subTest(source=source, destination=destination, enabled=enabled):
                        x = self.tensor([0., 0.], destination)
                        y = self.tensor([3., -4.], source)
                        with (ferro.enable_grad() if enabled else ferro.no_grad()):
                            self.assertIs(x.copy_(y), x)
                        self.assertEqual(x.tolist(), [3., -4.])
                        self.assertFalse(x.requires_grad)

    def test_leaf_updates_and_stale_versions(self):
        for device in self.devices:
            with self.subTest(device=device):
                x = self.tensor([2., 3.], device, True)
                source = self.tensor([4., 5.], device)
                with self.assertRaises(ValueError):
                    x.copy_(source)
                with ferro.no_grad():
                    self.assertIs(x.copy_(source), x)
                self.assertTrue(x.requires_grad)
                self.assertEqual(x.tolist(), [4., 5.])
                y = (x * x).sum()
                with ferro.no_grad():
                    x.copy_(source)
                with self.assertRaisesRegex(ValueError, 'version|modified|in.place'):
                    y.backward()
                (x * x).sum().backward()
                self.assertEqual(x.grad.tolist(), [8., 10.])

    def test_nonleaf_and_cross_device_leaf_rejected(self):
        for device in self.devices:
            x = self.tensor([2., 3.], device, True)
            nonleaf = x * x
            with ferro.no_grad():
                with self.assertRaises(ValueError):
                    nonleaf.copy_(self.tensor([0., 0.], device))
                for other in self.devices:
                    if other != device:
                        with self.assertRaises(ValueError):
                            x.copy_(self.tensor([0., 0.], other))
            self.assertEqual(x.tolist(), [2., 3.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
