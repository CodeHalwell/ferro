"""Native optional convolution arguments agree with facade defaults."""
import unittest
import ferro
from ferro import _native
from ferro.nn import functional as F


class NativeConvolutionDefaultsTests(unittest.TestCase):
    def test_defaults_and_keyword_geometry_match_facade(self):
        x = ferro.Tensor([1., 2., 3., 4.], [1, 1, 2, 2]).requires_grad_(True)
        w = ferro.Tensor([2.], [1, 1, 1, 1]).requires_grad_(True)
        result = _native.conv2d_options(x, w)
        self.assertEqual(result.tolist(), F.conv2d(x, w).tolist())
        result.sum().backward()
        self.assertEqual(x.grad.tolist(), [[[[2., 2.], [2., 2.]]]])
        self.assertEqual(w.grad.tolist(), [[[[10.]]]])
        self.assertEqual(_native.conv2d_options(x, w, stride=[2, 2]).tolist(),
                         F.conv2d(x, w, stride=2).tolist())
        with self.assertRaises(ValueError):
            _native.conv2d_options(x, w, groups=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
