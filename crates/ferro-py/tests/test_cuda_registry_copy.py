"""Old CUDA allocations support host reads, not general cross-owner compute."""
import os
import subprocess
import sys
import textwrap
import unittest


class CUDARegistryCopy(unittest.TestCase):
    def test_old_allocations_remain_host_readable(self):
        child = textwrap.dedent('''
            import gc
            import os
            import unittest
            import ferro as fr
            check = unittest.TestCase()
            def check_values(actual, expected):
                check.assertIs(type(actual), type(expected))
                if isinstance(expected, list):
                    check.assertEqual(len(actual), len(expected))
                    for got, want in zip(actual, expected):
                        check_values(got, want)
                else:
                    check.assertEqual(actual, expected)
            try:
                initialized = fr.cuda_init(0)
            except Exception as exc:
                if os.environ.get("FERRO_REQUIRE_CUDA"):
                    raise
                print("CUDA unavailable:", exc)
                raise SystemExit(77)
            check.assertTrue(initialized)
            values = [1., 2., 3., 4., 5., 6.]
            x = fr.Tensor(values, [2, 3]).cuda()
            view = x.transpose(0, 1)
            ints = [-(1 << 63), -(1 << 63) + 1, -(1 << 54) - 1,
                    (1 << 53) + 1, (1 << 54) + 1, (1 << 63) - 1]
            ix = fr.Tensor.from_i64(ints, [2, 3]).cuda()
            iview = ix.transpose(0, 1)
            rank3 = ix.reshape([1, 2, 3]).transpose(0, 2)
            scalar = fr.Tensor.from_i64([(1 << 63) - 1], []).cuda()
            empty = fr.Tensor([], [2, 0]).cuda()
            empty_i64 = fr.Tensor.from_i64([], [0, 3]).cuda()
            empty_i64_view = empty_i64.transpose(0, 1)
            cases = [
                (x, [[1., 2., 3.], [4., 5., 6.]]),
                (view, [[1., 4.], [2., 5.], [3., 6.]]),
                (ix, [ints[:3], ints[3:]]),
                (iview, [[ints[i], ints[i + 3]] for i in range(3)]),
                (rank3, [[[ints[i]], [ints[i + 3]]] for i in range(3)]),
                (scalar, (1 << 63) - 1),
                (empty, [[], []]), (empty_i64, []),
                (empty_i64_view, [[], [], []]),
            ]
            for generation in range(4):
                if generation:
                    check.assertTrue(fr.cuda_init(0))
                    gc.collect()
                for tensor, expected in cases:
                    check_values(tensor.tolist(), expected)
                    check_values(tensor.cpu().tolist(), expected)
                fresh = fr.Tensor([3., -2.], [2]).cuda()
                check_values(fresh.tolist(), [3., -2.])
            print("old f32/i64/strided/empty host reads and fresh allocations passed")
        ''')
        flags = ['-B'] + (['-O'] if sys.flags.optimize else [])
        result = subprocess.run([sys.executable, *flags, '-c', child], capture_output=True,
                                text=True, timeout=90, env=os.environ.copy())
        if result.returncode == 77 and not os.environ.get('FERRO_REQUIRE_CUDA'):
            self.skipTest(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
