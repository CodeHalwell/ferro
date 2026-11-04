import itertools
import unittest
import ferro as fr
import torch


class TensorShapeMethods(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=1e-12 if actual.dtype == 'float64' else 1e-6, atol=1e-12 if actual.dtype == 'float64' else 1e-6, equal_nan=True)

    def test_reshape_sequence_variadic_inference_and_strided_gradient(self):
        for dtype in ['float32', 'float64', 'int64', 'bool']:
            x = fr.arange(6).to(dtype=dtype).reshape(2, 3)
            tx = torch.arange(6).to(dtype=getattr(torch, dtype)).reshape(2, 3)
            for args in [([3, 2],), ((3, 2),), (3, 2), (-1, 2), ([-1],), (6,)]:
                self.same(x.reshape(*args), tx.reshape(*args))
            self.same(x.transpose(0, 1).reshape(-1), tx.T.reshape(-1))
            self.same(x.reshape(shape=(3, 2)), tx.reshape(shape=(3, 2)))
            with self.assertRaises(TypeError): x.reshape(3, 2, shape=(3, 2))
        x = fr.tensor([[.2, .4, .6], [.8, 1., 1.2]], dtype='float64', requires_grad=True)
        tx = torch.tensor(x.tolist(), dtype=torch.float64, requires_grad=True)
        x.transpose(0, 1).reshape(2, -1).square().sum().backward()
        tx.T.reshape(2, -1).square().sum().backward()
        self.same(x.grad, tx.grad)
        self.same(fr.tensor(1.).reshape([]), torch.tensor(1.).reshape([]))
        self.same(fr.zeros(0).reshape(2, -1), torch.zeros(0).reshape(2, -1))
        for args in [(), (-1, -1), (0, -1), (-2,), (True, 6), ([2.0, 3],), (5,), (1 << 80,)]:
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError, OverflowError)):
                fr.zeros(6).reshape(*args)

    def test_squeeze_tuple_none_nonunit_and_strided_aliases(self):
        for dims in [None, 0, 1, -1, (0, 2), (), [0, 2]]:
            a = fr.ones(1, 2, 1, 3)
            x = torch.ones(1, 2, 1, 3)
            self.same(a.squeeze() if dims is None else a.squeeze(dims), x.squeeze() if dims is None else x.squeeze(dims))
        for dtype in ['float32', 'float64', 'int64', 'bool']:
            a = fr.zeros((2, 3), dtype=dtype)
            x = torch.zeros((2, 3), dtype=getattr(torch, dtype))
            av = a.transpose(0, 1).unsqueeze(1)
            xv = x.T.unsqueeze(1)
            removed, tr = av.squeeze(1), xv.squeeze(1)
            a.copy_(fr.ones((2, 3), dtype=dtype)); x.copy_(torch.ones((2, 3), dtype=getattr(torch, dtype)))
            self.same(av, xv); self.same(removed, tr)
        for dims in [None, 0, -1, ()]:
            self.same(fr.tensor(2.).squeeze(dims), torch.tensor(2.).squeeze(dims) if dims is not None else torch.tensor(2.).squeeze())
        for dims in [(0, 0), 4, True]:
            with self.assertRaises((TypeError, ValueError)): fr.ones(1, 2, 1).squeeze(dims)

    def test_sum_mean_dim_tuple_dtype_keepdim_and_gradients(self):
        for name, input_dtype, output_dtype, dim, keep in itertools.product(
                ['sum', 'mean'], ['float32', 'float64', 'int64', 'bool'], [None, 'float32', 'float64', 'int64', 'bool'],
                [None, 0, -1, (0, 2), ()], [False, True]):
            with self.subTest(name=name, input_dtype=input_dtype, output_dtype=output_dtype, dim=dim, keepdim=keep):
                a = fr.arange(12).reshape(2, 2, 3).to(dtype=input_dtype)
                x = torch.arange(12).reshape(2, 2, 3).to(dtype=getattr(torch, input_dtype))
                if name == 'mean' and (output_dtype in ['int64', 'bool'] or input_dtype in ['int64', 'bool'] and output_dtype is None):
                    with self.assertRaises((TypeError, ValueError, NotImplementedError)): a.mean(dim, keep, dtype=output_dtype)
                    continue
                kwargs = {} if output_dtype is None else {'dtype': output_dtype}
                tk = {} if output_dtype is None else {'dtype': getattr(torch, output_dtype)}
                self.same(getattr(a, name)(dim, keep, **kwargs), getattr(x, name)(dim, keep, **tk))
        for name in ['sum', 'mean']:
            a = fr.tensor([[[.2, .4], [.6, .8]], [[1., 1.2], [1.4, 1.6]]], requires_grad=True)
            x = torch.tensor(a.tolist(), requires_grad=True)
            getattr(a, name)((0, 2), True, dtype='float64').square().sum().backward()
            getattr(x, name)((0, 2), True, dtype=torch.float64).square().sum().backward()
            self.same(a.grad, x.grad)
        self.same(fr.zeros((2, 0), dtype='float64').mean(1), torch.zeros((2, 0), dtype=torch.float64).mean(1))


if __name__ == '__main__':
    unittest.main()
