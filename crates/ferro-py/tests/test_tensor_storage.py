import unittest
import ferro as fr
import ferro.autograd as ad
import torch


class TensorStorage(unittest.TestCase):
    def same(self, actual, expected, layout=False):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected.detach(), rtol=1e-12 if actual.dtype == 'float64' else 1e-6,
                                   atol=1e-12 if actual.dtype == 'float64' else 1e-6, equal_nan=True)
        if layout:
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())

    def test_detach_shares_values_versions_but_not_gradient_state(self):
        for dtype in ['float32', 'float64', 'int64', 'bool']:
            a = fr.ones((2, 2), dtype=dtype)
            x = torch.ones((2, 2), dtype=getattr(torch, dtype))
            if dtype.startswith('float'):
                a = a.requires_grad_(True); x.requires_grad_(True)
            detached, td = a.transpose(0, 1).detach(), x.T.detach()
            self.same(detached, td, True)
            self.assertFalse(detached.requires_grad)
            self.assertTrue(detached.is_leaf)
            with fr.no_grad(), torch.no_grad():
                a.copy_(fr.zeros((2, 2), dtype=dtype)); x.copy_(torch.zeros((2, 2), dtype=x.dtype))
            self.same(detached, td, True)
            self.assertIsNone(detached.grad)
        a = fr.tensor([2., 3.], dtype='float64', requires_grad=True)
        x = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
        loss, tloss = a.square().sum(), x.square().sum()
        a.detach().copy_(fr.ones(2, dtype='float64')); x.detach().copy_(torch.ones(2, dtype=torch.float64))
        with self.assertRaises(ValueError): loss.backward()
        with self.assertRaises(RuntimeError): tloss.backward()
        self.assertIsNone(a.grad)

    def test_clone_and_contiguous_preserve_or_materialize_actual_layouts(self):
        for dtype in ['float32', 'float64', 'int64', 'bool']:
            a = fr.arange(12).to(dtype=dtype).reshape(3, 4)
            x = torch.arange(12).to(dtype=getattr(torch, dtype)).reshape(3, 4)
            cases = [(a, x), (a.transpose(0, 1), x.T), (a.narrow(0, 1, 2), x.narrow(0, 1, 2)),
                     (a[:, ::2], x[:, ::2]), (a.narrow(0, 0, 1).expand(5, 4), x[:1].expand(5, 4))]
            for av, xv in cases:
                self.same(av, xv, True)
                for fmt, tfmt in [('preserve_format', torch.preserve_format), ('contiguous_format', torch.contiguous_format)]:
                    copied = av.clone(memory_format=fmt)
                    self.same(copied, xv.clone(memory_format=tfmt), True)
                    self.assertIsNot(copied, av)
                packed = av.contiguous()
                self.same(packed, xv.contiguous(), True)
                self.assertEqual(packed is av, xv.contiguous() is xv)
            copied = a.clone()
            a.copy_(fr.zeros((3, 4), dtype=dtype))
            self.same(copied, x, True)
        for shape in [(2, 0, 3), (0, 3, 2), (2, 3, 0), ()]:
            a, x = fr.zeros(shape), torch.zeros(shape)
            if len(shape) > 1: a, x = a.transpose(0, -1), x.transpose(0, -1)
            self.same(a, x, True)
            self.assertIs(a.contiguous(), a)
            self.same(a.clone(), x.clone(), True)
        with self.assertRaises(ValueError): fr.zeros(2, 3).transpose(0, 1).contiguous(memory_format='preserve_format')

    def test_clone_has_identity_adjoint_without_saving_original_values(self):
        for dtype in ['float32', 'float64']:
            a = fr.tensor([2., 3.], dtype=dtype, requires_grad=True)
            x = torch.tensor([2., 3.], dtype=getattr(torch, dtype), requires_grad=True)
            copied, tc = a.clone(), x.clone()
            loss, tloss = copied.square().sum(), tc.square().sum()
            with fr.no_grad(), torch.no_grad():
                a.copy_(fr.full((2,), 7., dtype=dtype)); x.copy_(torch.full((2,), 7., dtype=x.dtype))
            loss.backward(); tloss.backward()
            self.same(a.grad, x.grad)
            self.same(copied, tc)
            stale, ts = copied.square().sum(), tc.square().sum()
            copied.detach().copy_(fr.zeros(2, dtype=dtype)); tc.detach().copy_(torch.zeros(2, dtype=x.dtype))
            with self.assertRaises(ValueError): stale.backward()
            with self.assertRaises(RuntimeError): ts.backward()
        a = fr.tensor([[.2, .4], [.6, .8]], dtype='float64', requires_grad=True)
        x = torch.tensor(a.tolist(), dtype=torch.float64, requires_grad=True)
        a.transpose(0, 1).contiguous().sum().backward(); x.T.contiguous().sum().backward()
        self.same(a.grad, x.grad)

    def test_empty_offset_views_materialize_and_compute_without_reading_storage(self):
        for dtype in ['float32', 'float64', 'int64', 'bool']:
            with self.subTest(dtype=dtype):
                a = fr.zeros((0, 8), dtype=dtype)
                x = torch.zeros((0, 8), dtype=getattr(torch, dtype))
                av, xv = a.narrow(1, 4, 4), x.narrow(1, 4, 4)
                self.same(av, xv, True)
                self.assertEqual(av.storage_offset(), 4)
                self.same(av.detach(), xv.detach(), True)
                self.same(av.clone(), xv.clone(), True)
                dst, td = fr.zeros((0, 4), dtype=dtype), torch.zeros((0, 4), dtype=x.dtype)
                dst.copy_(av); td.copy_(xv)
                self.same(dst, td)
                if dtype.startswith('float'):
                    a.requires_grad_(True); x.requires_grad_(True)
                    av, xv = a.narrow(1, 4, 4), x.narrow(1, 4, 4)
                    w = fr.ones((4, 2), dtype=dtype, requires_grad=True)
                    tw = torch.ones((4, 2), dtype=x.dtype, requires_grad=True)
                    out, reference = av.sigmoid() * av + 1, xv.sigmoid() * xv + 1
                    self.same(av.sum(), xv.sum())
                    self.same(out, reference)
                    loss = (out @ w).sum() + (av.reshape(0, 4) @ w).sum()
                    expected = (reference @ tw).sum() + (xv.reshape(0, 4) @ tw).sum()
                    loss.backward(); expected.backward()
                    self.same(a.grad, x.grad); self.same(w.grad, tw.grad)
                    if dtype == 'float32':
                        dst.add_(av.detach()); td.add_(xv.detach())
                        self.same(dst, td)

    def test_no_grad_view_flags_and_missing_edges_match_torch(self):
        for operation, torch_operation in [(lambda x:x.reshape(4), lambda x:x.reshape(4)),
                (lambda x:x.transpose(0, 1), lambda x:x.T), (lambda x:x.unsqueeze(0), lambda x:x.unsqueeze(0)),
                (lambda x:x.permute(1, 0), lambda x:x.permute(1, 0)), (lambda x:x.narrow(0, 0, 1), lambda x:x.narrow(0, 0, 1))]:
            a = fr.ones((2, 2), requires_grad=True)
            x = torch.ones((2, 2), requires_grad=True)
            with fr.no_grad(), torch.no_grad(): av, xv = operation(a), torch_operation(x)
            self.assertEqual(av.requires_grad, xv.requires_grad)
            self.assertEqual(av.is_leaf, xv.is_leaf)
            av.sum().backward(); xv.sum().backward()
            self.assertIsNone(av.grad); self.assertIsNone(xv.grad)
            self.assertIsNone(a.grad); self.assertIsNone(x.grad)
            with self.assertRaises(ValueError): av.backward_with(fr.ones(av.shape))
            with self.assertRaises(RuntimeError): xv.backward(torch.ones_like(xv))
            enabled, te = av.requires_grad_(True), xv.requires_grad_(True)
            enabled.sum().backward(); te.sum().backward()
            self.same(enabled.grad, te.grad)
            self.assertIsNone(a.grad); self.assertIsNone(x.grad)
        with fr.no_grad(), torch.no_grad():
            a = fr.ones((2, 3), requires_grad=True); x = torch.ones((2, 3), requires_grad=True)
            self.assertFalse(a.clone().requires_grad)
            self.assertFalse(a.transpose(0, 1).contiguous().requires_grad)
            self.assertEqual(a.transpose(0, 1).reshape(-1).requires_grad, x.T.reshape(-1).requires_grad)
            self.assertIs(a.contiguous(), a)

    def test_requires_grad_same_flag_preserves_identity_history_and_gradients(self):
        for dtype in ['float32', 'float64']:
            a = fr.tensor([2., 3.], dtype=dtype, requires_grad=True)
            x = torch.tensor([2., 3.], dtype=getattr(torch, dtype), requires_grad=True)
            loss, expected = (a * a).sum(), (x * x).sum()
            identity = a._id()
            self.assertIs(a.requires_grad_(True), a)
            self.assertIs(x.requires_grad_(True), x)
            self.assertEqual(a._id(), identity)
            loss.backward(); expected.backward(retain_graph=True)
            self.same(a.grad, x.grad)
            gradient = a.grad._id()
            a.requires_grad_(True); x.requires_grad_(True)
            self.assertEqual(a.grad._id(), gradient)
            loss.backward(); expected.backward()
            self.same(a.grad, x.grad)
            out, reference = a * a, x * x
            self.assertIs(out.requires_grad_(True), out)
            self.assertIs(reference.requires_grad_(True), reference)
            with self.assertRaises(ValueError): out.requires_grad_(False)
            with self.assertRaises(RuntimeError): reference.requires_grad_(False)
            if dtype == 'float32':
                parameter, expected_parameter = fr.nn.Parameter(out), torch.nn.Parameter(reference)
                self.assertEqual(parameter.is_leaf, expected_parameter.is_leaf)
                parameter.sum().backward(); expected_parameter.sum().backward()
                self.same(parameter.grad, expected_parameter.grad)
            out.sum().backward(); reference.sum().backward()
            self.same(a.grad, x.grad)
        constant = fr.ones(2, dtype='int64')
        identity = constant._id()
        self.assertIs(constant.requires_grad_(False), constant)
        self.assertEqual(constant._id(), identity)

    def test_saved_output_versions_are_checked_without_blanket_output_rejection(self):
        operations = ['relu', 'exp', 'exp2', 'expm1', 'sigmoid', 'tanh', 'sqrt', 'reciprocal', 'softmax', 'log_softmax', 'max']
        for dtype in ['float32', 'float64']:
            for name in operations:
                with self.subTest(dtype=dtype, op=name):
                    a = fr.tensor([.5, 1.5], dtype=dtype, requires_grad=True)
                    x = torch.tensor([.5, 1.5], dtype=getattr(torch, dtype), requires_grad=True)
                    args = (0,) if 'softmax' in name else ()
                    out, reference = getattr(a, name)(*args), getattr(x, name)(*args)
                    out.detach().copy_(fr.zeros(out.shape, dtype=dtype)); reference.detach().copy_(torch.zeros_like(reference))
                    with self.assertRaises(ValueError): ad.grad(out.sum(), [a])
                    with self.assertRaises(RuntimeError): torch.autograd.grad(reference.sum(), [x])
                    self.assertIsNone(a.grad); self.assertIsNone(x.grad)
        a = fr.tensor([2., 3.], requires_grad=True); x = torch.tensor([2., 3.], requires_grad=True)
        out, reference = a * a, x * x
        out.detach().copy_(fr.zeros(2)); reference.detach().copy_(torch.zeros(2))
        out.sum().backward(); reference.sum().backward()
        self.same(a.grad, x.grad)

    def test_clone_replay_returns_independent_values(self):
        a = fr.tensor([2., 3.], dtype='float64')
        cloned = fr.capture(lambda:a.clone())
        replay = cloned.compile_fused()
        a.copy_(fr.tensor([4., 5.], dtype='float64'))
        first = replay.replay()
        a.copy_(fr.tensor([6., 7.], dtype='float64'))
        self.assertEqual(first.tolist(), [4., 5.])
        self.assertEqual(replay.replay().tolist(), [6., 7.])


if __name__ == '__main__':
    unittest.main()
