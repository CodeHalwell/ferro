"""Square derivatives and target-aware unsigned alpha follow actual Torch."""
import unittest
import torch
import ferro as fr


def native(value, grad=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(grad)


class ArithmeticAdjointRepairTests(unittest.TestCase):
    def same(self, actual, expected, exact=False):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=0 if exact else 4e-6, rtol=0 if exact else 4e-5, equal_nan=True)
        zero = expected.detach() == 0
        self.assertEqual(torch.signbit(value[zero]).tolist(), torch.signbit(expected.detach()[zero]).tolist())

    def test_unsigned_alpha_forward_and_only_requested_right_adjoint_rejection(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((), (2,), (0, 2)):
                for operation in ('add', 'sub'):
                    for alpha in (2**63, 2**64-1):
                        with self.subTest(dtype=dtype, shape=shape, operation=operation, alpha=alpha):
                            x = torch.ones(shape, dtype=dtype, requires_grad=True)
                            y = torch.ones(shape, dtype=dtype, requires_grad=True)
                            a, b = native(x, True), native(y, True)
                            actual = getattr(fr, operation)(a, b, alpha=alpha)
                            expected = getattr(torch, operation)(x, y, alpha=alpha)
                            self.same(actual, expected, True)
                            for graph in (False, True) if dtype == torch.float32 else (False,):
                                left = fr.autograd.vjp(actual, [a], fr.ones(shape, dtype=a.dtype), create_graph=graph)[0]
                                reference = torch.autograd.grad(expected, x, torch.ones_like(x), create_graph=graph, retain_graph=True)[0]
                                self.same(left, reference, True)
                                with self.assertRaises(ValueError):
                                    fr.autograd.vjp(actual, [b], fr.ones(shape, dtype=a.dtype), create_graph=graph)
                                with self.assertRaises(RuntimeError):
                                    torch.autograd.grad(expected, y, torch.ones_like(y), create_graph=graph, retain_graph=True)
                            self.assertIsNone(a.grad); self.assertIsNone(b.grad)

    def test_unsigned_alpha_backward_failure_is_atomic_and_jvp_rejects(self):
        a = fr.ones(2, requires_grad=True); b = fr.ones(2, requires_grad=True)
        a.sum().backward(); b.sum().backward()
        before = [(p.grad._id(), p.grad.tolist(), p.tolist()) for p in (a, b)]
        result = fr.add(a, b, alpha=2**63).sum()
        with self.assertRaises(ValueError): result.backward()
        self.assertEqual([(p.grad._id(), p.grad.tolist(), p.tolist()) for p in (a, b)], before)
        a.grad = None
        fr.add(a, b.detach(), alpha=2**63).sum().backward()
        self.assertEqual(a.grad.tolist(), [1., 1.])
        for side in ('left', 'right', 'both'):
            with self.subTest(side=side):
                fun = (lambda x: fr.add(x, fr.ones(2), alpha=2**63)) if side == 'left' else (lambda y: fr.add(fr.ones(2), y, alpha=2**63)) if side == 'right' else (lambda x, y: fr.add(x, y, alpha=2**63))
                reference = (lambda x: torch.add(x, torch.ones(2), alpha=2**63)) if side == 'left' else (lambda y: torch.add(torch.ones(2), y, alpha=2**63)) if side == 'right' else (lambda x, y: torch.add(x, y, alpha=2**63))
                count = 2 if side == 'both' else 1
                with self.assertRaises(ValueError): fr.func.jvp(fun, tuple(fr.ones(2) for _ in range(count)), tuple(fr.ones(2) for _ in range(count)))
                with self.assertRaises(RuntimeError): torch.func.jvp(reference, tuple(torch.ones(2) for _ in range(count)), tuple(torch.ones(2) for _ in range(count)))
        self.assertTrue(fr.is_grad_enabled())

    def test_square_composed_higher_derivatives_and_live_cotangents(self):
        for alpha in (0., .3, -2.):
            x = torch.tensor([[.2, -.4], [.7, .9]], requires_grad=True)
            y = torch.tensor([[.3], [.6]], requires_grad=True)
            a, b = native(x, True), native(y, True)
            actual = fr.add(a, b, alpha=alpha).tanh().square().sum()
            expected = torch.add(x, y, alpha=alpha).tanh().square().sum()
            for order in range(3):
                gradients = fr.autograd.grad(actual, (a, b), create_graph=order < 2)
                references = torch.autograd.grad(expected, (x, y), create_graph=order < 2)
                for value, target in zip(gradients, references): self.same(value, target)
                if order < 2:
                    actual = gradients[0].sum() + gradients[1].sum()
                    expected = references[0].sum() + references[1].sum()
        x = torch.tensor([.2, -.7], requires_grad=True); seed = torch.tensor([.3, .4], requires_grad=True)
        a, g = native(x, True), native(seed, True)
        first = fr.autograd.vjp(a.square(), [a], g, create_graph=True)[0]
        target = torch.autograd.grad(x.square(), x, seed, create_graph=True)[0]
        for value, expected in zip(fr.autograd.grad(first.sum(), [a, g]), torch.autograd.grad(target.sum(), (x, seed))): self.same(value, expected)

    def test_square_adjoint_and_forward_rounding_match_torch_extremes(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([torch.finfo(dtype).max, torch.finfo(dtype).max, -0., 1e-30], dtype=dtype, requires_grad=True)
            seed = torch.tensor([0., .25, 1., 1e30], dtype=dtype)
            a, g = native(x, True), native(seed)
            for graph in (False, True) if dtype == torch.float32 else (False,):
                actual = fr.autograd.vjp(a.square(), [a], g, create_graph=graph)[0]
                expected = torch.autograd.grad(x.square(), x, seed, create_graph=graph)[0]
                self.same(actual, expected, True)
        x = torch.tensor([torch.finfo(torch.float32).max, torch.finfo(torch.float32).max, -.0])
        for dtype in (torch.float32, torch.float64):
            v = torch.tensor([0., .25, 1.], dtype=dtype)
            actual = fr.func.jvp(lambda a: a.square(), (native(x),), (native(v),))
            expected = torch.func.jvp(lambda a: a.square(), (x,), (v,))
            for value, target in zip(actual, expected): self.same(value, target, True)

    def test_square_input_versions_remain_guarded_and_output_values_are_not_saved(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([.2, .4], dtype=dtype, requires_grad=True); a = native(x, True)
            actual, expected = a.square(), x.square()
            with fr.no_grad(), torch.no_grad(): actual.detach().zero_(); expected.detach().zero_()
            self.same(fr.autograd.grad(actual.sum(), [a])[0], torch.autograd.grad(expected.sum(), x, retain_graph=True)[0], True)
            with fr.no_grad(), torch.no_grad(): a.fill_(9); x.fill_(9)
            with self.assertRaises(ValueError): fr.autograd.grad(actual.sum(), [a])
            with self.assertRaises(RuntimeError): torch.autograd.grad(expected.sum(), x)
            self.assertIsNone(a.grad)


if __name__ == '__main__': unittest.main()
