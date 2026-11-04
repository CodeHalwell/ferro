"""Empty broadcast adjoints retain provenance without huge intermediate axes."""
import subprocess
import sys
import unittest

import torch
import ferro as fr
from ferro.autograd import vjp


def native(value, requires_grad=False):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).split('.')[-1]).reshape(list(value.shape)).requires_grad_(requires_grad)


def operation(module, a, b, kind):
    if kind == 'mul': return a * b
    if kind == 'add': return a + b
    if kind == 'alpha': return module.add(a, b, alpha=2.5)
    return module.sub(a, b, alpha=2.5)


def close(test, actual, expected):
    test.assertEqual(actual.shape, list(expected.shape))
    test.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
    value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(value, expected.detach(), atol=0, rtol=0)


def huge_worker():
    torch.set_num_threads(1)
    count = 0
    for dtype in (torch.float32, torch.float64):
        for kind in ('mul', 'add', 'alpha', 'sub_alpha'):
            ta = torch.empty(0, 1 << 60, dtype=dtype, requires_grad=True)
            tb = torch.full((1, 1), 2., dtype=dtype, requires_grad=True)
            a, b = native(ta, True), native(tb, True)
            output, expected = operation(fr, a, b, kind), operation(torch, ta, tb, kind)
            output.sum().backward(); expected.sum().backward()
            if a.grad.shape != list(ta.grad.shape) or b.grad.tolist() != tb.grad.tolist():
                raise RuntimeError('huge empty ordinary adjoint differs')
            if dtype == torch.float32:
                seed = torch.empty_like(ta, requires_grad=True); v = native(seed, True)
                output, expected = operation(fr, a, b, kind), operation(torch, ta, tb, kind)
                first = vjp(output, [b], v, create_graph=True)[0]
                reference = torch.autograd.grad(expected, tb, seed, create_graph=True)[0]
                if not first.requires_grad or not reference.requires_grad or first.tolist() != reference.tolist():
                    raise RuntimeError('huge empty live seed was detached')
                second = first.sum().grad_wrt([v], create_graph=True)[0]
                expected_second = torch.autograd.grad(reference.sum(), seed, create_graph=True)[0]
                if second.shape != list(expected_second.shape):
                    raise RuntimeError('huge empty live seed shape differs')
            count += 1
    print('PASS', count, 'huge empty actual Torch ordinary/higher cases')


class EmptyBroadcastAdjointTests(unittest.TestCase):
    def test_empty_ordinary_and_functional_gradients_match_torch(self):
        cases = (((0, 7), (1, 1)), ((2, 0, 7), (1, 7)), ((0, 2, 7), (0, 1, 7)), ((0, 2, 7), ()), ((0, 7), (7,)))
        for dtype in (torch.float32, torch.float64):
            for ashape, bshape in cases:
                for kind in ('mul', 'add', 'alpha', 'sub_alpha'):
                    for functional in (False, True):
                        ta = torch.empty(ashape, dtype=dtype, requires_grad=True)
                        tb = torch.full(bshape, 2., dtype=dtype, requires_grad=True)
                        a, b = native(ta, True), native(tb, True)
                        actual, expected = operation(fr, a, b, kind), operation(torch, ta, tb, kind)
                        if functional:
                            gradients = actual.sum().grad_wrt([a, b])
                            references = torch.autograd.grad(expected.sum(), [ta, tb])
                            self.assertIsNone(a.grad); self.assertIsNone(b.grad)
                        else:
                            actual.sum().backward(); expected.sum().backward()
                            gradients, references = (a.grad, b.grad), (ta.grad, tb.grad)
                        for got, wanted in zip(gradients, references): close(self, got, wanted)

    def test_empty_higher_vjp_keeps_live_cotangent_provenance(self):
        for kind in ('mul', 'add', 'alpha', 'sub_alpha'):
            ta = torch.empty(0, 7, requires_grad=True); tb = torch.full((1, 1), 2., requires_grad=True)
            seed = torch.empty_like(ta, requires_grad=True)
            a, b, v = native(ta, True), native(tb, True), native(seed, True)
            actual, expected = operation(fr, a, b, kind), operation(torch, ta, tb, kind)
            first = vjp(actual, [a, b], v, create_graph=True)
            references = torch.autograd.grad(expected, [ta, tb], seed, create_graph=True)
            for got, wanted in zip(first, references): close(self, got, wanted)
            self.assertTrue(first[1].requires_grad)
            self.assertTrue(references[1].requires_grad)
            second = first[1].sum().grad_wrt([v, a], create_graph=True)
            expected_second = torch.autograd.grad(references[1].sum(), [seed, ta], create_graph=True, allow_unused=True)
            # Tensor.grad_wrt explicitly returns zero for an unused coordinate.
            for got, wanted, original in zip(second, expected_second, (seed, ta)):
                close(self, got, torch.zeros_like(original) if wanted is None else wanted)
            self.assertIsNone(a.grad); self.assertIsNone(b.grad); self.assertIsNone(v.grad)

    def test_empty_jvp_and_reverse_over_tangent_match_torch(self):
        for kind in ('mul', 'add', 'alpha', 'sub_alpha'):
            ta = torch.empty(0, 7, requires_grad=True); tb = torch.full((1, 1), 2., requires_grad=True)
            tva = torch.empty_like(ta, requires_grad=True); tvb = torch.full_like(tb, .3, requires_grad=True)
            a, b, va, vb = native(ta, True), native(tb, True), native(tva, True), native(tvb, True)
            actual = fr.func.jvp(lambda x, y: operation(fr, x, y, kind), (a, b), (va, vb))
            expected = torch.func.jvp(lambda x, y: operation(torch, x, y, kind), (ta, tb), (tva, tvb))
            for got, wanted in zip(actual, expected): close(self, got, wanted)
            gradients = actual[1].sum().grad_wrt([b, vb])
            references = torch.autograd.grad(expected[1].sum(), [tb, tvb], allow_unused=True)
            for got, wanted, original in zip(gradients, references, (tb, tvb)):
                close(self, got, torch.zeros_like(original) if wanted is None else wanted)
            self.assertIsNone(a.grad); self.assertIsNone(b.grad); self.assertIsNone(va.grad); self.assertIsNone(vb.grad)

    def test_empty_shape_changing_pullback_flags_and_exact_seed_identity(self):
        for graph in (False, True):
            ta = torch.empty(0, 2, 7, requires_grad=True); tb = torch.empty(0, 1, 7, requires_grad=True)
            seed = torch.empty_like(ta, requires_grad=True)
            a, b, v = native(ta, True), native(tb, True), native(seed, True)
            _, pullback = fr.func.vjp(lambda x, y: x + y, a, b)
            _, reference_pullback = torch.func.vjp(lambda x, y: x + y, ta, tb)
            result, expected = pullback(v, create_graph=graph), reference_pullback(seed, create_graph=graph)
            self.assertIs(result[0], v); self.assertIs(expected[0], seed)
            for value, wanted in zip(result, expected):
                close(self, value, wanted)
                self.assertEqual(value.requires_grad, wanted.requires_grad)
                self.assertEqual(value.is_leaf, wanted.is_leaf)

    def test_huge_empty_gradients_finish_in_bounded_subprocess(self):
        result = subprocess.run([sys.executable, __file__, '--huge-worker'], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS 8 huge empty', result.stdout)


if __name__ == '__main__':
    if sys.argv[1:] == ['--huge-worker']:
        huge_worker()
    else:
        unittest.main(verbosity=2)
