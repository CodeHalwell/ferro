"""Actual Torch unused-gradient layout/ownership on the canonical public grad API.

These are ordinary CPU materialized zeros, not structural-zero forward tangents.
F64 is first-order only; graph-building and fresh-leaf history controls use F32.
"""
import itertools
import os
import subprocess
import sys
import unittest

import torch
import ferro as fr


KINDS = ('dense', 'transpose', 'gap', 'broadcast', 'offset',
         'empty', 'empty_offset', 'empty_broadcast')


def mode(lib, enabled):
    return torch.set_grad_enabled(enabled) if lib is torch else fr.enable_grad() if enabled else fr.no_grad()


def make(lib, kind, dtype='float32'):
    typ = getattr(torch, dtype) if lib is torch else dtype
    base = lib.ones((3, 4, 5), dtype=typ).requires_grad_()
    if kind == 'dense':
        value = base
    elif kind == 'transpose':
        value = base.permute(2, 0, 1)
    elif kind == 'gap':
        value = base[:, ::2, ::2]
    elif kind == 'broadcast':
        value = base[:1, :1, :1].expand(2, 3, 4)
    elif kind == 'offset':
        value = base[1:, 1:, 1:]
    elif kind == 'empty':
        value = base[:0].permute(2, 0, 1)
    elif kind == 'empty_offset':
        value = base[:, :, 2:2].permute(2, 0, 1)
    elif kind == 'empty_broadcast':
        value = base[:1, :0, :1].expand(7, 0, 3)
    else:
        raise ValueError(kind)
    return base, value


def aliases(lib, first, second):
    return torch._C._is_alias_of(first, second) if lib is torch else first._shares_storage(second)


class MaterializedGradientLayoutTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual is None, expected is None)
        if expected is None:
            return
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertFalse(actual._is_structural_zero())
        self.assertFalse(torch._is_zerotensor(expected))
        values = torch.tensor(actual.reshape([-1]).tolist(), dtype=expected.dtype)
        torch.testing.assert_close(values, expected.detach().reshape(-1), rtol=0, atol=0)

    def test_materialized_layouts_match_torch_independent_of_caller_grad_mode(self):
        for kind, dtype, graph, enabled in itertools.product(KINDS, ('float32', 'float64'), (False, True), (False, True)):
            if dtype == 'float64' and graph:
                continue
            with self.subTest(kind=kind, dtype=dtype, graph=graph, enabled=enabled):
                cases = []
                for lib in (fr, torch):
                    base, unused = make(lib, kind, dtype)
                    typ = getattr(torch, dtype) if lib is torch else dtype
                    active = lib.ones((2,), dtype=typ).requires_grad_()
                    seed = lib.ones((2,), dtype=typ).requires_grad_()
                    slot = lib.ones((2,), dtype=typ)
                    active.grad = slot
                    before = (unused.stride(), unused.storage_offset(), unused.requires_grad)
                    with mode(lib, enabled):
                        used, zero = lib.autograd.grad(active, (active, unused), seed,
                            create_graph=graph, materialize_grads=True)
                    self.assertIs(used, seed)
                    if lib is torch:
                        self.assertIs(active.grad, slot)
                    else:
                        self.assertEqual(active.grad._id(), slot._id())
                    self.assertIsNone(base.grad)
                    self.assertEqual(before, (unused.stride(), unused.storage_offset(), unused.requires_grad))
                    self.assertEqual(zero.storage_offset(), 0)
                    self.assertEqual(zero.requires_grad, graph)
                    self.assertTrue(zero.is_leaf)
                    self.assertFalse(aliases(lib, zero, unused))
                    self.assertFalse(aliases(lib, zero, base))
                    cases.append(zero)
                self.same(*cases)

    def test_duplicate_unused_results_own_storage_and_do_not_mutate_saved_inputs(self):
        for kind, dtype, graph in itertools.product(KINDS, ('float32', 'float64'), (False, True)):
            if dtype == 'float64' and graph:
                continue
            with self.subTest(kind=kind, dtype=dtype, graph=graph):
                cases = []
                for lib in (fr, torch):
                    base, unused = make(lib, kind, dtype)
                    saved = base.square().sum()
                    typ = getattr(torch, dtype) if lib is torch else dtype
                    active = lib.ones((), dtype=typ).requires_grad_()
                    first, second = lib.autograd.grad(active, (unused, unused),
                        create_graph=graph, materialize_grads=True)
                    self.assertIsNot(first, second)
                    self.assertFalse(aliases(lib, first, second))
                    self.assertFalse(aliases(lib, first, base))
                    with mode(lib, False):
                        physical = first.detach()
                        if not physical.is_contiguous():
                            axes = tuple(sorted(range(physical.ndim),
                                key=lambda axis: physical.stride()[axis], reverse=True))
                            physical = physical.permute(axes)
                        self.assertTrue(physical.is_contiguous())
                        self.assertTrue(aliases(lib, physical, first))
                        physical.fill_(7.)
                    self.assertEqual(second.reshape([-1]).tolist(), [0.] * second.numel())
                    # An empty write through an alias would also invalidate saved's version.
                    guard = lib.autograd.grad(saved, base)[0]
                    self.assertEqual(guard.reshape([-1]).tolist(), [2.] * base.numel())
                    cases.append((first, second))
                for got, want in zip(*cases):
                    self.same(got, want)

    def test_unused_none_validation_and_materialized_leaf_has_no_primal_history(self):
        for enabled, graph, materialize, live_seed in itertools.product((False, True), (False, True), (False, True), (False, True)):
            with self.subTest(enabled=enabled, graph=graph, materialize=materialize, live_seed=live_seed):
                cases = []
                for lib in (fr, torch):
                    base, unused = make(lib, 'empty_offset')
                    active = lib.ones((2,)).requires_grad_()
                    seed = lib.ones((2,)).requires_grad_(live_seed)
                    with mode(lib, enabled):
                        used, zero = lib.autograd.grad(active, (active, unused), seed,
                            create_graph=graph, allow_unused=True, materialize_grads=materialize)
                    self.assertIs(used, seed)
                    if not materialize:
                        self.assertIsNone(zero)
                    elif graph:
                        # Build this query outside caller no_grad: zero is a fresh leaf,
                        # not a zero-valued connection back to either original handle.
                        own, original, backing = lib.autograd.grad(zero.sum(), (zero, unused, base),
                            allow_unused=True)
                        self.assertEqual(own.shape, zero.shape)
                        self.assertIsNone(original)
                        self.assertIsNone(backing)
                    cases.append(zero)
                self.same(*cases)
        for lib in (fr, torch):
            active = lib.ones(()).requires_grad_()
            _, unused = make(lib, 'transpose')
            with self.assertRaises(RuntimeError):
                lib.autograd.grad(active, unused)
            with self.assertRaises(ValueError):
                lib.autograd.grad(active, unused, allow_unused=False, materialize_grads=True)
        active = fr.ones((), dtype='float64').requires_grad_()
        _, unused = make(fr, 'transpose', 'float64')
        with self.assertRaisesRegex(ValueError, 'float32'):
            fr.autograd.grad(active, unused, create_graph=True, materialize_grads=True)

    def test_huge_empty_materialization_is_bounded_and_uses_installed_api(self):
        script = r'''
import itertools
import unittest
import torch
import ferro as fr
case = unittest.TestCase()
for dtype, enabled in itertools.product(('float32', 'float64'), (False, True)):
    results = []
    for lib in (fr, torch):
        typ = getattr(torch, dtype) if lib is torch else dtype
        base = lib.ones((1, 0, 3), dtype=typ).requires_grad_()
        unused = base.expand(2**40, 0, 3).permute(2, 0, 1)
        active = lib.ones((), dtype=typ).requires_grad_()
        context = torch.set_grad_enabled(enabled) if lib is torch else fr.enable_grad() if enabled else fr.no_grad()
        with context:
            result = lib.autograd.grad(active, unused, materialize_grads=True)[0]
        case.assertEqual(result.numel(), 0)
        case.assertEqual(result.storage_offset(), 0)
        case.assertTrue(result.is_leaf)
        case.assertFalse(result.requires_grad)
        alias = torch._C._is_alias_of(result, unused) if lib is torch else result._shares_storage(unused)
        case.assertFalse(alias)
        case.assertEqual(result.reshape([-1]).tolist(), [])
        results.append(result)
    case.assertEqual(results[0].shape, list(results[1].shape))
    case.assertEqual(results[0].stride(), results[1].stride())
print('materialized huge-empty layouts pass')
'''
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        completed = subprocess.run([sys.executable, '-c', script], env=env,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('materialized huge-empty layouts pass', completed.stdout)


if __name__ == '__main__':
    unittest.main()
