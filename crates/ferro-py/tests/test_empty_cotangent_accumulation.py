"""Actual Torch empty cotangent accumulation without observer-altering hooks."""
import itertools
import unittest
import torch
import ferro as fr


def empty(lib, shape, dtype='float32', reverse=False, live=True):
    dtype = getattr(torch, dtype) if lib is torch else dtype
    value = lib.empty(tuple(reversed(shape)) if reverse else shape, dtype=dtype)
    if reverse:
        value = value.permute(tuple(reversed(range(len(shape)))))
    return value.requires_grad_(live)


def mode(lib, enabled):
    return torch.set_grad_enabled(enabled) if lib is torch else fr.enable_grad() if enabled else fr.no_grad()


def pair(lib, shape, dtype, permute, left_first, seed_kind):
    x = empty(lib, shape, dtype, reverse=True)
    g, w = empty(lib, shape, dtype), empty(lib, shape, dtype)
    dg, dw = empty(lib, (2, *shape), dtype), empty(lib, (2, *shape), dtype)
    left = dg
    if left_first and permute != 'none':
        left = dg.permute(tuple(range(dg.ndim)))
    negative = dg * w + g * dw
    if not left_first and permute != 'none':
        left = dg.permute(tuple(range(dg.ndim)))
    condition = x >= 0
    if permute == 'all':
        condition = condition.permute(tuple(range(condition.ndim)))
        negative = negative.permute(tuple(range(negative.ndim)))
    output = lib.where(condition, left, negative)
    if permute == 'all':
        output = output.permute(tuple(range(output.ndim)))
    direction = None if seed_kind == 'sum' else empty(lib, output.shape, dtype, reverse=True, live=seed_kind == 'live')
    query = output.sum() if direction is None else output
    return query, dg, direction, w


class EmptyCotangentAccumulationTests(unittest.TestCase):
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
        self.assertEqual(actual._is_structural_zero(), torch._is_zerotensor(expected))
        if not torch._is_zerotensor(expected):
            values = torch.tensor(actual.reshape([-1]).tolist(), dtype=expected.dtype)
            torch.testing.assert_close(values, expected.detach().reshape(-1), rtol=0, atol=0, equal_nan=True)

    def test_where_branches_order_identity_permutations_and_query_modes(self):
        configs = itertools.product(((3, 0, 2), (0, 3), (2, 0)), ('float32', 'float64'),
                                    ('none', 'left', 'all'), (False, True),
                                    ('sum', 'detached', 'live'), (False, True), (False, True))
        for shape, dtype, permute, left_first, seed_kind, graph, enabled in configs:
            if dtype == 'float64' and graph:
                continue
            with self.subTest(shape=shape, dtype=dtype, permute=permute, left_first=left_first,
                              seed=seed_kind, graph=graph, enabled=enabled):
                output, x, seed, _ = pair(fr, shape, dtype, permute, left_first, seed_kind)
                reference, tx, direction, _ = pair(torch, shape, dtype, permute, left_first, seed_kind)
                with mode(fr, enabled):
                    got = fr.autograd.grad(output, x, seed, create_graph=graph)[0]
                with mode(torch, enabled):
                    want = torch.autograd.grad(reference, tx, direction, create_graph=graph)[0]
                self.same(got, want)
                self.assertIsNone(x.grad)
                if seed is not None:
                    self.assertEqual(got._shares_storage(seed), torch._C._is_alias_of(want, direction))

    def test_independent_mul_add_branches_use_generic_accumulation(self):
        for dtype, graph, enabled, reverse in itertools.product(('float32', 'float64'), (False, True), (False, True), (False, True)):
            if dtype == 'float64' and graph:
                continue
            with self.subTest(dtype=dtype, graph=graph, enabled=enabled, reverse=reverse):
                results = []
                for lib in (fr, torch):
                    x = empty(lib, (2, 3, 0, 2), dtype)
                    a, b = empty(lib, (3, 0, 2), dtype, reverse=True), empty(lib, x.shape, dtype)
                    if reverse:
                        right = x * b
                        left = x * a
                    else:
                        left = x * a
                        right = x * b
                    loss = (left + right).sum()
                    with mode(lib, enabled):
                        results.append(lib.autograd.grad(loss, x, create_graph=graph)[0])
                self.same(*results)

    def test_external_offset_seeds_repeated_roots_and_saved_versions_are_untouched(self):
        for dtype, graph, enabled, live, repeated in itertools.product(('float32', 'float64'), (False, True), (False, True), (False, True), (False, True)):
            if dtype == 'float64' and graph:
                continue
            with self.subTest(dtype=dtype, graph=graph, enabled=enabled, live=live, repeated=repeated):
                cases = []
                for lib in (fr, torch):
                    typ = getattr(torch, dtype) if lib is torch else dtype
                    backing = lib.ones((2, 3, 2), dtype=typ).requires_grad_()
                    saved = backing.square().sum()
                    seed = backing[:, :, 1:1]
                    if not live:
                        seed = seed.detach()
                    x, weight = empty(lib, seed.shape, dtype), empty(lib, seed.shape, dtype)
                    roots = (x, x) if repeated else (x, x * weight)
                    before = (seed.stride(), seed.storage_offset(), seed.requires_grad)
                    with mode(lib, enabled):
                        result = lib.autograd.grad(roots, x, (seed, seed), create_graph=graph)[0]
                    self.assertEqual(before, (seed.stride(), seed.storage_offset(), seed.requires_grad))
                    # Even an empty write to the external alias would invalidate this saved input.
                    guard = lib.autograd.grad(saved, backing)[0]
                    self.assertEqual(guard.tolist(), [[[2., 2.]] * 3] * 2)
                    aliases = torch._C._is_alias_of(result, seed) if lib is torch else result._shares_storage(seed)
                    cases.append((result, aliases))
                self.same(cases[0][0], cases[1][0])
                self.assertEqual(cases[0][1], cases[1][1])

    def test_passthrough_identity_unused_slots_and_live_higher_seed_history(self):
        for graph, enabled, materialize in itertools.product((False, True), (False, True), (False, True)):
            cases = []
            for lib in (fr, torch):
                x, unused = empty(lib, (2, 0, 3)), empty(lib, (2, 0, 3))
                seed = empty(lib, x.shape, reverse=True)
                with mode(lib, enabled):
                    result = lib.autograd.grad(x, (x, unused), seed, create_graph=graph,
                                               allow_unused=True, materialize_grads=materialize)
                self.assertIs(result[0], seed)
                self.assertIsNone(x.grad)
                cases.append(result)
            for got, want in zip(*cases):
                self.same(got, want)
        for enabled in (False, True):
            cases = []
            for lib in (fr, torch):
                output, x, seed, weight = pair(lib, (3, 0, 2), 'float32', 'none', True, 'live')
                with mode(lib, enabled):
                    first = lib.autograd.grad(output, x, seed, create_graph=True)[0]
                # Build the follow-up query outside caller no_grad, preserving the actual graph.
                second = lib.autograd.grad(first.sum(), (seed, weight), allow_unused=True)
                cases.append(second)
            for got, want in zip(*cases):
                self.same(got, want)


if __name__ == '__main__':
    unittest.main()
