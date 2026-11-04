"""Embedding options, pretrained storage, dense adjoints and saved-state semantics."""
import inspect
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    result = fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(True) if grad else result


class EmbeddingOptionsTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        actual = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 3e-6 if expected.dtype == torch.float32 else 2e-12
        torch.testing.assert_close(actual, expected.detach(), atol=tolerance, rtol=tolerance, equal_nan=True)

    def test_arbitrary_id_shapes_padding_and_frequency_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for padding in (None, 0, -1):
                for scale in (False, True):
                    for shape in ((), (4,), (2, 2)):
                        with self.subTest(dtype=dtype, padding=padding, scale=scale, shape=shape):
                            indices = torch.tensor(1 if shape == () else [0, 1, 1, 3], dtype=torch.int64).reshape(shape)
                            weight = (torch.arange(12, dtype=dtype).reshape(4, 3) * .1 + .2).requires_grad_(True)
                            value = native(weight, True)
                            expected = torch.nn.functional.embedding(indices, weight, padding_idx=padding, scale_grad_by_freq=scale)
                            actual = F.embedding(native(indices), value, padding_idx=padding, scale_grad_by_freq=scale)
                            self.same(actual, expected)
                            seed = torch.arange(expected.numel(), dtype=dtype).reshape(expected.shape) * .1 + .3
                            (actual * native(seed)).sum().backward(); (expected * seed).sum().backward()
                            self.same(value.grad, weight.grad)

    def test_max_norm_all_norm_orders_and_repeated_forward_versions(self):
        for dtype in (torch.float32, torch.float64):
            for order in (0., 1., 2., 3., -2., float('inf'), -float('inf')):
                for maximum in (1., -1.):
                    with self.subTest(dtype=dtype, order=order, maximum=maximum):
                        weight = torch.tensor([[3., 4., 0.], [-2., 1., .5], [0., 0., 0.]], dtype=dtype, requires_grad=True)
                        value, ids = native(weight, True), torch.tensor([0, 0, 1, 2])
                        a = F.embedding(native(ids), value, 0, maximum, order, True)
                        b = F.embedding(native(ids), value, 0, maximum, order, True)
                        x = torch.nn.functional.embedding(ids, weight, 0, maximum, order, True)
                        y = torch.nn.functional.embedding(ids, weight, 0, maximum, order, True)
                        self.same(a, x); self.same(b, y); self.same(value, weight)
                        (a.sum() + b.sum()).backward(); (x.sum() + y.sum()).backward()
                        self.same(value.grad, weight.grad)

    def test_strided_weight_without_renorm_and_empty_shapes(self):
        for dtype in (torch.float32, torch.float64):
            weight = torch.arange(12, dtype=dtype).reshape(3, 4).t().detach().requires_grad_(True)
            value = native(weight.t()).transpose(0, 1).detach().requires_grad_(True)
            ids = torch.tensor([[2, 0], [1, 2]])
            actual, expected = F.embedding(native(ids), value, scale_grad_by_freq=True), torch.nn.functional.embedding(ids, weight, scale_grad_by_freq=True)
            self.same(actual, expected)
            actual.sum().backward(); expected.sum().backward(); self.same(value.grad, weight.grad)
            for shape, id_values in (((3, 0), [1, 2]), ((0, 3), []), ((3, 2), [])):
                weight = torch.empty(shape, dtype=dtype, requires_grad=True)
                value, ids = native(weight, True), torch.tensor(id_values, dtype=torch.int64)
                actual = F.embedding(native(ids), value, max_norm=1., scale_grad_by_freq=True)
                expected = torch.nn.functional.embedding(ids, weight, max_norm=1., scale_grad_by_freq=True)
                self.same(actual, expected)
                actual.sum().backward(); expected.sum().backward(); self.same(value.grad, weight.grad)

    def test_constructor_padding_reset_signatures_and_zero_dimensions(self):
        self.assertEqual(list(inspect.signature(fr.nn.Embedding).parameters), list(inspect.signature(torch.nn.Embedding).parameters))
        self.assertEqual(list(inspect.signature(fr.nn.Embedding.from_pretrained).parameters), list(inspect.signature(torch.nn.Embedding.from_pretrained).parameters))
        model = fr.nn.Embedding(4, 3, padding_idx=-1, max_norm=2., scale_grad_by_freq=True)
        self.assertEqual(model.padding_idx, 3)
        self.assertEqual(model.weight.tensor().tolist()[3], [0., 0., 0.])
        parameter, identity = model.weight, model.weight.tensor()._id()
        model.reset_parameters()
        self.assertIs(model.weight, parameter)
        self.assertEqual(model.weight.tensor()._id(), identity)
        self.assertEqual(model.weight.tensor().tolist()[3], [0., 0., 0.])
        for rows, width in ((0, 3), (3, 0), (0, 0)):
            model = fr.nn.Embedding(rows, width)
            self.assertEqual(model(fr.tensor([], dtype='int64')).shape, [0, width])

    def test_pretrained_alias_flags_and_padding_values_match_torch(self):
        for freeze in (False, True):
            source = torch.tensor([[.2, .7], [.5, -.3], [1.2, .4]], requires_grad=True)
            value = native(source, True)
            model = fr.nn.Embedding.from_pretrained(value, freeze=freeze, padding_idx=0)
            expected = torch.nn.Embedding.from_pretrained(source, freeze=freeze, padding_idx=0)
            self.assertTrue(model.weight.tensor()._shares_storage(value))
            self.assertEqual(model.weight.tensor().requires_grad, not freeze)
            self.assertTrue(model.weight.trainable)
            self.assertTrue(value.requires_grad)
            ids = torch.tensor([0, 1, 1])
            a, b = model(native(ids)), expected(ids)
            self.same(a, b); self.assertEqual(a.requires_grad, b.requires_grad)
            if not freeze:
                a.sum().backward(); b.sum().backward()
                self.same(model.weight.tensor().grad, expected.weight.grad)
            self.assertIsNone(value.grad)
            with fr.no_grad(): value.copy_(fr.ones(3, 2))
            with torch.no_grad(): source.copy_(torch.ones_like(source))
            self.same(model(native(ids)), expected(ids))

    def test_indices_version_is_saved_and_weight_version_is_not(self):
        value = fr.ones(3, 2, requires_grad=True)
        ids = fr.tensor([0, 1])
        result = F.embedding(ids, value)
        with fr.no_grad(): value.copy_(fr.zeros(3, 2))
        result.sum().backward()
        self.assertEqual(value.grad.tolist(), [[1., 1.], [1., 1.], [0., 0.]])
        result = F.embedding(ids, value)
        ids.copy_(fr.tensor([1, 2]))
        with self.assertRaises((ValueError, RuntimeError)):
            result.sum().backward()

    def test_invalid_options_reject_before_renorm_and_capture(self):
        value = fr.tensor([[3., 4.], [5., 6.]])
        before = value.tolist()
        for ids in (fr.tensor([-1]), fr.tensor([2])):
            with self.assertRaises(ValueError): F.embedding(ids, value, max_norm=1.)
            self.assertEqual(value.tolist(), before)
        with self.assertRaises(NotImplementedError): F.embedding(fr.tensor([0]), value, sparse=True)
        with self.assertRaises(ValueError): F.embedding(fr.tensor([0]), value, padding_idx=2)
        with self.assertRaises((ValueError, NotImplementedError)):
            F.embedding(fr.tensor([0]), value.transpose(0, 1), max_norm=1.)
        with self.assertRaises((ValueError, NotImplementedError)):
            fr.capture(lambda: F.embedding(fr.tensor([0]), value, max_norm=1.))
        self.assertEqual(value.tolist(), before)

    def test_training_checkpoint_preserves_options_shared_ties_and_continuation(self):
        initial = fr.tensor([[.2, .7], [.5, -.3], [1.2, .4]])
        model = fr.nn.Embedding.from_pretrained(initial, freeze=False, padding_idx=0, max_norm=1., scale_grad_by_freq=True)
        model.register_buffer('source', initial)
        optimizer, generator = fr.optim.AdamW(model.parameters(), lr=.02), fr.Generator(7)
        ids = fr.tensor([0, 1, 1, 2])
        def step(module, opt):
            loss = module(ids).square().sum(); loss.backward(); opt.step(); opt.zero_grad()
            return loss.tolist()
        step(model, optimizer)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
        other_source = fr.zeros(3, 2)
        other = fr.nn.Embedding.from_pretrained(other_source, freeze=False, padding_idx=0, max_norm=1., scale_grad_by_freq=True)
        other.register_buffer('source', other_source)
        target = fr.optim.AdamW(other.parameters(), lr=.02)
        saved.restore(other, optimizers={'main': target}, generator=fr.Generator(11))
        for _ in range(4):
            self.assertEqual(step(model, optimizer), step(other, target))
            self.assertEqual(model.weight.tensor().tolist(), other.weight.tensor().tolist())
            self.assertTrue(other.weight.tensor()._shares_storage(other.source))

    def test_freeze_thaw_and_checkpoint_restore_preserve_independent_flags(self):
        ids = fr.tensor([0, 1, 1])
        for source_freeze in (False, True):
            for target_freeze in (False, True):
                model = fr.nn.Embedding.from_pretrained(fr.ones(3, 2), freeze=source_freeze)
                saved = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(7))
                target = fr.nn.Embedding.from_pretrained(fr.zeros(3, 2), freeze=target_freeze)
                saved.restore(target, optimizers={}, generator=fr.Generator(11))
                self.assertEqual(target.weight.requires_grad, not source_freeze)
                self.assertTrue(target.weight.trainable)
                self.assertEqual(target(ids).requires_grad, not source_freeze)
                identity = target.weight.tensor()._id()
                self.assertIs(target.weight.requires_grad_(True), target.weight)
                self.assertEqual(target.weight.tensor()._id(), identity)
                target(ids).sum().backward()
                self.assertEqual(target.weight.grad.tolist(), [[1., 1.], [2., 2.], [0., 0.]])
        model = fr.nn.Embedding.from_pretrained(fr.ones(3, 2), freeze=False)
        model.register_buffer('gradient_buffer', fr.ones(1, requires_grad=True), persistent=False)
        model(ids).sum().backward()
        gradient = model.weight.grad._id()
        self.assertIs(model.requires_grad_(False), model)
        self.assertFalse(model.weight.requires_grad)
        self.assertEqual(model.weight.grad._id(), gradient)
        self.assertTrue(model.gradient_buffer.requires_grad)
        self.assertTrue(model.weight.trainable)
        model.zero_grad()
        self.assertIsNone(model.weight.grad)
        model.requires_grad_()
        self.assertTrue(model.weight.requires_grad)
