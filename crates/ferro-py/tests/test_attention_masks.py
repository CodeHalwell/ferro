"""Mask semantics, head weights and layouts against real PyTorch."""
import math
import unittest
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import functional as F


def native(value, grad=False):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class AttentionMasks(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        value = torch.tensor(actual.tolist()).reshape(expected.shape)
        torch.testing.assert_close(value, expected.detach(), atol=3e-6, rtol=4e-5, equal_nan=True)

    def sdpa(self, shape, mask_kind, scale=None, causal=False, all_masked=False):
        rng = torch.Generator().manual_seed(13)
        q = torch.randn(shape + (3, 4), generator=rng).requires_grad_()
        k = torch.randn(shape + (5, 4), generator=rng).requires_grad_()
        v = torch.randn(shape + (5, 2), generator=rng).requires_grad_()
        if mask_kind == 'bool':
            mask = torch.tensor([[True, True, False, False, True], [True, False, True, True, False], [False, True, True, False, True]])
            if all_masked:
                mask[1] = False
            actual_mask = fr.tensor(mask.tolist(), dtype='bool')
        else:
            mask = torch.randn(3, 5, generator=rng).requires_grad_()
            if all_masked:
                with torch.no_grad():
                    mask[1] = -math.inf
            actual_mask = native(mask, True)
        inputs = [native(t, True) for t in (q, k, v)]
        actual = F.scaled_dot_product_attention(*inputs, attn_mask=actual_mask, scale=scale, is_causal=causal)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, is_causal=causal)
        self.close(actual, expected)
        seed = torch.randn(expected.shape, generator=rng)
        actual.backward_with(native(seed))
        expected.backward(seed)
        for left, right in zip(inputs, (q, k, v)):
            self.close(left.grad, right.grad)
        if mask.dtype == torch.float32:
            self.close(actual_mask.grad, mask.grad)
        if all_masked:
            self.assertTrue(torch.equal(expected[..., 1, :].detach(), torch.zeros_like(expected[..., 1, :])))

    def test_sdpa_float_masks_custom_scales_ranks_and_mask_gradients(self):
        for shape in ((), (2,), (2, 3)):
            for scale in (None, .3, 0., -.2):
                with self.subTest(shape=shape, scale=scale):
                    self.sdpa(shape, 'float', scale)

    def test_sdpa_bool_masks_all_masked_rows_and_causal_rejection(self):
        for shape in ((), (2,), (2, 3)):
            for kind in ('float', 'bool'):
                with self.subTest(shape=shape, kind=kind):
                    self.sdpa(shape, kind, all_masked=True)
        inputs = [fr.ones(2, 3, 4)] * 3
        mask = fr.ones(3, 3, dtype='bool')
        with self.assertRaises(ValueError):
            F.scaled_dot_product_attention(*inputs, attn_mask=mask, is_causal=True)
        with self.assertRaises(RuntimeError):
            torch.nn.functional.scaled_dot_product_attention(*[torch.ones(2, 3, 4)] * 3,
                                                            attn_mask=torch.ones(3, 3, dtype=torch.bool), is_causal=True)

    def mha(self, batch_first=True, unbatched=False, need_weights=True, average=True, mask_kind='float', per_head=False):
        model = fr.nn.MultiheadAttention(8, 2, batch_first=batch_first)
        reference = torch.nn.MultiheadAttention(8, 2, batch_first=batch_first)
        parameters = dict(reference.named_parameters())
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(native(parameters[name]))
        batch, length, source = 2, 3, 4
        q_shape = (length, 8) if unbatched else (batch, length, 8) if batch_first else (length, batch, 8)
        kv_shape = (source, 8) if unbatched else (batch, source, 8) if batch_first else (source, batch, 8)
        rng = torch.Generator().manual_seed(29)
        inputs = [torch.randn(shape, generator=rng).requires_grad_() for shape in (q_shape, kv_shape, kv_shape)]
        actual_inputs = [native(t, True) for t in inputs]
        actual_batch = 1 if unbatched else batch
        shape = (actual_batch * 2, length, source) if per_head else (length, source)
        if mask_kind == 'bool':
            mask = torch.rand(shape, generator=rng) > .7
            mask[..., 0] = False
            padding = torch.zeros((source,) if unbatched else (batch, source), dtype=torch.bool)
            padding[..., -1] = True
            actual_mask, actual_padding = fr.tensor(mask.tolist(), dtype='bool'), fr.tensor(padding.tolist(), dtype='bool')
        else:
            mask = torch.randn(shape, generator=rng).requires_grad_()
            padding = torch.randn((source,) if unbatched else (batch, source), generator=rng).requires_grad_()
            actual_mask, actual_padding = native(mask, True), native(padding, True)
        actual, weights = model(*actual_inputs, attn_mask=actual_mask, key_padding_mask=actual_padding,
                                need_weights=need_weights, average_attn_weights=average)
        expected, ref_weights = reference(*inputs, attn_mask=mask, key_padding_mask=padding,
                                          need_weights=need_weights, average_attn_weights=average)
        self.close(actual, expected)
        seed = torch.randn(expected.shape, generator=rng)
        loss, ref_loss = (actual * native(seed)).sum(), (expected * seed).sum()
        if need_weights:
            self.close(weights, ref_weights)
            probe = torch.randn(ref_weights.shape, generator=rng)
            loss, ref_loss = loss + (weights * native(probe)).sum(), ref_loss + (ref_weights * probe).sum()
        else:
            self.assertIsNone(weights)
        loss.backward()
        ref_loss.backward()
        for left, right in zip(actual_inputs, inputs):
            self.close(left.grad, right.grad)
        for name, parameter in model.named_parameters():
            self.close(parameter.tensor().grad, parameters[name].grad)
        if mask_kind == 'float':
            self.close(actual_mask.grad, mask.grad)
            self.close(actual_padding.grad, padding.grad)

    def test_mha_float_masks_layouts_head_weights_and_all_gradients(self):
        for batch_first, unbatched, need, average, per_head in (
            (True, False, True, True, False), (False, False, True, False, True),
            (True, False, False, False, True), (False, True, True, False, True),
            (True, True, False, True, False)):
            with self.subTest(batch_first=batch_first, unbatched=unbatched, need=need):
                self.mha(batch_first, unbatched, need, average, per_head=per_head)

    def test_mha_bool_exclusion_masks(self):
        for need in (True, False):
            for unbatched in (True, False):
                self.mha(False, unbatched, need, False, 'bool', True)

    def test_invalid_mask_metadata_and_zero_scale_empty_inputs(self):
        q, k, v = fr.ones(2, 3, 4), fr.ones(2, 5, 4), fr.ones(2, 5, 2)
        for mask in (fr.ones(4, 5), fr.ones(2, 3, 3, 5), fr.ones(3, 5, dtype='int64')):
            with self.assertRaises(ValueError):
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        for probability in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                F.scaled_dot_product_attention(q, k, v, dropout_p=probability)
        for scale in (float('nan'), float('inf'), 'x'):
            with self.assertRaises(ValueError):
                F.scaled_dot_product_attention(q, k, v, scale=scale)
        for shapes in (((0, 3, 4), (0, 5, 4), (0, 5, 2)), ((2, 0, 4), (2, 5, 4), (2, 5, 2)),
                       ((2, 3, 4), (2, 0, 4), (2, 0, 2))):
            expected_inputs = [torch.ones(s, requires_grad=True) for s in shapes]
            actual_inputs = [native(t, True) for t in expected_inputs]
            actual = F.scaled_dot_product_attention(*actual_inputs, scale=0.)
            expected = torch.nn.functional.scaled_dot_product_attention(*expected_inputs, scale=0.)
            self.close(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
            for left, right in zip(actual_inputs, expected_inputs):
                self.close(left.grad, right.grad)

    def test_mha_dropout_same_mask_all_gradients_stream_and_eval(self):
        for need in (False, True):
            for layout in ('batch', 'sequence', 'unbatched'):
                for probability in (.3, 1.):
                    with self.subTest(need=need, layout=layout, probability=probability):
                        torch.manual_seed(431)
                        batch_first = layout != 'sequence'
                        model = fr.nn.MultiheadAttention(4, 2, batch_first=batch_first, dropout=probability)
                        oracle = torch.nn.MultiheadAttention(4, 2, batch_first=batch_first, dropout=probability)
                        reference_params = dict(oracle.named_parameters())
                        with fr.no_grad():
                            for name, parameter in model.named_parameters():
                                parameter.tensor().copy_(native(reference_params[name]))
                            model.attn_dropout.rng_state.copy_(fr.nn.Dropout._encode(17, 5))
                        values = [torch.randn(2, length, 4) for length in (3, 5, 5)]
                        if layout == 'sequence':
                            values = [value.transpose(0, 1) for value in values]
                        elif layout == 'unbatched':
                            values = [value[0] for value in values]
                        values = [value.detach().requires_grad_() for value in values]
                        inputs = [native(value, True) for value in values]
                        padding = torch.tensor([[False, True, False, False, False], [False, False, True, False, False]])
                        if layout == 'unbatched':
                            padding = padding[0]
                        mask = fr.tensor(padding.tolist(), dtype='bool')
                        batch = 1 if layout == 'unbatched' else 2
                        multiplier = F.dropout(fr.ones(batch * 2, 3, 5), probability, True, seed=17, offset=5)
                        multiplier = torch.tensor(multiplier.tolist())

                        def fixed_dropout(value, p=.5, training=True, inplace=False):
                            self.assertEqual(p, probability)
                            self.assertTrue(training)
                            return value * multiplier.reshape(value.shape)

                        actual, weights = model(*inputs, key_padding_mask=mask, need_weights=need, average_attn_weights=False)
                        with patch('torch.nn.functional.dropout', fixed_dropout):
                            expected, expected_weights = oracle(*values, key_padding_mask=padding,
                                                                 need_weights=True, average_attn_weights=False)
                        self.close(actual, expected)
                        if need:
                            self.close(weights, expected_weights)
                        else:
                            self.assertIsNone(weights)
                        probe = torch.randn(expected.shape)
                        loss, ref_loss = (actual * native(probe)).sum(), (expected * probe).sum()
                        if need:
                            weight_probe = torch.randn(expected_weights.shape)
                            loss = loss + (weights * native(weight_probe)).sum()
                            ref_loss = ref_loss + (expected_weights * weight_probe).sum()
                        loss.backward()
                        ref_loss.backward()
                        for left, right in zip(inputs, values):
                            self.close(left.grad, right.grad)
                        for name, parameter in model.named_parameters():
                            self.close(parameter.tensor().grad, reference_params[name].grad)
                        self.assertEqual(model.attn_dropout._offset, 5 if probability == 1 else 5 + batch * 2 * 3 * 5)
                        saved = model.attn_dropout.rng_state.tolist()
                        model.eval()
                        oracle.eval()
                        self.close(model(*inputs, key_padding_mask=mask, need_weights=need)[0],
                                   oracle(*values, key_padding_mask=padding, need_weights=need)[0])
                        self.assertEqual(model.attn_dropout.rng_state.tolist(), saved)

    def test_empty_attention_does_not_allocate_causal_mask_or_use_float_mask(self):
        model = fr.nn.MultiheadAttention(2, 1, batch_first=True)
        from unittest.mock import patch
        empty = fr.zeros(0, 2**24, 2)
        mask = fr.zeros(1).expand(2**24, 2**24)
        class TensorType(type):
            def __instancecheck__(cls, instance):
                return isinstance(instance, fr.Tensor)

        class NoAllocation(metaclass=TensorType):
            def __new__(cls, *args, **kwargs):
                raise AssertionError('allocated unused causal mask')

        with patch('ferro.nn._attention_ops.Tensor', NoAllocation):
            output, weights = model(empty, empty, empty, is_causal=True, attn_mask=mask)
        self.assertEqual(output.shape, [0, 2**24, 2])
        self.assertEqual(weights.shape, [0, 2**24, 2**24])
        with self.assertRaises(ValueError):
            fr.nn.MultiheadAttention(2**40, 1)
        for batch, length, source in ((0, 3, 5), (2, 0, 5), (2, 3, 0)):
            inputs = [torch.ones(batch, length, 4, requires_grad=True), torch.ones(batch, source, 4, requires_grad=True),
                      torch.ones(batch, source, 2, requires_grad=True)]
            actual_inputs = [native(t, True) for t in inputs]
            mask = torch.zeros(length, source, requires_grad=True)
            actual_mask = native(mask, True)
            actual = F.scaled_dot_product_attention(*actual_inputs, attn_mask=actual_mask)
            expected = torch.nn.functional.scaled_dot_product_attention(*inputs, attn_mask=mask)
            actual.sum().backward()
            expected.sum().backward()
            self.close(actual, expected)
            self.assertIsNone(actual_mask.grad)
            self.assertIsNone(mask.grad)

    def test_data_dependent_mask_paths_reject_capture_before_native_compute(self):
        q = fr.ones(1, 2, 4)
        mask = fr.zeros(2, 2)
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: F.scaled_dot_product_attention(q, q, q, attn_mask=mask))
        model = fr.nn.MultiheadAttention(4, 1)
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(q, q, q, attn_mask=mask)[0])


if __name__ == '__main__':
    unittest.main()
