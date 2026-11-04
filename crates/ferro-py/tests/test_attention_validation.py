"""Attention masking, preflight, empty shapes and checkpoint configuration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import attention as layers
from ferro.nn import functional as F


def native(value):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(True)


def checkpoint_values(model, optimizer, rng):
    state = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng)
    return json.dumps({name: [value.shape, value.tolist()] for name, value in state.tensors().items()},
                      sort_keys=True, allow_nan=False).encode('ascii')


class AttentionValidationTests(unittest.TestCase):
    def compare_sdpa(self, shapes, causal):
        generator = torch.Generator().manual_seed(43)
        reference = [torch.randn(shape, generator=generator, requires_grad=True) for shape in shapes]
        actual = [native(value) for value in reference]
        expected = torch.nn.functional.scaled_dot_product_attention(*reference, is_causal=causal)
        output = F.scaled_dot_product_attention(*actual, is_causal=causal)
        got = torch.tensor(output.tolist(), dtype=torch.float32).reshape(expected.shape)
        torch.testing.assert_close(got, expected.detach(), atol=2e-5, rtol=2e-4)
        cotangent = torch.randn(expected.shape, generator=generator)
        (output * fr.Tensor(cotangent.reshape(-1).tolist(), list(cotangent.shape))).sum().backward()
        (expected * cotangent).sum().backward()
        for value, wanted in zip(actual, reference):
            self.assertIsNotNone(value.grad)
            expected_grad = wanted.grad if wanted.grad is not None else torch.zeros_like(wanted)
            got = torch.tensor(value.grad.tolist(), dtype=torch.float32).reshape(wanted.shape)
            torch.testing.assert_close(got, expected_grad, atol=2e-5, rtol=2e-4)

    def test_nonsquare_rank3_and_rank4_values_and_all_input_gradients(self):
        for prefix in ([2], [2, 2]):
            for query_length, key_length in ((1, 4), (4, 2), (3, 3)):
                shapes = [prefix + [query_length, 4], prefix + [key_length, 4], prefix + [key_length, 3]]
                for causal in (False, True):
                    with self.subTest(shapes=shapes, causal=causal):
                        self.compare_sdpa(shapes, causal)

    def test_extreme_finite_logits_cannot_attend_to_future_keys(self):
        q = fr.Tensor([1e5], [1, 1, 1]).requires_grad_(True)
        k = fr.Tensor([0., 1e4], [1, 2, 1]).requires_grad_(True)
        v = fr.Tensor([3., 7.], [1, 2, 1]).requires_grad_(True)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(output.tolist(), [[[3.]]])
        output.sum().backward()
        self.assertEqual(q.grad.tolist(), [[[0.]]])
        self.assertEqual(k.grad.tolist(), [[[0.], [0.]]])
        self.assertEqual(v.grad.tolist(), [[[1.], [0.]]])

    def test_multihead_outputs_and_returned_weights_use_identical_causality(self):
        model = fr.nn.MultiheadAttention(1, 1, bias=False, batch_first=True)
        with fr.no_grad():
            model.in_proj_weight.tensor().copy_(fr.Tensor.ones([3, 1]))
            model.out_proj.weight.tensor().copy_(fr.Tensor.ones([1, 1]))
        q = fr.Tensor([1e5], [1, 1, 1]).requires_grad_(True)
        k = fr.Tensor([0., 1e4], [1, 2, 1]).requires_grad_(True)
        v = fr.Tensor([3., 7.], [1, 2, 1]).requires_grad_(True)
        output, weights = model(q, k, v, is_causal=True, attn_mask=fr.tensor([[False, True]], dtype='bool'))
        self.assertEqual(output.tolist(), [[[3.]]])
        self.assertEqual(weights.tolist(), [[[1., 0.]]])
        (output.sum() + (weights * fr.Tensor([2., 5.], [1, 1, 2])).sum()).backward()
        self.assertEqual(q.grad.tolist(), [[[0.]]])
        self.assertEqual(k.grad.tolist(), [[[0.], [0.]]])
        self.assertEqual(v.grad.tolist(), [[[1.], [0.]]])
        self.assertEqual(model.in_proj_weight.grad.tolist(), [[0.], [0.], [3.]])

    def test_rank_and_batch_head_mismatches_reject_before_native_folding(self):
        for shapes in [([2, 2, 1, 1], [1, 4, 1, 1], [1, 4, 1, 1]),
                       ([1, 2, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]),
                       ([1, 2, 1, 1], [1, 2, 1], [1, 2, 1]),
                       ([1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 2])]:
            values = [fr.Tensor.zeros(shape) for shape in shapes]
            with self.subTest(shapes=shapes), patch.object(F._native, 'scaled_dot_product_attention', side_effect=AssertionError('invalid metadata reached native computation')):
                with self.assertRaises(ValueError):
                    F.scaled_dot_product_attention(*values)

    def test_empty_batch_sequence_head_and_value_axes_preserve_gradients(self):
        for shapes in [([0, 2, 3], [0, 4, 3], [0, 4, 2]),
                       ([1, 0, 3], [1, 4, 3], [1, 4, 2]),
                       ([1, 2, 3], [1, 0, 3], [1, 0, 2]),
                       ([1, 2, 3], [1, 4, 3], [1, 4, 0]),
                       ([1, 0, 2, 3], [1, 0, 4, 3], [1, 0, 4, 2])]:
            for causal in (False, True):
                with self.subTest(shapes=shapes, causal=causal):
                    self.compare_sdpa(shapes, causal)
        model = fr.nn.MultiheadAttention(2, 1, batch_first=True)
        mask = fr.zeros(1).expand(1000, 1000)
        output, weights = model(fr.Tensor.zeros([0, 1000, 2]), fr.Tensor.zeros([0, 1000, 2]), fr.Tensor.zeros([0, 1000, 2]), is_causal=True, attn_mask=mask)
        self.assertEqual(output.shape, [0, 1000, 2])
        self.assertEqual(weights.shape, [0, 1000, 1000])

    def test_zero_query_width_and_nonfloat_inputs_reject(self):
        for batch in (0, 1):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, 'feature width must be positive'):
                F.scaled_dot_product_attention(fr.Tensor.zeros([batch, 2, 0]), fr.Tensor.zeros([batch, 3, 0]), fr.Tensor.zeros([batch, 3, 2]))
        floating = fr.Tensor.ones([1, 1, 1])
        integer = fr.Tensor.from_i64([1], [1, 1, 1])
        for values in ((integer, floating, floating), (floating, integer, floating), (floating, floating, integer)):
            with self.assertRaises(ValueError):
                F.scaled_dot_product_attention(*values)

    def test_invalid_constructor_configuration_rejects_before_parameters(self):
        invalid = [lambda: fr.nn.MultiheadAttention(0, 1), lambda: fr.nn.MultiheadAttention(4, 0),
                   lambda: fr.nn.MultiheadAttention(True, 1), lambda: fr.nn.SelfAttention(0, 1),
                   lambda: fr.nn.SelfAttention(4, 1, num_kv_heads=0), lambda: fr.nn.TransformerBlock(4, 3),
                   lambda: fr.nn.TransformerBlock(3, 1), lambda: fr.nn.SelfAttention(3, 1, rope_base=10000.)]
        invalid += [lambda base=base: fr.nn.SelfAttention(4, 1, rope_base=base) for base in (0., -1., float('nan'), float('inf'))]
        with patch.object(layers, 'Tensor') as tensor, patch.object(layers, 'Linear', side_effect=AssertionError('allocated projection')), patch.object(layers, 'RMSNorm', side_effect=AssertionError('allocated norm')):
            for construct in invalid:
                with self.assertRaises(ValueError):
                    construct()
            tensor.rand.assert_not_called()
            tensor.randn.assert_not_called()
        self.assertEqual(fr.nn.SelfAttention(3, 1)(fr.Tensor.ones([1, 2, 3])).shape, [1, 2, 3])

    def test_multihead_invalid_geometry_rejects_before_projection(self):
        model = fr.nn.MultiheadAttention(4, 2, batch_first=True)
        for shapes in [([1, 2, 4], [2, 2, 4], [2, 2, 4]),
                       ([1, 2, 4], [1, 3, 4], [1, 2, 4]),
                       ([1, 2, 4], [1, 3, 4], [1, 3, 3]),
                       ([1, 1, 2, 4], [1, 2, 4], [1, 2, 4])]:
            values = [fr.Tensor.zeros(shape) for shape in shapes]
            with self.subTest(shapes=shapes), patch.object(layers.F, 'linear', side_effect=AssertionError('invalid input reached projection')):
                with self.assertRaises(ValueError):
                    model(*values)

    def test_checkpoint_restores_same_geometry_and_rejects_semantic_changes_atomically(self):
        source = fr.nn.SelfAttention(8, 2, causal=True, rope_base=10000.)
        source.eval()
        checkpoint = fr.checkpoint.snapshot_training(source, optimizers={'main': fr.optim.AdamW(source.parameters())}, generator=fr.Generator(7))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint.save(Path(directory) / 'state')
            checkpoint = fr.checkpoint.load(Path(directory) / 'state')
            target = fr.nn.SelfAttention(8, 2, causal=True, rope_base=10000.)
            optimizer, rng = fr.optim.AdamW(target.parameters(), lr=.3), fr.Generator(11)
            checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
            self.assertFalse(target.training)
            x = fr.Tensor.randn([1, 3, 8], seed=43)
            with fr.no_grad():
                self.assertEqual(source(x).tolist(), target(x).tolist())
            for config in ({'num_heads': 4}, {'causal': False}, {'rope_base': 500.}):
                target = fr.nn.SelfAttention(**{'embed_dim': 8, 'num_heads': 2, 'causal': True, 'rope_base': 10000., **config})
                optimizer, rng = fr.optim.AdamW(target.parameters(), lr=.3), fr.Generator(11)
                before = checkpoint_values(target, optimizer, rng)
                with self.subTest(config=config), self.assertRaises(ValueError):
                    checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                self.assertEqual(checkpoint_values(target, optimizer, rng), before)


if __name__ == '__main__':
    unittest.main()
