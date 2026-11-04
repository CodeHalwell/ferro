"""Separate projections and source tokens against Torch's public MHA."""
import unittest
import inspect

import torch
import ferro as fr


def tensor(value, grad=False):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class MultiheadAttentionOptions(unittest.TestCase):
    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(expected.shape),
                                   expected.detach(), atol=4e-6, rtol=6e-5, equal_nan=True)

    def check(self, separate, bias_kv, zero, need, layout, boolean=False, source=4, causal=False, use_padding=True):
        torch.manual_seed(701)
        kdim, vdim = (3, 5) if separate else (4, 4)
        options = dict(batch_first=layout != 'sequence', kdim=kdim, vdim=vdim,
                       add_bias_kv=bias_kv, add_zero_attn=zero)
        actual = fr.nn.MultiheadAttention(4, 2, **options)
        expected = torch.nn.MultiheadAttention(4, 2, **options)
        parameters = dict(expected.named_parameters())
        self.assertEqual([name for name, _ in actual.named_parameters()], list(parameters))
        with fr.no_grad():
            for name, parameter in actual.named_parameters():
                parameter.tensor().copy_(tensor(parameters[name]))
        values = [torch.randn(2, length, width) for length, width in ((3, 4), (source, kdim), (source, vdim))]
        if layout == 'sequence':
            values = [value.transpose(0, 1) for value in values]
        elif layout == 'unbatched':
            values = [value[0] for value in values]
        values = [value.detach().requires_grad_() for value in values]
        inputs = [tensor(value, True) for value in values]
        mask = torch.zeros(3, source, dtype=torch.bool) if boolean else torch.randn(3, source, requires_grad=True)
        if boolean and source:
            mask[:, -1] = True
        padding_shape = (source,) if layout == 'unbatched' else (2, source)
        padding = torch.zeros(padding_shape, dtype=torch.bool) if boolean else torch.randn(padding_shape, requires_grad=True)
        native_mask = fr.tensor(mask.tolist(), dtype='bool') if boolean else tensor(mask, True)
        native_padding = fr.tensor(padding.tolist(), dtype='bool') if boolean else tensor(padding, True)
        if not use_padding:
            padding, native_padding = None, None
        output, weights = actual(*inputs, key_padding_mask=native_padding, attn_mask=native_mask,
                                  need_weights=need, average_attn_weights=False, is_causal=causal)
        wanted, wanted_weights = expected(*values, key_padding_mask=padding, attn_mask=mask,
                                          need_weights=need, average_attn_weights=False, is_causal=causal)
        self.close(output, wanted)
        probe = torch.randn(wanted.shape)
        loss, ref_loss = (output * tensor(probe)).sum(), (wanted * probe).sum()
        if need:
            self.close(weights, wanted_weights)
            weight_probe = torch.randn(wanted_weights.shape)
            loss, ref_loss = loss + (weights * tensor(weight_probe)).sum(), ref_loss + (wanted_weights * weight_probe).sum()
        else:
            self.assertIsNone(weights)
        loss.backward()
        ref_loss.backward()
        for value, reference in zip(inputs, values):
            self.close(value.grad, reference.grad)
        for name, parameter in actual.named_parameters():
            self.close(parameter.tensor().grad, parameters[name].grad)
        if not boolean and source:
            if mask.grad is None:
                self.assertIsNone(native_mask.grad)
            else:
                self.close(native_mask.grad, mask.grad)
            if padding is not None:
                self.close(native_padding.grad, padding.grad)

    def test_separate_projections_tokens_layouts_and_all_gradients(self):
        for separate in (False, True):
            for bias, zero in ((False, False), (True, False), (False, True), (True, True)):
                for need in (True, False):
                    for layout in ('batch', 'sequence', 'unbatched'):
                        with self.subTest(separate=separate, bias=bias, zero=zero, need=need, layout=layout):
                            self.check(separate, bias, zero, need, layout)

    def test_boolean_masks_and_empty_sources_with_added_tokens(self):
        for bias, zero in ((True, False), (False, True), (True, True)):
            for need in (True, False):
                self.check(True, bias, zero, need, 'batch', True)
                self.check(True, bias, zero, need, 'unbatched', False, 0)

    def test_explicit_mask_causal_hints_and_added_tokens(self):
        for need in (True, False):
            for padding in (True, False):
                for boolean in (True, False):
                    for bias, zero in ((False, False), (True, True)):
                        with self.subTest(need=need, padding=padding, boolean=boolean, bias=bias):
                            self.check(True, bias, zero, need, 'batch', boolean, causal=True, use_padding=padding)

    def test_validation_before_initialization_and_forward(self):
        for options in (dict(kdim=0), dict(vdim=True), dict(kdim=2**62), dict(add_bias_kv=1),
                        dict(add_zero_attn=None), dict(dtype='float64'), dict(device='cuda')):
            with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                fr.nn.MultiheadAttention(4, 2, **options)
        model = fr.nn.MultiheadAttention(4, 2, kdim=3, vdim=5, dropout=.2)
        before = model.attn_dropout.rng_state.tolist()
        with self.assertRaises(ValueError):
            model(fr.ones(2, 3, 4), fr.ones(2, 4, 4), fr.ones(2, 4, 5))
        self.assertEqual(model.attn_dropout.rng_state.tolist(), before)

    def test_torch_constructor_order_default_layout_and_class_signature(self):
        self.assertEqual(list(inspect.signature(fr.nn.MultiheadAttention).parameters),
                         list(inspect.signature(torch.nn.MultiheadAttention).parameters))
        self.assertFalse(inspect.signature(fr.nn.MultiheadAttention).parameters['batch_first'].default)
        for positional in (False, True):
            torch.manual_seed(73)
            if positional:
                model = fr.nn.MultiheadAttention(4, 2, .2, False, True, True, 3, 5, True, 'cpu', 'float32')
                reference = torch.nn.MultiheadAttention(4, 2, .2, False, True, True, 3, 5, True, 'cpu', torch.float32)
                shapes = ((2, 3, 4), (2, 5, 3), (2, 5, 5))
            else:
                model, reference = fr.nn.MultiheadAttention(4, 2), torch.nn.MultiheadAttention(4, 2)
                shapes = ((3, 2, 4), (5, 2, 4), (5, 2, 4))
            values = [torch.randn(shape, requires_grad=True) for shape in shapes]
            inputs = [tensor(value, True) for value in values]
            parameters = dict(reference.named_parameters())
            with fr.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.tensor().copy_(tensor(parameters[name]))
            model.eval()
            reference.eval()
            output, weights = model(*inputs)
            expected, expected_weights = reference(*values)
            self.close(output, expected)
            self.close(weights, expected_weights)
            output.square().sum().backward()
            expected.square().sum().backward()
            for value, wanted in zip(inputs, values):
                self.close(value.grad, wanted.grad)
            for name, parameter in model.named_parameters():
                self.close(parameter.tensor().grad, parameters[name].grad)
            with self.assertRaisesRegex(ValueError, 'explicit attention mask'):
                model(*inputs, is_causal=True)
            with self.assertRaises(RuntimeError):
                reference(*values, is_causal=True)


if __name__ == '__main__':
    unittest.main()
