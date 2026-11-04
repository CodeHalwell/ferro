"""Canonical dense Transformers against actual PyTorch and saved dropout state."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import torch
import ferro as fr
from ferro.nn import (Transformer, TransformerDecoder, TransformerDecoderLayer,
                      TransformerEncoder, TransformerEncoderLayer)


def tensor(value, requires_grad=False):
    if value.dtype == torch.bool:
        return fr.Tensor.from_bool(value.reshape(-1).tolist(), list(value.shape))
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


class TransformerTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected.detach(), atol=2e-5, rtol=3e-4)

    def copy(self, model, reference):
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(parameters[name]))

    def gradients(self, model, reference, actual, expected, inputs):
        self.close(actual, expected)
        probe = torch.randn(expected.shape, generator=torch.Generator().manual_seed(817))
        (actual * tensor(probe)).sum().backward()
        (expected * probe).sum().backward()
        parameters = dict(reference.named_parameters())
        for name, got, wanted in list(inputs) + [(name, p.tensor(), parameters[name]) for name, p in model.named_parameters()]:
            with self.subTest(gradient=name):
                if wanted.grad is None:
                    self.assertIsNone(got.grad)
                else:
                    self.assertIsNotNone(got.grad)
                    self.close(got.grad, wanted.grad)

    def masks(self, batch, length, source, heads=2, boolean=False, per_head=False, unbatched=False):
        shape = (batch * heads, length, source) if per_head else (length, source)
        if boolean:
            mask = torch.rand(shape) > .75
            mask[..., 0] = False
            padding = torch.zeros((source,) if unbatched else (batch, source), dtype=torch.bool)
            if source > 1:
                padding[..., -1] = True
        else:
            mask = (torch.randn(shape) * .1).requires_grad_()
            padding = (torch.randn((source,) if unbatched else (batch, source)) * .1).requires_grad_()
        return mask, padding

    def test_encoder_layer_layout_bias_activation_norm_and_mask_gradients(self):
        for first, pre, bias, activation, boolean, unbatched, per_head in (
                (False, False, True, 'relu', False, False, False),
                (True, True, True, 'gelu', False, False, True),
                (False, True, False, 'gelu', True, True, True),
                (True, False, False, 'relu', True, False, False)):
            with self.subTest(batch_first=first, norm_first=pre, bias=bias, activation=activation, boolean=boolean, unbatched=unbatched):
                torch.manual_seed(701)
                config = dict(dim_feedforward=7, dropout=0., activation=activation, batch_first=first, norm_first=pre, bias=bias)
                model, reference = TransformerEncoderLayer(4, 2, **config), torch.nn.TransformerEncoderLayer(4, 2, **config)
                self.copy(model, reference)
                batch, length = (1 if unbatched else 2), 4
                shape = (length, 4) if unbatched else (batch, length, 4) if first else (length, batch, 4)
                x = torch.randn(shape, requires_grad=True)
                mask, padding = self.masks(batch, length, length, boolean=boolean, per_head=per_head, unbatched=unbatched)
                sx, sm, sp = tensor(x, True), tensor(mask, not boolean), tensor(padding, not boolean)
                self.gradients(model, reference, model(sx, sm, sp), reference(x, mask, padding),
                               [('src', sx, x)] + ([] if boolean else [('mask', sm, mask), ('padding', sp, padding)]))

    def test_decoder_layer_self_cross_attention_all_mask_gradients(self):
        for first, pre, bias, boolean, unbatched in ((False, False, True, False, False),
                                                    (True, True, False, False, False),
                                                    (False, True, True, True, True)):
            with self.subTest(first=first, pre=pre, bias=bias, boolean=boolean, unbatched=unbatched):
                torch.manual_seed(709)
                config = dict(dim_feedforward=9, dropout=0., activation='gelu', batch_first=first, norm_first=pre, bias=bias)
                model, reference = TransformerDecoderLayer(4, 2, **config), torch.nn.TransformerDecoderLayer(4, 2, **config)
                self.copy(model, reference)
                batch, length, source = (1 if unbatched else 2), 3, 5
                shape = lambda n: (n, 4) if unbatched else (batch, n, 4) if first else (n, batch, 4)
                target, memory = torch.randn(shape(length), requires_grad=True), torch.randn(shape(source), requires_grad=True)
                tm, tp = self.masks(batch, length, length, boolean=boolean, per_head=True, unbatched=unbatched)
                mm, mp = self.masks(batch, length, source, boolean=boolean, per_head=True, unbatched=unbatched)
                st, smem = tensor(target, True), tensor(memory, True)
                native_masks = [tensor(value, not boolean) for value in (tm, mm, tp, mp)]
                actual = model(st, smem, *native_masks)
                expected = reference(target, memory, tm, mm, tp, mp)
                pairs = [('tgt', st, target), ('memory', smem, memory)]
                if not boolean:
                    pairs += [(name, actual_mask, expected_mask) for name, actual_mask, expected_mask in zip(('tgt mask', 'memory mask', 'tgt padding', 'memory padding'), native_masks, (tm, mm, tp, mp))]
                self.gradients(model, reference, actual, expected, pairs)

    def test_causal_hints_auto_detection_and_ignored_mask_gradients(self):
        for padding in (False, True):
            with self.subTest(padding=padding):
                torch.manual_seed(719)
                model = TransformerEncoder(TransformerEncoderLayer(4, 2, 7, 0., batch_first=True), 2)
                reference = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(4, 2, 7, 0., batch_first=True), 2, enable_nested_tensor=False)
                self.copy(model, reference)
                x = torch.randn(2, 3, 4, requires_grad=True)
                mask = torch.nn.Transformer.generate_square_subsequent_mask(3).requires_grad_()
                pad = torch.zeros(2, 3, requires_grad=True) if padding else None
                sx, sm, sp = tensor(x, True), tensor(mask, True), None if pad is None else tensor(pad, True)
                self.gradients(model, reference, model(sx, sm, sp), reference(x, mask, pad),
                               [('input', sx, x), ('mask', sm, mask)] + ([] if pad is None else [('padding', sp, pad)]))
                self.assertFalse(model.use_nested_tensor)
        model, reference = TransformerDecoderLayer(4, 2, 7, 0.), torch.nn.TransformerDecoderLayer(4, 2, 7, 0.)
        self.copy(model, reference)
        tgt, mem = torch.randn(3, 2, 4, requires_grad=True), torch.randn(5, 2, 4, requires_grad=True)
        tm = torch.nn.Transformer.generate_square_subsequent_mask(3).requires_grad_()
        mm = torch.tensor([[0. if j <= i else -float('inf') for j in range(5)] for i in range(3)], requires_grad=True)
        st, sm, stm, smm = tensor(tgt, True), tensor(mem, True), tensor(tm, True), tensor(mm, True)
        self.gradients(model, reference, model(st, sm, stm, smm, tgt_is_causal=True, memory_is_causal=True),
                       reference(tgt, mem, tm, mm, tgt_is_causal=True, memory_is_causal=True),
                       [('target', st, tgt), ('memory', sm, mem), ('target mask', stm, tm), ('memory mask', smm, mm)])

    def test_stacks_final_norm_and_full_transformer_gradients(self):
        for first, pre, bias, unbatched, dropout in ((False, False, True, False, 0.),
                                                   (True, True, False, False, 0.),
                                                   (False, True, True, True, .2)):
            with self.subTest(first=first, pre=pre, bias=bias, unbatched=unbatched, dropout=dropout):
                torch.manual_seed(727)
                config = dict(d_model=4, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=7,
                              dropout=dropout, activation='gelu', batch_first=first, norm_first=pre, bias=bias)
                model = Transformer(**config)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    reference = torch.nn.Transformer(**config)
                if dropout:
                    model.eval()
                    reference.eval()
                self.copy(model, reference)
                shape = lambda n: (n, 4) if unbatched else (2, n, 4) if first else (n, 2, 4)
                source, target = torch.randn(shape(4), requires_grad=True), torch.randn(shape(3), requires_grad=True)
                mask = (torch.randn(3, 4) * .1).requires_grad_()
                ss, st, sm = tensor(source, True), tensor(target, True), tensor(mask, True)
                self.gradients(model, reference, model(ss, st, memory_mask=sm), reference(source, target, memory_mask=mask),
                               [('source', ss, source), ('target', st, target), ('memory mask', sm, mask)])
        template = TransformerDecoderLayer(4, 2, 7, 0., bias=False)
        model = TransformerDecoder(template, 2, fr.nn.LayerNorm(4, bias=False))
        reference = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(4, 2, 7, 0., bias=False), 2, torch.nn.LayerNorm(4, bias=False))
        self.copy(model, reference)
        target, memory = torch.randn(3, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True)
        st, sm = tensor(target, True), tensor(memory, True)
        self.gradients(model, reference, model(st, sm), reference(target, memory), [('target', st, target), ('memory', sm, memory)])

    def test_template_clones_copy_weights_modes_freezing_and_independent_storage(self):
        template = TransformerEncoderLayer(4, 2, 7, .25, activation=fr.nn.GELU(approximate='tanh'))
        template.dropout1.eval()
        template.linear1.weight.trainable = False
        model = TransformerEncoder(template, 2)
        for layer in model.layers:
            self.assertEqual([p.tensor().tolist() for p in layer.parameters()], [p.tensor().tolist() for p in template.parameters()])
            self.assertFalse(layer.dropout1.training)
            self.assertFalse(layer.linear1.weight.trainable)
            self.assertIsNot(layer.linear1.weight, template.linear1.weight)
        self.assertIsNot(model.layers[0].linear1.weight, model.layers[1].linear1.weight)
        untouched = model.layers[1].linear1.weight.tensor().tolist()
        stream = model.layers[1].dropout.rng_state.tolist()
        with fr.no_grad():
            parameter = model.layers[0].linear1.weight.tensor()
            parameter.copy_(parameter + 1.)
            model.layers[0].dropout.rng_state.copy_(fr.nn.Dropout(.25, seed=17).rng_state)
        self.assertEqual(model.layers[1].linear1.weight.tensor().tolist(), untouched)
        self.assertEqual(template.linear1.weight.tensor().tolist(), untouched)
        self.assertEqual(model.layers[1].dropout.rng_state.tolist(), stream)
        self.assertEqual(template.dropout.rng_state.tolist(), stream)
        template(tensor(torch.randn(3, 2, 4))).sum().backward()
        gradients = [None if p.tensor().grad is None else p.tensor().grad.tolist() for p in template.parameters()]
        with self.assertRaises(ValueError):
            TransformerEncoder(template, 1)
        self.assertEqual([None if p.tensor().grad is None else p.tensor().grad.tolist() for p in template.parameters()], gradients)
        for parameter in template.parameters():
            parameter.zero_grad()
        template.norm2.weight = template.norm1.weight
        with self.assertRaises(ValueError):
            TransformerEncoder(template, 1)
        class CustomLayer(TransformerEncoderLayer):
            pass
        with self.assertRaises(NotImplementedError):
            TransformerEncoder(CustomLayer(4, 2, 7), 1)

    def test_known_functional_and_module_activations_match_torch(self):
        for activation, torch_activation in ((fr.nn.functional.relu, torch.nn.functional.relu),
                                               (fr.nn.functional.gelu, torch.nn.functional.gelu),
                                               (fr.nn.ReLU(), torch.nn.ReLU()),
                                               (fr.nn.GELU(approximate='tanh'), torch.nn.GELU(approximate='tanh'))):
            model = TransformerEncoderLayer(4, 2, 7, 0., activation=activation)
            reference = torch.nn.TransformerEncoderLayer(4, 2, 7, 0., activation=torch_activation)
            self.copy(model, reference)
            x = torch.randn(3, 2, 4, requires_grad=True)
            sx = tensor(x, True)
            self.gradients(model, reference, model(sx), reference(x), [('input', sx, x)])

    def test_fully_masked_attention_empty_memory_and_unit_dropout_gradients(self):
        for pre, dropout, empty_memory in ((False, 0., False), (True, 0., True), (False, 1., False), (True, 1., False)):
            with self.subTest(norm_first=pre, dropout=dropout, empty_memory=empty_memory):
                torch.manual_seed(743)
                options = dict(dim_feedforward=7, dropout=dropout, norm_first=pre, batch_first=True)
                model, reference = TransformerDecoderLayer(4, 2, **options), torch.nn.TransformerDecoderLayer(4, 2, **options)
                self.copy(model, reference)
                target = torch.randn(2, 3, 4, requires_grad=True)
                memory = torch.randn(2, 0 if empty_memory else 5, 4, requires_grad=True)
                mask = torch.ones(3, 3, dtype=torch.bool)
                st, sm, mask_native = tensor(target, True), tensor(memory, True), tensor(mask)
                self.gradients(model, reference, model(st, sm, tgt_mask=mask_native), reference(target, memory, tgt_mask=mask),
                               [('target', st, target), ('memory', sm, memory)])

    def test_custom_encoder_decoder_registration_and_forward_contract(self):
        class Encoder(fr.nn.Module):
            def build(self):
                self.proj = fr.nn.Linear(4, 4)
            def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
                self._hint = is_causal
                return self.proj(src)
        class Decoder(fr.nn.Module):
            def build(self):
                self.proj = fr.nn.Linear(4, 4)
            def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                        memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
                self._hints = (tgt_is_causal, memory_is_causal)
                return self.proj(tgt) + memory.mean_dim(1, True)
        class TorchEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
            def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
                return self.proj(src)
        class TorchDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
            def forward(self, tgt, memory, **kwargs):
                return self.proj(tgt) + memory.mean(1, keepdim=True)
        encoder, decoder = Encoder(), Decoder()
        model = Transformer(4, 2, 0, 0, custom_encoder=encoder, custom_decoder=decoder, batch_first=True)
        reference = torch.nn.Transformer(4, 2, 0, 0, custom_encoder=TorchEncoder(), custom_decoder=TorchDecoder(), batch_first=True)
        self.assertIs(model.encoder, encoder)
        self.assertIs(model.decoder, decoder)
        self.copy(model, reference)
        source, target = torch.randn(2, 5, 4, requires_grad=True), torch.randn(2, 3, 4, requires_grad=True)
        ss, st = tensor(source, True), tensor(target, True)
        kwargs = dict(src_is_causal=False, tgt_is_causal=True, memory_is_causal=True)
        self.gradients(model, reference, model(ss, st, **kwargs), reference(source, target, **kwargs), [('source', ss, source), ('target', st, target)])
        self.assertIs(encoder._hint, False)
        self.assertEqual(decoder._hints, (True, True))

    def test_dropout_checkpoint_continuation_preserves_all_streams_and_optimizer(self):
        torch.manual_seed(751)
        def factory():
            return Transformer(4, 2, 2, 2, 7, dropout=.25, batch_first=True, norm_first=True)
        model = factory()
        rng = fr.Generator(19)
        opt = fr.optim.AdamW(model.parameters(), lr=.01)
        source, target, labels = [tensor(torch.randn(2, 3, 4)) for _ in range(3)]
        def step(current, optimizer, generator):
            noisy = source + fr.Tensor.rand([2, 3, 4], generator=generator) * .01
            output = current(noisy, target)
            difference = output - labels
            loss = (difference * difference).mean()
            result = output.tolist(), loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return result
        step(model, opt, rng)
        state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            state.save(directory / 'mid')
            expected = [step(model, opt, rng) for _ in range(3)]
            end = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=4)
            other, other_rng = factory(), fr.Generator(999)
            other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
            self.assertEqual(fr.checkpoint.load_training(directory / 'mid', other, optimizers={'main': other_opt}, generator=other_rng), 1)
            self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
            actual = fr.checkpoint.snapshot_training(other, optimizers={'main': other_opt}, generator=other_rng, step=4)
            for checkpoint, name in ((end, 'expected'), (actual, 'actual')):
                checkpoint.save(directory / name)
            metadata = [json.loads((directory / name / 'checkpoint.json').read_bytes()) for name in ('expected', 'actual')]
            payloads = [(directory / name / record.pop('generation') / 'model.safetensors').read_bytes() for name, record in zip(('expected', 'actual'), metadata)]
            self.assertEqual(metadata[0], metadata[1])
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(fr.Tensor.rand([8], generator=rng).tolist(), fr.Tensor.rand([8], generator=other_rng).tolist())
        model.eval()
        buffers = {name: buffer.tolist() for name, buffer in model.named_buffers()}
        self.assertEqual(model(source, target).tolist(), model(source, target).tolist())
        self.assertEqual(buffers, {name: buffer.tolist() for name, buffer in model.named_buffers()})

    def test_mask_factory_empties_strided_input_and_invalid_boundaries(self):
        for size in (0, 4):
            for dtype, native_dtype in ((torch.float32, 'float32'), (torch.bool, 'bool')):
                actual = Transformer.generate_square_subsequent_mask(size, dtype=native_dtype)
                expected = torch.nn.Transformer.generate_square_subsequent_mask(size, dtype=dtype)
                self.assertEqual(actual.tolist(), expected.tolist())
        model, reference = TransformerEncoderLayer(4, 2, 7, 0.), torch.nn.TransformerEncoderLayer(4, 2, 7, 0.)
        self.copy(model, reference)
        raw = torch.randn(4, 2, 3, requires_grad=True)
        native = tensor(raw, True)
        self.gradients(model, reference, model(native.transpose(0, 2)), reference(raw.transpose(0, 2)), [('strided', native, raw)])
        self.assertEqual(model(fr.Tensor.zeros([0, 2, 4])).shape, [0, 2, 4])
        for factory in (lambda: TransformerEncoderLayer(0, 1), lambda: TransformerEncoderLayer(4, 3),
                        lambda: TransformerDecoderLayer(4, 2, dim_feedforward=0), lambda: TransformerEncoderLayer(4, 2, dropout=float('nan')),
                        lambda: TransformerEncoderLayer(4, 2, layer_norm_eps=-1), lambda: Transformer(4, 2, 0, 1),
                        lambda: TransformerEncoderLayer(sys.maxsize, 1), lambda: Transformer.generate_square_subsequent_mask(sys.maxsize)):
            with self.assertRaises((ValueError, TypeError)):
                factory()
        for factory in (lambda: TransformerEncoderLayer(4, 2, dtype='float64'), lambda: TransformerDecoderLayer(4, 2, device='cuda'),
                        lambda: TransformerEncoderLayer(4, 2, activation=lambda x: x.tanh())):
            with self.assertRaises(NotImplementedError):
                factory()
        x = tensor(torch.randn(3, 2, 4))
        for call in (lambda: model(x, is_causal=True), lambda: model(x, src_mask=tensor(torch.ones(3, 4))),
                     lambda: model(x, src_key_padding_mask=tensor(torch.zeros(3, 2))), lambda: model([x, x]),
                     lambda: model(fr.tensor([[[1., 2., 3., 4.]]], dtype='float64'))):
            with self.assertRaises((ValueError, TypeError)):
                call()
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: model(x))
        with self.assertRaises(AttributeError):
            model.norm_first = True
        for parameter in model.parameters():
            parameter.zero_grad()
        state = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(1))
        with self.assertRaises(ValueError):
            state.restore(TransformerEncoderLayer(4, 2, 7, 0., norm_first=True), optimizers={}, generator=fr.Generator(2))

    def test_invalid_attention_metadata_preserves_dropout_streams(self):
        target, memory = tensor(torch.randn(2, 3, 4)), tensor(torch.randn(2, 5, 4))
        layer = TransformerDecoderLayer(4, 2, 7, .3, batch_first=True)
        transformer = Transformer(4, 2, 2, 2, 7, .3, batch_first=True)
        for model, call in ((layer, lambda **kw: layer(target, memory, **kw)),
                            (transformer, lambda **kw: transformer(memory, target, **kw))):
            before = {name: value.tolist() for name, value in model.named_buffers()}
            for keywords in ({'memory_mask': tensor(torch.randn(3, 4))},
                             {'memory_key_padding_mask': tensor(torch.randn(1, 5))},
                             {'memory_is_causal': True}, {'tgt_is_causal': True}):
                with self.assertRaises(ValueError):
                    call(**keywords)
                self.assertEqual(before, {name: value.tolist() for name, value in model.named_buffers()})
        before = {name: value.tolist() for name, value in layer.named_buffers()}
        with self.assertRaises(ValueError):
            layer(target, tensor(torch.randn(1, 5, 4)))
        self.assertEqual(before, {name: value.tolist() for name, value in layer.named_buffers()})


if __name__ == '__main__':
    unittest.main(verbosity=2)
