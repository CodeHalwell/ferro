"""Packed sequence semantics, projected recurrent gradients and dropout restart."""
from pathlib import Path
import sys
import tempfile
import unittest

import ferro as fr
from ferro.nn import RNN, GRU, LSTM
from ferro.nn.utils.rnn import (PackedSequence, pack_padded_sequence, pad_packed_sequence,
                                pack_sequence, pad_sequence, unpad_sequence)
import torch
from torch.nn.utils import rnn as reference_utils


def tensor(value, requires_grad=False):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def close(actual, expected, name):
    if actual.shape != list(expected.shape):
        raise AssertionError(f'{name}: shape {actual.shape} differs from {list(expected.shape)}')
    got = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
    torch.testing.assert_close(got, expected.detach(), atol=1e-5, rtol=1e-4,
                               msg=lambda message: name + ': ' + message)


def copy_weights(model, reference):
    actual, expected = dict(model.named_parameters()), dict(reference.named_parameters())
    if list(actual) != list(expected):
        raise AssertionError('recurrent parameter names/order differ from PyTorch')
    with fr.no_grad():
        for name, parameter in actual.items():
            parameter.tensor().copy_(tensor(expected[name]))


def gradients(actual, expected, leaves):
    generator = torch.Generator().manual_seed(91)
    loss, reference_loss = None, None
    for index, (result, reference) in enumerate(zip(actual, expected)):
        close(result, reference, f'output {index}')
        coefficients = torch.randn(reference.shape, generator=generator)
        term = (result * tensor(coefficients)).sum() + (result * result).sum() * 0.03
        wanted = (reference * coefficients).sum() + (reference * reference).sum() * 0.03
        loss = term if loss is None else loss + term
        reference_loss = wanted if reference_loss is None else reference_loss + wanted
    close(loss, reference_loss, 'loss')
    loss.backward()
    reference_loss.backward()
    for name, actual, expected in leaves:
        if actual.grad is None or expected.grad is None:
            raise AssertionError('missing gradient: ' + name)
        close(actual.grad, expected.grad, name + ' gradient')


class PackedRecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_recurrent(self, kind, options, supplied=True, packed=True, unbatched=False, batch=4, train=True):
        cls = {'rnn': RNN, 'relu': RNN, 'gru': GRU, 'lstm': LSTM}[kind]
        reference_cls = {'rnn': torch.nn.RNN, 'relu': torch.nn.RNN, 'gru': torch.nn.GRU, 'lstm': torch.nn.LSTM}[kind]
        options = dict(options, **({'nonlinearity': 'relu'} if kind == 'relu' else {}))
        model = cls(2, 4, **options)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            reference = reference_cls(2, 4, **options)
        copy_weights(model, reference)
        model.train(train)
        reference.train(train)
        generator = torch.Generator().manual_seed(43)
        tx = torch.randn([6, 2] if unbatched else [6, batch, 2], generator=generator, requires_grad=True)
        x = tensor(tx, True)
        inputs, torch_inputs = x, tx
        if options.get('batch_first') and not unbatched:
            inputs, torch_inputs = x.transpose(0, 1), tx.transpose(0, 1)
        if packed:
            lengths = [2, 5, 3, 3]
            inputs = pack_padded_sequence(inputs, lengths, batch_first=options.get('batch_first', False), enforce_sorted=False)
            torch_inputs = reference_utils.pack_padded_sequence(torch_inputs, lengths, batch_first=options.get('batch_first', False), enforce_sorted=False)
            for field in ('data', 'batch_sizes', 'sorted_indices', 'unsorted_indices'):
                close(getattr(inputs, field), getattr(torch_inputs, field), 'packed input ' + field)
        count = options.get('num_layers', 1) * (2 if options.get('bidirectional') else 1)
        leaves, states, torch_states = [('input', x, tx)], [], []
        if supplied:
            for name in ('h_0', 'c_0') if kind == 'lstm' else ('h_0',):
                width = options.get('proj_size', 0) or 4 if name == 'h_0' else 4
                shape = [count, width] if unbatched else [batch, count, width]
                ts = torch.randn(shape, generator=generator, requires_grad=True)
                state = tensor(ts, True)
                leaves.append((name, state, ts))
                states.append(state if unbatched else state.transpose(0, 1))
                torch_states.append(ts if unbatched else ts.transpose(0, 1))
        hx = None if not supplied else tuple(states) if kind == 'lstm' else states[0]
        thx = None if not supplied else tuple(torch_states) if kind == 'lstm' else torch_states[0]
        result, state = model(inputs, hx)
        expected, reference_state = reference(torch_inputs, thx)
        if packed:
            self.assertIsInstance(result, PackedSequence)
            for field in ('batch_sizes', 'sorted_indices', 'unsorted_indices'):
                close(getattr(result, field), getattr(expected, field), field)
            padded, lengths = pad_packed_sequence(result, batch_first=True, padding_value=0.3, total_length=7)
            torch_padded, torch_lengths = reference_utils.pad_packed_sequence(expected, batch_first=True, padding_value=0.3, total_length=7)
            close(lengths, torch_lengths, 'original lengths')
            outputs, wanted = [result.data, padded], [expected.data, torch_padded]
        else:
            outputs, wanted = [result], [expected]
        outputs.extend(state if kind == 'lstm' else [state])
        wanted.extend(reference_state if kind == 'lstm' else [reference_state])
        leaves.extend((name, parameter.tensor(), dict(reference.named_parameters())[name]) for name, parameter in model.named_parameters())
        gradients(outputs, wanted, leaves)
        if packed:
            for index, length in enumerate([2, 5, 3, 3]):
                self.assertEqual(x.grad.narrow(1, index, 1).narrow(0, length, 6 - length).sum().item(), 0.)

    def test_packed_outputs_order_and_all_gradients_against_torch(self):
        options = ({}, {'num_layers': 2, 'bidirectional': True, 'batch_first': True},
                   {'num_layers': 3, 'bidirectional': True, 'bias': False})
        for kind in ('rnn', 'relu', 'gru', 'lstm'):
            for settings in options:
                for supplied in (False, True):
                    with self.subTest(kind=kind, options=settings, supplied=supplied):
                        self.check_recurrent(kind, settings, supplied)

    def test_projected_lstm_dense_packed_unbatched_and_empty_gradients(self):
        for packed, unbatched, batch in ((True, False, 4), (False, False, 3), (False, True, 1), (False, False, 0)):
            for bias in (False, True):
                with self.subTest(packed=packed, unbatched=unbatched, batch=batch, bias=bias):
                    self.check_recurrent('lstm', {'proj_size': 2, 'num_layers': 2, 'bidirectional': True,
                                                 'batch_first': True, 'bias': bias}, packed=packed,
                                         unbatched=unbatched, batch=batch)

    def test_dropout_one_and_eval_match_torch_without_rng_sequence_assumptions(self):
        for kind in ('rnn', 'gru', 'lstm'):
            for packed in (False, True):
                for p, train in ((1., True), (0.35, False)):
                    options = {'num_layers': 2, 'bidirectional': True, 'dropout': p}
                    if kind == 'lstm':
                        options['proj_size'] = 2
                    with self.subTest(kind=kind, packed=packed, p=p, train=train):
                        self.check_recurrent(kind, options, packed=packed, train=train)

    def test_pack_padding_and_unpadding_preserve_values_and_gradients(self):
        generator = torch.Generator().manual_seed(19)
        for batch_first in (False, True):
            for side in ('left', 'right'):
                with self.subTest(batch_first=batch_first, side=side):
                    sources = [torch.randn([length, 2, 3], generator=generator, requires_grad=True) for length in (2, 0, 4)]
                    actual = [tensor(value, True) for value in sources]
                    result = pad_sequence(actual, batch_first, -0.4, padding_side=side)
                    expected = reference_utils.pad_sequence(sources, batch_first, -0.4, padding_side=side)
                    gradients([result], [expected], [(f'sequence {index}', value, sources[index]) for index, value in enumerate(actual)])
        for batch_first in (False, True):
            tx = torch.randn([5, 3, 2], generator=generator, requires_grad=True)
            x = tensor(tx, True)
            value, reference = (x.transpose(0, 1), tx.transpose(0, 1)) if batch_first else (x, tx)
            lengths = fr.Tensor.from_i64([3, 5, 0], [3])
            outputs = unpad_sequence(value, lengths, batch_first)
            # Torch unpad_sequence transposes its argument in place.
            expected = reference_utils.unpad_sequence(reference.clone(), torch.tensor([3, 5, 0]), batch_first)
            gradients(outputs, expected, [('padded input', x, tx)])
        sources = [torch.randn([length, 2], generator=generator, requires_grad=True) for length in (2, 4, 3)]
        actual = [tensor(value, True) for value in sources]
        packed = pack_sequence(actual, enforce_sorted=False)
        expected = reference_utils.pack_sequence(sources, enforce_sorted=False)
        gradients([packed.data], [expected.data], [(f'packed sequence {i}', value, sources[i]) for i, value in enumerate(actual)])

    def test_sorted_packing_scalar_features_and_typed_padding(self):
        for dtype, values in (('bool', [[True, False], [False, True], [True, True]]),
                              ('int64', [[(1 << 53) + 1, 3], [4, 5], [6, 7]]),
                              ('float64', [[1e-300, 3.], [4., 5.], [6., 7.]])):
            with self.subTest(dtype=dtype):
                x = fr.tensor(values, dtype=dtype)
                tx = torch.tensor(values, dtype=getattr(torch, dtype))
                packed = pack_padded_sequence(x, [3, 1])
                expected = reference_utils.pack_padded_sequence(tx, [3, 1])
                self.assertIsNone(packed.sorted_indices)
                self.assertIsNone(packed.unsorted_indices)
                self.assertEqual(packed.data.dtype, dtype)
                self.assertEqual(packed.data.tolist(), expected.data.tolist())
                result, lengths = pad_packed_sequence(packed, total_length=4, padding_value=1)
                wanted, tl = reference_utils.pad_packed_sequence(expected, total_length=4, padding_value=1)
                self.assertEqual(result.tolist(), wanted.tolist())
                self.assertEqual(lengths.tolist(), tl.tolist())
                self.assertIs(packed.cpu(), packed)
                self.assertIs(packed.to('cpu'), packed)
                self.assertFalse(packed.is_pinned())
        sequence = fr.Tensor.zeros([0, 2]).requires_grad_(True)
        padded = pad_sequence([sequence, sequence], batch_first=True)
        self.assertEqual(padded.shape, [2, 0, 2])
        padded.sum().backward()
        self.assertEqual(sequence.grad.shape, [0, 2])

    def test_invalid_lengths_metadata_projection_and_bounds(self):
        x = fr.Tensor.zeros([4, 3, 2])
        for lengths in ([4, 0, 2], [5, 2, 1], [4, 2], [2, 4, 1], [4., 3., 2.], [True, 2, 1], fr.Tensor.zeros([3])):
            with self.subTest(lengths=lengths), self.assertRaises((TypeError, ValueError)):
                pack_padded_sequence(x, lengths)
        with self.assertRaises(ValueError):
            pack_padded_sequence(fr.Tensor.zeros([4, 0, 2]), [])
        data = fr.Tensor.zeros([5, 2])
        for values in ([3, 2, 1], [2, 3], [3, 0, 2], []):
            with self.subTest(batch_sizes=values), self.assertRaises(ValueError):
                PackedSequence(data, fr.Tensor.from_i64(values, [len(values)]))
        sizes = fr.Tensor.from_i64([3, 2], [2])
        for order in ([0, 0, 2], [0, 1], [-1, 1, 2]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                PackedSequence(data, sizes, fr.Tensor.from_i64(order, [len(order)]))
        order = fr.Tensor.from_i64([2, 0, 1], [3])
        with self.assertRaises(ValueError):
            PackedSequence(data, sizes, order, order)
        packed = PackedSequence(data, sizes, order)
        self.assertEqual(packed.unsorted_indices.tolist(), [1, 2, 0])
        with self.assertRaises(ValueError):
            GRU(2, 3)(packed._replace(batch_sizes=fr.Tensor.from_i64([5], [1])))
        with self.assertRaises(ValueError):
            pad_packed_sequence(packed, total_length=1)
        with self.assertRaises(ValueError):
            pad_packed_sequence(packed, total_length=sys.maxsize)
        for projection in (-1, 4, 5, True, 1.5):
            with self.subTest(projection=projection), self.assertRaises(ValueError):
                LSTM(2, 4, proj_size=projection)
        for build in (lambda: LSTM(sys.maxsize, 4, proj_size=2), lambda: LSTM(2, sys.maxsize, proj_size=2),
                      lambda: GRU(1, 1, num_layers=sys.maxsize), lambda: fr.nn.LSTMCell(sys.maxsize, 2)):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                build()
        huge_empty = fr.Tensor.zeros([sys.maxsize, 0, 1])
        with self.assertRaisesRegex(ValueError, 'capacity'):
            GRU(1, 2)(huge_empty)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            RNN(1, 1)(huge_empty)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            pack_padded_sequence(fr.Tensor.zeros([sys.maxsize, 1, 0]), [sys.maxsize])
        projected = LSTM(2, 4, proj_size=2)
        with self.assertRaises(ValueError):
            projected(x, (fr.Tensor.zeros([1, 3, 4]), fr.Tensor.zeros([1, 3, 4])))
        with self.assertRaises(ValueError):
            projected(x, (fr.Tensor.zeros([1, 3, 2]), fr.Tensor.zeros([1, 3, 2])))
        for sequences in ([], [fr.Tensor.zeros([])], [fr.Tensor.zeros([2, 3]), fr.Tensor.zeros([3, 4])]):
            with self.assertRaises(ValueError):
                pad_sequence(sequences)
        with self.assertRaises(ValueError):
            unpad_sequence(x, [4, 3, 5])
        with self.assertRaises(NotImplementedError):
            packed.to('cuda')

    def test_dropout_streams_are_registered_advance_only_valid_rows_and_eval_is_pure(self):
        model = LSTM(2, 4, num_layers=3, bidirectional=True, proj_size=2, dropout=0.3)
        self.assertEqual([name for name, _ in model.named_buffers()],
                         ['interlayer_dropout.0.rng_state', 'interlayer_dropout.1.rng_state'])
        x = fr.Tensor.randn([6, 4, 2], seed=2)
        packed = pack_padded_sequence(x, [2, 5, 3, 3], enforce_sorted=False)
        initial = [layer.rng_state.tolist() for layer in model.interlayer_dropout]
        model.eval()
        before = model(packed)[0].data.tolist()
        self.assertEqual(model(packed)[0].data.tolist(), before)
        self.assertEqual([layer.rng_state.tolist() for layer in model.interlayer_dropout], initial)
        model.train()
        model(packed)
        for layer in model.interlayer_dropout:
            self.assertEqual(layer._offset, sum([2, 5, 3, 3]) * 2 * 2)
        self.assertEqual(list(GRU(2, 3, num_layers=1, dropout=0.5).named_buffers()), [])
        self.assertEqual(list(GRU(2, 3, num_layers=2, dropout=0.).named_buffers()), [])

    def test_stochastic_packed_interlayer_dropout_matches_explicit_torch_mask_oracle(self):
        model = GRU(2, 3, num_layers=2, bidirectional=True, dropout=0.35)
        model.reset_parameters(generator=fr.Generator(71))
        layers = [torch.nn.GRU(2, 3, bidirectional=True), torch.nn.GRU(6, 3, bidirectional=True)]
        parameter_pairs = []
        with torch.no_grad():
            for index, layer in enumerate(layers):
                for name, parameter in layer.named_parameters():
                    actual = dict(model.named_parameters())[name.replace('_l0', f'_l{index}')].tensor()
                    parameter.copy_(torch.tensor(actual.tolist()))
                    parameter_pairs.append((name + f' layer {index}', actual, parameter))
        generator = torch.Generator().manual_seed(29)
        tx = torch.randn([5, 3, 2], generator=generator, requires_grad=True)
        th = torch.randn([4, 3, 3], generator=generator, requires_grad=True)
        x, h = tensor(tx, True), tensor(th, True)
        packed = pack_padded_sequence(x, [3, 5, 2], enforce_sorted=False)
        torch_packed = reference_utils.pack_padded_sequence(tx, [3, 5, 2], enforce_sorted=False)
        seed, offset = model.interlayer_dropout[0]._stream()
        mask = fr.nn.functional.dropout(fr.Tensor.ones([10, 6]), 0.35, True, seed=seed, offset=offset)
        self.assertEqual(set(round(value, 6) for row in mask.tolist() for value in row), {0., round(1 / 0.65, 6)})
        first, first_h = layers[0](torch_packed, th[:2])
        masked = reference_utils.PackedSequence(first.data * torch.tensor(mask.tolist()), first.batch_sizes,
                                                first.sorted_indices, first.unsorted_indices)
        expected, second_h = layers[1](masked, th[2:])
        result, final = model(packed, h)
        gradients([result.data, final], [expected.data, torch.cat([first_h, second_h], 0)],
                  [('input', x, tx), ('initial h', h, th), *parameter_pairs])

    def test_dropout_probability_changes_affect_forward_modes_and_checkpoint_configuration(self):
        model, reference = GRU(2, 3, num_layers=2, dropout=0.2), torch.nn.GRU(2, 3, num_layers=2, dropout=1.)
        copy_weights(model, reference)
        identities = [id(parameter) for parameter in model.parameters()]
        model.dropout = 1.
        tx = torch.randn([4, 2, 2], generator=torch.Generator().manual_seed(67), requires_grad=True)
        x = tensor(tx, True)
        actual, ah = model(x)
        expected, eh = reference(tx)
        gradients([actual, ah], [expected, eh], [('input', x, tx)] +
                  [(name, p.tensor(), dict(reference.named_parameters())[name]) for name, p in model.named_parameters()])
        for parameter in model.parameters():
            parameter.zero_grad()
        checkpoint = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(9))
        target = GRU(2, 3, num_layers=2, dropout=1.)
        checkpoint.restore(target, optimizers={}, generator=fr.Generator(11))
        self.assertEqual(target(x)[0].tolist(), model(x)[0].tolist())
        model.dropout = 0.
        self.assertEqual(list(model.named_buffers()), [])
        model.eval()
        model.dropout = 0.4
        self.assertFalse(model.interlayer_dropout[0].training)
        self.assertEqual(model.interlayer_dropout[0].p, 0.4)
        self.assertEqual([id(parameter) for parameter in model.parameters()], identities)
        before = model.dropout
        for value in (-1, 2, float('nan'), True):
            with self.assertRaises(ValueError):
                model.dropout = value
            self.assertEqual(model.dropout, before)

    def test_packed_projected_stochastic_checkpoint_exact_continuation(self):
        options = {'num_layers': 2, 'bidirectional': True, 'proj_size': 2, 'dropout': 0.35}
        model = LSTM(2, 4, **options)
        model.reset_parameters(generator=fr.Generator(7))
        optimizer, rng = fr.optim.AdamW(model.parameters(), lr=0.01), fr.Generator(11)
        def step(module, opt, stream):
            opt.zero_grad()
            x = fr.Tensor.randn([5, 3, 2], generator=stream)
            packed = pack_padded_sequence(x, [2, 5, 3], enforce_sorted=False)
            output, (h, c) = module(packed)
            loss = output.data.pow(2).mean() + h.pow(2).mean() * 0.2 + c.pow(2).mean() * 0.1
            value = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return value
        def snapshot(module, opt, stream, number):
            return fr.checkpoint.snapshot_training(module, optimizers={'main': opt}, generator=stream, step=number)
        def state(checkpoint):
            return {name: (value.dtype, value.shape, value.tolist()) for name, value in checkpoint.tensors().items()}
        for _ in range(2):
            step(model, optimizer, rng)
        checkpoint = snapshot(model, optimizer, rng, 2)
        expected = [step(model, optimizer, rng) for _ in range(3)]
        expected_state = state(snapshot(model, optimizer, rng, 5))
        expected_draw = fr.Tensor.randn([7], generator=rng).tolist()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint.save(Path(directory) / 'packed')
            saved = fr.checkpoint.load(Path(directory) / 'packed')
            target = LSTM(2, 4, **options)
            target.eval()
            target_opt, target_rng = fr.optim.AdamW(target.parameters(), lr=0.5), fr.Generator(99)
            self.assertEqual(saved.restore(target, optimizers={'main': target_opt}, generator=target_rng), 2)
            self.assertTrue(target.training)
            self.assertTrue(target.interlayer_dropout[0].training)
            self.assertEqual([step(target, target_opt, target_rng) for _ in range(3)], expected)
            self.assertEqual(state(snapshot(target, target_opt, target_rng, 5)), expected_state)
            self.assertEqual(fr.Tensor.randn([7], generator=target_rng).tolist(), expected_draw)
            bad = LSTM(2, 4, **{**options, 'dropout': 0.2})
            bad_opt, bad_rng = fr.optim.AdamW(bad.parameters()), fr.Generator(13)
            before = state(snapshot(bad, bad_opt, bad_rng, 0))
            with self.assertRaises(ValueError):
                saved.restore(bad, optimizers={'main': bad_opt}, generator=bad_rng)
            self.assertEqual(state(snapshot(bad, bad_opt, bad_rng, 0)), before)


if __name__ == '__main__':
    unittest.main()
