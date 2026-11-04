"""Dense recurrent module layouts, gradients and restart versus actual PyTorch."""
import math
from pathlib import Path
import tempfile
import unittest

import ferro as fr
from ferro.nn.recurrent import RNN, GRU, LSTM, RNNCell, GRUCell, LSTMCell
import torch


def tensor(value, requires_grad=False):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def close(actual, expected, name):
    if actual.shape != list(expected.shape):
        raise AssertionError(f'{name}: shape {actual.shape} differs from {list(expected.shape)}')
    got = torch.tensor(actual.tolist(), dtype=torch.float32).reshape(expected.shape)
    torch.testing.assert_close(got, expected.detach(), atol=1e-5, rtol=1e-4,
                               msg=lambda message: name + ': ' + message)


def copy_weights(model, reference):
    actual = dict(model.named_parameters())
    expected = dict(reference.named_parameters())
    if list(actual) != list(expected):
        raise AssertionError('Recurrent parameter names/order differ from PyTorch')
    with fr.no_grad():
        for name, parameter in actual.items():
            parameter.tensor().copy_(tensor(expected[name]))


def loss_and_gradients(actual, expected, model, reference, leaves):
    generator = torch.Generator().manual_seed(491)
    loss, torch_loss = None, None
    for index, (got, wanted) in enumerate(zip(actual, expected)):
        close(got, wanted, f'output/state {index}')
        weight = torch.randn(wanted.shape, generator=generator)
        term = (got * tensor(weight)).sum() + (got * got).sum() * 0.03
        torch_term = (wanted * weight).sum() + (wanted * wanted).sum() * 0.03
        loss = term if loss is None else loss + term
        torch_loss = torch_term if torch_loss is None else torch_loss + torch_term
    close(loss, torch_loss, 'loss')
    loss.backward()
    torch_loss.backward()
    pairs = leaves + [(name, parameter.tensor(), dict(reference.named_parameters())[name])
                      for name, parameter in model.named_parameters()]
    for name, got, wanted in pairs:
        if got.grad is None or wanted.grad is None:
            raise AssertionError('Missing recurrent gradient: ' + name)
        close(got.grad, wanted.grad, name + ' gradient')


class RecurrentModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_sequence(self, kind, options, unbatched=False, supplied=True, batch=2):
        classes = {'rnn': (RNN, torch.nn.RNN), 'relu': (RNN, torch.nn.RNN),
                   'gru': (GRU, torch.nn.GRU), 'lstm': (LSTM, torch.nn.LSTM)}
        actual_type, expected_type = classes[kind]
        options = dict(options)
        if kind == 'relu':
            options['nonlinearity'] = 'relu'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            reference = expected_type(2, 3, **options)
        model = actual_type(2, 3, **options)
        copy_weights(model, reference)
        generator = torch.Generator().manual_seed(57)
        tx = torch.randn([4, 2] if unbatched else [4, batch, 2], generator=generator, requires_grad=True)
        x = tensor(tx, True)
        inputs, torch_inputs = x, tx
        if options.get('batch_first') and not unbatched:
            inputs, torch_inputs = x.transpose(0, 1), tx.transpose(0, 1)
        leaves, states, torch_states = [('input', x, tx)], [], []
        count = options.get('num_layers', 1) * (2 if options.get('bidirectional') else 1)
        if supplied:
            for name in ('h_0', 'c_0') if kind == 'lstm' else ('h_0',):
                shape = [count, 3] if unbatched else [batch, count, 3]
                source = torch.randn(shape, generator=generator, requires_grad=True)
                value = tensor(source, True)
                leaves.append((name, value, source))
                states.append(value if unbatched else value.transpose(0, 1))
                torch_states.append(source if unbatched else source.transpose(0, 1))
        hx = None if not supplied else tuple(states) if kind == 'lstm' else states[0]
        thx = None if not supplied else tuple(torch_states) if kind == 'lstm' else torch_states[0]
        output, final = model(inputs, hx)
        wanted, tfinal = reference(torch_inputs, thx)
        actual_outputs = [output, *final] if kind == 'lstm' else [output, final]
        expected_outputs = [wanted, *tfinal] if kind == 'lstm' else [wanted, tfinal]
        loss_and_gradients(actual_outputs, expected_outputs, model, reference, leaves)
        self.assertEqual(len(model.all_weights), count)
        self.assertEqual([len(group) for group in model.all_weights], [4 if options.get('bias', True) else 2] * count)
        self.assertIsNone(model.flatten_parameters())

    def test_sequence_outputs_and_all_gradients_match_torch(self):
        cases = (
            ({}, False),
            ({'num_layers': 2, 'bidirectional': True, 'batch_first': True}, False),
            ({'num_layers': 3, 'bidirectional': True, 'bias': False}, True),
            ({'num_layers': 2, 'bias': False, 'batch_first': True}, True),
        )
        for kind in ('rnn', 'relu', 'gru', 'lstm'):
            for options, unbatched in cases:
                for supplied in (False, True):
                    with self.subTest(kind=kind, options=options, unbatched=unbatched, supplied=supplied):
                        self.check_sequence(kind, options, unbatched, supplied)

    def test_empty_batch_keeps_input_state_and_parameter_gradient_paths(self):
        for kind in ('rnn', 'relu', 'gru', 'lstm'):
            with self.subTest(kind=kind):
                self.check_sequence(kind, {'num_layers': 2, 'bidirectional': True, 'batch_first': True}, batch=0)

    def test_cell_outputs_and_all_gradients_match_torch(self):
        classes = ((RNNCell, torch.nn.RNNCell, {}), (RNNCell, torch.nn.RNNCell, {'nonlinearity': 'relu'}),
                   (GRUCell, torch.nn.GRUCell, {}), (LSTMCell, torch.nn.LSTMCell, {}))
        for actual_type, expected_type, options in classes:
            for batch in (None, 2, 0):
                for bias in (False, True):
                    for supplied in (False, True):
                        with self.subTest(kind=actual_type.__name__, options=options, batch=batch, bias=bias, supplied=supplied):
                            with torch.random.fork_rng(devices=[]):
                                torch.manual_seed(91)
                                reference = expected_type(2, 3, bias=bias, **options)
                            model = actual_type(2, 3, bias=bias, **options)
                            copy_weights(model, reference)
                            generator = torch.Generator().manual_seed(33)
                            tx = torch.randn([2] if batch is None else [batch, 2], generator=generator, requires_grad=True)
                            x = tensor(tx, True)
                            leaves, states, torch_states = [('input', x, tx)], [], []
                            lstm = actual_type is LSTMCell
                            if supplied:
                                for name in ('h', 'c') if lstm else ('h',):
                                    ts = torch.randn([3] if batch is None else [batch, 3], generator=generator, requires_grad=True)
                                    state = tensor(ts, True)
                                    states.append(state)
                                    torch_states.append(ts)
                                    leaves.append((name, state, ts))
                            hx = None if not supplied else tuple(states) if lstm else states[0]
                            thx = None if not supplied else tuple(torch_states) if lstm else torch_states[0]
                            result, wanted = model(x, hx), reference(tx, thx)
                            loss_and_gradients(list(result) if lstm else [result], list(wanted) if lstm else [wanted],
                                               model, reference, leaves)

    def test_invalid_and_unsupported_options_fail_explicitly(self):
        for cls in (RNN, GRU, LSTM, RNNCell, GRUCell, LSTMCell):
            for sizes in ((0, 3), (2, 0), (-1, 3), (True, 3)):
                with self.subTest(cls=cls.__name__, sizes=sizes), self.assertRaises(ValueError):
                    cls(*sizes)
            for options in ({'device': 'cuda'}, {'dtype': torch.float64}):
                with self.subTest(cls=cls.__name__, options=options), self.assertRaises(NotImplementedError):
                    cls(2, 3, **options)
            with self.assertRaises(TypeError):
                cls(2, 3, bias=1)
        for cls in (RNN, GRU, LSTM):
            for options in ({'num_layers': 0}, {'dropout': float('nan')}, {'dropout': -0.1}):
                with self.subTest(cls=cls.__name__, options=options), self.assertRaises(ValueError):
                    cls(2, 3, **options)
            output, _ = cls(2, 3, dropout=0.1)(fr.Tensor.zeros([4, 2, 2]))
            self.assertEqual(output.shape, [4, 2, 3])
            with self.assertRaises(TypeError):
                cls(2, 3, batch_first=1)
        for cls in (RNN, RNNCell):
            with self.assertRaises(ValueError):
                cls(2, 3, nonlinearity='sigmoid')
        output, (hidden, cell) = LSTM(2, 3, proj_size=1)(fr.Tensor.zeros([4, 2, 2]))
        self.assertEqual((output.shape, hidden.shape, cell.shape), ([4, 2, 1], [1, 2, 1], [1, 2, 3]))
        with self.assertRaises(ValueError):
            LSTM(2, 3, proj_size=-1)

    def test_invalid_input_and_state_shapes(self):
        for cls in (RNN, GRU, LSTM):
            model = cls(2, 3, num_layers=2, bidirectional=True, batch_first=True)
            for shape in ([2], [1, 2, 3, 2], [2, 4, 5], [2, 0, 2]):
                with self.subTest(cls=cls.__name__, shape=shape), self.assertRaises(ValueError):
                    model(fr.Tensor.zeros(shape))
            with self.assertRaises(TypeError):
                model([[0., 0.]])
            x = fr.Tensor.zeros([2, 4, 2])
            for shape in ([4, 3], [2, 4, 3], [4, 2, 2]):
                state = fr.Tensor.zeros(shape)
                with self.assertRaises(ValueError):
                    model(x, (state, state) if cls is LSTM else state)
            with self.assertRaises((ValueError, TypeError)):
                model(fr.Tensor.from_i64([0] * 16, [2, 4, 2]))
            if cls is LSTM:
                for state in (fr.Tensor.zeros([4, 2, 3]), (fr.Tensor.zeros([4, 2, 3]),),
                              (fr.Tensor.zeros([4, 2, 3]), None)):
                    with self.assertRaises(TypeError):
                        model(x, state)
        for cls in (RNNCell, GRUCell, LSTMCell):
            model = cls(2, 3)
            for shape in ([], [3], [1, 1, 2]):
                with self.assertRaises(ValueError):
                    model(fr.Tensor.zeros(shape))
            bad = fr.Tensor.zeros([1, 3])
            with self.assertRaises(ValueError):
                model(fr.Tensor.zeros([2]), (bad, bad) if cls is LSTMCell else bad)

    def test_float32_input_and_state_are_required(self):
        for cls in (RNN, GRU, LSTM, RNNCell, GRUCell, LSTMCell):
            with self.subTest(cls=cls.__name__):
                cell = cls in (RNNCell, GRUCell, LSTMCell)
                model = cls(2, 3)
                shape = [2, 2] if cell else [4, 2, 2]
                with self.assertRaisesRegex(ValueError, 'float32'):
                    model(fr.Tensor.from_f64([0.] * math.prod(shape), shape))
                state_shape = [2, 3] if cell else [1, 2, 3]
                state = fr.Tensor.from_i64([0] * math.prod(state_shape), state_shape)
                with self.assertRaisesRegex(ValueError, 'float32'):
                    model(fr.Tensor.zeros(shape), (state, state) if cls in (LSTM, LSTMCell) else state)

    def test_recorded_rnn_nonlinearity_matches_forward_and_checkpoint(self):
        for cls, reference_cls in ((RNN, torch.nn.RNN), (RNNCell, torch.nn.RNNCell)):
            with self.subTest(cls=cls.__name__):
                model = cls(2, 3, nonlinearity='tanh')
                model.reset_parameters(generator=fr.Generator(31))
                model.nonlinearity = 'relu'
                reference = reference_cls(2, 3, nonlinearity='relu')
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        dict(reference.named_parameters())[name].copy_(torch.tensor(parameter.tensor().tolist()))
                inputs = fr.Tensor.randn([2, 2] if cls is RNNCell else [4, 2, 2], seed=7)
                actual, expected = model(inputs), reference(torch.tensor(inputs.tolist()))
                if cls is RNN:
                    for got, wanted in zip(actual, expected):
                        close(got, wanted, 'recorded RNN activation')
                else:
                    close(actual, expected, 'recorded RNNCell activation')
                optimizer, rng = fr.optim.AdamW(model.parameters()), fr.Generator(9)
                checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng)
                target = cls(2, 3, nonlinearity='relu')
                target_optimizer, target_rng = fr.optim.AdamW(target.parameters()), fr.Generator(99)
                checkpoint.restore(target, optimizers={'main': target_optimizer}, generator=target_rng)
                restored = target(inputs)
                if cls is RNN:
                    self.assertEqual([value.tolist() for value in restored], [value.tolist() for value in actual])
                else:
                    self.assertEqual(restored.tolist(), actual.tolist())
                model.nonlinearity = 'invalid'
                with self.assertRaises(ValueError):
                    model(inputs)

    def test_reset_parameters_modes_and_registration(self):
        for cls in (RNN, GRU, LSTM, RNNCell, GRUCell, LSTMCell):
            with self.subTest(cls=cls.__name__):
                left, right = cls(2, 3), cls(2, 3)
                identities = [id(parameter) for parameter in left.parameters()]
                left.reset_parameters(generator=fr.Generator(7))
                right.reset_parameters(generator=fr.Generator(7))
                self.assertEqual(identities, [id(parameter) for parameter in left.parameters()])
                for a, b in zip(left.parameters(), right.parameters()):
                    self.assertEqual(a.tensor().tolist(), b.tensor().tolist())
                    values = torch.tensor(a.tensor().tolist())
                    self.assertLessEqual(float(values.abs().max()), 1 / math.sqrt(3))
                self.assertIs(left.eval(), left)
                self.assertFalse(left.training)
                inputs = fr.Tensor.randn([2, 2] if cls in (RNNCell, GRUCell, LSTMCell) else [3, 2, 2], seed=3)
                evaluated = left(inputs)
                self.assertIs(left.train(), left)
                trained = left(inputs)
                def values(value):
                    return [values(item) for item in value] if isinstance(value, tuple) else value.tolist()
                self.assertEqual(values(evaluated), values(trained))

    def test_checkpoint_config_modes_and_exact_training_restart(self):
        options = {'num_layers': 2, 'bias': False, 'batch_first': True, 'bidirectional': True}
        model = LSTM(2, 3, **options)
        model.reset_parameters(generator=fr.Generator(7))
        optimizer, rng = fr.optim.AdamW(model.parameters(), lr=0.02), fr.Generator(99)
        def step(module, opt, stream):
            opt.zero_grad()
            inputs = fr.Tensor.randn([2, 3, 2], generator=stream)
            output, (hidden, cell) = module(inputs)
            loss = (output * output).mean() + (hidden * hidden).mean() * 0.2 + (cell * cell).mean() * 0.1
            value = loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return value
        def snapshot(module, opt, stream, number):
            return fr.checkpoint.snapshot_training(module, optimizers={'main': opt}, generator=stream, step=number)
        def state(checkpoint):
            return {name: (value.shape, value.tolist()) for name, value in checkpoint.tensors().items()}
        for _ in range(2):
            step(model, optimizer, rng)
        model.eval()
        checkpoint = snapshot(model, optimizer, rng, 2)
        expected_losses = [step(model, optimizer, rng) for _ in range(3)]
        expected_state = state(snapshot(model, optimizer, rng, 5))
        expected_draws = fr.Tensor.randn([7], generator=rng).tolist()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint.save(Path(directory) / 'training')
            loaded = fr.checkpoint.load(Path(directory) / 'training')
            target = LSTM(2, 3, **options)
            target_optimizer, target_rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(1234)
            self.assertEqual(loaded.restore(target, optimizers={'main': target_optimizer}, generator=target_rng), 2)
            self.assertFalse(target.training)
            self.assertEqual([step(target, target_optimizer, target_rng) for _ in range(3)], expected_losses)
            self.assertEqual(state(snapshot(target, target_optimizer, target_rng, 5)), expected_state)
            self.assertEqual(fr.Tensor.randn([7], generator=target_rng).tolist(), expected_draws)
            mismatch = LSTM(2, 3, **{**options, 'batch_first': False})
            bad_optimizer, bad_rng = fr.optim.AdamW(mismatch.parameters(), lr=0.4), fr.Generator(5)
            before = state(snapshot(mismatch, bad_optimizer, bad_rng, 0))
            with self.assertRaises(ValueError):
                loaded.restore(mismatch, optimizers={'main': bad_optimizer}, generator=bad_rng)
            self.assertEqual(state(snapshot(mismatch, bad_optimizer, bad_rng, 0)), before)


if __name__ == '__main__':
    unittest.main()
