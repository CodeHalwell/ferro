"""Typed native RMSprop/Adagrad, measured against actual CPU Torch updates."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import ferro as fr
import torch


def tensor(values, dtype='float64', requires_grad=False):
    return fr.tensor(values, dtype=dtype, requires_grad=requires_grad)


def record(value):
    if isinstance(value, fr.Tensor):
        data = value.reshape([value.numel()]).tolist()
        return [value.dtype, value.shape, struct.pack('<'+{'float32':'f', 'float64':'d', 'int64':'q', 'bool':'?'}[value.dtype]*len(data), *data).hex()]
    if isinstance(value, dict):
        return {str(k): record(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [record(v) for v in value]
    return value


def options(name):
    common = dict(lr=.01234567891234567, weight_decay=.0234567891234567, eps=1e-100, maximize=True)
    return dict(common, alpha=.9123456789123456, momentum=.7123456789123456, centered=True) if name == 'RMSprop' else dict(common, lr_decay=.1234567891234567, initial_accumulator_value=1e-200)


class Model(fr.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = fr.nn.Parameter(tensor([.7, -.2]))
        self.b = fr.nn.Parameter(tensor([.4, -.3], 'float32'))


def setup(name):
    model = Model()
    opt = getattr(fr.optim, name)([{'params': [model.a], **options(name)}, {'params': [model.b], 'lr': .017}], lr=.03)
    return model, opt


def update(model, opt, rng, index):
    opt.zero_grad()
    noise = fr.randn([2], generator=rng).tolist()
    loss = model.a.square().sum()*.13 + (model.a*tensor(noise)).sum()
    loss.backward()
    if index % 3:
        model.b.square().sum().backward()
    opt.step(); opt.zero_grad()
    return loss.item()


def continuation(model, opt, rng, name):
    losses = [update(model, opt, rng, i) for i in range(5, 11)]
    saved = fr.checkpoint.snapshot_training(model, optimizers={name: opt}, generator=rng, step=11)
    digest = hashlib.sha256(json.dumps(record(saved.tensors()), sort_keys=True).encode()).hexdigest()
    return dict(losses=losses, checkpoint=digest, state=record(opt.state_dict()), draws=fr.randn([9], generator=rng).tolist())


def worker(directory, name):
    model, opt = setup(name); rng = fr.Generator(999)
    fr.checkpoint.load(Path(directory)/'state').restore(model, optimizers={name: opt}, generator=rng)
    (Path(directory)/'result.json').write_text(json.dumps(continuation(model, opt, rng, name)))


class AdaptiveF64Tests(unittest.TestCase):
    def same(self, actual, expected, *, exact=False):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        result = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 4e-6 if actual.dtype == 'float32' else 4e-14
        torch.testing.assert_close(result, expected.detach(), rtol=0 if exact else tolerance,
                                   atol=0 if exact else tolerance*1e-3, equal_nan=True)

    def states(self, opt, ref, parameters):
        states = opt.state_dict()['state']
        for i, p in enumerate(parameters):
            if not ref.state[p]:
                self.assertNotIn(i, states)
                continue
            self.assertEqual(set(states[i]), set(ref.state[p]))
            for key, value in ref.state[p].items():
                self.same(states[i][key], value)
            self.assertEqual(states[i]['step'].dtype, 'float32')

    def test_actual_torch_multistep_full_precision_options_and_skips(self):
        for name in ('RMSprop', 'Adagrad'):
            for maximize in (False, True):
                for variant in (False, True):
                    with self.subTest(name=name, maximize=maximize, variant=variant):
                        config = options(name); config['maximize'] = maximize
                        if name == 'RMSprop':
                            config.update(centered=variant, momentum=.7123456789123456 if variant else 0.)
                        else:
                            config['initial_accumulator_value'] = 1e-200 if variant else .7000000000000001
                        actual = [tensor([.7, -.3, 1e-200, 1e150], requires_grad=True), tensor([.4, -.8], requires_grad=True)]
                        expected = [torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True) for p in actual]
                        opt = getattr(fr.optim, name)(actual, **config)
                        ref = getattr(torch.optim, name)(expected, **config, foreach=False)
                        ids = [p._id() for p in actual]
                        for step in range(12):
                            opt.zero_grad(); ref.zero_grad()
                            for i, (p, q) in enumerate(zip(actual, expected)):
                                if (step+i) % 4 != 1:
                                    values = [.2+step*.03, -.4, 1e-200, 1e100][:p.numel()]
                                    p.grad = tensor(values); q.grad = torch.tensor(values, dtype=torch.float64)
                            held = [None if p.grad is None else p.grad._id() for p in actual]
                            opt.step(); ref.step()
                            for p, q in zip(actual, expected): self.same(p, q)
                            self.states(opt, ref, expected)
                            self.assertEqual([p._id() for p in actual], ids)
                            self.assertEqual([None if p.grad is None else p.grad._id() for p in actual], held)

    def test_mixed_groups_preserve_legacy_float32_and_typed_state(self):
        for name in ('RMSprop', 'Adagrad'):
            a = tensor([.7, -.2], 'float32', True); b = tensor([.4, -.3], requires_grad=True)
            control = tensor(a.tolist(), 'float32', True)
            opt = getattr(fr.optim, name)([a, b], lr=.017)
            old = getattr(fr.optim, name)([control], lr=.017)
            expected = torch.tensor(b.tolist(), dtype=torch.float64, requires_grad=True)
            ref = getattr(torch.optim, name)([expected], lr=.017, foreach=False)
            for step in range(7):
                values = [.2+step*.1, -.3]
                a.grad = tensor(values, 'float32'); control.grad = tensor(values, 'float32')
                b.grad = tensor(values); expected.grad = torch.tensor(values, dtype=torch.float64)
                opt.step(); old.step(); ref.step()
                self.assertEqual(record(a), record(control)); self.same(b, expected)
                self.assertEqual(record(opt.state_dict()['state'][0]), record(old.state_dict()['state'][0]))
                for key, value in ref.state[expected].items(): self.same(opt.state_dict()['state'][1][key], value)
            opt.zero_grad(False)
            self.assertEqual(a.grad.tolist(), [0., 0.]); self.assertEqual(b.grad.dtype, 'float64')
            self.assertEqual(b.grad.tolist(), [0., 0.])

    def test_adagrad_constructor_defaults_and_lazy_added_groups(self):
        a = tensor([.7, -.3], requires_grad=True); q = torch.tensor(a.tolist(), dtype=torch.float64, requires_grad=True)
        opt = fr.optim.Adagrad([{'params': [a], 'initial_accumulator_value': 3.}], lr=.01, initial_accumulator_value=.7000000000000001)
        ref = torch.optim.Adagrad([{'params': [q], 'initial_accumulator_value': 3.}], lr=.01, initial_accumulator_value=.7000000000000001, foreach=False)
        self.states(opt, ref, [q])
        b = tensor([.4, -.1], requires_grad=True); r = torch.tensor(b.tolist(), dtype=torch.float64, requires_grad=True)
        opt.add_param_group({'params': [b], 'initial_accumulator_value': 4.})
        ref.add_param_group({'params': [r], 'initial_accumulator_value': 4.})
        self.states(opt, ref, [q, r])
        opt.defaults['initial_accumulator_value'] = .12500000000000003
        ref.defaults['initial_accumulator_value'] = .12500000000000003
        b.grad = tensor([.5, -.25]); r.grad = torch.tensor([.5, -.25], dtype=torch.float64)
        opt.step(); ref.step(); self.same(b, r); self.states(opt, ref, [q, r])
        opt.defaults['initial_accumulator_value'] = .9; ref.defaults['initial_accumulator_value'] = .9
        opt.step(); ref.step(); self.same(b, r); self.states(opt, ref, [q, r])
        tiny = fr.optim.Adagrad([a], initial_accumulator_value=1e-300)
        self.assertEqual(tiny.state_dict()['state'][0]['sum'].tolist(), [1e-300]*2)
        with self.assertRaises(ValueError): fr.optim.Adagrad([tensor([0.], 'float32', True)], initial_accumulator_value=1e100)
        # Lazy initialization does not invoke user state/load hooks.
        traces = []
        for api in (fr, torch):
            dtype = 'float64' if api is fr else torch.float64
            first = api.tensor([.7], dtype=dtype, requires_grad=True)
            added = api.tensor([.3], dtype=dtype, requires_grad=True)
            local = api.optim.Adagrad([first], lr=.01, initial_accumulator_value=.125)
            local.add_param_group({'params': [added]})
            events = []
            local.register_state_dict_pre_hook(lambda *_: events.append('state'))
            local.register_load_state_dict_pre_hook(lambda *_: events.append('load'))
            added.grad = api.tensor([.5], dtype=dtype)
            local.step(); self.assertEqual(events, [])
            saved = local.state_dict(); local.load_state_dict(saved)
            self.assertEqual(events, ['state', 'load']); traces.append(added.tolist())
        torch.testing.assert_close(torch.tensor(traces[0], dtype=torch.float64), torch.tensor(traces[1], dtype=torch.float64), rtol=2e-14, atol=0)
        # Unused added F32 slots do not constrain a later Double-only default.
        first = tensor([.7], requires_grad=True); pending32 = tensor([.3], 'float32', True); pending64 = tensor([.2], requires_grad=True)
        mixed = fr.optim.Adagrad([first], initial_accumulator_value=0.)
        mixed.add_param_group({'params': [pending32]}); mixed.add_param_group({'params': [pending64]})
        mixed.defaults['initial_accumulator_value'] = 1e100
        pending64.grad = tensor([.5]); mixed.step()
        state = mixed.state_dict()['state']; self.assertNotIn(1, state)
        self.assertEqual(state[2]['sum'].tolist(), [1e100])
        pending32.grad = tensor([.1], 'float32'); before = [record(pending32), record(pending64), record(mixed.state_dict())]
        with self.assertRaises(ValueError): mixed.step()
        self.assertEqual([record(pending32), record(pending64), record(mixed.state_dict())], before)

    def test_typed_restore_counter_and_multigroup_failures_are_atomic(self):
        for name in ('RMSprop', 'Adagrad'):
            a = tensor([.7, -.2], requires_grad=True); b = tensor([.4, -.3], requires_grad=True)
            opt = getattr(fr.optim, name)([a, b], **options(name))
            a.grad = tensor([.2, .1]); b.grad = tensor([-.3, .4]); opt.step()
            saved = opt.state_dict(); before = record(saved)
            for replacement in (fr.zeros([2]), tensor([0.])):
                bad = {'state': {i: dict(v) for i, v in saved['state'].items()}, 'param_groups': [dict(g) for g in saved['param_groups']]}
                key = 'square_avg' if name == 'RMSprop' else 'sum'; bad['state'][1][key] = replacement
                with self.assertRaises(ValueError): opt.load_state_dict(bad)
                self.assertEqual(record(opt.state_dict()), before)
            opt.load_state_dict(saved); self.assertEqual(record(opt.state_dict()), before)
            bad = {'state': {i: dict(v) for i, v in saved['state'].items()}, 'param_groups': [dict(g) for g in saved['param_groups']]}
            bad['state'][1]['step'] = fr.tensor(float(2**24))
            opt.load_state_dict(bad); held = [record(a), record(b)]; old = record(opt.state_dict())
            with self.assertRaises(ValueError): opt.step()
            self.assertEqual([record(a), record(b)], held); self.assertEqual(record(opt.state_dict()), old)
            opt.load_state_dict(saved)
            c = tensor([.7], 'float32', True); grouped = getattr(fr.optim, name)([{'params': [a]}, {'params': [c]}], lr=.01)
            a.grad = tensor([.2, .1]); c.grad = tensor([.3], 'float32'); grouped.step()
            native = grouped._native.configurations(); values = [record(a), record(c)]
            grouped.param_groups[0]['lr'] = .1234567891234567; grouped.param_groups[1]['lr'] = 1e200
            with self.assertRaises(ValueError): grouped.step()
            self.assertEqual(grouped._native.configurations(), native); self.assertEqual([record(a), record(c)], values)

    def test_live_strided_empty_scalar_and_nonfinite_gradients(self):
        for name in ('RMSprop', 'Adagrad'):
            p = tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
            q = torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True)
            seed = tensor([[.1, .2, .3], [.4, .5, .6]], requires_grad=True)
            reference = torch.tensor(seed.tolist(), dtype=torch.float64, requires_grad=True)
            g = seed.T; h = reference.T; p.grad = g; q.grad = h
            opt = getattr(fr.optim, name)([p], lr=.1); ref = getattr(torch.optim, name)([q], lr=.1, foreach=False)
            opt.step(); ref.step(); self.same(p, q); self.assertEqual(p.grad._id(), g._id()); self.assertTrue(g.requires_grad)
            self.same(fr.autograd.grad(g.sum(), seed)[0], torch.autograd.grad(h.sum(), reference)[0], exact=True)
            for shape in ([], [0, 3]):
                p = fr.zeros(shape, dtype='float64', requires_grad=True); q = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
                p.grad = fr.ones(shape, dtype='float64'); q.grad = torch.ones_like(q)
                opt = getattr(fr.optim, name)([p], lr=.1); ref = getattr(torch.optim, name)([q], lr=.1, foreach=False)
                opt.step(); ref.step(); self.same(p, q); self.states(opt, ref, [q])
            p = tensor([.2]*5, requires_grad=True); q = torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True)
            values = [0., -0., math.inf, -math.inf, math.nan]
            p.grad = tensor(values); q.grad = torch.tensor(values, dtype=torch.float64)
            config = dict(lr=.1, eps=0.)
            if name == 'RMSprop': config.update(alpha=0., centered=True, momentum=.5)
            opt = getattr(fr.optim, name)([p], **config); ref = getattr(torch.optim, name)([q], **config, foreach=False)
            opt.step(); ref.step(); self.same(p, q); self.states(opt, ref, [q])

    def test_double_linear_all_gradients_and_optimizer_updates(self):
        for name in ('RMSprop', 'Adagrad'):
            for scale in (1e-100, 1e100):
                model = fr.nn.Linear(2, 2, dtype='float64'); expected = torch.nn.Linear(2, 2, dtype=torch.float64)
                with torch.no_grad():
                    expected.weight.copy_(torch.tensor(model.weight.tolist(), dtype=torch.float64)); expected.bias.copy_(torch.tensor(model.bias.tolist(), dtype=torch.float64))
                opt = getattr(fr.optim, name)(model.parameters(), lr=1e-100)
                ref = getattr(torch.optim, name)(expected.parameters(), lr=1e-100, foreach=False)
                for _ in range(2):
                    opt.zero_grad(); ref.zero_grad()
                    x = tensor([[scale, -scale*.3]], requires_grad=True); tx = torch.tensor(x.tolist(), dtype=torch.float64, requires_grad=True)
                    loss = model(x).sum(); reference = expected(tx).sum(); loss.backward(); reference.backward()
                    self.same(x.grad, tx.grad)
                    for p, q in zip(model.parameters(), expected.parameters()): self.same(p.grad, q.grad)
                    opt.step(); ref.step()
                    for p, q in zip(model.parameters(), expected.parameters()): self.same(p, q)
                    self.states(opt, ref, list(expected.parameters()))

    def test_training_checkpoint_restarts_exactly_in_fresh_process(self):
        for name in ('RMSprop', 'Adagrad'):
            model, opt = setup(name); rng = fr.Generator(17)
            for i in range(5): update(model, opt, rng, i)
            model.eval()
            with tempfile.TemporaryDirectory() as directory:
                fr.checkpoint.save_training(Path(directory)/'state', model, optimizers={name: opt}, generator=rng, step=5)
                expected = continuation(model, opt, rng, name)
                env = dict(os.environ); env.pop('PYTHONPATH', None)
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', directory, name],
                                        env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(json.loads((Path(directory)/'result.json').read_text()), expected)

    def test_shared_cells_freezing_and_validation_boundaries(self):
        for name in ('RMSprop', 'Adagrad'):
            data = tensor([.7, -.3]); a = fr.nn.Parameter._from_shared(data, True); b = fr.nn.Parameter._from_shared(data, True)
            other = torch.tensor(data.tolist(), dtype=torch.float64); x = torch.nn.Parameter(other); y = torch.nn.Parameter(other)
            config = options(name); opt = getattr(fr.optim, name)([a, b], **config)
            ref = getattr(torch.optim, name)([x, y], **config, foreach=False)
            for step in range(8):
                opt.zero_grad(); ref.zero_grad()
                for i, (p, q) in enumerate(((a, x), (b, y))):
                    p.trainable = not (i == 1 and step % 3 == 1)
                    if (step+i) % 4:
                        values = [step*.1+i, -.2]; p.grad = tensor(values)
                        if p.trainable: q.grad = torch.tensor(values, dtype=torch.float64)
                opt.step(); ref.step(); self.same(data, other); self.states(opt, ref, [x, y])
            for kwargs in ({'lr': math.inf}, {'eps': -1.}, {'lr': math.nan}):
                with self.assertRaises(ValueError): getattr(fr.optim, name)([a], **kwargs)
            before = record(data)
            with self.assertRaises((ValueError, NotImplementedError)):
                fr.func.jvp(lambda z: (opt.step(), z)[1], (fr.tensor([1.]),), (fr.tensor([1.]),))
            self.assertEqual(record(data), before)
            for unsupported in ('Adamax', 'Adadelta', 'Adafactor'):
                with self.assertRaises((ValueError, TypeError)): getattr(fr.optim, unsupported)([a], lr=.01)
        # Torch accepts alpha > 1; only the new Double path widens this legacy
        # scalar limit, while retaining finite-value validation.
        p = tensor([.7], requires_grad=True); q = torch.tensor([.7], dtype=torch.float64, requires_grad=True)
        p.grad = tensor([.2]); q.grad = torch.tensor([.2], dtype=torch.float64)
        opt = fr.optim.RMSprop([p], alpha=1.1); ref = torch.optim.RMSprop([q], alpha=1.1, foreach=False)
        opt.step(); ref.step(); self.same(p, q)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        import test_adaptive_f64
        test_adaptive_f64.worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
