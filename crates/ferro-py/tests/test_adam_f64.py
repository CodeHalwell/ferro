"""Double Adam/AdamW use native typed moments and exact CPU scalar settings."""
import copy
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
        return [value.dtype, value.shape, struct.pack('<'+{'float32':'f','float64':'d','int64':'q','bool':'?'}[value.dtype]*len(data), *data).hex()]
    if isinstance(value, dict):
        return {str(k): record(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [record(v) for v in value]
    return value


def digest(checkpoint):
    return hashlib.sha256(json.dumps(record(checkpoint.tensors()), sort_keys=True).encode()).hexdigest()


class Model(fr.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = fr.nn.Parameter(tensor([.7, -.2]))
        self.b = fr.nn.Parameter(tensor([.4, -.3], 'float32'))


def setup(name):
    model = Model()
    opt = getattr(fr.optim, name)([
        {'params': [model.a], 'lr': .10000000000000002, 'betas': (.9123456789123456, .9876543219876543),
         'eps': 1e-100, 'weight_decay': .030000000000000002, 'amsgrad': True, 'maximize': True},
        {'params': [model.b], 'lr': .017, 'betas': (.7, .93)}], lr=.03)
    return model, opt


def update(model, opt, rng, index):
    opt.zero_grad()
    noise = fr.randn([2], generator=rng).tolist()
    loss = model.a.square().sum() * .13 + (model.a * tensor(noise)).sum()
    loss.backward()
    if index % 3:
        model.b.square().sum().backward()
    opt.step(); opt.zero_grad()
    return loss.item()


def continuation(model, opt, rng, name):
    losses = [update(model, opt, rng, i) for i in range(5, 11)]
    saved = fr.checkpoint.snapshot_training(model, optimizers={name: opt}, generator=rng, step=11)
    return dict(losses=losses, checkpoint=digest(saved), state=record(opt.state_dict()),
                draws=fr.randn([9], generator=rng).tolist())


def worker(directory, name):
    model, opt = setup(name); rng = fr.Generator(999)
    fr.checkpoint.load(Path(directory)/'state').restore(model, optimizers={name: opt}, generator=rng)
    (Path(directory)/'result.json').write_text(json.dumps(continuation(model, opt, rng, name)))


class AdamF64Tests(unittest.TestCase):
    def same(self, actual, expected, *, exact=False):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        result = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 3e-6 if actual.dtype == 'float32' else 4e-14
        torch.testing.assert_close(result, expected.detach(), rtol=0 if exact else tolerance,
                                   atol=0 if exact else tolerance*1e-3, equal_nan=True)

    def states(self, opt, ref, expected):
        state = opt.state_dict()['state']
        for i, q in enumerate(expected):
            if not ref.state[q]:
                self.assertNotIn(i, state)
                continue
            for key, value in ref.state[q].items():
                self.same(state[i][key], value)
            self.assertEqual(state[i]['step'].dtype, 'float32')

    def test_actual_torch_multistep_options_skips_and_full_precision(self):
        for name in ('Adam', 'AdamW'):
            for amsgrad, maximize in ((False, False), (True, False), (False, True), (True, True)):
                config = dict(lr=.01234567891234567, betas=(.9123456789123456, .9876543219876543),
                              eps=1e-19, weight_decay=.1234567891234567, amsgrad=amsgrad, maximize=maximize)
                actual = [tensor([.7, -.3, .2, -.1], requires_grad=True) for _ in range(2)]
                expected = [torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True) for p in actual]
                opt = getattr(fr.optim, name)(actual, **config)
                ref = getattr(torch.optim, name)(expected, **config, foreach=False, fused=False)
                ids = [p._id() for p in actual]
                for step in range(11):
                    opt.zero_grad(); ref.zero_grad()
                    for i, (p, q) in enumerate(zip(actual, expected)):
                        if (step+i) % 4 != 2:
                            values = [math.sin(step+j+.3)*.7 for j in range(4)]
                            p.grad = tensor(values); q.grad = torch.tensor(values, dtype=torch.float64)
                    held = [p.grad for p in actual]
                    opt.step(); ref.step()
                    self.assertEqual([p._id() for p in actual], ids)
                    for p, q, g in zip(actual, expected, held):
                        self.same(p, q)
                        self.assertEqual(None if g is None else g._id(), None if p.grad is None else p.grad._id())
                    self.states(opt, ref, expected)
                group = opt.state_dict()['param_groups'][0]
                self.assertEqual(group['lr'], config['lr']); self.assertEqual(group['betas'], config['betas'])

    def test_mixed_cpu_group_keeps_existing_float32_kernel_bits(self):
        for name in ('Adam', 'AdamW'):
            config = dict(lr=.017, betas=(.7, .93), eps=1e-8, weight_decay=.03, amsgrad=True)
            a = tensor([.7, -.3], 'float32', True); b = tensor([.4, -.2], requires_grad=True)
            control = tensor(a.tolist(), 'float32', True)
            opt = getattr(fr.optim, name)([a, b], **config)
            legacy = getattr(fr.optim, name)([control], **config)
            for step in range(7):
                g = tensor([step*.03, -.2], 'float32')
                a.grad = g; control.grad = g; b.grad = tensor([.2, .3])
                opt.step(); legacy.step()
                self.assertEqual(record(a), record(control))
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    self.assertEqual(record(opt.state_dict()['state'][0][key]), record(legacy.state_dict()['state'][0][key]))
                opt.zero_grad(); legacy.zero_grad()
            self.assertEqual(opt.state_dict()['state'][1]['exp_avg'].dtype, 'float64')

    def test_typed_state_owns_buffers_and_late_invalid_restore_is_atomic(self):
        for name in ('Adam', 'AdamW'):
            model, opt = setup(name); rng = fr.Generator(7); update(model, opt, rng, 1)
            saved = opt.state_dict(); before = record(saved)
            update(model, opt, rng, 2)
            self.assertEqual(record(saved), before)
            fresh, restored = setup(name); restored.load_state_dict(saved)
            self.assertEqual(record(restored.state_dict()), before)
            for bad_value in (tensor([0., 0.], 'float32'), tensor([0.]), tensor([0., 0.], 'int64')):
                bad = dict(state={i: dict(v) for i, v in saved['state'].items()}, param_groups=[dict(g) for g in saved['param_groups']])
                bad['state'][0]['exp_avg_sq'] = bad_value
                prior = record(restored.state_dict())
                with self.assertRaises(ValueError): restored.load_state_dict(bad)
                self.assertEqual(record(restored.state_dict()), prior)
            bad = dict(state={i: dict(v) for i, v in saved['state'].items()}, param_groups=[dict(g) for g in saved['param_groups']])
            bad['state'][1]['step'] = fr.tensor(float(2**24+2))
            with self.assertRaises(ValueError): restored.load_state_dict(bad)
            self.assertEqual(record(restored.state_dict()), before)

        # Native training checkpoints preserve exact u32 counters; public
        # default-F32 step tensors must reject instead of silently rounding.
        for name in ('Adam', 'AdamW'):
            p = tensor([1.], requires_grad=True)
            opt = getattr(fr.optim, name)([p], lr=.1)
            config = opt._native.configurations()[0]
            for count in (2**24, 2**24+1, 2**24+2):
                arrays = opt._native.state_tensors()
                arrays = [(key, fr.tensor([float(count)] if name == 'AdamW' else float(count), dtype='float64')
                           if key == ('t' if name == 'AdamW' else 't.0') else value) for key, value in arrays]
                opt._native.restore_state(config, arrays)
                before_native = record(dict(opt._native.state_tensors()))
                if count == 2**24:
                    public = opt.state_dict(); self.assertEqual(public['state'][0]['step'].item(), count)
                    opt.load_state_dict(public)
                else:
                    with self.assertRaisesRegex(ValueError, 'public float32 step bound'): opt.state_dict()
                self.assertEqual(record(dict(opt._native.state_tensors())), before_native)

    def test_training_checkpoint_restarts_exactly_in_a_fresh_process(self):
        for name in ('Adam', 'AdamW'):
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

    def test_live_strided_gradients_and_empty_scalar_parameters(self):
        for name in ('Adam', 'AdamW'):
            p = tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
            q = torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True)
            seed = tensor([[.1, .2, .3], [.4, .5, .6]], requires_grad=True)
            reference = torch.tensor(seed.tolist(), dtype=torch.float64, requires_grad=True)
            g = seed.T; h = reference.T; p.grad = g; q.grad = h
            opt = getattr(fr.optim, name)([p], lr=.1, amsgrad=True)
            ref = getattr(torch.optim, name)([q], lr=.1, amsgrad=True, foreach=False, fused=False)
            opt.step(); ref.step(); self.same(p, q)
            self.assertEqual(p.grad._id(), g._id()); self.assertIs(q.grad, h)
            self.assertTrue(g.requires_grad); self.assertEqual(g.stride(), h.stride())
            self.same(fr.autograd.grad(g.sum(), seed)[0], torch.autograd.grad(h.sum(), reference)[0])
            opt.zero_grad(); ref.zero_grad(); self.assertIsNone(p.grad)
            for shape in ([], [0], [2, 0]):
                p = fr.zeros(shape, dtype='float64', requires_grad=True)
                q = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
                p.grad = fr.ones(shape, dtype='float64'); q.grad = torch.ones_like(q)
                opt = getattr(fr.optim, name)([p], lr=.13, amsgrad=True)
                ref = getattr(torch.optim, name)([q], lr=.13, amsgrad=True, foreach=False, fused=False)
                opt.step(); ref.step(); self.same(p, q); self.states(opt, ref, [q])

    def test_double_linear_training_extremes_and_zero_epsilon(self):
        for name in ('Adam', 'AdamW'):
            for scale in (1e-100, 1e100):
                model = fr.nn.Linear(2, 2, dtype='float64'); expected = torch.nn.Linear(2, 2, dtype=torch.float64)
                with fr.no_grad(), torch.no_grad():
                    model.weight.copy_(tensor([[.7, -.2], [.4, .9]])); model.bias.copy_(tensor([.1, -.3]))
                    expected.weight.copy_(torch.tensor(model.weight.tolist(), dtype=torch.float64)); expected.bias.copy_(torch.tensor(model.bias.tolist(), dtype=torch.float64))
                opt = getattr(fr.optim, name)(model.parameters(), lr=1e-100, betas=(math.nextafter(1., 0.), .9123456789123456), eps=0.)
                ref = getattr(torch.optim, name)(expected.parameters(), lr=1e-100, betas=(math.nextafter(1., 0.), .9123456789123456), eps=0., foreach=False, fused=False)
                for _ in range(2):
                    x = tensor([[scale, -scale*.3]], requires_grad=True); tx = torch.tensor(x.tolist(), dtype=torch.float64, requires_grad=True)
                    opt.zero_grad(); ref.zero_grad()
                    y = model(x); ty = expected(tx); self.same(y, ty)
                    y.sum().backward(); ty.sum().backward(); self.same(x.grad, tx.grad)
                    for p, q in zip(model.parameters(), expected.parameters()): self.same(p.grad, q.grad)
                    opt.step(); ref.step()
                    for p, q in zip(model.parameters(), expected.parameters()): self.same(p, q)
                    self.states(opt, ref, list(expected.parameters()))
            p = tensor([.2]*5, requires_grad=True); q = torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True)
            values = [0., -0., math.inf, -math.inf, math.nan]
            p.grad = tensor(values); q.grad = torch.tensor(values, dtype=torch.float64)
            opt = getattr(fr.optim, name)([p], lr=.1, betas=(0., 0.), eps=0., amsgrad=True)
            ref = getattr(torch.optim, name)([q], lr=.1, betas=(0., 0.), eps=0., amsgrad=True, foreach=False, fused=False)
            opt.step(); ref.step(); self.same(p, q); self.states(opt, ref, [q])

    def test_shared_storage_and_frozen_slots_keep_separate_moments(self):
        for name in ('Adam', 'AdamW'):
            data = tensor([.7, -.3]); a = fr.nn.Parameter._from_shared(data, True); b = fr.nn.Parameter._from_shared(data, True)
            other = torch.tensor(data.tolist(), dtype=torch.float64); x = torch.nn.Parameter(other); y = torch.nn.Parameter(other)
            config = dict(lr=.13, betas=(.7, .93), weight_decay=.2, amsgrad=True)
            opt = getattr(fr.optim, name)([a, b], **config)
            ref = getattr(torch.optim, name)([x, y], **config, foreach=False, fused=False)
            for step in range(9):
                opt.zero_grad(); ref.zero_grad()
                for index, (p, q) in enumerate(((a, x), (b, y))):
                    p.trainable = not (index == 1 and step % 3 == 1)
                    if (step+index) % 4 != 0:
                        values = [step*.1+index, -.2]; p.grad = tensor(values)
                        if p.trainable: q.grad = torch.tensor(values, dtype=torch.float64)
                opt.step(); ref.step(); self.same(data, other); self.states(opt, ref, [x, y])

    def test_configuration_preflight_retains_parameters_state_and_gradients(self):
        for name in ('Adam', 'AdamW'):
            a = tensor([.7], requires_grad=True); b = tensor([.3], 'float32', True)
            opt = getattr(fr.optim, name)([{'params': [a]}, {'params': [b]}], lr=.01)
            a.grad = tensor([.2]); b.grad = tensor([.1], 'float32'); opt.step()
            state = record(opt.state_dict()); values = [record(a), record(b)]; grads = [a.grad._id(), b.grad._id()]
            opt.param_groups[0]['lr'] = .1234567891234567; opt.param_groups[1]['lr'] = 1e200
            native_before = opt._native.configurations()
            with self.assertRaises(ValueError): opt.step()
            self.assertEqual(opt._native.configurations(), native_before); self.assertEqual([record(a), record(b)], values)
            self.assertEqual([a.grad._id(), b.grad._id()], grads)
            opt.param_groups[0]['lr'] = .01; opt.param_groups[1]['lr'] = .01
            self.assertEqual(record(opt.state_dict()), state)
            for kwargs in ({'betas': (1., .9)}, {'eps': -1.}, {'lr': math.inf}):
                with self.assertRaises(ValueError): getattr(fr.optim, name)([a], **kwargs)
            with self.assertRaises((ValueError, NotImplementedError)):
                fr.func.jvp(lambda z: (opt.step(), z)[1], (fr.tensor([1.]),), (fr.tensor([1.]),))
            self.assertEqual([record(a), record(b)], values)
            for unsupported in ('Adadelta', 'Adamax', 'Adafactor'):
                with self.assertRaises((ValueError, TypeError)): getattr(fr.optim, unsupported)([a], lr=.01)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        # Import under the stable module name used by checkpoint schemas.
        import test_adam_f64
        test_adam_f64.worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
