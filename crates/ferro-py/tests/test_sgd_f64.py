"""CPU float64 SGD: actual Torch trajectories and owning typed restart state."""
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


def flat_bits(value):
    data = value.reshape([value.numel()]).tolist()
    code = {'float32': 'f', 'float64': 'd', 'int64': 'q', 'bool': '?'}[value.dtype]
    return struct.pack('<' + code * len(data), *data).hex()


def state_bits(value):
    if isinstance(value, fr.Tensor):
        return [value.dtype, value.shape, flat_bits(value)]
    if isinstance(value, dict):
        return {str(k): state_bits(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [state_bits(v) for v in value]
    return value


def digest(checkpoint):
    return hashlib.sha256(json.dumps(state_bits(checkpoint.tensors()), sort_keys=True).encode()).hexdigest()


class F64Model(fr.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = fr.nn.Parameter(tensor([.7, -.2]))
        self.b = fr.nn.Parameter(tensor([.4, -.3], 'float32'))


def setup():
    model = F64Model()
    optimizer = fr.optim.SGD([
        {'params': [model.a], 'lr': .10000000000000002, 'momentum': .9000000000000001,
         'weight_decay': .030000000000000002, 'dampening': .10000000000000002},
        {'params': [model.b], 'lr': .017, 'momentum': .7, 'nesterov': True}], lr=.03)
    return model, optimizer


def update(model, optimizer, generator, index):
    optimizer.zero_grad()
    noise = fr.randn([2], generator=generator).tolist()
    a = model.a.square().sum() * .13 + (model.a * tensor(noise)).sum()
    b = model.b.square().sum() * .07
    a.backward()
    if index % 3:
        b.backward()
    result = [a.item(), b.item()]
    optimizer.step()
    optimizer.zero_grad()
    return result


def continuation(model, optimizer, generator):
    losses = [update(model, optimizer, generator, i) for i in range(5, 13)]
    checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'sgd': optimizer}, generator=generator, step=13)
    return dict(losses=losses, checkpoint=digest(checkpoint), state=state_bits(optimizer.state_dict()),
                next_draws=fr.randn([11], generator=generator).tolist())


def worker(path):
    model, optimizer = setup()
    generator = fr.Generator(999)
    saved = fr.checkpoint.load(Path(path) / 'training')
    saved.restore(model, optimizers={'sgd': optimizer}, generator=generator)
    (Path(path) / 'result.json').write_text(json.dumps(continuation(model, optimizer, generator)))


class SgdF64Tests(unittest.TestCase):
    def assert_tensor(self, actual, expected, *, exact=True):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype)
        torch.testing.assert_close(values, expected.detach(), atol=0. if exact else 3e-6, rtol=0. if exact else 3e-6, equal_nan=True)

    def test_torch_multistep_groups_skips_decay_dampening_nesterov_maximize(self):
        base = dict(lr=.10000000000000002, momentum=.9000000000000001,
                    weight_decay=.030000000000000002, dampening=.10000000000000002)
        configs = [base, dict(base, maximize=True), dict(base, dampening=0., nesterov=True),
                   dict(base, momentum=0., dampening=0.), dict(base, weight_decay=0.),
                   dict(base, lr=1+2**-27, momentum=1-2**-27, dampening=2**-27)]
        for config in configs:
            actual = [tensor([-.7, .2, 1.1, -0.], d, True) for d in ('float32', 'float64')]
            expected = [torch.tensor([-.7, .2, 1.1, -0.], dtype=d, requires_grad=True) for d in (torch.float32, torch.float64)]
            opt = fr.optim.SGD(actual, **config)
            ref = torch.optim.SGD(expected, **config, foreach=False, fused=False)
            held_ids = [p._id() for p in actual]
            for step in range(12):
                opt.zero_grad(); ref.zero_grad()
                for i, (p, q) in enumerate(zip(actual, expected)):
                    if (step+i) % 5 != 2:
                        values = [math.sin(step+j+.3)*.7 for j in range(4)]
                        p.grad = tensor(values, p.dtype); q.grad = torch.tensor(values, dtype=q.dtype)
                gradients = [p.grad for p in actual]
                opt.step(); ref.step()
                self.assertEqual([p._id() for p in actual], held_ids)
                for i, (p, q) in enumerate(zip(actual, expected)):
                    self.assertEqual(None if p.grad is None else p.grad._id(), None if gradients[i] is None else gradients[i]._id()); self.assert_tensor(p, q, exact=i == 1)
                    state = opt.state_dict()['state'].get(i, {})
                    if 'momentum_buffer' in ref.state[q]:
                        self.assert_tensor(state['momentum_buffer'], ref.state[q]['momentum_buffer'], exact=i == 1)

    def test_exact_scalar_bits_fused_cancellation_and_closure_updates(self):
        p = tensor([1.], requires_grad=True)
        opt = fr.optim.SGD([p], lr=1+2**-27)
        p.grad = tensor([1-2**-27]); opt.step()
        self.assertEqual(p.item(), 2**-54)
        p = tensor([0.], requires_grad=True)
        opt = fr.optim.SGD([p], lr=1e-200)
        def closure():
            opt.param_groups[0]['lr'] = 1e-201
            p.grad = tensor([1e200])
            return p
        self.assertIs(opt.step(closure), p)
        q = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        q.grad = torch.tensor([1e200], dtype=torch.float64)
        torch.optim.SGD([q], lr=1e-201).step()
        self.assert_tensor(p, q)
        self.assertEqual(opt.state_dict()['param_groups'][0]['lr'], 1e-201)
        self.assertEqual(len(opt._native.configurations()[0]), 25)
        with self.assertRaises(ValueError):
            opt.add_param_group({'params': [tensor([1.], 'float32', True)], 'lr': 1e100})
        self.assertEqual(len(opt.param_groups), 1)
        layer = fr.nn.Linear(1, 1, bias=False, dtype='float64')
        oracle = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with fr.no_grad(): layer.weight.copy_(tensor([[1e-100]]))
        with torch.no_grad(): oracle.weight.fill_(1e-100)
        native = fr.optim.SGD(layer.parameters(), lr=1e-200, momentum=.7, dampening=.3)
        expected = torch.optim.SGD(oracle.parameters(), lr=1e-200, momentum=.7, dampening=.3, foreach=False)
        for _ in range(2):
            native.zero_grad(); expected.zero_grad()
            loss = layer(tensor([[1e100]])).square().sum()
            reference_loss = oracle(torch.tensor([[1e100]], dtype=torch.float64)).square().sum()
            self.assert_tensor(loss, reference_loss)
            loss.backward(); reference_loss.backward()
            self.assert_tensor(layer.weight.grad, oracle.weight.grad)
            native.step(); expected.step()
            self.assert_tensor(layer.weight, oracle.weight)
        for rate, momentum in ((0., 0.), (0., .9), (.1, .9)):
            p = tensor([-0., float('inf'), -float('inf'), float('nan')], requires_grad=True)
            q = torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True)
            values = [float('inf'), 0., -0., float('nan')]
            p.grad = tensor(values); q.grad = torch.tensor(values, dtype=torch.float64)
            fr.optim.SGD([p], lr=rate, momentum=momentum).step()
            torch.optim.SGD([q], lr=rate, momentum=momentum, foreach=False).step()
            self.assert_tensor(p, q)
        for shape in ([], [0], [2, 0]):
            p = fr.zeros(shape, dtype='float64', requires_grad=True)
            q = torch.zeros(shape, dtype=torch.float64, requires_grad=True)
            p.grad = fr.ones(shape, dtype='float64'); q.grad = torch.ones_like(q)
            opt = fr.optim.SGD([p], lr=.13, momentum=.7, dampening=.3)
            ref = torch.optim.SGD([q], lr=.13, momentum=.7, dampening=.3, foreach=False)
            opt.step(); ref.step(); self.assert_tensor(p, q)
            self.assert_tensor(opt.state_dict()['state'][0]['momentum_buffer'], ref.state[q]['momentum_buffer'])

    def test_live_strided_gradient_identity_zero_grad_and_first_buffer(self):
        p = tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
        q = torch.tensor(p.tolist(), dtype=torch.float64, requires_grad=True)
        seed = tensor([[.1, .2, .3], [.4, .5, .6]], requires_grad=True)
        reference = torch.tensor(seed.tolist(), dtype=torch.float64, requires_grad=True)
        grad = seed.transpose(0, 1); oracle_grad = reference.T
        p.grad = grad; q.grad = oracle_grad
        opt = fr.optim.SGD([p], lr=.1, momentum=.9, dampening=.7)
        ref = torch.optim.SGD([q], lr=.1, momentum=.9, dampening=.7, foreach=False)
        opt.step(); ref.step()
        self.assertEqual(p.grad._id(), grad._id()); self.assertIs(q.grad, oracle_grad)
        self.assertTrue(grad.requires_grad); self.assertEqual(grad.stride(), oracle_grad.stride())
        self.assert_tensor(p, q)
        self.assert_tensor(opt.state_dict()['state'][0]['momentum_buffer'], ref.state[q]['momentum_buffer'])
        # Existing checked in-place API does not implement detach_ on a live
        # manual gradient. Step preserves it; clearing its slot is supported.
        self.assert_tensor(fr.autograd.grad(grad.sum(), seed)[0], torch.autograd.grad(oracle_grad.sum(), reference)[0])
        opt.zero_grad(); ref.zero_grad(); self.assertIsNone(p.grad)
        p.grad = tensor([[1., 2.], [3., 4.], [5., 6.]])
        q.grad = torch.tensor(p.grad.tolist(), dtype=torch.float64)
        identity = p.grad._id()
        opt.zero_grad(set_to_none=False); ref.zero_grad(set_to_none=False)
        self.assertEqual(p.grad._id(), identity); self.assert_tensor(p.grad, q.grad)
        self.assertEqual(p.grad.dtype, 'float64')
        opt.zero_grad(); self.assertIsNone(p.grad)

    def test_state_dictionary_owns_typed_buffers_and_rejects_late_corruption(self):
        model, opt = setup(); generator = fr.Generator(7)
        update(model, opt, generator, 1)
        saved = opt.state_dict(); old = state_bits(saved)
        update(model, opt, generator, 2)
        self.assertEqual(state_bits(saved), old)
        fresh, restored = setup(); restored.load_state_dict(saved)
        self.assertEqual(state_bits(restored.state_dict()), old)
        for bad_value in (tensor([0., 0.], 'float32'), tensor([0.]), tensor([0., 0.], 'int64')):
            bad = {'state': {i: dict(v) for i, v in saved['state'].items()}, 'param_groups': [dict(g) for g in saved['param_groups']]}
            bad['state'][0]['momentum_buffer'] = bad_value
            before = state_bits(restored.state_dict())
            with self.assertRaises(ValueError): restored.load_state_dict(bad)
            self.assertEqual(state_bits(restored.state_dict()), before)
        p = tensor([0.], requires_grad=True)
        opt = fr.optim.SGD([p], lr=.1, momentum=.9)
        opt.load_state_dict(opt.state_dict())
        p.grad = tensor([-0.]); opt.step()
        self.assertEqual(flat_bits(opt.state_dict()['state'][0]['momentum_buffer']), struct.pack('<d', -0.).hex())

    def test_all_training_state_and_rng_resume_in_a_fresh_process(self):
        model, opt = setup(); rng = fr.Generator(17)
        for i in range(5): update(model, opt, rng, i)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fr.checkpoint.save_training(path / 'training', model, optimizers={'sgd': opt}, generator=rng, step=5)
            expected = continuation(model, opt, rng)
            env = dict(os.environ); env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', directory],
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((path / 'result.json').read_text()), expected)

    def test_exact_config_and_late_checkpoint_faults_leave_every_target_unchanged(self):
        from test_training_checkpoint import rewrite_checkpoint
        source, optimizer = setup(); rng = fr.Generator(7); update(source, optimizer, rng, 1)
        saved = fr.checkpoint.snapshot_training(source, optimizers={'sgd': optimizer}, generator=rng)
        target, restored = setup(); other = fr.Generator(999); update(target, restored, other, 2)
        keys = saved.tensors()
        config_key = next(k for k in keys if k.startswith('optim.') and k.endswith('config'))
        config = keys[config_key].tolist()
        corrupt = list(config); corrupt[10] = .5
        cases = {config_key: fr.Tensor(corrupt, [len(corrupt)]),
                 'optim.sgd.g0.velocity.0': tensor([0., 0.], 'float32'),
                 'optim.sgd.g0.initialized.0': fr.Tensor([.5], [])}
        for key, value in cases.items():
            before = digest(fr.checkpoint.snapshot_training(target, optimizers={'sgd': restored}, generator=other))
            ids = [p._id() for p in target.parameters()]
            with tempfile.TemporaryDirectory() as directory:
                bad = rewrite_checkpoint(saved, directory, {key: value})
                with self.assertRaises(ValueError): bad.restore(target, optimizers={'sgd': restored}, generator=other)
            self.assertEqual([p._id() for p in target.parameters()], ids)
            self.assertEqual(digest(fr.checkpoint.snapshot_training(target, optimizers={'sgd': restored}, generator=other)), before)
        restored.param_groups[0]['lr'] = .031000000000000003
        scheduler = fr.optim.lr_scheduler.StepLR(restored, 2, gamma=.7)
        scheduler.step(); scheduler.step()
        self.assertEqual(restored.state_dict()['param_groups'][0]['lr'], .031000000000000003 * .7)

    def test_every_other_optimizer_rejects_f64_before_updates(self):
        names = ('Adamax', 'Adadelta', 'ASGD',
                 'Rprop', 'RAdam', 'NAdam', 'Adafactor', 'Muon', 'LBFGS')
        for name in names:
            p = tensor([[1., 2.], [3., 4.]], requires_grad=True)
            before = flat_bits(p)
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                getattr(fr.optim, name)([p], lr=.01)
            self.assertEqual(flat_bits(p), before); self.assertIsNone(p.grad)

    def test_shared_parameters_and_frozen_missing_gradients_keep_separate_state(self):
        data = tensor([.7, -.3])
        a = fr.nn.Parameter._from_shared(data, True); b = fr.nn.Parameter._from_shared(data, True)
        other = torch.tensor([.7, -.3], dtype=torch.float64)
        x = torch.nn.Parameter(other); y = torch.nn.Parameter(other)
        opt = fr.optim.SGD([a, b], lr=.13, momentum=.7, weight_decay=.2, dampening=.3)
        ref = torch.optim.SGD([x, y], lr=.13, momentum=.7, weight_decay=.2, dampening=.3, foreach=False)
        for step in range(9):
            opt.zero_grad(); ref.zero_grad()
            for index, (p, q) in enumerate(((a, x), (b, y))):
                p.trainable = not (index == 1 and step % 3 == 1)
                if (step+index) % 4 != 0:
                    values = [step*.1+index, -.2]
                    p.grad = tensor(values)
                    if p.trainable: q.grad = torch.tensor(values, dtype=torch.float64)
            opt.step(); ref.step(); self.assert_tensor(data, other)
            for i, q in enumerate((x, y)):
                if 'momentum_buffer' in ref.state[q]:
                    self.assert_tensor(opt.state_dict()['state'][i]['momentum_buffer'], ref.state[q]['momentum_buffer'])


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        import test_sgd_f64
        test_sgd_f64.worker(sys.argv[2])
    else:
        unittest.main()
