"""Native group updates, scheduler parity, and exact independent-process restart."""
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

import torch
import ferro as fr
from ferro.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


class Model(fr.nn.Module):
    def build(self):
        self.a = fr.nn.Parameter(fr.Tensor([1., -.5], [2]))
        self.b = fr.nn.Parameter(fr.Tensor([.2, .7, 1.3], [3]))
        self.c = fr.nn.Parameter(fr.Tensor([-.3], [1]))


def groups(parameters, kind, wrong=False):
    p, q, r = parameters
    options = [{'params': [p, r], 'lr': .02}, {'params': [q], 'lr': .005}]
    defaults = {'lr': .03}
    if kind == 'SGD':
        defaults.update(momentum=.7, nesterov=True)
        options[0].update(momentum=.4)
        options[1].update(momentum=.9, nesterov=False)
    else:
        defaults.update(betas=(.8, .93), eps=1e-7)
        options[0].update(eps=1e-6)
        options[1].update(betas=(.6, .85))
        if kind == 'AdamW':
            defaults['weight_decay'] = .03
            options[0]['weight_decay'], options[1]['weight_decay'] = 0., .12
    if wrong:
        for group in options:
            group['lr'] = .7
    return options, defaults


def setup(kind='AdamW', wrong=False):
    model = Model()
    params = list(model.parameters())
    specs, defaults = groups(params, kind, wrong)
    return model, getattr(fr.optim, kind)(specs, **defaults)


def plain(value):
    if isinstance(value, fr.Tensor):
        return {'shape': value.shape, 'dtype': value.dtype, 'values': value.tolist()}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def checkpoint_digest(checkpoint):
    h = hashlib.sha256()
    for name, value in sorted(checkpoint.tensors().items()):
        h.update(json.dumps([name, value.dtype, value.shape], separators=(',', ':')).encode())
        data = value.reshape([value.numel()]).tolist()
        h.update(struct.pack('<' + 'f' * len(data), *data))
    return h.hexdigest()


def stochastic_update(model, optimizer, scheduler, rng, index):
    optimizer.zero_grad()
    loss = None
    for pi, parameter in enumerate(model.parameters()):
        if (index + pi) % 4 == 0:
            continue
        term = (parameter.tensor() + fr.Tensor.randn(parameter.tensor().shape, generator=rng) * .03).square().sum()
        loss = term if loss is None else loss + term
    loss.backward()
    value = loss.item()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return value


def continuation(model, optimizer, scheduler, rng, start):
    losses = [stochastic_update(model, optimizer, scheduler, rng, i) for i in range(start, 14)]
    cp = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=14)
    return {'losses': losses, 'state': checkpoint_digest(cp), 'optimizer': plain(optimizer.state_dict()),
            'scheduler': scheduler.state_dict(), 'draws': fr.Tensor.randn([11], generator=rng).tolist()}


def checkpoint_worker(directory):
    directory = Path(directory)
    model, optimizer = setup(wrong=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=7, eta_min=.001)
    rng = fr.Generator(999)
    checkpoint = fr.checkpoint.load(directory / 'training')
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=rng)
    scheduler.load_state_dict(json.loads((directory / 'scheduler.json').read_text()))
    (directory / 'result.json').write_text(json.dumps(continuation(model, optimizer, scheduler, rng, checkpoint.step), allow_nan=False))


class OptimizerGroupSchedulerTests(unittest.TestCase):
    def compare_states(self, actual, expected):
        a, b = actual.state_dict(), expected.state_dict()
        self.assertEqual(set(a['state']), set(b['state']))
        for key, state in a['state'].items():
            self.assertEqual(set(state), set(b['state'][key]))
            for name, value in state.items():
                got = torch.tensor(value.tolist()).reshape(value.shape)
                torch.testing.assert_close(got, b['state'][key][name], atol=3e-6, rtol=5e-5)
        for group, reference in zip(a['param_groups'], b['param_groups']):
            for key, value in group.items():
                self.assertEqual(value, reference[key])

    def test_group_updates_moments_skips_freezing_and_mutable_rates_match_torch(self):
        for kind in ('SGD', 'Adam', 'AdamW'):
            with self.subTest(optimizer=kind):
                model, opt = setup(kind)
                params = list(model.parameters())
                reference = [torch.nn.Parameter(torch.tensor(p.tensor().tolist())) for p in params]
                specifications, defaults = groups(reference, kind)
                oracle = getattr(torch.optim, kind)(specifications, **defaults)
                self.compare_states(opt, oracle)
                for step in range(15):
                    opt.zero_grad()
                    oracle.zero_grad(set_to_none=True)
                    if step == 5:
                        opt.param_groups[0]['lr'] = oracle.param_groups[0]['lr'] = .011
                    for pi, (p, tp) in enumerate(zip(params, reference)):
                        p.trainable = not (pi == 1 and step % 3 == 1)
                        if (step + pi) % 4 == 0:
                            continue
                        scale, shift = .2 + (pi + step) * .03, -.1 + pi * .04
                        (p.tensor().square().sum() * scale + p.tensor().sum() * shift).backward()
                        (tp.square().sum() * scale + tp.sum() * shift).backward()
                        if not p.trainable:
                            tp.grad = None
                    opt.step()
                    oracle.step()
                    for p, tp in zip(params, reference):
                        torch.testing.assert_close(torch.tensor(p.tensor().tolist()), tp.detach(), atol=3e-6, rtol=5e-5)
                    self.compare_states(opt, oracle)

    def test_add_group_duplicate_rejection_and_zero_grad_modes(self):
        p, q = fr.nn.Parameter(fr.Tensor([1.], [1])), fr.nn.Parameter(fr.Tensor([2.], [1]))
        opt = fr.optim.SGD([p, p], lr=.1, momentum=.5)
        self.assertEqual(len(opt.params), 1)
        p.tensor().sum().backward()
        opt.step()
        before = plain(opt.state_dict())
        with self.assertRaises(ValueError):
            opt.add_param_group({'params': [p]})
        self.assertEqual(plain(opt.state_dict()), before)
        opt.add_param_group({'params': [q], 'lr': .03, 'momentum': 0.})
        (p.tensor().sum() + q.tensor().sum()).backward()
        opt.zero_grad(set_to_none=False)
        self.assertEqual(p.grad.tolist(), [0.])
        self.assertEqual(q.grad.tolist(), [0.])
        opt.zero_grad()
        self.assertIsNone(p.grad)
        self.assertIsNone(q.grad)
        (p.tensor().sum() + q.tensor().sum()).backward()
        opt.step()
        self.assertAlmostEqual(q.tensor().item(), 1.97, places=6)
        with self.assertRaises(ValueError):
            fr.optim.Adam([{'params': [p]}, {'params': [p]}])

    def test_state_dict_owns_values_and_load_is_atomic_across_groups(self):
        for kind in ('SGD', 'Adam', 'AdamW'):
            model, opt = setup(kind)
            for p in model.parameters():
                p.tensor().square().sum().backward()
            opt.step()
            owned = opt.state_dict()
            saved = plain(owned)
            for p in model.parameters():
                p.tensor().sum().backward()
            opt.step()
            self.assertEqual(plain(owned), saved)
            other_model, other = setup(kind, wrong=True)
            other.load_state_dict(owned)
            self.assertEqual(plain(other.state_dict()), saved)
            for fault in ('late_shape', 'late_dtype', 'late_lr', 'duplicate_id', 'extra_state'):
                bad = opt.state_dict()
                if fault == 'late_shape':
                    key = 'momentum_buffer' if kind == 'SGD' else 'exp_avg'
                    bad['state'][2][key] = fr.Tensor.ones([99])
                elif fault == 'late_dtype':
                    key = 'momentum_buffer' if kind == 'SGD' else 'exp_avg'
                    bad['state'][2][key] = fr.Tensor.from_i64([1, 2, 3], [3])
                elif fault == 'late_lr':
                    bad['param_groups'][1]['lr'] = float('nan')
                elif fault == 'duplicate_id':
                    bad['param_groups'][1]['params'] = [0]
                else:
                    bad['state'][99] = {}
                before = plain(other.state_dict())
                with self.subTest(kind=kind, fault=fault), self.assertRaises(ValueError):
                    other.load_state_dict(bad)
                self.assertEqual(plain(other.state_dict()), before)

    def test_schedulers_match_torch_multistep_and_closed_form(self):
        cases = [('StepLR', {'step_size': 3, 'gamma': .3}),
                 ('ExponentialLR', {'gamma': .87}),
                 ('CosineAnnealingLR', {'T_max': 4, 'eta_min': .001})]
        for name, options in cases:
            for explicit in (False, True):
                model, opt = setup('AdamW')
                specs, defaults = groups([torch.nn.Parameter(torch.ones(p.tensor().shape)) for p in model.parameters()], 'AdamW')
                oracle = torch.optim.AdamW(specs, **defaults)
                sched = getattr(fr.optim.lr_scheduler, name)(opt, **options)
                reference = getattr(torch.optim.lr_scheduler, name)(oracle, **options)
                self.assertEqual(sched.state_dict(), reference.state_dict())
                for step in range(13):
                    opt.step()
                    oracle.step()
                    if step == 3 and not explicit:
                        opt.param_groups[1]['lr'] = oracle.param_groups[1]['lr'] = .009
                    epoch = step + 2 if explicit else None
                    sched.step(epoch)
                    reference.step(epoch)
                    with self.subTest(name=name, explicit=explicit, step=step):
                        self.assertEqual(sched.get_last_lr(), reference.get_last_lr())
                        self.assertEqual(sched.state_dict(), reference.state_dict())

    def test_plateau_threshold_patience_cooldown_floors_and_nonfinite_metrics(self):
        metrics = [1., .98, .979, 1.1, 1.2, 1.3, 1.4, .4, .4, .4, float('nan'), float('inf'), .4, .4]
        for mode in ('min', 'max'):
            for threshold_mode in ('rel', 'abs'):
                model, opt = setup('SGD')
                specs, defaults = groups([torch.nn.Parameter(torch.ones(p.tensor().shape)) for p in model.parameters()], 'SGD')
                oracle = torch.optim.SGD(specs, **defaults)
                options = dict(mode=mode, factor=.4, patience=1, cooldown=2, threshold=.03,
                               threshold_mode=threshold_mode, min_lr=[.002, .001], eps=1e-9)
                sched, reference = ReduceLROnPlateau(opt, **options), torch.optim.lr_scheduler.ReduceLROnPlateau(oracle, **options)
                for value in metrics:
                    sched.step(value)
                    reference.step(value)
                    self.assertEqual(sched.state_dict(), reference.state_dict())
                    self.assertEqual(sched.get_last_lr(), reference.get_last_lr())

    def test_scheduler_state_load_rejects_atomically_and_does_not_change_optimizer_lr(self):
        for cls, options in [(StepLR, {'step_size': 2}), (ExponentialLR, {'gamma': .9}),
                             (CosineAnnealingLR, {'T_max': 3}), (ReduceLROnPlateau, {'patience': 1})]:
            _, opt = setup()
            sched = cls(opt, **options)
            for i in range(3):
                sched.step(float(i)) if cls is ReduceLROnPlateau else sched.step()
            saved = sched.state_dict()
            _, other = setup(wrong=True)
            restored = cls(other, **options)
            rates = [g['lr'] for g in other.param_groups]
            restored.load_state_dict(saved)
            self.assertEqual(restored.state_dict(), saved)
            self.assertEqual([g['lr'] for g in other.param_groups], rates)
            for key, value in [('_last_lr', [float('nan'), .1]), ('last_epoch', -20), ('_last_lr', [.1])]:
                bad = copy.deepcopy(saved)
                bad[key] = value
                with self.assertRaises(ValueError):
                    restored.load_state_dict(bad)
                self.assertEqual(restored.state_dict(), saved)
                self.assertEqual([g['lr'] for g in other.param_groups], rates)
            snapshot = restored.state_dict()
            snapshot['_last_lr'][0] = 999.
            self.assertEqual(restored.state_dict(), saved)

    def test_group_configuration_preflight_and_step_failure_do_not_update_parameters(self):
        model, opt = setup()
        for p in model.parameters():
            p.tensor().sum().backward()
        before = [p.tensor().tolist() for p in model.parameters()]
        opt.param_groups[1]['lr'] = float('nan')
        with self.assertRaises(ValueError):
            opt.step()
        self.assertEqual([p.tensor().tolist() for p in model.parameters()], before)
        opt.param_groups[1]['lr'] = .01
        opt.param_groups[1]['params'] = [model.a]
        with self.assertRaises(ValueError):
            opt.step()
        self.assertEqual([p.tensor().tolist() for p in model.parameters()], before)

    def test_checkpoint_metadata_corruption_and_late_group_state_reject_atomically(self):
        from test_training_checkpoint import rewrite_checkpoint
        model, opt = setup()
        rng = fr.Generator(7)
        for p in model.parameters():
            p.tensor().square().sum().backward()
        opt.step()
        opt.zero_grad()
        cp = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng)
        replacement = {
            'scalars.optim_meta.main.g1.lr': fr.Tensor([0., 0., 0., 0.], [4]),
            'optim.main.g1.m.0': fr.Tensor.ones([2]),
            'optim.main.g1.t': fr.Tensor([-1.], [1]),
        }
        target, other = setup(wrong=True)
        other_rng = fr.Generator(41)
        for name, value in replacement.items():
            with self.subTest(key=name), tempfile.TemporaryDirectory() as directory:
                malformed = rewrite_checkpoint(cp, directory, {name: value})
                before = checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': other}, generator=other_rng))
                group_before = plain(other.state_dict())
                with self.assertRaises(ValueError):
                    malformed.restore(target, optimizers={'main': other}, generator=other_rng)
                self.assertEqual(plain(other.state_dict()), group_before)
                self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': other}, generator=other_rng)), before)
        # Same flattened parameter order, different group partition must reject.
        repartitioned = fr.optim.AdamW([{'params': [target.a]}, {'params': [target.c, target.b]}], lr=.7)
        before = checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': repartitioned}, generator=other_rng))
        with self.assertRaises(ValueError):
            cp.restore(target, optimizers={'main': repartitioned}, generator=other_rng)
        self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': repartitioned}, generator=other_rng)), before)

    def test_legacy_checkpoint_without_python_optimizer_metadata_keeps_native_updates(self):
        for kind in ('SGD', 'Adam', 'AdamW'):
            model, rng = Model(), fr.Generator(23)
            options = {'lr': .031, 'momentum': .7} if kind == 'SGD' else {'lr': .031}
            optimizer = getattr(fr.optim, kind)(model.parameters(), **options)
            for _ in range(2):
                for p in model.parameters():
                    p.tensor().square().sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            # Recreate the exact pre-group Python adapter: no optim_meta scalars.
            params, buffers, scalars, opts, generator, _ = fr.checkpoint._state(model, {'main': optimizer}, rng, False)
            raw = fr._native._TrainingCheckpoint.snapshot(params, buffers, scalars, opts, generator, 2)
            legacy = fr.checkpoint.TrainingCheckpoint(raw)
            self.assertFalse(any('optim_meta.' in name for name in legacy.tensors()))
            target, other_rng = Model(), fr.Generator(999)
            restored = getattr(fr.optim, kind)(target.parameters(), **{**options, 'lr': .9})
            with tempfile.TemporaryDirectory() as directory:
                legacy.save(directory)
                fr.checkpoint.load_training(directory, target, optimizers={'main': restored}, generator=other_rng)
            self.assertEqual(restored._native.configurations(), optimizer._native.configurations())
            for live, opt in ((model, optimizer), (target, restored)):
                for p in live.parameters():
                    (p.tensor().square().sum() * .37 + p.tensor().sum()).backward()
                opt.step()
            self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in target.parameters()])
            self.assertEqual(plain(dict(optimizer._native.state_tensors())), plain(dict(restored._native.state_tensors())))
            self.assertEqual(fr.Tensor.randn([7], generator=rng).tolist(), fr.Tensor.randn([7], generator=other_rng).tolist())

    def test_fresh_process_group_and_scheduler_continuation_is_bit_exact(self):
        model, opt = setup()
        sched = CosineAnnealingLR(opt, T_max=7, eta_min=.001)
        rng = fr.Generator(19)
        for i in range(5):
            stochastic_update(model, opt, sched, rng, i)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fr.checkpoint.save_training(path / 'training', model, optimizers={'main': opt}, generator=rng, step=5)
            (path / 'scheduler.json').write_text(json.dumps(sched.state_dict(), allow_nan=False))
            expected = continuation(model, opt, sched, rng, 5)
            code = 'import sys; sys.path.insert(0, sys.argv[1]); import test_optimizer_groups_schedulers as proof; proof.checkpoint_worker(sys.argv[2])'
            subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parent), str(path)],
                           check=True, timeout=60, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertEqual(json.loads((path / 'result.json').read_text()), expected)


if __name__ == '__main__':
    unittest.main()
