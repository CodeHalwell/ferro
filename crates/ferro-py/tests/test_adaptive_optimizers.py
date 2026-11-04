"""Actual Torch optimizer trajectories and independently resumed native state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from test_optimizer_groups_schedulers import Model, plain, checkpoint_digest


KINDS = ('RMSprop', 'Adagrad', 'Adamax', 'Adadelta', 'ASGD', 'Rprop', 'RAdam', 'NAdam')


def specifications(params, kind, wrong=False):
    a, b, c = params
    groups = [{'params': [a, c], 'lr': .023, 'weight_decay': .07},
              {'params': [b], 'lr': .011, 'weight_decay': .13, 'maximize': True}]
    defaults = dict(lr=.017)
    if kind == 'RMSprop':
        defaults.update(alpha=.85, eps=2e-5, centered=True, momentum=.6)
        groups[1].update(alpha=.72, momentum=.2, centered=False)
    elif kind == 'Adagrad':
        defaults.update(lr_decay=.04, initial_accumulator_value=.2, eps=1e-7)
        groups[1].update(lr_decay=.11, initial_accumulator_value=.7)
    elif kind == 'Adamax':
        defaults.update(betas=(.7, .89), eps=2e-5)
        groups[1].update(betas=(.5, .77))
    elif kind == 'Adadelta':
        defaults.update(rho=.8, eps=2e-4)
        groups[0]['lr'], groups[1]['lr'] = .7, .3
        groups[1]['rho'] = .6
    elif kind == 'ASGD':
        defaults.update(lambd=.03, alpha=.65, t0=3.)
        groups[1].update(lambd=.06, alpha=.5, t0=1.)
    elif kind == 'Rprop':
        defaults.update(etas=(.5, 1.3), step_sizes=(.002, .08))
        groups[1].update(etas=(.4, 1.4), step_sizes=(.001, .07))
        for group in groups:
            group.pop('weight_decay')
    else:
        defaults.update(betas=(.7, .89), eps=2e-5)
        groups[1].update(betas=(.5, .77), decoupled_weight_decay=True)
        if kind == 'NAdam':
            defaults['momentum_decay'] = .013
            groups[1]['momentum_decay'] = .027
    if wrong:
        for group in groups:
            group['lr'] = .8
    return groups, defaults


def setup(kind, wrong=False):
    model = Model()
    groups, defaults = specifications(list(model.parameters()), kind, wrong)
    return model, getattr(fr.optim, kind)(groups, **defaults)


def update(model, optimizer, rng, index):
    optimizer.zero_grad()
    loss = None
    for pi, parameter in enumerate(model.parameters()):
        parameter.trainable = not (pi == 1 and index % 4 == 2)
        if (index + pi) % 5 == 0 or not parameter.trainable:
            continue
        term = (parameter.tensor() + fr.Tensor.randn(parameter.tensor().shape, generator=rng) * .05).square().sum()
        loss = term if loss is None else loss + term
    loss.backward()
    value = loss.item()
    optimizer.step()
    optimizer.zero_grad()
    return value


def continuation(model, optimizer, scheduler, rng, start):
    losses = []
    for index in range(start, 19):
        losses.append(update(model, optimizer, rng, index))
        scheduler.step()
    snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=19)
    return {'losses': losses, 'checkpoint': checkpoint_digest(snapshot),
            'optimizer': plain(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
            'draws': fr.Tensor.randn([9], generator=rng).tolist()}


def worker(kind, path):
    path = Path(path)
    model, optimizer = setup(kind, wrong=True)
    scheduler = fr.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.8)
    rng = fr.Generator(999)
    cp = fr.checkpoint.load(path / 'training')
    cp.restore(model, optimizers={'main': optimizer}, generator=rng)
    scheduler.load_state_dict(json.loads((path / 'scheduler.json').read_text()))
    (path / 'result.json').write_text(json.dumps(continuation(model, optimizer, scheduler, rng, cp.step), allow_nan=False))


class AdaptiveOptimizerTests(unittest.TestCase):
    def compare_state(self, optimizer, reference):
        actual, expected = optimizer.state_dict(), reference.state_dict()
        self.assertEqual(set(actual['state']), set(expected['state']))
        for identifier, payload in actual['state'].items():
            other = expected['state'][identifier]
            self.assertEqual(set(payload), set(other))
            for name, tensor in payload.items():
                torch.testing.assert_close(torch.tensor(tensor.tolist()).reshape(tensor.shape), other[name],
                                           atol=1e-18 if name == 'mu_product' else 4e-6, rtol=6e-5,
                                           msg=lambda message: f'{type(optimizer).__name__} {identifier}/{name}: {message}')
        for group, other in zip(actual['param_groups'], expected['param_groups']):
            for key, value in group.items():
                self.assertEqual(value, other[key])

    def test_every_state_and_parameter_matches_torch_with_groups_skips_freezing_and_maximize(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                model, optimizer = setup(kind)
                parameters = list(model.parameters())
                reference = [torch.nn.Parameter(torch.tensor(p.tensor().tolist())) for p in parameters]
                groups, defaults = specifications(reference, kind)
                oracle = getattr(torch.optim, kind)(groups, **defaults, foreach=False)
                self.compare_state(optimizer, oracle)
                for step in range(35):
                    optimizer.zero_grad()
                    oracle.zero_grad(set_to_none=True)
                    if step == 11:
                        optimizer.param_groups[0]['lr'] = oracle.param_groups[0]['lr'] = .019
                    for pi, (parameter, other) in enumerate(zip(parameters, reference)):
                        parameter.trainable = not (pi == 1 and step % 3 == 1)
                        if (pi + step) % 4 == 0:
                            continue
                        scale, shift = .23 + step * .017 + pi * .05, -.17 + pi * .06
                        (parameter.tensor().square().sum() * scale + parameter.tensor().sum() * shift).backward()
                        (other.square().sum() * scale + other.sum() * shift).backward()
                        if not parameter.trainable:
                            other.grad = None
                    optimizer.step()
                    oracle.step()
                    for parameter, other in zip(parameters, reference):
                        torch.testing.assert_close(torch.tensor(parameter.tensor().tolist()), other.detach(), atol=4e-6, rtol=6e-5)
                        if parameter.trainable:
                            if other.grad is None:
                                self.assertIsNone(parameter.grad)
                            else:
                                torch.testing.assert_close(torch.tensor(parameter.grad.tolist()), other.grad, atol=4e-6, rtol=6e-5)
                    self.compare_state(optimizer, oracle)

    def test_boundary_options_scalar_and_empty_parameters_match_torch(self):
        variants = [('RMSprop', dict(alpha=0., momentum=0., centered=False, eps=0.)),
                    ('RMSprop', dict(alpha=1., momentum=.3, centered=True, eps=.2)),
                    ('Adagrad', dict(lr_decay=0., initial_accumulator_value=0., eps=0.)),
                    ('Adamax', dict(betas=(0., 0.), eps=0.)),
                    ('Adadelta', dict(rho=0., eps=0.)),
                    ('Adadelta', dict(rho=1., eps=.01)),
                    ('ASGD', dict(lambd=0., alpha=0., t0=0.)),
                    ('Rprop', dict(etas=(.5, 1.2), step_sizes=(.01, .02))),
                    ('RAdam', dict(betas=(0., 0.), eps=0.)),
                    ('RAdam', dict(betas=(.8, .9), decoupled_weight_decay=True, weight_decay=.1)),
                    ('NAdam', dict(betas=(0., 0.), momentum_decay=0., eps=0.)),
                    ('NAdam', dict(betas=(.6, .9), momentum_decay=.02, decoupled_weight_decay=True, weight_decay=.1))]
        for kind, options in variants:
            with self.subTest(kind=kind, options=options):
                parameters = [fr.nn.Parameter(fr.Tensor([.7], [])), fr.nn.Parameter(fr.Tensor([], [0]))]
                others = [torch.nn.Parameter(torch.tensor(.7)), torch.nn.Parameter(torch.empty(0))]
                optimizer = getattr(fr.optim, kind)(parameters, lr=.03, **options)
                oracle = getattr(torch.optim, kind)(others, lr=.03, **options, foreach=False)
                for step in range(5):
                    optimizer.zero_grad()
                    oracle.zero_grad(set_to_none=True)
                    for parameter, other in zip(parameters, others):
                        (parameter.tensor().sum() * (step + .4)).backward()
                        (other.sum() * (step + .4)).backward()
                    optimizer.step()
                    oracle.step()
                    self.compare_state(optimizer, oracle)
                    for parameter, other in zip(parameters, others):
                        torch.testing.assert_close(torch.tensor(parameter.tensor().tolist()).reshape(parameter.tensor().shape), other.detach(), atol=4e-6, rtol=6e-5)

    def test_state_dict_ownership_and_late_group_failures_are_atomic(self):
        for kind in KINDS:
            model, optimizer = setup(kind)
            update(model, optimizer, fr.Generator(7), 1)
            saved, before = optimizer.state_dict(), plain(optimizer.state_dict())
            update(model, optimizer, fr.Generator(11), 2)
            self.assertEqual(plain(saved), before)
            target, restored = setup(kind, wrong=True)
            restored.load_state_dict(saved)
            self.assertEqual(plain(restored.state_dict()), before)
            for fault in ('shape', 'dtype', 'step', 'config', 'key'):
                bad = optimizer.state_dict()
                entry = bad['state'][2]
                buffer = next(key for key in entry if key not in ('step', 'eta', 'mu'))
                if fault == 'shape':
                    entry[buffer] = fr.Tensor.ones([99])
                elif fault == 'dtype':
                    entry[buffer] = fr.Tensor.from_i64([1, 2, 3], [3])
                elif fault == 'step':
                    entry['step'] = fr.Tensor([2**24 + 2.], [])
                elif fault == 'config':
                    bad['param_groups'][1]['lr'] = float('nan')
                else:
                    entry['unexpected'] = fr.Tensor([0.], [])
                snapshot = plain(restored.state_dict())
                weights = [p.tensor().tolist() for p in target.parameters()]
                with self.subTest(kind=kind, fault=fault), self.assertRaises(ValueError):
                    restored.load_state_dict(bad)
                self.assertEqual(plain(restored.state_dict()), snapshot)
                self.assertEqual([p.tensor().tolist() for p in target.parameters()], weights)

    def test_checkpoint_late_moment_counter_and_metadata_corruption_are_atomic(self):
        from test_training_checkpoint import rewrite_checkpoint
        for kind in KINDS:
            model, optimizer = setup(kind)
            update(model, optimizer, fr.Generator(7), 1)
            rng = fr.Generator(23)
            checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng)
            names = optimizer._all_state_names
            moment = next(name for name in names if name not in ('eta', 'mu'))
            faults = {f'optim.main.g1.{moment}.0': fr.Tensor.ones([99]),
                      'optim.main.g1.t.0': fr.Tensor([-1.], []),
                      'scalars.optim_meta.main.g1.lr': fr.Tensor([0., 0., 0., 0.], [4])}
            target, restored = setup(kind, wrong=True)
            other_rng = fr.Generator(999)
            for key, value in faults.items():
                before = checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng))
                with self.subTest(kind=kind, key=key), tempfile.TemporaryDirectory() as directory:
                    bad = rewrite_checkpoint(checkpoint, directory, {key: value})
                    with self.assertRaises(ValueError):
                        bad.restore(target, optimizers={'main': restored}, generator=other_rng)
                self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng)), before)

    def test_every_family_resumes_exactly_in_fresh_process_with_scheduler_and_rng(self):
        for kind in KINDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                model, optimizer = setup(kind)
                scheduler = fr.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.8)
                rng = fr.Generator(23)
                for index in range(7):
                    update(model, optimizer, rng, index)
                    scheduler.step()
                fr.checkpoint.save_training(path / 'training', model, optimizers={'main': optimizer}, generator=rng, step=7)
                (path / 'scheduler.json').write_text(json.dumps(scheduler.state_dict(), allow_nan=False))
                expected = continuation(model, optimizer, scheduler, rng, 7)
                env = dict(os.environ)
                env.pop('PYTHONPATH', None)
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', kind, str(path)], env=env,
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(child.returncode, 0, child.stdout)
                self.assertEqual(json.loads((path / 'result.json').read_text()), expected)

    def test_add_group_and_eager_adagrad_initial_state_match_torch(self):
        for kind in KINDS:
            p, q = (fr.nn.Parameter(fr.Tensor([value], [1])) for value in (.3, -.2))
            a, b = (torch.nn.Parameter(torch.tensor([value])) for value in (.3, -.2))
            optimizer, oracle = getattr(fr.optim, kind)([p], lr=.03), getattr(torch.optim, kind)([a], lr=.03, foreach=False)
            p.tensor().square().sum().backward()
            a.square().sum().backward()
            optimizer.step()
            oracle.step()
            optimizer.add_param_group({'params': [q], 'lr': .01})
            oracle.add_param_group({'params': [b], 'lr': .01})
            self.compare_state(optimizer, oracle)
            optimizer.zero_grad()
            oracle.zero_grad(set_to_none=True)
            (p.tensor().sum() + q.tensor().sum()).backward()
            (a.sum() + b.sum()).backward()
            optimizer.step()
            oracle.step()
            self.compare_state(optimizer, oracle)

    def test_invalid_configuration_and_late_group_preflight_preserve_parameters(self):
        for kind in KINDS:
            model, optimizer = setup(kind)
            for parameter in model.parameters():
                parameter.tensor().sum().backward()
            before = [p.tensor().tolist() for p in model.parameters()]
            for option, value in (('lr', float('nan')), ('maximize', 1), ('eps', -1.)):
                if option not in optimizer.param_groups[1]:
                    continue
                old = optimizer.param_groups[1][option]
                optimizer.param_groups[1][option] = value
                with self.subTest(kind=kind, option=option), self.assertRaises(ValueError):
                    optimizer.step()
                self.assertEqual([p.tensor().tolist() for p in model.parameters()], before)
                optimizer.param_groups[1][option] = old
            with self.assertRaises(TypeError):
                getattr(fr.optim, kind)(model.parameters(), foreach=True)

    def test_closure_and_leaf_identity_survive_updates(self):
        for kind in KINDS:
            model, optimizer = setup(kind)
            original = model.a.tensor()
            calls = []
            def closure():
                optimizer.zero_grad()
                loss = model.a.tensor().square().sum()
                loss.backward()
                calls.append(loss.item())
                return loss
            with fr.no_grad():
                loss = optimizer.step(closure)
            self.assertEqual(calls, [1.25])
            self.assertEqual(loss.item(), 1.25)
            self.assertEqual(original.tolist(), model.a.tensor().tolist())
            self.assertNotEqual(original.tolist(), [1., -.5])
            self.assertEqual(model.a.grad.tolist(), [2., -1.])

    def test_rmsprop_state_layout_flags_cannot_silently_change(self):
        model, optimizer = setup('RMSprop')
        update(model, optimizer, fr.Generator(7), 1)
        for option, value in (('centered', False), ('momentum', 0.)):
            previous = optimizer.param_groups[0][option]
            optimizer.param_groups[0][option] = value
            before = [p.tensor().tolist() for p in model.parameters()]
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                optimizer.step()
            self.assertEqual([p.tensor().tolist() for p in model.parameters()], before)
            optimizer.param_groups[0][option] = previous

    def test_asgd_initialized_zero_step_and_pristine_checkpoint_preserve_learning_rate_semantics(self):
        parameter = fr.nn.Parameter(fr.Tensor([.7], [1]))
        reference = torch.nn.Parameter(torch.tensor([.7]))
        optimizer, oracle = fr.optim.ASGD([parameter], lr=.04), torch.optim.ASGD([reference], lr=.04, foreach=False)
        for current, tensor in ((optimizer, fr.Tensor), (oracle, torch.tensor)):
            state = current.state_dict()
            scalar = (lambda value: tensor([value], [])) if current is optimizer else (lambda value: tensor(float(value)))
            vector = (lambda value: tensor([value], [1])) if current is optimizer else (lambda value: tensor([value]))
            state['state'][0] = {'step': scalar(0.), 'eta': scalar(.013), 'mu': scalar(.25), 'ax': vector(-.4)}
            current.load_state_dict(state)
        parameter.tensor().sum().backward()
        reference.sum().backward()
        optimizer.step()
        oracle.step()
        self.compare_state(optimizer, oracle)
        torch.testing.assert_close(torch.tensor(parameter.tensor().tolist()), reference.detach())
        model, original = setup('ASGD')
        target, restored = setup('ASGD', wrong=True)
        rng, other_rng = fr.Generator(7), fr.Generator(99)
        checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': original}, generator=rng)
        checkpoint.restore(target, optimizers={'main': restored}, generator=other_rng)
        self.assertEqual(original.state_dict()['state'], {})
        self.assertEqual(restored.state_dict()['state'], {})
        for live, current in ((model, original), (target, restored)):
            current.param_groups[0]['lr'] = .117
            for p in live.parameters():
                p.tensor().sum().backward()
            current.step()
        self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in target.parameters()])
        self.assertEqual(plain(original.state_dict()), plain(restored.state_dict()))

    def test_rprop_manual_step_zero_and_nadam_scalar_state_validation(self):
        parameter = fr.nn.Parameter(fr.Tensor([.7], [1]))
        reference = torch.nn.Parameter(torch.tensor([.7]))
        optimizer, oracle = fr.optim.Rprop([parameter], lr=.04), torch.optim.Rprop([reference], lr=.04, foreach=False)
        state = optimizer.state_dict()
        state['state'][0] = {'step': fr.Tensor([0.], []), 'prev': fr.Tensor([0.], [1]), 'step_size': fr.Tensor([.013], [1])}
        optimizer.load_state_dict(state)
        expected = oracle.state_dict()
        expected['state'][0] = {'step': torch.tensor(0.), 'prev': torch.tensor([0.]), 'step_size': torch.tensor([.013])}
        oracle.load_state_dict(expected)
        parameter.tensor().sum().backward()
        reference.sum().backward()
        optimizer.step(); oracle.step()
        self.compare_state(optimizer, oracle)
        torch.testing.assert_close(torch.tensor(parameter.tensor().tolist()), reference.detach())
        model, nadam = setup('NAdam')
        update(model, nadam, fr.Generator(7), 1)
        for value in (float('nan'), -1., 1.01):
            before = plain(nadam.state_dict())
            bad = nadam.state_dict()
            bad['state'][2]['mu_product'] = fr.Tensor([value], [])
            with self.assertRaises(ValueError):
                nadam.load_state_dict(bad)
            self.assertEqual(plain(nadam.state_dict()), before)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
