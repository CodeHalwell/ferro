"""Extended SGD/Adam/AdamW trajectories and legacy checkpoint schema migration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from test_optimizer_groups_schedulers import Model, checkpoint_digest, plain
import test_adaptive_optimizers as adaptive_tests
from test_adaptive_optimizers import update, continuation


def groups(parameters, kind, plain_options=False):
    a, b, c = parameters
    groups = [{'params': [a, c], 'lr': .023}, {'params': [b], 'lr': .011}]
    if kind == 'SGD':
        defaults = dict(lr=.017, momentum=.7)
        groups[1].update(momentum=.4, nesterov=True)
        if not plain_options:
            groups[0].update(dampening=.3, weight_decay=.07, maximize=True)
    else:
        defaults = dict(lr=.017, betas=(.8, .93), eps=1e-7)
        groups[1]['betas'] = (.6, .85)
        if not plain_options:
            groups[0].update(amsgrad=True, weight_decay=.07, maximize=True)
    return groups, defaults


def setup(kind, plain_options=False):
    model = Model()
    specs, defaults = groups(list(model.parameters()), kind, plain_options)
    return model, getattr(fr.optim, kind)(specs, **defaults)


def worker(kind, path):
    path = Path(path)
    model, optimizer = setup(kind, plain_options=True)
    scheduler = fr.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.8)
    generator = fr.Generator(999)
    checkpoint = fr.checkpoint.load(path / 'training')
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=generator)
    scheduler.load_state_dict(json.loads((path / 'scheduler.json').read_text()))
    (path / 'result.json').write_text(json.dumps(continuation(model, optimizer, scheduler, generator, checkpoint.step), allow_nan=False))


class OptimizerOptionTests(unittest.TestCase):
    compare_state = adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_grouped_weight_decay_dampening_amsgrad_maximize_and_skips_match_torch(self):
        for kind in ('SGD', 'Adam', 'AdamW'):
            model, optimizer = setup(kind)
            parameters = list(model.parameters())
            reference = [torch.nn.Parameter(torch.tensor(p.tensor().tolist())) for p in parameters]
            specs, defaults = groups(reference, kind)
            oracle = getattr(torch.optim, kind)(specs, **defaults, foreach=False)
            self.compare_state(optimizer, oracle)
            for step in range(35):
                optimizer.zero_grad(); oracle.zero_grad(set_to_none=True)
                if step == 11:
                    optimizer.param_groups[0]['lr'] = oracle.param_groups[0]['lr'] = .019
                for pi, (p, tp) in enumerate(zip(parameters, reference)):
                    p.trainable = not (pi == 1 and step % 3 == 1)
                    if (step + pi) % 4 == 0:
                        continue
                    scale, shift = .21 + step * .02 + pi * .03, -.13 + pi * .05
                    (p.tensor().square().sum() * scale + p.tensor().sum() * shift).backward()
                    (tp.square().sum() * scale + tp.sum() * shift).backward()
                    if not p.trainable:
                        tp.grad = None
                optimizer.step(); oracle.step()
                for p, tp in zip(parameters, reference):
                    torch.testing.assert_close(torch.tensor(p.tensor().tolist()), tp.detach(), atol=4e-6, rtol=6e-5)
                self.compare_state(optimizer, oracle)

    def test_each_extended_group_schema_resumes_in_a_fresh_process(self):
        for kind in ('SGD', 'Adam', 'AdamW'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                model, optimizer = setup(kind)
                scheduler = fr.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.8)
                generator = fr.Generator(23)
                for index in range(7):
                    update(model, optimizer, generator, index); scheduler.step()
                fr.checkpoint.save_training(path / 'training', model, optimizers={'main': optimizer}, generator=generator, step=7)
                (path / 'scheduler.json').write_text(json.dumps(scheduler.state_dict(), allow_nan=False))
                expected = continuation(model, optimizer, scheduler, generator, 7)
                env = dict(os.environ); env.pop('PYTHONPATH', None)
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', kind, str(path)],
                                       env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(child.returncode, 0, child.stdout)
                self.assertEqual(json.loads((path / 'result.json').read_text()), expected)

    def test_prior_python_metadata_and_state_dicts_restore_neutral_options(self):
        for kind in ('SGD', 'Adam', 'AdamW'):
            source, optimizer = setup(kind, plain_options=True)
            generator = fr.Generator(7)
            for index in range(3):
                update(source, optimizer, generator, index)
            # Recreate prior public metadata exactly: old config widths and
            # moment keys, with no fields for the new neutral options.
            params, buffers, scalars, opts, native_rng, _ = fr.checkpoint._state(source, {'main': optimizer}, generator)
            additions = optimizer._legacy_defaults()
            removed = {f'optim_meta.main.g{gi}.{key}' for gi in range(2) for key in additions}
            legacy_scalars = [(name, value) for name, value in scalars if name not in removed]
            raw = fr._native._TrainingCheckpoint.snapshot(params, buffers, legacy_scalars, opts, native_rng, 3)
            legacy = fr.checkpoint.TrainingCheckpoint(raw)
            target, restored = setup(kind)
            other_rng = fr.Generator(999)
            legacy.restore(target, optimizers={'main': restored}, generator=other_rng)
            self.assertEqual(plain(restored.state_dict()), plain(optimizer.state_dict()))
            for index in range(3, 8):
                self.assertEqual(update(source, optimizer, generator, index), update(target, restored, other_rng, index))
            self.assertEqual(plain(restored.state_dict()), plain(optimizer.state_dict()))
            self.assertEqual(fr.Tensor.randn([7], generator=generator).tolist(), fr.Tensor.randn([7], generator=other_rng).tolist())
            old_state = optimizer.state_dict()
            for group in old_state['param_groups']:
                for key in additions:
                    group.pop(key)
            restored.load_state_dict(old_state)
            self.assertEqual(plain(restored.state_dict()), plain(optimizer.state_dict()))

    def test_late_namespace_state_and_metadata_faults_reject_atomically(self):
        from test_training_checkpoint import rewrite_checkpoint
        for kind in ('SGD', 'Adam', 'AdamW'):
            source, optimizer = setup(kind)
            generator = fr.Generator(7)
            update(source, optimizer, generator, 1)
            checkpoint = fr.checkpoint.snapshot_training(source, optimizers={'main': optimizer}, generator=generator)
            target, restored = setup(kind, plain_options=True)
            other_rng = fr.Generator(999)
            fields = {'optim.main.unexpected': fr.Tensor([1.], []),
                      'optim.other.t.0': fr.Tensor([1.], []),
                      'scalars.optim_meta.main.g0.maximize': fr.Tensor([2., 0., 0., 0.], [4])}
            name = 'initialized.1' if kind == 'SGD' else 'vmax.1'
            fields['optim.main.g0.' + name] = fr.Tensor.ones([99])
            for key, value in fields.items():
                before = checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng))
                with self.subTest(kind=kind, key=key), tempfile.TemporaryDirectory() as directory:
                    malformed = rewrite_checkpoint(checkpoint, directory, {key: value})
                    with self.assertRaises(ValueError):
                        malformed.restore(target, optimizers={'main': restored}, generator=other_rng)
                self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng)), before)
            # Omitting only part of the new metadata is corruption, not a
            # complete legacy schema eligible for migration.
            params, buffers, scalars, opts, native_rng, _ = fr.checkpoint._state(source, {'main': optimizer}, generator)
            partial = [(name, value) for name, value in scalars if name != 'optim_meta.main.g0.maximize']
            malformed = fr.checkpoint.TrainingCheckpoint(fr._native._TrainingCheckpoint.snapshot(params, buffers, partial, opts, native_rng, 0))
            before = checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng))
            with self.assertRaises(ValueError):
                malformed.restore(target, optimizers={'main': restored}, generator=other_rng)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng)), before)

    def test_option_validation_and_tiny_decay_configuration_roundtrip(self):
        parameter = fr.nn.Parameter(fr.Tensor([.7], [1]))
        for options in (dict(nesterov=True, momentum=.7, dampening=.3), dict(dampening=-.1), dict(momentum=1e-50)):
            with self.assertRaises(ValueError):
                fr.optim.SGD([parameter], lr=.1, **options)
        with self.assertRaises(ValueError):
            fr.optim.Adam([parameter], amsgrad=1)
        for kind in ('SGD', 'Adam', 'AdamW'):
            model = Model()
            optimizer = getattr(fr.optim, kind)(model.parameters(), lr=.03, weight_decay=1e-50)
            generator = fr.Generator(7)
            saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
            saved.restore(model, optimizers={'main': optimizer}, generator=generator)
            self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-50)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
