"""Callable schedules use real Torch references and saved callback attributes."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import torch
import ferro as fr
from test_lr_scheduler_extended import update
from test_optimizer_groups_schedulers import Model, checkpoint_digest, plain


class Factor:
    def __init__(self, gamma):
        self.gamma, self.epochs = gamma, []

    def __call__(self, epoch):
        self.epochs.append(epoch)
        return self.gamma ** (1 + epoch % 3)


def setup(name):
    model = Model()
    a, b, c = list(model.parameters())
    optimizer = fr.optim.AdamW([{'params': [a, c], 'lr': .06}, {'params': [b], 'lr': .015}], lr=.1)
    callbacks = [Factor(.9), Factor(.8)]
    scheduler = getattr(fr.optim.lr_scheduler, name)(optimizer, callbacks)
    return model, optimizer, scheduler


def suffix(model, optimizer, scheduler, generator, start):
    trace = []
    for index in range(start, 11):
        loss = update(model, optimizer, generator)
        scheduler.step(index + 2 if index % 3 == 1 else None)
        trace.append([loss, scheduler.get_last_lr()])
    return dict(trace=trace, optimizer=plain(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                checkpoint=checkpoint_digest(fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=11)),
                draws=fr.Tensor.randn([7], generator=generator).tolist())


def worker(name, directory):
    path = Path(directory)
    model, optimizer, scheduler = setup(name)
    generator = fr.Generator(999)
    checkpoint = fr.checkpoint.load(path / 'training')
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=generator)
    scheduler.load_state_dict(json.loads((path / 'scheduler.json').read_text()))
    (path / 'result.json').write_text(json.dumps(suffix(model, optimizer, scheduler, generator, checkpoint.step), allow_nan=False))


class CallbackSchedulerTests(unittest.TestCase):
    def test_resumed_constructor_callbacks_and_failed_initial_metadata(self):
        for name in ('LambdaLR', 'MultiplicativeLR'):
            p = fr.nn.Parameter(fr.Tensor([1.], [1]))
            optimizer = fr.optim.SGD([{'params': [p], 'lr': .03, 'initial_lr': .08}], lr=.1)
            tp = torch.nn.Parameter(torch.tensor([1.]))
            target_optimizer = torch.optim.SGD([{'params': [tp], 'lr': .03, 'initial_lr': .08}], lr=.1)
            callback, target_callback = Factor(.8), Factor(.8)
            scheduler = getattr(fr.optim.lr_scheduler, name)(optimizer, callback, last_epoch=4)
            target = getattr(torch.optim.lr_scheduler, name)(target_optimizer, target_callback, last_epoch=4)
            for _ in range(3):
                self.assertEqual(callback.epochs, target_callback.epochs)
                self.assertEqual(scheduler.state_dict(), target.state_dict())
                optimizer.step(); target_optimizer.step(); scheduler.step(); target.step()
        optimizer = fr.optim.SGD([p], lr=.1)
        before = plain(optimizer.state_dict())
        with self.assertRaises(ValueError):
            fr.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: float('nan'))
        self.assertEqual(plain(optimizer.state_dict()), before)

    def test_torch_functions_objects_explicit_epochs_and_external_rates(self):
        for name in ('LambdaLR', 'MultiplicativeLR'):
            for objects in (False, True):
                with self.subTest(name=name, objects=objects), warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = Model(); a, b, c = list(model.parameters())
                    optimizer = fr.optim.SGD([{'params': [a, c], 'lr': .06}, {'params': [b], 'lr': .015}], lr=.1)
                    p, q = torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))
                    target_optimizer = torch.optim.SGD([{'params': [p], 'lr': .06}, {'params': [q], 'lr': .015}], lr=.1)
                    callbacks = [Factor(.9), Factor(.8)] if objects else [lambda epoch: 1./(epoch+1), lambda epoch: .8**epoch]
                    target_callbacks = [Factor(.9), Factor(.8)] if objects else callbacks
                    scheduler = getattr(fr.optim.lr_scheduler, name)(optimizer, callbacks)
                    target = getattr(torch.optim.lr_scheduler, name)(target_optimizer, target_callbacks)
                    for index, epoch in enumerate([None, None, 5, None, 0, 8, None, None]):
                        self.assertEqual(scheduler.last_epoch, target.last_epoch)
                        self.assertEqual(scheduler.state_dict()['lr_lambdas'], target.state_dict()['lr_lambdas'])
                        for actual, expected in zip(scheduler.get_last_lr(), target.get_last_lr()):
                            self.assertAlmostEqual(actual, expected, places=14)
                        if index == 2:
                            optimizer.param_groups[0]['lr'] = target_optimizer.param_groups[0]['lr'] = .023
                        optimizer.step(); target_optimizer.step(); scheduler.step(epoch); target.step(epoch)
                    self.assertEqual(scheduler.state_dict(), target.state_dict())

    def test_state_owns_attributes_and_rejects_late_corruption_atomically(self):
        for name in ('LambdaLR', 'MultiplicativeLR'):
            _, optimizer, scheduler = setup(name)
            scheduler.step(); saved = scheduler.state_dict(); snapshot = copy.deepcopy(saved)
            scheduler.step(); self.assertEqual(saved, snapshot)
            before, rates = scheduler.state_dict(), [g['lr'] for g in optimizer.param_groups]
            faults = [('_last_lr', [.1]), ('last_epoch', -.5), ('base_lrs', [.1, float('nan')]),
                      ('lr_lambdas', [{'gamma': .1}, None]), ('lr_lambdas', [{'gamma': .1}, {3: 'invalid'}]),
                      ('lr_lambdas', [])]
            for key, value in faults:
                bad = copy.deepcopy(saved); bad[key] = value
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    scheduler.load_state_dict(bad)
                self.assertEqual(scheduler.state_dict(), before)
                self.assertEqual([g['lr'] for g in optimizer.param_groups], rates)
            scheduler.load_state_dict(saved)
            saved['lr_lambdas'][0]['epochs'].append(999)
            self.assertEqual(scheduler.state_dict(), snapshot)
            self.assertEqual([g['lr'] for g in optimizer.param_groups], rates)
            function_scheduler = getattr(fr.optim.lr_scheduler, name)(optimizer, lambda epoch: .8)
            self.assertEqual(function_scheduler.state_dict()['lr_lambdas'], [None, None])
            bad = function_scheduler.state_dict(); bad['lr_lambdas'][1] = {}
            with self.assertRaises(ValueError): function_scheduler.load_state_dict(bad)

    def test_callback_failures_leave_scheduler_and_optimizer_state_unmodified(self):
        for name in ('LambdaLR', 'MultiplicativeLR'):
            _, optimizer, _ = setup(name)
            def broken(epoch):
                if epoch > 0: raise LookupError('callback failure')
                return 1.
            scheduler = getattr(fr.optim.lr_scheduler, name)(optimizer, [lambda epoch: .9, broken])
            before, native = scheduler.state_dict(), plain(optimizer.state_dict())
            with self.assertRaisesRegex(LookupError, 'callback failure'): scheduler.step()
            self.assertEqual(scheduler.state_dict(), before)
            self.assertEqual(plain(optimizer.state_dict()), native)
            scheduler.lr_lambdas[1] = lambda epoch: float('nan')
            with self.assertRaises(ValueError): scheduler.step()
            self.assertEqual(scheduler.state_dict(), before)
            self.assertEqual(plain(optimizer.state_dict()), native)
            for callbacks in ([lambda epoch: 1.], [lambda epoch: 1., 3], 2):
                with self.assertRaises((ValueError, TypeError)):
                    getattr(fr.optim.lr_scheduler, name)(optimizer, callbacks)
                self.assertEqual(plain(optimizer.state_dict()), native)

    def test_each_callback_schedule_exact_fresh_process_resume(self):
        for name in ('LambdaLR', 'MultiplicativeLR'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                model, optimizer, scheduler = setup(name)
                generator = fr.Generator(43)
                for index in range(4):
                    update(model, optimizer, generator); scheduler.step(index+2 if index % 3 == 1 else None)
                fr.checkpoint.save_training(path / 'training', model, optimizers={'main': optimizer}, generator=generator, step=4)
                (path / 'scheduler.json').write_text(json.dumps(scheduler.state_dict(), allow_nan=False))
                expected = suffix(model, optimizer, scheduler, generator, 4)
                env = dict(os.environ); env.pop('PYTHONPATH', None)
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name, str(path)], env=env,
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(child.returncode, 0, child.stdout)
                self.assertEqual(json.loads((path / 'result.json').read_text()), expected)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
