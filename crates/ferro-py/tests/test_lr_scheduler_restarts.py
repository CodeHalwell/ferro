"""Actual Torch cosine restart trajectories, atomic loads and process resumes."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from test_lr_scheduler_extended import update
from test_optimizer_groups_schedulers import Model, checkpoint_digest, plain


def setup(options=None):
    model = Model()
    a, b, c = list(model.parameters())
    optimizer = fr.optim.AdamW([{'params': [a, c], 'lr': .06}, {'params': [b], 'lr': .015}], lr=.1)
    scheduler = fr.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **(options or dict(T_0=3, T_mult=2, eta_min=.002)))
    return model, optimizer, scheduler


EPOCHS = [0.25, 1.75, None, 3., 4.25, None, 8.9, None, 9., 12.5, None, 21.25]


def suffix(model, optimizer, scheduler, generator, start):
    trace = []
    for index in range(start, len(EPOCHS)):
        loss = update(model, optimizer, generator)
        scheduler.step(EPOCHS[index])
        trace.append([loss, scheduler.get_last_lr(), scheduler.T_cur, scheduler.T_i])
    return dict(trace=trace, optimizer=plain(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                checkpoint=checkpoint_digest(fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=len(EPOCHS))),
                draws=fr.Tensor.randn([7], generator=generator).tolist())


def worker(directory):
    path = Path(directory)
    model, optimizer, scheduler = setup()
    generator = fr.Generator(999)
    checkpoint = fr.checkpoint.load(path / 'training')
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=generator)
    scheduler.load_state_dict(json.loads((path / 'scheduler.json').read_text()))
    (path / 'result.json').write_text(json.dumps(suffix(model, optimizer, scheduler, generator, checkpoint.step), allow_nan=False))


class WarmRestartTests(unittest.TestCase):
    def compare(self, actual, target):
        self.assertEqual(actual.last_epoch, target.last_epoch)
        self.assertEqual(actual.T_i, target.T_i)
        self.assertEqual(actual.T_cur, target.T_cur)
        for got, expected in zip(actual.get_last_lr(), target.get_last_lr()):
            self.assertAlmostEqual(got, expected, places=14)

    def test_torch_automatic_fractional_and_interleaved_cycles(self):
        for multiplier in (1, 2, 3):
            for epochs in ([None] * 36, EPOCHS + [27., 0., .125, 37.5, None]):
                with self.subTest(multiplier=multiplier, epochs=epochs):
                    _, optimizer, scheduler = setup(dict(T_0=3, T_mult=multiplier, eta_min=.002))
                    p, q = torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))
                    target_optimizer = torch.optim.AdamW([{'params': [p], 'lr': .06}, {'params': [q], 'lr': .015}], lr=.1)
                    target = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(target_optimizer, T_0=3, T_mult=multiplier, eta_min=.002)
                    self.compare(scheduler, target)
                    for index, epoch in enumerate(epochs):
                        if index == 4:
                            optimizer.param_groups[0]['lr'] = target_optimizer.param_groups[0]['lr'] = .01234
                        optimizer.step(); target_optimizer.step()
                        scheduler.step(epoch); target.step(epoch)
                        self.compare(scheduler, target)

    def test_resumed_constructor_and_single_epoch_cycles_match_torch(self):
        for first, multiplier, last_epoch in ((1, 1, -1), (1, 3, 4), (3, 2, 2), (3, 2, 9)):
            with self.subTest(first=first, multiplier=multiplier, last_epoch=last_epoch):
                p = fr.nn.Parameter(fr.Tensor([1.], [1]))
                optimizer = fr.optim.SGD([{'params': [p], 'lr': .04, 'initial_lr': .07}], lr=.1)
                tp = torch.nn.Parameter(torch.tensor([1.]))
                target_optimizer = torch.optim.SGD([{'params': [tp], 'lr': .04, 'initial_lr': .07}], lr=.1)
                options = dict(T_0=first, T_mult=multiplier, eta_min=.001, last_epoch=last_epoch)
                scheduler = fr.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **options)
                target = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(target_optimizer, **options)
                self.compare(scheduler, target)
                for _ in range(8):
                    optimizer.step(); target_optimizer.step(); scheduler.step(); target.step()
                    self.compare(scheduler, target)

    def test_invalid_options_steps_and_state_are_atomic(self):
        model, optimizer, scheduler = setup()
        for epoch in (1.5, 3., 4.25):
            scheduler.step(epoch)
        before, rates = scheduler.state_dict(), [g['lr'] for g in optimizer.param_groups]
        native_before = plain(optimizer.state_dict())
        faults = [('T_0', 0), ('T_mult', 1.5), ('eta_min', float('nan')), ('T_i', 5),
                  ('T_cur', -1), ('T_cur', 6.), ('T_cur', float('inf')), ('last_epoch', 1.5),
                  ('_is_initial', True), ('_last_lr', [.1]), ('base_lrs', [.1, float('nan')])]
        for key, value in faults:
            bad = copy.deepcopy(before); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                scheduler.load_state_dict(bad)
            self.assertEqual(scheduler.state_dict(), before)
            self.assertEqual(plain(optimizer.state_dict()), native_before)
        for epoch in (-.5, float('nan'), float('inf'), True, '2'):
            with self.assertRaises((ValueError, TypeError)):
                scheduler.step(epoch)
            self.assertEqual(scheduler.state_dict(), before)
            self.assertEqual(plain(optimizer.state_dict()), native_before)
        for options in (dict(T_0=0), dict(T_0=2, T_mult=0), dict(T_0=2, T_mult=1.5), dict(T_0=2, eta_min=float('inf'))):
            with self.assertRaises(ValueError):
                fr.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **options)
            self.assertEqual([g['lr'] for g in optimizer.param_groups], rates)
        scheduler.step()
        saved = copy.deepcopy(before)
        scheduler.load_state_dict(saved)
        saved['base_lrs'][0] = 999.
        self.assertEqual(scheduler.state_dict(), before)
        self.assertNotEqual([g['lr'] for g in optimizer.param_groups], rates)

    def test_fractional_state_exact_fresh_process_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, optimizer, scheduler = setup()
            generator = fr.Generator(43)
            for index in range(5):
                update(model, optimizer, generator); scheduler.step(EPOCHS[index])
            self.assertEqual((scheduler.T_i, scheduler.T_cur), (6, 1.25))
            fr.checkpoint.save_training(path / 'training', model, optimizers={'main': optimizer}, generator=generator, step=5)
            (path / 'scheduler.json').write_text(json.dumps(scheduler.state_dict(), allow_nan=False))
            expected = suffix(model, optimizer, scheduler, generator, 5)
            env = dict(os.environ); env.pop('PYTHONPATH', None)
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)], env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout)
            self.assertEqual(json.loads((path / 'result.json').read_text()), expected)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        unittest.main()
