"""Actual Torch shared-storage slot order, independent moments and saved restart."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
import test_adaptive_optimizers as adaptive_tests
from test_optimizer_groups_schedulers import checkpoint_digest, plain


KINDS = ('SGD', 'Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta', 'ASGD', 'Rprop', 'RAdam', 'NAdam', 'Adafactor')


class Shared(fr.nn.Module):
    def build(self):
        self.register_buffer('table', fr.tensor([[.7, -1.2], [.4, .2]]))
        self.left = fr.nn.Embedding.from_pretrained(self.table, freeze=False)
        self.right = fr.nn.Embedding.from_pretrained(self.table, freeze=False)


def setup(kind):
    model = Shared()
    return model, getattr(fr.optim, kind)(model.parameters(), lr=.01, weight_decay=.03)


def suffix(model, optimizer, generator, start):
    losses = []
    for index in range(start, 10):
        optimizer.zero_grad()
        loss = None
        for pi, parameter in enumerate(model.parameters()):
            if (index+pi) % 4 == 1:
                continue
            term = (parameter.tensor()+fr.randn(2, 2, generator=generator)*.03).square().sum()*(.3+.1*pi)
            loss = term if loss is None else loss+term
        loss.backward(); optimizer.step(); optimizer.zero_grad(); losses.append(loss.item())
    return dict(losses=losses, state=checkpoint_digest(fr.checkpoint.snapshot_training(
        model, optimizers={'main': optimizer}, generator=generator)), optimizer=plain(optimizer.state_dict()),
        draws=fr.randn(7, generator=generator).tolist())


def worker(directory, kind):
    path = Path(directory); model, optimizer = setup(kind); generator = fr.Generator(100)
    saved = fr.checkpoint.load(path/'checkpoint')
    saved.restore(model, optimizers={'main': optimizer}, generator=generator)
    (path/'actual.json').write_text(json.dumps(suffix(model, optimizer, generator, saved.step), allow_nan=False))


class SharedOptimizerTests(unittest.TestCase):
    compare_state = adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_manual_gradient_parameter_aliases_reject_all_groups_before_config_or_update(self):
        for kind in KINDS+('LBFGS',):
            for direct in (False, True):
                with self.subTest(kind=kind, direct=direct):
                    parameters = [fr.nn.Parameter(fr.ones(2, 2)), fr.nn.Parameter(fr.ones(2, 2)*2.)]
                    groups = parameters if kind == 'LBFGS' else [{'params': [p]} for p in parameters]
                    optimizer = getattr(fr.optim, kind)(groups, lr=.01)
                    parameters[0].tensor().sum().backward()
                    alias = parameters[0].tensor().detach().transpose(0, 1)
                    parameters[1].tensor()._set_leaf_grad(alias)
                    before = (optimizer._native.configurations(), plain(dict(optimizer._native.state_tensors())),
                              [p.tensor().tolist() for p in parameters], [p.tensor()._id() for p in parameters])
                    optimizer.param_groups[0]['lr'] = .023
                    calls = []
                    def closure():
                        calls.append(1)
                        return fr.tensor(1.)
                    with self.assertRaisesRegex(ValueError, 'share parameter storage'):
                        if direct and kind == 'LBFGS':
                            fr._native._Optimizer.lbfgs_step(optimizer._native, closure, preserve_grads=True)
                        elif direct:
                            optimizer._native.step(preserve_grads=True)
                        else:
                            optimizer.step(closure)
                    self.assertEqual(calls, [])
                    self.assertEqual((optimizer._native.configurations(), plain(dict(optimizer._native.state_tensors())),
                                      [p.tensor().tolist() for p in parameters], [p.tensor()._id() for p in parameters]), before)
                    self.assertEqual(parameters[1].tensor().grad._id(), alias._id())

    def test_closure_installed_alias_rejects_before_native_update_and_hidden_alias_is_unused(self):
        for kind in KINDS+('LBFGS',):
            with self.subTest(kind=kind):
                parameters = [fr.nn.Parameter(fr.ones(2, 2)), fr.nn.Parameter(fr.ones(2, 2)*2.)]
                groups = parameters if kind == 'LBFGS' else [{'params': [p]} for p in parameters]
                optimizer = getattr(fr.optim, kind)(groups, lr=.01)
                before = (optimizer._native.configurations(), plain(dict(optimizer._native.state_tensors())),
                          [p.tensor().tolist() for p in parameters])
                def closure():
                    optimizer.zero_grad()
                    loss = sum(p.tensor().square().sum() for p in parameters)
                    loss.backward()
                    parameters[1].tensor()._set_leaf_grad(parameters[0].tensor().detach())
                    optimizer.param_groups[0]['lr'] = .023
                    return loss
                with self.assertRaisesRegex(ValueError, 'share parameter storage'):
                    optimizer.step(closure)
                self.assertEqual((optimizer._native.configurations(), plain(dict(optimizer._native.state_tensors())),
                                  [p.tensor().tolist() for p in parameters]), before)
                self.assertEqual(optimizer.param_groups[0]['lr'], .023)
                if kind != 'LBFGS':
                    parameters[1].trainable = False
                    parameters[1].tensor()._set_leaf_grad(parameters[0].tensor().detach())
                    optimizer._native.validate_gradient_storage()
                    optimizer.step()

    def test_actual_torch_every_family_shared_leaves_groups_skips_and_moments(self):
        for kind in KINDS:
            for grouped in (False, True):
                with self.subTest(kind=kind, grouped=grouped):
                    model = Shared(); parameters = list(model.parameters())
                    storage = torch.tensor(model.table.tolist())
                    expected = [torch.nn.Parameter(storage), torch.nn.Parameter(storage)]
                    groups = [{'params': [p], 'lr': .01+.003*i} for i, p in enumerate(parameters)] if grouped else parameters
                    targets = [{'params': [p], 'lr': .01+.003*i} for i, p in enumerate(expected)] if grouped else expected
                    options = dict(lr=.01)
                    if kind != 'Rprop': options['weight_decay'] = .03
                    if kind == 'RMSprop': options.update(alpha=.8, momentum=.4, centered=True)
                    if kind == 'SGD': options.update(momentum=.4, dampening=.2)
                    if kind == 'ASGD': options.update(t0=3.)
                    optimizer = getattr(fr.optim, kind)(groups, **options)
                    oracle = getattr(torch.optim, kind)(targets, **options)
                    for index in range(15):
                        optimizer.zero_grad(); oracle.zero_grad()
                        for pi, (parameter, target) in enumerate(zip(parameters, expected)):
                            parameter.trainable = not (pi == 1 and index % 7 == 3)
                            if (index+pi) % 5 == 1 or not parameter.trainable:
                                continue
                            (parameter.tensor().square().sum()*(.3+.1*pi)+parameter.tensor().sum()*(.02*index-.1*pi)).backward()
                            (target.square().sum()*(.3+.1*pi)+target.sum()*(.02*index-.1*pi)).backward()
                        optimizer.step(); oracle.step()
                        self.compare_state(optimizer, oracle)
                        torch.testing.assert_close(torch.tensor(model.table.tolist()), storage, atol=5e-6, rtol=8e-5)
                        self.assertTrue(parameters[0].tensor()._shares_storage(parameters[1].tensor()))

    def test_actual_torch_shared_lbfgs_closure_and_line_search_order(self):
        for line_search in (None, 'strong_wolfe'):
            with self.subTest(line_search=line_search):
                model = Shared(); parameters = list(model.parameters())
                storage = torch.tensor(model.table.tolist()); expected = [torch.nn.Parameter(storage), torch.nn.Parameter(storage)]
                options = dict(lr=.2, max_iter=3, max_eval=5, history_size=4, tolerance_grad=0., tolerance_change=0., line_search_fn=line_search)
                optimizer = fr.optim.LBFGS(parameters, **options); oracle = torch.optim.LBFGS(expected, **options)
                actual_calls, expected_calls = [], []
                def actual():
                    optimizer.zero_grad()
                    loss = (parameters[0].tensor()-.3).square().sum()+.7*(parameters[1].tensor()+.2).square().sum()
                    loss.backward(); actual_calls.append(loss.item()); return loss
                def reference():
                    oracle.zero_grad()
                    loss = (expected[0]-.3).square().sum()+.7*(expected[1]+.2).square().sum()
                    loss.backward(); expected_calls.append(loss.item()); return loss
                optimizer.step(actual); oracle.step(reference)
                torch.testing.assert_close(torch.tensor(actual_calls), torch.tensor(expected_calls), atol=6e-6, rtol=8e-5)
                torch.testing.assert_close(torch.tensor(model.table.tolist()), storage, atol=5e-6, rtol=8e-5)
                for parameter, target in zip(parameters, expected):
                    torch.testing.assert_close(torch.tensor(parameter.grad.tolist()), target.grad, atol=8e-6, rtol=8e-5)

    def test_fresh_process_shared_checkpoint_exact_continuation_and_all_moments(self):
        for kind in ('RMSprop', 'Adafactor'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory); model, optimizer = setup(kind); generator = fr.Generator(19)
                for index in range(4):
                    optimizer.zero_grad()
                    loss = sum((p.tensor()+fr.randn(2, 2, generator=generator)*.03).square().sum() for p in model.parameters())
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                fr.checkpoint.save_training(path/'checkpoint', model, optimizers={'main': optimizer}, generator=generator, step=4)
                expected = json.loads(json.dumps(suffix(model, optimizer, generator, 4)))
                child = subprocess.run([sys.executable, __file__, '--worker', str(path), kind], capture_output=True, text=True, timeout=60)
                self.assertEqual(child.returncode, 0, child.stdout+child.stderr)
                self.assertEqual(json.loads((path/'actual.json').read_text()), expected)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        from test_optimizer_shared_parameters import worker
        worker(sys.argv[2], sys.argv[3])
    else:
        unittest.main(verbosity=2)
