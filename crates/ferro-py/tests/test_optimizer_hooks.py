"""Actual Torch optimizer lifecycle callbacks and Ferro state boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr


FAMILIES = ('SGD', 'Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta',
            'ASGD', 'Rprop', 'RAdam', 'NAdam', 'Adafactor', 'Muon', 'LBFGS')


def setup(api, family='SGD'):
    parameter = api.nn.Parameter(api.tensor([[.4, -.6], [.7, -.2]]))
    options = {'lr': .017}
    if family == 'SGD': options['momentum'] = .6
    if family == 'LBFGS': options.update(max_iter=1, history_size=4)
    return parameter, getattr(api.optim, family)([parameter], **options)


def plain(value):
    if isinstance(value, (fr.Tensor, torch.Tensor)): return value.tolist()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [plain(item) for item in value]
    return value


def native(optimizer):
    return optimizer._native.configurations(), [(key, value.shape, value.tolist()) for key, value in optimizer._native.state_tensors()]


class Model(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.tensor([[.4, -.6], [.7, -.2]]))


def configure_hooks(optimizer, events):
    def pre(opt, args, kwargs):
        events.append('pre')
        opt.param_groups[0]['lr'] *= .9
    def post(opt, args, kwargs): events.append('post')
    optimizer.register_step_pre_hook(pre)
    optimizer.register_step_post_hook(post)


def checkpoint_suffix(model, optimizer, generator, events):
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = (model.weight + fr.randn(2, 2, generator=generator) * .03).square().sum()
        loss.backward(); losses.append(loss.item()); optimizer.step(); optimizer.zero_grad()
    state = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
    tensors = {key: (value.dtype, value.shape, value.tolist()) for key, value in state.tensors().items()}
    return {'losses': losses, 'events': events, 'state': tensors, 'draws': fr.randn(5, generator=generator).tolist()}


def worker(directory):
    path = Path(directory); model = Model(); optimizer = fr.optim.AdamW(model.parameters(), lr=.02)
    generator = fr.Generator(999); events = []; configure_hooks(optimizer, events)
    fr.checkpoint.load_training(path / 'checkpoint', model, optimizers={'main': optimizer}, generator=generator)
    (path / 'actual.json').write_text(json.dumps(checkpoint_suffix(model, optimizer, generator, events), sort_keys=True))


class OptimizerHookTests(unittest.TestCase):
    def test_all_families_step_argument_replacement_return_identity_and_order_match_torch(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                outputs = []
                for api in (fr, torch):
                    parameter, optimizer = setup(api, family); events = []; returned = []
                    def closure():
                        events.append('closure'); optimizer.zero_grad()
                        loss = parameter.square().sum() + parameter.sum() * .03
                        loss.backward(); returned.append(loss); return loss
                    def first(opt, args, kwargs):
                        self.assertIs(opt, optimizer); self.assertIs(args[0], optimizer)
                        self.assertEqual(len(args), 1); self.assertEqual(kwargs, {})
                        events.append('pre1'); opt.param_groups[0]['lr'] = .013
                        return (args, {'closure': closure})
                    def second(opt, args, kwargs):
                        self.assertIs(kwargs['closure'], closure); events.append('pre2')
                    def post(opt, args, kwargs):
                        self.assertIs(args[0], optimizer); self.assertIs(kwargs['closure'], closure)
                        events.append('post1'); return 'ignored'
                    optimizer.register_step_pre_hook(first); optimizer.register_step_pre_hook(second)
                    optimizer.register_step_post_hook(post)
                    optimizer.register_step_post_hook(lambda *args: events.append('post2'))
                    for _ in range(4):
                        loss = optimizer.step(); self.assertIs(loss, returned[-1])
                    self.assertEqual(events, ['pre1', 'pre2', 'closure', 'post1', 'post2'] * 4)
                    outputs.append(parameter.tolist())
                torch.testing.assert_close(torch.tensor(outputs[0]), torch.tensor(outputs[1]), atol=3e-6, rtol=3e-5)

    def test_state_and_load_hooks_prepend_replacement_shallow_copy_and_owning_state(self):
        traces = []
        for api in (fr, torch):
            parameter, optimizer = setup(api); parameter.square().sum().backward(); optimizer.step()
            events = []
            optimizer.register_state_dict_pre_hook(lambda opt: events.append('state-pre1'))
            optimizer.register_state_dict_pre_hook(lambda opt: events.append('state-pre0'), prepend=True)
            optimizer.register_state_dict_post_hook(lambda opt, value: events.append('state-post1'))
            def replace(opt, value):
                events.append('state-post0'); return {'payload': value}
            optimizer.register_state_dict_post_hook(replace, prepend=True)
            outer = optimizer.state_dict(); self.assertEqual(set(outer), {'payload'})
            state = outer['payload']; original_lr = state['param_groups'][0]['lr']
            target_parameter, target = setup(api)
            def load0(opt, value):
                self.assertIsNot(value, state)
                self.assertIs(value['param_groups'], state['param_groups'])
                value['temporary'] = 3; value['param_groups'][0]['lr'] = .031
                events.append('load-pre0'); return {'state': value['state'], 'param_groups': value['param_groups']}
            target.register_load_state_dict_pre_hook(lambda opt, value: events.append('load-pre1'))
            target.register_load_state_dict_pre_hook(load0, prepend=True)
            target.register_load_state_dict_post_hook(lambda opt: events.append('load-post1'))
            target.register_load_state_dict_post_hook(lambda opt: events.append('load-post0'), prepend=True)
            target.load_state_dict(state)
            self.assertNotIn('temporary', state); self.assertEqual(state['param_groups'][0]['lr'], .031)
            self.assertEqual(target.param_groups[0]['lr'], .031)
            self.assertEqual(optimizer.param_groups[0]['lr'], original_lr)
            traces.append(events)
            if api is fr:
                before = native(target)
                state['state'][0]['momentum_buffer'].fill_(99)
                self.assertEqual(native(target), before)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], ['state-pre0', 'state-pre1', 'state-post0', 'state-post1',
                                     'load-pre0', 'load-pre1', 'load-post0', 'load-post1'])

    def test_handle_removal_context_and_live_iteration_match_torch(self):
        for api in (fr, torch):
            parameter, optimizer = setup(api); events = []
            handle = optimizer.register_step_post_hook(lambda *args: events.append('persistent'))
            self.assertIsInstance(handle.id, int)
            with optimizer.register_step_pre_hook(lambda *args: events.append('temporary')) as temporary:
                self.assertIsInstance(temporary.id, int); optimizer.step()
            optimizer.step(); handle.remove(); handle.remove(); optimizer.step()
            self.assertEqual(events, ['temporary', 'persistent', 'persistent'])
            first = optimizer.register_step_pre_hook(lambda *args: first.remove())
            second = optimizer.register_step_pre_hook(lambda *args: None)
            with self.assertRaisesRegex(RuntimeError, 'mutated during iteration'):
                optimizer.step()
            second.remove()

    def test_hook_error_boundaries_match_torch_without_rollback_claim(self):
        def fail(*args): raise RuntimeError('callback failed')
        for api in (fr, torch):
            parameter, optimizer = setup(api); parameter.square().sum().backward()
            before = parameter.tolist()
            bad = optimizer.register_step_pre_hook(lambda *args: ['invalid'])
            with self.assertRaises(RuntimeError): optimizer.step()
            self.assertEqual(parameter.tolist(), before); bad.remove()
            post = optimizer.register_step_post_hook(fail)
            with self.assertRaisesRegex(RuntimeError, 'callback failed'): optimizer.step()
            self.assertNotEqual(parameter.tolist(), before); post.remove()
            state = optimizer.state_dict(); target_parameter, target = setup(api)
            target.register_load_state_dict_post_hook(fail)
            with self.assertRaisesRegex(RuntimeError, 'callback failed'): target.load_state_dict(state)
            self.assertEqual(plain(target.state_dict()), plain(state))
            state_handle = optimizer.register_state_dict_post_hook(lambda opt, state: 17)
            self.assertEqual(optimizer.state_dict(), 17); state_handle.remove()
            bad = optimizer.register_load_state_dict_pre_hook(lambda opt, state: {'param_groups': []})
            before = plain(optimizer.state_dict())
            with self.assertRaises((ValueError, KeyError)): optimizer.load_state_dict(state)
            self.assertEqual(plain(optimizer.state_dict()), before); bad.remove()

    def test_overrides_dispatch_state_and_load_exactly_once(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                for api in (fr, torch):
                    parameter, optimizer = setup(api, family); events = []
                    optimizer.register_state_dict_pre_hook(lambda opt: events.append('state-pre'))
                    optimizer.register_state_dict_post_hook(lambda opt, state: events.append('state-post'))
                    optimizer.register_load_state_dict_pre_hook(lambda opt, state: events.append('load-pre'))
                    optimizer.register_load_state_dict_post_hook(lambda opt: events.append('load-post'))
                    state = optimizer.state_dict(); optimizer.load_state_dict(state)
                    self.assertEqual(events, ['state-pre', 'state-post', 'load-pre', 'load-post'])
                    self.assertEqual(set(state), {'state', 'param_groups'})

    def test_capture_and_lbfgs_busy_guard_precede_callbacks(self):
        parameter, optimizer = setup(fr); events = []
        optimizer.register_step_pre_hook(lambda *args: events.append('pre'))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: optimizer.step())
        self.assertEqual(events, [])
        parameter, optimizer = setup(fr, 'LBFGS'); events = []
        optimizer.register_step_pre_hook(lambda *args: events.append('pre'))
        optimizer.register_state_dict_pre_hook(lambda *args: events.append('state'))
        optimizer.register_load_state_dict_pre_hook(lambda *args: events.append('load'))
        state = optimizer.state_dict(); events.clear()
        def closure():
            for attempt in (lambda: optimizer.step(closure), optimizer.state_dict,
                            lambda: optimizer.load_state_dict(state)):
                with self.assertRaisesRegex(RuntimeError, 'closure'): attempt()
            optimizer.zero_grad(); loss = parameter.square().sum(); loss.backward(); return loss
        optimizer.step(closure); self.assertEqual(events, ['pre'])

    def test_native_checkpoint_bypasses_hooks_and_fresh_process_reconstruction_is_exact(self):
        model = Model(); optimizer = fr.optim.AdamW(model.parameters(), lr=.02)
        generator = fr.Generator(7); events = []; configure_hooks(optimizer, events)
        checkpoint_suffix(model, optimizer, generator, events); events.clear()
        def fail(*args): self.fail('native checkpoint must not execute Python state hooks')
        optimizer.register_state_dict_pre_hook(fail); optimizer.register_state_dict_post_hook(fail)
        optimizer.register_load_state_dict_pre_hook(fail); optimizer.register_load_state_dict_post_hook(fail)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
            checkpoint.save(path / 'checkpoint')
            self.assertFalse(any('hook' in key for key in checkpoint.tensors()))
            checkpoint.restore(model, optimizers={'main': optimizer}, generator=generator)
            self.assertEqual(events, [])
            expected = checkpoint_suffix(model, optimizer, generator, events)
            subprocess.run([sys.executable, __file__, '--worker', directory], check=True)
            self.assertEqual(json.loads((path / 'actual.json').read_text()), json.loads(json.dumps(expected)))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        from test_optimizer_hooks import worker
        worker(sys.argv[2])
    else: unittest.main(module='test_optimizer_hooks')
