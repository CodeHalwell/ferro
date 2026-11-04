"""Plain leaf optimization retains public tensor objects, slots and flags."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
import test_adaptive_optimizers as adaptive_tests


FAMILIES = ('SGD', 'Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adamax', 'Adadelta',
            'ASGD', 'Rprop', 'RAdam', 'NAdam', 'Adafactor', 'Muon')


def plain(value):
    if isinstance(value, fr.Tensor): return value.tolist()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def native(optimizer):
    return optimizer._native.configurations(), [(key, value.shape, value.tolist()) for key, value in optimizer._native.state_tensors()]


def setup():
    values = [fr.tensor([[.4, -.6], [.7, -.2]], requires_grad=True),
              fr.tensor([[.1, .3], [-.2, .5]], requires_grad=True)]
    optimizer = fr.optim.AdamW([{'params': values[0], 'lr': .02}, {'params': [values[1]], 'lr': .03}])
    scheduler = fr.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
    return values, optimizer, scheduler


def continuation(values, optimizer, scheduler, start):
    losses = []
    for iteration in range(start, 9):
        optimizer.zero_grad()
        loss = sum(value.square().sum() * (.2 + iteration * .01 + pi * .1) for pi, value in enumerate(values))
        loss.backward(); losses.append(loss.item()); optimizer.step(); scheduler.step()
    return {'values': plain(values), 'optimizer': plain(optimizer.state_dict()),
            'scheduler': scheduler.state_dict(), 'losses': losses, 'grads': [value.grad.tolist() for value in values]}


def serialize(state, tensors):
    if isinstance(state, fr.Tensor):
        key = f'tensor{len(tensors)}'; tensors[key] = state; return {'__tensor__': key}
    if isinstance(state, dict): return {'__dict__': [[key, serialize(value, tensors)] for key, value in state.items()]}
    if isinstance(state, (list, tuple)): return [serialize(value, tensors) for value in state]
    return state


def deserialize(state, tensors):
    if isinstance(state, dict):
        if '__tensor__' in state: return tensors[state['__tensor__']]
        return {key: deserialize(value, tensors) for key, value in state['__dict__']}
    if isinstance(state, list): return [deserialize(value, tensors) for value in state]
    return state


def worker(directory):
    path = Path(directory); values, optimizer, scheduler = setup()
    tensors = fr.load_safetensors(str(path / 'tensors'))
    for index, value in enumerate(values):
        with fr.no_grad(): value.copy_(tensors[f'parameter{index}'])
    metadata = json.loads((path / 'state.json').read_text())
    optimizer.load_state_dict(deserialize(metadata['optimizer'], tensors))
    scheduler.load_state_dict(metadata['scheduler'])
    (path / 'actual.json').write_text(json.dumps(continuation(values, optimizer, scheduler, 4), sort_keys=True))


class OptimizerTensorInputTests(unittest.TestCase):
    compare_state = adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_strided_and_offset_gradient_history_preserves_parameter_base_and_original_grad_handle(self):
        for family in ('SGD', 'Adam', 'AdamW'):
            for parameter_type in (False, True):
                for history in (False, True):
                    with self.subTest(family=family, parameter_type=parameter_type, history=history):
                        value = fr.tensor([[1., 2.], [3., 4.]], requires_grad=True)
                        if parameter_type: value = fr.nn.Parameter(value)
                        held = value.tensor() if parameter_type else value
                        target = torch.nn.Parameter(torch.tensor(value.tolist()))
                        options = dict(lr=.1)
                        if family == 'SGD': options['momentum'] = .5
                        optimizer = getattr(fr.optim, family)([value], **options)
                        reference = getattr(torch.optim, family)([target], **options)
                        for iteration in range(4):
                            source = fr.tensor([[.1, .2], [.3, .4], [.5, .6]], requires_grad=history)
                            oracle = torch.tensor(source.tolist(), requires_grad=history)
                            gradient = source[1:3] if iteration % 2 == 0 else source[:2].transpose(0, 1)
                            expected = oracle[1:3] if iteration % 2 == 0 else oracle[:2].T
                            value.grad = gradient; target.grad = expected; gradient_id = gradient._id()
                            optimizer.step(); reference.step()
                            self.assertEqual(value._id(), held._id())
                            if parameter_type: self.assertEqual(value.tensor()._id(), held._id())
                            self.assertEqual(value.grad._id(), gradient_id); self.assertEqual(value.grad.requires_grad, history)
                            torch.testing.assert_close(torch.tensor(held.tolist()), target.detach(), atol=2e-6, rtol=2e-5)
                            self.compare_state(optimizer, reference)
                            if history:
                                gradient.sum().backward(); expected.sum().backward()
                                self.assertEqual(source.grad.tolist(), oracle.grad.tolist())

    def test_forward_ad_rejects_all_families_before_closure_configuration_or_state_changes(self):
        for family in (*FAMILIES, 'LBFGS'):
            with self.subTest(family=family):
                parameter = fr.nn.Parameter(fr.tensor([[.4, -.6], [.7, -.2]]))
                optimizer = getattr(fr.optim, family)([parameter], lr=.02)
                parameter.square().sum().backward()
                before = native(optimizer); values = parameter.tolist(); identity = parameter._id()
                gradient = parameter.grad; optimizer.param_groups[0]['lr'] = .1
                def transformed(value):
                    optimizer.step(lambda: self.fail('forward AD must reject before closure invocation'))
                    return value * 2.
                with self.assertRaisesRegex(ValueError, 'forward AD'):
                    fr.func.jvp(transformed, (fr.tensor([.3, .4]),), (fr.ones(2),))
                self.assertEqual(native(optimizer), before); self.assertEqual(parameter.tolist(), values)
                self.assertEqual(parameter._id(), identity); self.assertEqual(parameter.grad._id(), gradient._id())

    def test_all_ordinary_families_mixed_groups_match_torch_and_retain_exact_leaf_slots(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                data = [[[.4, -.6], [.7, -.2]], [[.1, .3], [-.2, .5]]]
                params = [fr.tensor(data[0], requires_grad=True), fr.nn.Parameter(fr.tensor(data[1]))]
                expected = [torch.tensor(data[0], requires_grad=True), torch.nn.Parameter(torch.tensor(data[1]))]
                options = {'lr': .017}
                if family in ('SGD', 'RMSprop'): options['momentum'] = .6
                optimizer = getattr(fr.optim, family)([{'params': params[0]}, {'params': [params[1]], 'lr': .023}], **options)
                reference = getattr(torch.optim, family)([{'params': expected[0]}, {'params': [expected[1]], 'lr': .023}], **options)
                identities = [parameter._id() for parameter in params]
                for iteration in range(9):
                    optimizer.zero_grad(); reference.zero_grad()
                    for index, (parameter, target) in enumerate(zip(params, expected)):
                        if (iteration + index) % 4 == 1: continue
                        scale = .2 + iteration * .03 + index * .1
                        (parameter.square().sum() * scale + parameter.sum() * .02).backward()
                        (target.square().sum() * scale + target.sum() * .02).backward()
                    slots = [None if p.grad is None else p.grad._id() for p in params]
                    optimizer.step(); reference.step(); self.compare_state(optimizer, reference)
                    for index, (parameter, target) in enumerate(zip(params, expected)):
                        self.assertIs(optimizer.param_groups[index]['params'][0], parameter)
                        self.assertEqual(parameter._id(), identities[index]); self.assertTrue(parameter.requires_grad)
                        self.assertEqual(None if parameter.grad is None else parameter.grad._id(), slots[index])
                        torch.testing.assert_close(torch.tensor(parameter.tolist()), target.detach(), atol=3e-5, rtol=3e-4)

    def test_non_grad_leaves_manual_gradients_flags_and_parameter_masks_remain_independent(self):
        for requires_grad in (False, True):
            parameter = fr.tensor([1., 2.], requires_grad=requires_grad)
            target = torch.tensor([1., 2.], requires_grad=requires_grad)
            optimizer = fr.optim.SGD([parameter], lr=.1, momentum=.5)
            reference = torch.optim.SGD([target], lr=.1, momentum=.5)
            parameter.grad = fr.tensor([2., 3.]); target.grad = torch.tensor([2., 3.])
            identity = parameter._id(); gradient_id = parameter.grad._id()
            optimizer.step(); reference.step()
            self.assertEqual(parameter._id(), identity); self.assertEqual(parameter.grad._id(), gradient_id)
            self.assertEqual(parameter.requires_grad, requires_grad)
            torch.testing.assert_close(torch.tensor(parameter.tolist()), target.detach())
            optimizer.zero_grad(set_to_none=False); reference.zero_grad(set_to_none=False)
            self.assertEqual(parameter.grad._id(), gradient_id); self.assertEqual(parameter.grad.tolist(), [0., 0.])
            optimizer.zero_grad(); self.assertIsNone(parameter.grad)
        frozen = fr.nn.Parameter(fr.tensor([1., 2.])); frozen.trainable = False
        value = fr.tensor([.3, -.4], requires_grad=True)
        optimizer = fr.optim.AdamW([frozen, value], lr=.02)
        (frozen.square().sum() + value.square().sum()).backward()
        hidden = frozen.tensor().grad._id(); before = frozen.tolist()
        optimizer.step()
        self.assertEqual(frozen.tolist(), before); self.assertEqual(frozen.tensor().grad._id(), hidden)
        self.assertIsNone(frozen.grad); self.assertIsNotNone(value.grad)
        optimizer.zero_grad(set_to_none=False)
        self.assertEqual(frozen.tensor().grad._id(), hidden); self.assertEqual(frozen.tensor().grad.tolist(), [0., 0.])

    def test_lbfgs_plain_leaves_closures_history_and_retained_gradients_match_torch(self):
        for line_search in (None, 'strong_wolfe'):
            parameter = fr.tensor([-.7, 1.3], requires_grad=True)
            target = torch.tensor([-.7, 1.3], requires_grad=True)
            options = {'lr': .3, 'max_iter': 4, 'history_size': 5, 'line_search_fn': line_search}
            optimizer = fr.optim.LBFGS([parameter], **options); reference = torch.optim.LBFGS([target], **options)
            identity = parameter._id()
            def closure():
                optimizer.zero_grad(); loss = (parameter - .2).square().sum(); loss.backward(); return loss
            def oracle():
                reference.zero_grad(); loss = (target - .2).square().sum(); loss.backward(); return loss
            for _ in range(4):
                self.assertAlmostEqual(optimizer.step(closure).item(), reference.step(oracle).item(), places=5)
                torch.testing.assert_close(torch.tensor(parameter.tolist()), target.detach(), atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(torch.tensor(parameter.grad.tolist()), target.grad, atol=3e-6, rtol=3e-5)
                self.assertEqual(parameter._id(), identity)
                self.assertIs(optimizer.param_groups[0]['params'][0], parameter)
                self.assertEqual(optimizer.state_dict()['state'][0]['n_iter'], reference.state_dict()['state'][0]['n_iter'])

    def test_validation_duplicate_slots_add_group_and_manual_storage_alias_reject_atomically(self):
        value = fr.tensor([.4, -.6], requires_grad=True)
        for invalid in (value * 2., fr.tensor([1], dtype='int64'),
                        fr.ones(2, 3).transpose(0, 1), fr.ones(4)[1:3]):
            with self.assertRaises((TypeError, ValueError)): fr.optim.AdamW([invalid])
        double = fr.tensor([1.], dtype='float64', requires_grad=True)
        typed = fr.optim.AdamW([double])
        identity = double._id()
        double.grad = fr.tensor([.2], dtype='float64')
        typed.step()
        self.assertEqual(double._id(), identity)
        self.assertEqual(typed.state_dict()['state'][0]['exp_avg'].dtype, 'float64')
        with self.assertRaises(TypeError): fr.optim.SGD(value, lr=.1)
        _, tangent = fr.func.jvp(lambda argument: argument * 2., (value,), (fr.ones(2),))
        zero = fr.autograd.grad(tangent.sum(), [value])[0]
        self.assertTrue(zero._is_structural_zero())
        with self.assertRaisesRegex(ValueError, 'clone'): fr.optim.SGD([zero], lr=.1)
        with self.assertRaisesRegex(ValueError, 'clone'): fr.nn.Embedding.from_pretrained(zero.reshape([1, 2]))
        mutable = zero.clone(); fr.optim.SGD([mutable], lr=.1)
        fr.nn.Embedding.from_pretrained(mutable.reshape([1, 2]))
        optimizer = fr.optim.AdamW([value]); value.square().sum().backward(); optimizer.step(); optimizer.zero_grad()
        before = native(optimizer); old = value.tolist()
        with self.assertRaises((TypeError, ValueError)): optimizer.add_param_group({'params': [value * 2.]})
        self.assertEqual(native(optimizer), before); self.assertEqual(value.tolist(), old)
        new = fr.tensor([.2, .3], requires_grad=True); optimizer.add_param_group({'params': new})
        self.assertIs(optimizer.param_groups[1]['params'][0], new)
        with self.assertRaises(ValueError): optimizer.add_param_group({'params': [value]})
        value.grad = new.detach(); new.grad = fr.ones(2)
        before = native(optimizer); values = [value.tolist(), new.tolist()]
        with self.assertRaisesRegex(ValueError, 'storage'): optimizer.step()
        self.assertEqual(native(optimizer), before); self.assertEqual([value.tolist(), new.tolist()], values)
        parameter = fr.nn.Parameter(fr.ones(2))
        with self.assertRaises(ValueError): fr.optim.SGD([{'params': [parameter]}, {'params': [parameter.tensor()]}], lr=.1)

    def test_shared_storage_distinct_leaf_slots_follow_torch_order(self):
        for family in ('SGD', 'AdamW', 'RMSprop', 'Adafactor'):
            storage = fr.tensor([[.4, -.6], [.7, -.2]])
            target = torch.tensor(storage.tolist())
            params = [storage.detach().requires_grad_(True) for _ in range(2)]
            expected = [target.detach().requires_grad_(True) for _ in range(2)]
            optimizer = getattr(fr.optim, family)(params, lr=.01)
            reference = getattr(torch.optim, family)(expected, lr=.01)
            for iteration in range(5):
                optimizer.zero_grad(); reference.zero_grad()
                for index, (parameter, oracle) in enumerate(zip(params, expected)):
                    (parameter.square().sum() * (.3 + index * .1 + iteration * .02)).backward()
                    (oracle.square().sum() * (.3 + index * .1 + iteration * .02)).backward()
                optimizer.step(); reference.step()
                torch.testing.assert_close(torch.tensor(storage.tolist()), target, atol=3e-6, rtol=3e-5)
                self.compare_state(optimizer, reference)

    def test_owning_state_and_fresh_process_schedule_continuation(self):
        values, optimizer, scheduler = setup()
        for iteration in range(4):
            optimizer.zero_grad(); sum(value.square().sum() * (.2 + iteration * .01 + pi * .1) for pi, value in enumerate(values)).backward()
            optimizer.step(); scheduler.step()
        optimizer.zero_grad(); state = optimizer.state_dict()
        tensors = {f'parameter{index}': value.detach().clone() for index, value in enumerate(values)}
        encoded = serialize(state, tensors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); fr.save_safetensors(str(path / 'tensors'), tensors)
            (path / 'state.json').write_text(json.dumps({'optimizer': encoded, 'scheduler': scheduler.state_dict()}))
            before = plain(state); expected = continuation(values, optimizer, scheduler, 4)
            self.assertEqual(plain(state), before)
            subprocess.run([sys.executable, __file__, '--worker', directory], check=True)
            self.assertEqual(json.loads((path / 'actual.json').read_text()), json.loads(json.dumps(expected)))
        before = native(optimizer); state['state'][1]['exp_avg'] = fr.zeros(3)
        with self.assertRaises(ValueError): optimizer.load_state_dict(state)
        self.assertEqual(native(optimizer), before)
        class Model(fr.nn.Module):
            def build(self): self.parameter = fr.nn.Parameter(fr.ones(2))
        optimizer.zero_grad()
        with self.assertRaisesRegex(ValueError, 'outside model'):
            fr.checkpoint.snapshot_training(Model(), optimizers={'external': optimizer}, generator=fr.Generator(1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker': worker(sys.argv[2])
    else: unittest.main()
