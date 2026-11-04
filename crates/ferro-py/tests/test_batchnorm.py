"""Canonical BatchNorm classes, typed counters and restart against actual Torch."""
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr


def native(value, grad=False):
    factory = fr.Tensor.from_f64 if value.dtype == torch.float64 else fr.Tensor
    return factory(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def state(model):
    result = {name: [value.dtype, value.shape, value.tolist()] for name, value in model.named_buffers()}
    result.update({name: [value.tensor().dtype, value.tensor().shape, value.tensor().tolist()] for name, value in model.named_parameters()})
    return result


def setup():
    model = fr.nn.BatchNorm1d(3, momentum=None)
    optimizer = fr.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    return model, optimizer


def step(model, optimizer, generator, index):
    x = fr.Tensor.randn([3 + index % 2, 3, 2], generator=generator)
    output = model(x)
    loss = output.square().mean() + output.mean() * .3
    values = output.tolist(), loss.item()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    return values


def continuation(model, optimizer, generator, start):
    values = [step(model, optimizer, generator, index) for index in range(start, 7)]
    snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=7)
    return {'values': values, 'state': {key: [value.dtype, value.shape, value.tolist()] for key, value in snapshot.tensors().items()},
            'draws': fr.Tensor.randn([5], generator=generator).tolist()}


def worker(directory):
    model, optimizer = setup()
    generator = fr.Generator(999)
    saved = fr.checkpoint.load(Path(directory) / 'checkpoint')
    saved.restore(model, optimizers={'main': optimizer}, generator=generator)
    (Path(directory) / 'result.json').write_text(json.dumps(continuation(model, optimizer, generator, saved.step)))


class BatchNormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1889)

    def close(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        value = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 3e-11 if expected.dtype == torch.float64 else 3e-5
        torch.testing.assert_close(value, expected.detach(), atol=tolerance, rtol=tolerance, equal_nan=True)

    def same_state(self, model, reference):
        self.assertEqual(set(dict(model.named_parameters())), set(dict(reference.named_parameters())))
        self.assertEqual(set(dict(model.named_buffers())), set(dict(reference.named_buffers())))
        for name, value in model.named_parameters():
            self.close(value.tensor(), dict(reference.named_parameters())[name])
        for name, value in model.named_buffers():
            self.close(value, dict(reference.named_buffers())[name])

    def compare(self, model, reference, shape, dtype=torch.float32, strided=False):
        if strided:
            shape = list(shape)
            shape[0], shape[-1] = shape[-1], shape[0]
            base = torch.randn(shape, dtype=dtype)
            x = base.transpose(0, -1).detach().requires_grad_(True)
            nx = native(base).transpose(0, len(shape) - 1).requires_grad_(True)
        else:
            x = torch.randn(shape, dtype=dtype, requires_grad=True)
            nx = native(x, True)
        actual, expected = model(nx), reference(x)
        self.close(actual, expected)
        probe = torch.randn(expected.shape, dtype=dtype)
        (actual * native(probe)).sum().backward(); (expected * probe).sum().backward()
        self.close(nx.grad, x.grad)
        for name, value in model.named_parameters():
            self.close(value.tensor().grad, dict(reference.named_parameters())[name].grad)
            value.tensor().zero_grad()
        reference.zero_grad(set_to_none=True)
        self.same_state(model, reference)

    def test_all_ranks_strided_training_evaluation_and_every_gradient(self):
        cases = [(fr.nn.BatchNorm1d, torch.nn.BatchNorm1d, (5, 3)),
                 (fr.nn.BatchNorm1d, torch.nn.BatchNorm1d, (4, 3, 2)),
                 (fr.nn.BatchNorm2d, torch.nn.BatchNorm2d, (3, 3, 2, 4)),
                 (fr.nn.BatchNorm3d, torch.nn.BatchNorm3d, (2, 3, 2, 3, 2))]
        for cls, oracle, shape in cases:
            for bias in (True, False):
                with self.subTest(layer=cls.__name__, shape=shape, bias=bias):
                    model, reference = cls(3, bias=bias), oracle(3, bias=bias)
                    with fr.no_grad(), torch.no_grad():
                        model.weight.tensor().copy_(fr.tensor([.7, 1.2, -.4])); reference.weight.copy_(torch.tensor([.7, 1.2, -.4]))
                        if bias:
                            model.bias.tensor().copy_(fr.tensor([.2, -.3, .4])); reference.bias.copy_(torch.tensor([.2, -.3, .4]))
                    self.compare(model, reference, shape, strided=True)
                    model.eval(); reference.eval()
                    self.compare(model, reference, shape, strided=True)

    def test_cumulative_and_fixed_momentum_with_variable_batch_sizes(self):
        for momentum in (None, 0., .3, 1., -.2, 1.2):
            model, reference = fr.nn.BatchNorm1d(3, momentum=momentum), torch.nn.BatchNorm1d(3, momentum=momentum)
            for shape in ((4, 3, 2), (2, 3, 3), (5, 3, 1)):
                self.compare(model, reference, shape)
            self.assertEqual(model.num_batches_tracked.dtype, 'int64')
            self.assertEqual(model.num_batches_tracked.item(), 3)

    def test_nonaffine_float64_optional_statistics_and_all_input_gradients(self):
        for cls, oracle, shape in ((fr.nn.BatchNorm1d, torch.nn.BatchNorm1d, (3, 3, 2)),
                                    (fr.nn.BatchNorm2d, torch.nn.BatchNorm2d, (2, 3, 2, 2)),
                                    (fr.nn.BatchNorm3d, torch.nn.BatchNorm3d, (2, 3, 2, 2, 2))):
            for tracking in (True, False):
                model = cls(3, affine=False, track_running_stats=tracking, dtype='float64', momentum=None)
                reference = oracle(3, affine=False, track_running_stats=tracking, dtype=torch.float64, momentum=None)
                self.compare(model, reference, shape, torch.float64, True)
                model.eval(); reference.eval()
                self.compare(model, reference, shape, torch.float64)
                if not tracking:
                    self.assertIsNone(model.running_mean)
                    self.assertIsNone(model.running_var)
                    self.assertIsNone(model.num_batches_tracked)

    def test_empty_and_singleton_counter_order_rank_failure_and_exact_int64(self):
        for cls, oracle, shapes, singleton in ((fr.nn.BatchNorm1d, torch.nn.BatchNorm1d, ((0, 3), (2, 3, 0)), (1, 3)),
                                               (fr.nn.BatchNorm2d, torch.nn.BatchNorm2d, ((0, 3, 2, 2), (2, 3, 0, 4)), (1, 3, 1, 1)),
                                               (fr.nn.BatchNorm3d, torch.nn.BatchNorm3d, ((0, 3, 2, 2, 2), (2, 3, 0, 2, 2)), (1, 3, 1, 1, 1))):
            model, reference = cls(3, momentum=None), oracle(3, momentum=None)
            for shape in shapes:
                self.compare(model, reference, shape)
            before_mean, before_var = model.running_mean.tolist(), model.running_var.tolist()
            for layer, value in ((model, fr.ones(singleton)), (reference, torch.ones(singleton))):
                with self.assertRaises(ValueError): layer(value)
            self.same_state(model, reference)
            self.assertEqual(model.num_batches_tracked.item(), 3)
            self.assertEqual(model.running_mean.tolist(), before_mean)
            self.assertEqual(model.running_var.tolist(), before_var)
            for layer, value in ((model, fr.ones(3)), (reference, torch.ones(3))):
                with self.assertRaises(ValueError): layer(value)
            self.assertEqual(model.num_batches_tracked.item(), 3)
        model, reference = fr.nn.BatchNorm1d(3), torch.nn.BatchNorm1d(3)
        for count in (2**60 + 3, 2**63 - 1):
            model.num_batches_tracked.copy_(fr.tensor(count, dtype='int64'))
            reference.num_batches_tracked.fill_(count)
            self.compare(model, reference, (2, 3))

    def test_mutable_tracking_flag_and_eval_requires_both_buffers_absent(self):
        for initial in (False, True):
            model, reference = fr.nn.BatchNorm1d(3, track_running_stats=initial), torch.nn.BatchNorm1d(3, track_running_stats=initial)
            for training, tracking in ((True, initial), (True, not initial), (False, not initial), (False, initial)):
                model.train(training); reference.train(training)
                model.track_running_stats = reference.track_running_stats = tracking
                self.compare(model, reference, (4, 3, 2))
        model, reference = fr.nn.BatchNorm1d(3).eval(), torch.nn.BatchNorm1d(3).eval()
        model.running_mean = reference.running_mean = None
        for layer, value in ((model, fr.ones([2, 3])), (reference, torch.ones(2, 3))):
            with self.assertRaises((ValueError, RuntimeError)): layer(value)
        self.assertEqual(model.num_batches_tracked.item(), 0)
        model.running_var = reference.running_var = None
        self.compare(model, reference, (4, 3))

    def test_reset_parameters_running_statistics_and_identity(self):
        for affine, tracking in ((True, True), (False, True), (True, False), (False, False)):
            model, reference = fr.nn.BatchNorm1d(3, affine=affine, track_running_stats=tracking), torch.nn.BatchNorm1d(3, affine=affine, track_running_stats=tracking)
            self.compare(model, reference, (4, 3))
            held = {name: value._id() for name, value in model.named_buffers()}
            if affine:
                with fr.no_grad(), torch.no_grad():
                    model.weight.tensor().copy_(fr.zeros(3)); reference.weight.zero_()
                    model.bias.tensor().copy_(fr.ones(3)); reference.bias.fill_(1)
            model.reset_running_stats(); reference.reset_running_stats()
            self.same_state(model, reference)
            model.reset_parameters(); reference.reset_parameters()
            self.same_state(model, reference)
            self.assertEqual(held, {name: value._id() for name, value in model.named_buffers()})

    def test_constructor_dtype_capacity_and_capture_boundaries(self):
        for constructor in (lambda: fr.nn.BatchNorm1d(0), lambda: fr.nn.BatchNorm1d(True), lambda: fr.nn.BatchNorm1d(3, eps=0),
                            lambda: fr.nn.BatchNorm1d(3, momentum=float('nan')), lambda: fr.nn.BatchNorm1d(3, affine=1),
                            lambda: fr.nn.BatchNorm1d(sys.maxsize), lambda: fr.nn.BatchNorm1d(3, bias=1)):
            with self.assertRaises((TypeError, ValueError)): constructor()
        for constructor in (lambda: fr.nn.BatchNorm1d(3, dtype='bool'), lambda: fr.nn.BatchNorm1d(3, dtype='int64'),
                            lambda: fr.nn.BatchNorm1d(3, device='cuda')):
            with self.assertRaises(NotImplementedError): constructor()
        model = fr.nn.BatchNorm1d(3)
        before = state(model)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(fr.ones([2, 3])))
        self.assertEqual(state(model), before)

    def test_inherited_public_signatures_match_torch(self):
        for model, reference in ((fr.nn.BatchNorm1d, torch.nn.BatchNorm1d), (fr.nn.BatchNorm2d, torch.nn.BatchNorm2d),
                                 (fr.nn.BatchNorm3d, torch.nn.BatchNorm3d)):
            self.assertEqual(list(inspect.signature(model).parameters), list(inspect.signature(reference).parameters))
        class DerivedBatchNorm(fr.nn.BatchNorm1d):
            def reset_parameters(self):
                super().reset_parameters()
                with fr.no_grad():
                    self.weight.tensor().copy_(fr.ones(self.weight.tensor().shape) * 2.)
        class TorchDerived(torch.nn.BatchNorm1d):
            def reset_parameters(self):
                super().reset_parameters()
                with torch.no_grad():
                    self.weight.fill_(2.)
        self.assertEqual(inspect.signature(DerivedBatchNorm), inspect.signature(fr.nn.BatchNorm1d))
        self.compare(DerivedBatchNorm(3, momentum=None, bias=False), TorchDerived(3, momentum=None, bias=False), (4, 3, 2))


class BatchNormCheckpointTests(unittest.TestCase):
    # Separate state tests allow first math checks before the typed native build.
    def test_checkpoint_exact_fresh_process_suffix_and_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer = setup(); generator = fr.Generator(181)
            for index in range(3): step(model, optimizer, generator, index)
            fr.checkpoint.save_training(Path(directory) / 'checkpoint', model, optimizers={'main': optimizer}, generator=generator, step=3)
            expected = continuation(model, optimizer, generator, 3)
            env = dict(os.environ); env.pop('PYTHONPATH', None)
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', directory], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout)
            self.assertEqual(json.loads((Path(directory) / 'result.json').read_text()), json.loads(json.dumps(expected)))

    def test_checkpoint_typed_optional_buffers_and_configuration_rejection(self):
        for dtype, affine, tracking in (('float32', True, True), ('float64', False, True), ('float32', False, False)):
            model = fr.nn.BatchNorm1d(3, momentum=None, dtype=dtype, affine=affine, track_running_stats=tracking)
            model(fr.ones([2, 3], dtype=dtype))
            saved = fr.checkpoint.snapshot_training(model, optimizers={}, generator=fr.Generator(3))
            restored = fr.nn.BatchNorm1d(3, momentum=None, dtype=dtype, affine=affine, track_running_stats=tracking)
            saved.restore(restored, optimizers={}, generator=fr.Generator(99))
            self.assertEqual(state(model), state(restored))
            changed = fr.nn.BatchNorm1d(3, momentum=.2, dtype=dtype, affine=affine, track_running_stats=tracking)
            before = state(changed)
            with self.assertRaises(ValueError): saved.restore(changed, optimizers={}, generator=fr.Generator(4))
            self.assertEqual(state(changed), before)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        unittest.main(verbosity=2)
