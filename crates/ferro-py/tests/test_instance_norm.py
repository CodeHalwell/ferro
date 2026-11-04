"""Actual Torch instance normalization contracts and native gradient checks."""
import inspect
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
from ferro.nn import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
from ferro.nn.functional import instance_norm
import test_batchnorm as helpers

native = helpers.native


def setup():
    model = InstanceNorm1d(3, affine=True, track_running_stats=True, momentum=.2)
    return model, fr.optim.AdamW(model.parameters(), lr=.004, weight_decay=.01)


def step(model, optimizer, generator, index):
    value = fr.Tensor.randn([2 + index % 2, 3, 4], generator=generator)
    output = model(value)
    loss = output.square().mean() + output.mean() * .2
    observed = output.tolist(), loss.item()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    return observed


def continuation(model, optimizer, generator, start):
    values = [step(model, optimizer, generator, index) for index in range(start, 7)]
    snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=7)
    return {'values': values, 'state': {key: [value.dtype, value.shape, value.tolist()] for key, value in snapshot.tensors().items()},
            'draws': fr.Tensor.randn([5], generator=generator).tolist()}


def worker(directory):
    model, optimizer = setup(); generator = fr.Generator(999)
    saved = fr.checkpoint.load(Path(directory) / 'checkpoint')
    saved.restore(model, optimizers={'main': optimizer}, generator=generator)
    (Path(directory) / 'result.json').write_text(json.dumps(continuation(model, optimizer, generator, saved.step)))


class InstanceNormTests(unittest.TestCase):
    close = helpers.BatchNormTests.close
    same_state = helpers.BatchNormTests.same_state
    compare = helpers.BatchNormTests.compare

    def setUp(self):
        torch.manual_seed(1907)

    def test_untracked_all_ranks_unbatched_strided_and_affine_gradients(self):
        for cls, oracle, shapes in ((InstanceNorm1d, torch.nn.InstanceNorm1d, ((3, 5), (2, 3, 5))),
                                   (InstanceNorm2d, torch.nn.InstanceNorm2d, ((3, 2, 3), (2, 3, 2, 3))),
                                   (InstanceNorm3d, torch.nn.InstanceNorm3d, ((3, 2, 2, 3), (2, 3, 2, 2, 3)))):
            for shape in shapes:
                for affine, bias in ((False, True), (True, True), (True, False)):
                    for training in (True, False):
                        with self.subTest(layer=cls.__name__, shape=shape, affine=affine, bias=bias, training=training):
                            model, reference = cls(3, affine=affine, bias=bias), oracle(3, affine=affine, bias=bias)
                            model.train(training); reference.train(training)
                            self.compare(model, reference, shape, strided=True)

    def test_float64_nonaffine_untracked_and_functional_affine_gradients(self):
        model, reference = InstanceNorm2d(3, dtype='float64'), torch.nn.InstanceNorm2d(3, dtype=torch.float64)
        self.compare(model, reference, (2, 3, 2, 3), torch.float64, True)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
            weight = torch.tensor([.8, -.4, 1.1], dtype=dtype, requires_grad=True)
            bias = torch.tensor([.2, -.3, .4], dtype=dtype, requires_grad=True)
            nx, nw, nb = native(x, True), native(weight, True), native(bias, True)
            output = instance_norm(nx, weight=nw, bias=nb)
            expected = torch.nn.functional.instance_norm(x, weight=weight, bias=bias)
            self.close(output, expected)
            probe = torch.randn(output.shape, dtype=dtype)
            (output * native(probe)).sum().backward(); (expected * probe).sum().backward()
            for left, right in ((nx, x), (nw, weight), (nb, bias)):
                self.close(left.grad, right.grad)

    def test_spatial_empty_singleton_zero_batch_and_warning_contracts(self):
        for shape in ((2, 3, 0), (3, 0)):
            self.compare(InstanceNorm1d(3, affine=True), torch.nn.InstanceNorm1d(3, affine=True), shape)
        for shape in ((2, 3, 1), (3, 1)):
            for layer, value in ((InstanceNorm1d(3), fr.zeros(shape)), (torch.nn.InstanceNorm1d(3), torch.zeros(shape))):
                with self.assertRaises((ValueError, IndexError)):
                    layer(value)
        for shape in ((0, 3, 3), (0, 3, 0)):
            self.compare(InstanceNorm1d(3), torch.nn.InstanceNorm1d(3), shape)
            for layer, value in ((InstanceNorm1d(3, affine=True), fr.zeros(shape)), (torch.nn.InstanceNorm1d(3, affine=True), torch.zeros(shape))):
                with self.assertRaises((ValueError, IndexError)): layer(value)
        for cls, tensor in ((InstanceNorm1d, fr.ones([2, 4, 3])), (torch.nn.InstanceNorm1d, torch.ones(2, 4, 3))):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always'); result = cls(3)(tensor)
                self.assertEqual(len(seen), 1)
            self.assertEqual(list(result.shape), [2, 4, 3])
            with self.assertRaises(ValueError): cls(3, affine=True)(tensor)
        model = InstanceNorm1d(3)
        self.assertFalse(model(fr.ones([2, 3, 4])).requires_grad)
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(fr.ones([2, 3, 4])))
        with self.assertRaises(NotImplementedError): InstanceNorm1d(3, affine=True, dtype='bool')
        for cls, reference in ((InstanceNorm1d, torch.nn.InstanceNorm1d), (InstanceNorm2d, torch.nn.InstanceNorm2d), (InstanceNorm3d, torch.nn.InstanceNorm3d)):
            self.assertEqual(list(inspect.signature(cls).parameters), list(inspect.signature(reference).parameters))

    def test_tracked_statistics_average_unbiased_instance_variance_and_never_count(self):
        for momentum in (None, 0., .3, 1., -.2):
            for dtype in (torch.float32, torch.float64):
                model = InstanceNorm1d(2, track_running_stats=True, momentum=momentum, dtype=str(dtype).split('.')[-1])
                reference = torch.nn.InstanceNorm1d(2, track_running_stats=True, momentum=momentum, dtype=dtype)
                value = torch.tensor([[[0., 1., 2.], [3., 4., 5.]], [[6., 8., 10.], [11., 15., 19.]]], dtype=dtype)
                actual, expected = model(native(value)), reference(value)
                self.close(actual, expected); self.same_state(model, reference)
                factor = 0. if momentum is None else momentum
                self.close(model.running_mean, value.mean(2).mean(0) * factor)
                self.close(model.running_var, torch.ones(2, dtype=dtype) * (1-factor) + value.var(2, unbiased=True).mean(0) * factor)
                self.assertEqual(model.num_batches_tracked.item(), 0)
                model.eval(); reference.eval()
                self.compare(model, reference, (2, 2, 1), dtype)

    def test_mutable_tracking_flags_and_empty_stats_reset(self):
        for initial in (False, True):
            model, reference = InstanceNorm2d(3, affine=True, track_running_stats=initial), torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=initial)
            for training, tracking in ((True, initial), (True, not initial), (False, not initial), (False, initial)):
                model.train(training); reference.train(training)
                model.track_running_stats = reference.track_running_stats = tracking
                if not initial and not training and tracking:
                    for layer, value in ((model, fr.ones([2,3,2,2])), (reference, torch.ones(2,3,2,2))):
                        with self.assertRaises((ValueError, RuntimeError)): layer(value)
                else:
                    self.compare(model, reference, (2, 3, 2, 2))
            model.track_running_stats = reference.track_running_stats = initial
            model.reset_parameters(); reference.reset_parameters()
            self.same_state(model, reference)
        model, reference = InstanceNorm3d(3, track_running_stats=True), torch.nn.InstanceNorm3d(3, track_running_stats=True)
        self.compare(model, reference, (2, 3, 0, 2, 2))
        self.assertEqual(model.num_batches_tracked.item(), 0)
        for training in (False, True):
            model, reference = InstanceNorm1d(3, track_running_stats=True), torch.nn.InstanceNorm1d(3, track_running_stats=True)
            model.train(training); reference.train(training)
            self.compare(model, reference, (0,3,4))

    def test_running_write_preflight_capacity_and_gradient_boundaries(self):
        x = fr.ones([2,3,4])
        mean, variance = fr.zeros(3), fr.ones(3)
        bad = fr.ones(6).narrow(0, 0, 3)
        for args in ((mean, bad), (mean, mean)):
            before = mean.tolist()
            with self.assertRaises((ValueError, NotImplementedError)): instance_norm(x, *args)
            self.assertEqual(mean.tolist(), before)
        for training in (False, True):
            for args, reference in (((mean, None), (torch.zeros(3), None)), ((None, variance), (None, torch.ones(3)))):
                with self.assertRaises(ValueError): instance_norm(x, *args, use_input_stats=training)
                with self.assertRaises((ValueError, RuntimeError)): torch.nn.functional.instance_norm(torch.ones(2,3,4), *reference, use_input_stats=training)
                self.assertEqual(mean.tolist(), [0., 0., 0.])
                self.assertEqual(variance.tolist(), [1., 1., 1.])
        with self.assertRaises(RuntimeError): instance_norm(x, mean.requires_grad_(True), variance)
        value = fr.ones([1,1,1]).expand(2, sys.maxsize//4, 0)
        with self.assertRaises(ValueError): instance_norm(value)
        differentiable = fr.ones([2,3,4]).requires_grad_(True)
        output = instance_norm(differentiable)
        with self.assertRaises((ValueError, RuntimeError)): fr.autograd.grad(output.sum(), differentiable, create_graph=True)

    def test_typed_checkpoint_optimizer_statistics_and_rng_continuation(self):
        model, optimizer = setup(); generator = fr.Generator(189)
        for index in range(3): step(model, optimizer, generator, index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'instance'
            fr.checkpoint.save_training(path, model, optimizers={'main': optimizer}, generator=generator, step=3)
            restored, restored_optimizer = setup(); restored_generator = fr.Generator(998)
            saved = fr.checkpoint.load(path)
            saved.restore(restored, optimizers={'main': restored_optimizer}, generator=restored_generator)
            self.assertEqual(helpers.state(model), helpers.state(restored))
            for index in range(3, 7):
                self.assertEqual(step(model, optimizer, generator, index), step(restored, restored_optimizer, restored_generator, index))
            for current, current_optimizer, current_generator in ((model, optimizer, generator), (restored, restored_optimizer, restored_generator)):
                snapshot = fr.checkpoint.snapshot_training(current, optimizers={'main': current_optimizer}, generator=current_generator, step=7)
                observed = {key: [value.dtype, value.shape, value.tolist()] for key, value in snapshot.tensors().items()}
                if current is model: expected = observed
                else: self.assertEqual(observed, expected)
            self.assertEqual(fr.Tensor.randn([5], generator=generator).tolist(), fr.Tensor.randn([5], generator=restored_generator).tolist())
            self.assertEqual(restored.num_batches_tracked.item(), 0)
        for dtype, tracking in (('float64', True), ('float32', False)):
            original = InstanceNorm2d(3, dtype=dtype, track_running_stats=tracking, momentum=None)
            original(fr.ones([2, 3, 2, 2], dtype=dtype))
            saved = fr.checkpoint.snapshot_training(original, optimizers={}, generator=fr.Generator(3))
            restored = InstanceNorm2d(3, dtype=dtype, track_running_stats=tracking, momentum=None)
            saved.restore(restored, optimizers={}, generator=fr.Generator(9))
            self.assertEqual(helpers.state(original), helpers.state(restored))
            changed = InstanceNorm2d(3, dtype=dtype, track_running_stats=tracking, momentum=.1)
            before = helpers.state(changed)
            with self.assertRaises(ValueError): saved.restore(changed, optimizers={}, generator=fr.Generator(9))
            self.assertEqual(helpers.state(changed), before)

    def test_checkpoint_exact_fresh_process_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer = setup(); generator = fr.Generator(189)
            for index in range(3): step(model, optimizer, generator, index)
            fr.checkpoint.save_training(Path(directory) / 'checkpoint', model, optimizers={'main': optimizer}, generator=generator, step=3)
            expected = continuation(model, optimizer, generator, 3)
            env = dict(os.environ); env.pop('PYTHONPATH', None)
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', directory], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout)
            self.assertEqual(json.loads((Path(directory) / 'result.json').read_text()), json.loads(json.dumps(expected)))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        unittest.main(verbosity=2)
