"""Dropout math, persistent counter state and atomic fresh-process restore."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from test_optimizer_groups_schedulers import checkpoint_digest, plain


class DropoutModel(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.Tensor([.2, -.4, .7], [3]))
        self.dropout = fr.nn.Dropout(.4, seed=2**64 - 17)

    def forward(self, x):
        return self.dropout(x * self.weight.tensor())


def setup():
    model = DropoutModel()
    return model, fr.optim.Adam(model.parameters(), lr=.017), fr.Generator(13)


def update(model, optimizer, generator):
    optimizer.zero_grad()
    x = fr.Tensor.randn([4, 3], generator=generator)
    output = model(x)
    loss = output.square().sum() / 12.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return {'output': output.tolist(), 'loss': loss.item()}


def continuation(model, optimizer, generator, start):
    outputs = [update(model, optimizer, generator) for _ in range(start, 9)]
    masks = [model.dropout(fr.Tensor.ones([2, 3])).tolist() for _ in range(3)]
    snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=9)
    return {'outputs': outputs, 'masks': masks, 'state': checkpoint_digest(snapshot),
            'optimizer': plain(optimizer.state_dict()), 'rng': fr.Tensor.randn([7], generator=generator).tolist()}


def worker(path):
    path = Path(path)
    model, optimizer, generator = setup()
    model.dropout.rng_state.copy_(fr.Tensor([123.] + [0.] * 7, [8]))
    with fr.no_grad():
        model.weight.tensor().copy_(fr.Tensor([5., 6., 7.], [3]))
    checkpoint = fr.checkpoint.load(path / 'training')
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=generator)
    (path / 'result.json').write_text(json.dumps(continuation(model, optimizer, generator, checkpoint.step), allow_nan=False))


class DropoutStateTests(unittest.TestCase):
    def test_known_philox_mask_and_manual_torch_value_input_gradient(self):
        # Philox-4x32-10 zero-key/zero-counter published known answer:
        # 6627e8d5 e169c58d bc57ac4c 9b00dbd8. At p=.5 its keep mask is 0,1,1,1.
        x = fr.Tensor([1., -2., 0., 4.], [4]).requires_grad_(True)
        tx = torch.tensor([1., -2., 0., 4.], requires_grad=True)
        output = fr.nn.functional.dropout(x, .5, seed=0, offset=0)
        oracle = tx * torch.tensor([0., 2., 2., 2.])
        cotangent = [1., .7, -.3, 2.]
        (output * fr.Tensor(cotangent, [4])).sum().backward()
        oracle.backward(torch.tensor(cotangent))
        torch.testing.assert_close(torch.tensor(output.tolist()), oracle.detach(), rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(x.grad.tolist()), tx.grad, rtol=0, atol=0)

    def test_zero_one_eval_scalar_empty_and_nonfinite_match_torch(self):
        for p in (0., 1.):
            for training in (False, True):
                x = fr.Tensor([1., float('inf'), float('-inf'), float('nan')], [4]).requires_grad_(True)
                tx = torch.tensor([1., float('inf'), float('-inf'), float('nan')], requires_grad=True)
                output = fr.nn.functional.dropout(x, p, training=training)
                oracle = torch.nn.functional.dropout(tx, p, training=training)
                output.sum().backward()
                oracle.sum().backward()
                torch.testing.assert_close(torch.tensor(output.tolist()), oracle.detach(), equal_nan=True)
                torch.testing.assert_close(torch.tensor(x.grad.tolist()), tx.grad, rtol=0, atol=0)
                if not training or p == 0:
                    self.assertIs(output, x)
        for shape in ([], [0], [2, 0, 3]):
            values = [.7] if shape == [] else []
            x = fr.Tensor(values, shape).requires_grad_(True)
            dropout = fr.nn.Dropout(.5, seed=0)
            output = dropout(x)
            output.sum().backward()
            self.assertEqual(output.shape, shape)
            self.assertEqual(x.grad.shape, shape)
            self.assertEqual(dropout._offset, 1 if shape == [] else 0)
            if shape == []:
                self.assertEqual((output.item(), x.grad.item()), (0., 0.))

    def test_persistent_uint64_stream_modes_and_replay(self):
        dropout = fr.nn.Dropout(.25, seed=2**64 - 17)
        self.assertEqual(dropout.rng_state.tolist(), [65519., 65535., 65535., 65535., 0., 0., 0., 0.])
        self.assertEqual(set(dict(dropout.named_buffers())), {'rng_state'})
        replay = fr.nn.Dropout(.25, seed=2**64 - 17)
        x = fr.Tensor.ones([17])
        first, second = dropout(x).tolist(), dropout(x).tolist()
        self.assertEqual(replay(x).tolist(), first)
        self.assertEqual(replay(x).tolist(), second)
        self.assertNotEqual(first, second)
        self.assertEqual(dropout._offset, 34)
        before = dropout.rng_state.tolist()
        dropout.eval()
        self.assertIs(dropout(x), x)
        self.assertEqual(dropout.rng_state.tolist(), before)
        for probability in (0., 1.):
            other = fr.nn.Dropout(probability, seed=99)
            other(x)
            self.assertEqual(other._offset, 0)

    def test_corrupt_saved_chunks_reject_before_model_optimizer_rng_mutation(self):
        from test_training_checkpoint import rewrite_checkpoint
        model, optimizer, generator = setup()
        update(model, optimizer, generator)
        checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
        target, restored, other_rng = setup()
        update(target, restored, other_rng)
        target.dropout(fr.Tensor.ones([11]))
        for bad in (float('nan'), float('inf'), -.5, .5, 65536.):
            corrupt = checkpoint.tensors()['buffers.dropout.rng_state'].tolist()
            corrupt[7] = bad
            before = checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng))
            with self.subTest(chunk=bad), tempfile.TemporaryDirectory() as directory:
                malformed = rewrite_checkpoint(checkpoint, directory, {'buffers.dropout.rng_state': fr.Tensor(corrupt, [8])})
                with self.assertRaisesRegex(ValueError, 'dropout RNG'):
                    malformed.restore(target, optimizers={'main': restored}, generator=other_rng)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng)), before)
        # Validate live buffers on snapshot as well, even in eval mode.
        target.dropout.rng_state.copy_(fr.Tensor([.5] + [0.] * 7, [8]))
        target.eval()
        with self.assertRaisesRegex(ValueError, 'dropout RNG'):
            fr.checkpoint.snapshot_training(target, optimizers={'main': restored}, generator=other_rng)

    def test_counter_overflow_invalid_options_and_capture_do_not_advance_stream(self):
        dropout = fr.nn.Dropout(.5, seed=7)
        dropout.rng_state.copy_(fr.Tensor([7., 0., 0., 0.] + [65535.] * 4, [8]))
        before = dropout.rng_state.tolist()
        with self.assertRaisesRegex(ValueError, 'counter'):
            dropout(fr.Tensor.ones([1]))
        self.assertEqual(dropout.rng_state.tolist(), before)
        self.assertEqual(dropout(fr.Tensor.zeros([0])).shape, [0])
        self.assertEqual(dropout.rng_state.tolist(), before)
        for probability in (-.1, 1.1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                fr.nn.Dropout(probability)
        with self.assertRaises(NotImplementedError):
            fr.nn.Dropout(.5, inplace=True)
        for seed in (-1, 2**64, True):
            with self.assertRaises(ValueError):
                fr.nn.Dropout(.5, seed=seed)
        fresh = fr.nn.Dropout(.5, seed=7)
        with self.assertRaises(NotImplementedError):
            fr.capture(lambda: fresh(fr.Tensor.ones([4])))
        self.assertEqual(fresh._offset, 0)
        fresh.eval()
        fresh.p = 2.
        with self.assertRaises(ValueError):
            fresh(fr.Tensor.ones([4]))
        fresh.p = 0.
        with self.assertRaises(TypeError):
            fresh(object())
        self.assertEqual(fresh._offset, 0)

    def test_modes_configuration_and_owning_snapshot_restore_exactly(self):
        model, optimizer, generator = setup()
        model.dropout(fr.Tensor.ones([5]))
        model.dropout.eval()
        checkpoint = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
        saved = checkpoint_digest(checkpoint)
        model.dropout.train()
        model.dropout(fr.Tensor.ones([9]))
        self.assertEqual(checkpoint_digest(checkpoint), saved)
        target, restored, other_rng = setup()
        checkpoint.restore(target, optimizers={'main': restored}, generator=other_rng)
        self.assertFalse(target.dropout.training)
        self.assertEqual(target.dropout._offset, 5)
        target.dropout.p = .6
        before = target.dropout.rng_state.tolist()
        with self.assertRaises(ValueError):
            checkpoint.restore(target, optimizers={'main': restored}, generator=other_rng)
        self.assertEqual(target.dropout.rng_state.tolist(), before)

    def test_fresh_process_restores_training_suffix_and_next_masks_bit_exactly(self):
        model, optimizer, generator = setup()
        for _ in range(4):
            update(model, optimizer, generator)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fr.checkpoint.save_training(path / 'training', model, optimizers={'main': optimizer}, generator=generator, step=4)
            expected = continuation(model, optimizer, generator, 4)
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                                   env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout)
            self.assertEqual(json.loads((path / 'result.json').read_text()), expected)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        from test_dropout_state import worker as restore_worker
        restore_worker(sys.argv[2])
    else:
        unittest.main(module='test_dropout_state')
