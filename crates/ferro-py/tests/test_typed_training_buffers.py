"""Typed buffer payloads, alias transactions, optional slots and process resume."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from test_optimizer_groups_schedulers import plain


class TypedModel(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.tensor([.8, -.3]))
        self.bias = fr.nn.Parameter(fr.tensor([.1]))
        payload = struct.unpack('<d', struct.pack('<Q', 0x7ff8000000000042))[0]
        self.register_buffer('double', fr.tensor([-0., 1.0000000000000002, payload], dtype='float64'))
        self.register_buffer('double_alias', self.double.reshape([3]))
        self.register_buffer('count', fr.tensor([(1 << 60) + 3, -(1 << 59) - 7], dtype='int64'))
        self.register_buffer('count_alias', self.count.reshape([2]))
        self.register_buffer('num_batches_tracked', fr.tensor(0, dtype='int64'))
        self.register_buffer('flag', fr.tensor([True, False], dtype='bool'))
        self.register_buffer('flag_alias', self.flag.reshape([2]))
        self.register_buffer('optional', None)
        self.register_buffer('scratch', None, persistent=False)

    def forward(self, x):
        shift = .03 * (self.count.tolist()[0] % 7)
        sign = 1. if self.flag.tolist()[0] else -1.
        return (x * self.weight.tensor()).sum() + self.bias.tensor().sum() + shift * sign


def setup():
    model = TypedModel()
    optimizer = fr.optim.AdamW([{'params': [model.weight], 'lr': .02}, {'params': [model.bias], 'lr': .01}], weight_decay=.03)
    return model, optimizer


def tensor_bytes(tensor):
    formats = {'float32': 'f', 'float64': 'd', 'int64': 'q', 'bool': '?'}
    values = tensor.reshape([tensor.numel()]).tolist()
    return struct.pack('<' + formats[tensor.dtype] * len(values), *values)


def digest(checkpoint):
    checksum = hashlib.sha256()
    for name, value in sorted(checkpoint.tensors().items()):
        checksum.update(json.dumps([name, value.dtype, value.shape], separators=(',', ':')).encode())
        checksum.update(tensor_bytes(value))
    return checksum.hexdigest()


def snapshot(model, optimizer, generator):
    return fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=9)


def update(model, optimizer, generator):
    optimizer.zero_grad()
    x = fr.Tensor.randn([2], generator=generator)
    loss = model(x).square(); loss.backward(); optimizer.step(); optimizer.zero_grad()
    values = model.count.tolist()
    model.count.copy_(fr.tensor([values[0]+1, values[1]-1], dtype='int64'))
    model.num_batches_tracked.copy_(fr.tensor(model.num_batches_tracked.item()+1, dtype='int64'))
    model.flag.copy_(fr.tensor([not value for value in model.flag.tolist()], dtype='bool'))
    return loss.item()


def suffix(model, optimizer, generator, start):
    losses = [update(model, optimizer, generator) for _ in range(start, 9)]
    return dict(losses=losses, checkpoint=digest(snapshot(model, optimizer, generator)), optimizer=plain(optimizer.state_dict()),
                buffers={name: [value.dtype, value.shape, tensor_bytes(value).hex()] for name, value in model.named_buffers()},
                draws=fr.Tensor.randn([7], generator=generator).tolist())


def worker(directory):
    path = Path(directory); model, optimizer = setup(); generator = fr.Generator(999)
    checkpoint = fr.checkpoint.load(path / 'training')
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=generator)
    (path / 'result.json').write_text(json.dumps(suffix(model, optimizer, generator, checkpoint.step), allow_nan=False))


def corrupt(checkpoint, directory, key, tensor):
    path = Path(directory)
    checkpoint.save(path)
    pointer = path / 'checkpoint.json'
    metadata = json.loads(pointer.read_text())
    payload = path / metadata['generation'] / 'model.safetensors'
    arrays = checkpoint.tensors(); arrays[key] = tensor
    fr.save_safetensors(str(payload), arrays)
    checksum = 0xcbf29ce484222325
    for byte in payload.read_bytes():
        checksum = ((checksum ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    metadata['checksum'] = checksum
    pointer.write_text(json.dumps(metadata, indent=2))
    return fr.checkpoint.load(path)


class TypedTrainingBufferTests(unittest.TestCase):
    def test_optional_buffer_registration_assignment_and_iteration_match_torch(self):
        model, target = fr.nn.Module(), torch.nn.Module()
        for native in (model, target):
            native.register_buffer('optional', None)
            native.register_buffer('scratch', None, persistent=False)
        self.assertEqual(list(model.named_buffers()), list(target.named_buffers()))
        model.optional = fr.tensor([3], dtype='int64'); target.optional = torch.tensor([3])
        self.assertEqual([name for name, _ in model.named_buffers()], [name for name, _ in target.named_buffers()])
        model.optional = None; target.optional = None
        for native in (model, target):
            with self.assertRaises(TypeError): native.optional = 3
        self.assertEqual(list(model.named_buffers()), list(target.named_buffers()))
        model._structure_locked = True
        with self.assertRaises(RuntimeError): model.optional = fr.tensor([1])
        self.assertIsNone(model.optional)

    def test_owning_typed_snapshot_restores_exact_payloads_and_aliases(self):
        model, optimizer = setup(); generator = fr.Generator(43)
        update(model, optimizer, generator)
        saved = snapshot(model, optimizer, generator); before = digest(saved)
        model.double.copy_(fr.tensor([9., 8., 7.], dtype='float64'))
        saved.tensors()['buffers.count'].copy_(fr.tensor([1, 2], dtype='int64'))
        self.assertEqual(digest(saved), before)
        fresh, other = setup(); fresh_generator = fr.Generator(99)
        fresh.scratch = fr.tensor([8], dtype='int64')
        handles = dict(fresh.named_buffers()); identities = {name: value._id() for name, value in handles.items()}
        saved.restore(fresh, optimizers={'main': other}, generator=fresh_generator)
        self.assertEqual(digest(snapshot(fresh, other, fresh_generator)), before)
        self.assertEqual({name: value._id() for name, value in fresh.named_buffers()}, identities)
        self.assertEqual(tensor_bytes(fresh.double), struct.pack('<QQQ', 1 << 63, 0x3ff0000000000001, 0x7ff8000000000042))
        self.assertEqual(fresh.count.tolist(), [(1 << 60) + 4, -(1 << 59) - 8])
        self.assertEqual(fresh.num_batches_tracked.item(), 1)
        fresh.count.copy_(fr.tensor([5, 6], dtype='int64'))
        self.assertEqual(handles['count_alias'].tolist(), [5, 6])
        self.assertEqual(fresh.flag.tolist(), [False, True])
        self.assertIsNone(fresh.optional)
        self.assertEqual(fresh.scratch.tolist(), [8])

    def test_typed_alias_dtype_and_late_optimizer_faults_reject_atomically(self):
        source, optimizer = setup(); generator = fr.Generator(43)
        update(source, optimizer, generator); saved = snapshot(source, optimizer, generator)
        target, other = setup(); target_generator = fr.Generator(7)
        update(target, other, target_generator); update(target, other, target_generator)
        payload = struct.unpack('<d', struct.pack('<Q', 0x7ff8000000000042))[0]
        faults = [('buffers.double_alias', fr.tensor([0., 1.0000000000000002, payload], dtype='float64')),
                  ('buffers.count_alias', fr.tensor([(1 << 60)+5, -(1 << 59)-8], dtype='int64')),
                  ('buffers.flag_alias', fr.tensor([True, True], dtype='bool')),
                  ('buffers.count', fr.tensor([1., 2.])), ('optim.main.g1.v.0', fr.zeros(9))]
        for key, tensor in faults:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                malformed = corrupt(saved, directory, key, tensor)
                before = digest(snapshot(target, other, target_generator))
                with self.assertRaises(ValueError): malformed.restore(target, optimizers={'main': other}, generator=target_generator)
                self.assertEqual(digest(snapshot(target, other, target_generator)), before)
        target.optional = fr.tensor([1.])
        before = digest(snapshot(target, other, target_generator))
        with self.assertRaises(ValueError): saved.restore(target, optimizers={'main': other}, generator=target_generator)
        self.assertEqual(digest(snapshot(target, other, target_generator)), before)

    def test_pending_typed_gradients_and_noncontiguous_destination_reject(self):
        model, optimizer = setup(); generator = fr.Generator(3)
        model.register_buffer('differentiable', fr.tensor([2.], dtype='float64', requires_grad=True))
        model.differentiable.square().sum().backward()
        with self.assertRaisesRegex(ValueError, 'pending gradients'): snapshot(model, optimizer, generator)
        model.differentiable.zero_grad(); snapshot(model, optimizer, generator)
        del model.differentiable
        saved = snapshot(model, optimizer, generator)
        # A same-shaped offset view does not cover its entire destination storage.
        base = fr.tensor([4., 5., 6., 7.], dtype='float64')
        model.double = base.narrow(0, 1, 3); model.double_alias = model.double.reshape([3])
        original = base.tolist()
        with self.assertRaises(ValueError): saved.restore(model, optimizers={'main': optimizer}, generator=generator)
        self.assertEqual(base.tolist(), original)

    def test_full_typed_training_state_exact_fresh_process_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); model, optimizer = setup(); generator = fr.Generator(43)
            for _ in range(4): update(model, optimizer, generator)
            fr.checkpoint.save_training(path / 'training', model, optimizers={'main': optimizer}, generator=generator, step=4)
            expected = suffix(model, optimizer, generator, 4)
            env = dict(os.environ); env.pop('PYTHONPATH', None)
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)], env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(child.returncode, 0, child.stdout)
            self.assertEqual(json.loads((path / 'result.json').read_text()), expected)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        from test_typed_training_buffers import worker as run_worker
        run_worker(sys.argv[2])
    else:
        unittest.main(module='test_typed_training_buffers')
