import tempfile
import unittest
import ferro as fr


class TrainingCheckpointTests(unittest.TestCase):
    def test_stochastic_resume(self):
        self.assertTrue(hasattr(fr, 'checkpoint'), 'training checkpoint facade missing')
        class Model(fr.nn.Module):
            def build(self):
                self.w = fr.nn.Parameter(fr.Tensor([0.7], [1]))
            def forward(self, x):
                return self.w.tensor() * x
        model, rng = Model(), fr.Generator(43)
        opt = fr.optim.AdamW(model.parameters(), lr=0.03)
        def step():
            opt.zero_grad()
            x = fr.Tensor.randn([1], generator=rng)
            loss = (model(x) - 0.2).pow(2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return loss.item()
        step()
        snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=1)
        expected = [step() for _ in range(4)]
        value = model.w.tensor().tolist()
        with tempfile.TemporaryDirectory() as directory:
            snapshot.save(directory)
            saved = fr.checkpoint.load(directory)
            self.assertEqual(saved.restore(model, optimizers={'main': opt}, generator=rng), 1)
        self.assertEqual([step() for _ in range(4)], expected)
        self.assertEqual(model.w.tensor().tolist(), value)

    def test_frozen_parameter_roundtrip(self):
        p = fr.nn.Parameter(fr.Tensor([1.0], [1]))
        self.assertTrue(hasattr(p, 'trainable'), 'native parameter freezing missing')
        p.trainable = False
        self.assertFalse(p.trainable)
        (p.tensor() * 2).sum().backward()
        fr.optim.Adam([p], lr=0.1).step()
        self.assertEqual(p.tensor().item(), 1.0)
        p.zero_grad()

    def test_ties_private_children_buffers_modes_multiple_optimizers(self):
        self.assertTrue(hasattr(fr, 'checkpoint'), 'training checkpoint facade missing')
        class Child(fr.nn.Module):
            width: int = 1
            def build(self):
                self.p = fr.nn.Parameter(fr.Tensor([0.4], [1]))
                self.register_buffer('running', fr.Tensor([2.0], [1]))
                self.register_buffer('temporary', fr.Tensor([3.0], [1]), persistent=False)
        class Model(fr.nn.Module):
            def build(self):
                self._child = Child()
                self.alias = self._child.p
                self.q = fr.nn.Parameter(fr.Tensor([0.6], [1]))
        m, g = Model(), fr.Generator(56)
        m._child.eval()
        m.q.trainable = False
        opts = {'a': fr.optim.Adam([m.alias], lr=0.04), 'b': fr.optim.SGD([m.q], lr=0.3, momentum=0.6)}
        cp = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g, step=2**53 + 1)
        target = Model()
        target.eval()
        target._child.train()
        target_opts = {'a': fr.optim.Adam([target.alias], lr=0.9), 'b': fr.optim.SGD([target.q], lr=0.1)}
        with fr.no_grad():
            target._child.running.copy_(fr.Tensor([8.0], [1]))
            target._child.temporary.copy_(fr.Tensor([9.0], [1]))
        cp.restore(target, optimizers=target_opts, generator=fr.Generator(8))
        self.assertTrue(target.training)
        self.assertFalse(target._child.training)
        self.assertIs(target.alias, target._child.p)
        self.assertFalse(target.q.trainable)
        self.assertEqual(target._child.running.item(), 2.0)
        self.assertEqual(target._child.temporary.item(), 9.0)
        self.assertEqual(cp.step, 2**53 + 1)
        copied = cp.tensors()
        with fr.no_grad():
            copied['model.alias'].copy_(fr.Tensor([9.0], [1]))
        self.assertEqual(cp.tensors()['model.alias'].item(), m.alias.tensor().item())

    def test_pending_gradients_rejected(self):
        self.assertTrue(hasattr(fr, 'checkpoint'), 'training checkpoint facade missing')
        m = fr.nn.Linear(1, 1)
        m.weight.tensor().sum().backward()
        with self.assertRaisesRegex(ValueError, 'pending gradients'):
            fr.checkpoint.snapshot_training(m, optimizers={}, generator=fr.Generator(1))

    def test_late_optimizer_failure_does_not_mutate_any_state(self):
        self.assertTrue(hasattr(fr, 'checkpoint'), 'training checkpoint facade missing')
        m = fr.nn.Linear(1, 1)
        g = fr.Generator(1)
        opts = {'a': fr.optim.Adam([m.weight], lr=0.01), 'b': fr.optim.AdamW([m.bias], lr=0.02)}
        cp = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g)
        with fr.no_grad():
            m.weight.tensor().copy_(fr.Tensor([9.0], [1, 1]))
        bad = {'a': opts['a'], 'b': fr.optim.Adam([m.bias], lr=0.02)}
        before = fr.checkpoint.snapshot_training(m, optimizers=bad, generator=g).tensors()
        with self.assertRaises(ValueError):
            cp.restore(m, optimizers=bad, generator=g)
        after = fr.checkpoint.snapshot_training(m, optimizers=bad, generator=g).tensors()
        self.assertEqual({k:v.tolist() for k,v in before.items()}, {k:v.tolist() for k,v in after.items()})

    def test_all_optimizers_resume_stochastic_training_into_fresh_objects(self):
        class Model(fr.nn.Module):
            def build(self):
                self.p = fr.nn.Parameter(fr.Tensor([0.7, -0.4], [2]))
                self.tied = self.p
                self.q = fr.nn.Parameter(fr.Tensor([0.2, 0.1], [2]))
        def setup(seed, alternate=False):
            m, g = Model(), fr.Generator(seed)
            opts = {'a': fr.optim.Adam([m.p, m.tied], lr=0.05 if not alternate else 0.9),
                    'w': fr.optim.AdamW([m.q], lr=0.03, weight_decay=0.2),
                    's': fr.optim.SGD([m.p], lr=0.01, momentum=0.7, nesterov=True)}
            return m, g, opts
        def step(m, g, opts):
            for o in opts.values(): o.zero_grad()
            x = fr.Tensor.randn([2], generator=g)
            loss = ((m.p.tensor() + m.tied.tensor()) * x + m.q.tensor()).pow(2).sum()
            loss.backward()
            for o in opts.values(): o.step()
            for o in opts.values(): o.zero_grad()
            return loss.item()
        m, g, opts = setup(987)
        for _ in range(3): step(m, g, opts)
        cp = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g, step=3)
        losses = [step(m, g, opts) for _ in range(7)]
        expected = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g, step=10).tensors()
        target, rng, target_opts = setup(123, True)
        with tempfile.TemporaryDirectory() as directory:
            cp.save(directory)
            fr.checkpoint.load_training(directory, target, optimizers=target_opts, generator=rng)
        self.assertEqual([step(target, rng, target_opts) for _ in range(7)], losses)
        actual = fr.checkpoint.snapshot_training(target, optimizers=target_opts, generator=rng, step=10).tensors()
        self.assertEqual({k:v.tolist() for k,v in expected.items()}, {k:v.tolist() for k,v in actual.items()})

    def test_corrupt_components_prevalidate_before_live_mutation(self):
        import json
        from pathlib import Path
        m, g = fr.nn.Linear(1, 1), fr.Generator(1)
        opts = {'a': fr.optim.Adam(m.parameters(), lr=0.02)}
        cp = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g)
        with fr.no_grad(): m.weight.tensor().copy_(fr.Tensor([4.0], [1, 1]))
        before = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g).tensors()
        mutations = {
            'model.weight': fr.Tensor([1.0, 2.0], [2]),
            'optim.a.config': fr.Tensor([-1.0], [1]),
            'optim.a.m.0': fr.Tensor([1.0, 2.0], [2]),
            'rng.xorshift128': fr.Tensor.zeros([8]),
            'scalars.mode.': fr.Tensor([2.0, 0.0, 0.0, 0.0], [4]),
            'scalars.schema.0': fr.Tensor([0.0, 0.0, 0.0, 0.0], [4]),
        }
        for key, tensor in mutations.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                cp.save(directory)
                pointer = Path(directory) / 'checkpoint.json'
                meta = json.loads(pointer.read_text())
                payload = Path(directory) / meta['generation'] / 'model.safetensors'
                tensors = cp.tensors()
                self.assertIn(key, tensors)
                tensors[key] = tensor
                fr.save_safetensors(str(payload), tensors)
                checksum = 0xcbf29ce484222325
                for byte in payload.read_bytes():
                    checksum = ((checksum ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
                meta['checksum'] = checksum
                pointer.write_text(json.dumps(meta, indent=2))
                invalid = fr.checkpoint.load(directory)
                with self.assertRaises(ValueError):
                    invalid.restore(m, optimizers=opts, generator=g)
                after = fr.checkpoint.snapshot_training(m, optimizers=opts, generator=g).tensors()
                self.assertEqual({k:v.tolist() for k,v in before.items()}, {k:v.tolist() for k,v in after.items()})

    def test_unsupported_cursor_and_dtype_rejected(self):
        import struct
        import json
        from pathlib import Path
        m = fr.nn.Linear(1, 1)
        with self.assertRaises(TypeError):
            fr.checkpoint.snapshot_training(m, optimizers={}, generator=fr.Generator(1), cursor=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'double.safetensors'
            data = struct.pack('<d', 1.0000000000000002)
            header = json.dumps({'x': {'dtype': 'F64', 'shape': [1], 'data_offsets': [0, len(data)]}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header + data)
            double = fr.load_safetensors(str(path))['x']
            m.register_buffer('double', double)
            with self.assertRaisesRegex(ValueError, 'CPU f32'):
                fr.checkpoint.snapshot_training(m, optimizers={}, generator=fr.Generator(1))

    def test_generation_republication_keeps_old_payload(self):
        import json
        from pathlib import Path
        m, g = fr.nn.Linear(1, 1), fr.Generator(1)
        with tempfile.TemporaryDirectory() as directory:
            fr.checkpoint.save_training(directory, m, optimizers={}, generator=g, step=1)
            pointer = Path(directory) / 'checkpoint.json'
            old = json.loads(pointer.read_text())
            payload = Path(directory) / old['generation'] / 'model.safetensors'
            previous = payload.read_bytes()
            with fr.no_grad(): m.weight.tensor().copy_(fr.Tensor([4.0], [1, 1]))
            fr.checkpoint.save_training(directory, m, optimizers={}, generator=g, step=2)
            self.assertEqual(payload.read_bytes(), previous)
            self.assertNotEqual(json.loads(pointer.read_text())['generation'], old['generation'])
            self.assertEqual(fr.checkpoint.load(directory).step, 2)

    def test_adam_variants_match_torch_cpu_updates(self):
        import torch
        for kind, kwargs in [(fr.optim.Adam, {}), (fr.optim.AdamW, {'weight_decay': 0.17})]:
            with self.subTest(kind=kind.__name__):
                p = fr.nn.Parameter(fr.Tensor([0.4, -0.7], [2]))
                q = torch.tensor([0.4, -0.7], dtype=torch.float32, requires_grad=True)
                opt = kind([p], lr=0.02, betas=(0.8, 0.95), eps=1e-5, **kwargs)
                reference = getattr(torch.optim, kind.__name__)([q], lr=0.02, betas=(0.8, 0.95), eps=1e-5, **kwargs)
                for i in range(9):
                    opt.zero_grad()
                    reference.zero_grad()
                    (p.tensor() * (i + 1)).pow(2).sum().backward()
                    (q * (i + 1)).pow(2).sum().backward()
                    opt.step()
                    reference.step()
                    for actual, expected in zip(p.tensor().tolist(), q.detach().tolist()):
                        self.assertAlmostEqual(actual, expected, places=6)

    def test_model_schema_and_optimizer_binding_rejections(self):
        class Model(fr.nn.Module):
            scale: int = 1
            def build(self):
                self.p = fr.nn.Parameter(fr.Tensor([0.5], [1]))
                self.q = fr.nn.Parameter(fr.Tensor([0.2], [1]))
                self.alias = self.p
        m, g = Model(), fr.Generator(1)
        opt = fr.optim.Adam([m.p, m.q], lr=0.1)
        cp = fr.checkpoint.snapshot_training(m, optimizers={'main': opt}, generator=g)
        wrong_config = Model(scale=2)
        split_tie = Model()
        split_tie.alias = fr.nn.Parameter(fr.Tensor([0.5], [1]))
        for target, optimizer in [(m, fr.optim.Adam([m.q, m.p], lr=0.1)),
                (wrong_config, fr.optim.Adam(wrong_config.parameters(), lr=0.1)),
                (split_tie, fr.optim.Adam([split_tie.p, split_tie.q], lr=0.1))]:
            with self.subTest(target=target, optimizer=optimizer):
                before = [p.tensor().tolist() for p in target.parameters()]
                with self.assertRaises(ValueError):
                    cp.restore(target, optimizers={'main': optimizer}, generator=g)
                self.assertEqual([p.tensor().tolist() for p in target.parameters()], before)
        with self.assertRaises(ValueError):
            fr.checkpoint.snapshot_training(m, optimizers={'a': opt, 'b': opt}, generator=g)
        foreign = fr.nn.Parameter(fr.Tensor([0.3], [1]))
        with self.assertRaises(ValueError):
            fr.checkpoint.snapshot_training(m, optimizers={'x': fr.optim.Adam([foreign])}, generator=g)

    def test_checkpoint_io_preserves_exact_f64_bits_without_restore_coercion(self):
        import struct
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            bits = bytes.fromhex('010000000000f03f')
            header = json.dumps({'x': {'dtype': 'F64', 'shape': [1], 'data_offsets': [0, 8]}}).encode()
            (path / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bits)
            (path / 'checkpoint.json').write_text(json.dumps({'version': 1, 'step': 4, 'rng_seed': None, 'rng_offset': None}, indent=2))
            cp = fr.checkpoint.load(path)
            output = path / 'copy.safetensors'
            fr.save_safetensors(str(output), cp.tensors())
            saved = output.read_bytes()
            header_len = struct.unpack('<Q', saved[:8])[0]
            self.assertEqual(json.loads(saved[8:8 + header_len])['x']['dtype'], 'F64')
            self.assertEqual(saved[8 + header_len:], bits)
            cp.save(path / 'new')
            self.assertEqual(fr.checkpoint.load(path / 'new').step, 4)

    def test_adam_updates_shared_native_parameter(self):
        self.assertTrue(hasattr(fr.optim, 'Adam'), 'native Adam facade missing')
        p = fr.nn.Parameter(fr.Tensor([1.0], [1]))
        opt = fr.optim.Adam([p, p], lr=0.1)
        (p.tensor() * p.tensor()).sum().backward()
        opt.step()
        self.assertAlmostEqual(p.tensor().item(), 0.9, places=5)
        self.assertEqual(len(opt.params), 1)


if __name__ == '__main__':
    unittest.main()
