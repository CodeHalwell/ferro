import tempfile
import unittest
from pathlib import Path
import ferro as fr


class Buffers(fr.nn.Module):
    def build(self):
        self.p = fr.nn.Parameter(fr.Tensor([1.0], [1]))
        self.register_buffer('a', fr.Tensor([1.0], [1]))
        self.register_buffer('b', self.a)


class DerivedLinear(fr.nn.Linear):
    def forward(self, x):
        return super().forward(x) * self.in_features


class Intermediate(DerivedLinear):
    pass


class Leaf(Intermediate):
    pass


class Nested(fr.nn.Module):
    def build(self):
        self.child = Leaf(1, 1)
        self.register_buffer('running', fr.Tensor([1.0], [1]))


def snapshot(m, opts, rng):
    return fr.checkpoint.snapshot_training(m, optimizers=opts, generator=rng)


def bits(cp):
    # Safetensors writes typed owning inspection copies without float-list coercion.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'bits.safetensors'
        fr.save_safetensors(str(path), cp.tensors())
        return path.read_bytes()


class CheckpointSchemaTests(unittest.TestCase):
    def assert_atomic_rejection(self, source, target):
        sr, tr = fr.Generator(12), fr.Generator(42)
        so = {'main': fr.optim.Adam(source.parameters(), lr=0.01)}
        to = {'main': fr.optim.Adam(target.parameters(), lr=0.5)}
        for p in target.parameters():
            p.tensor().sum().backward()
        to['main'].step()
        to['main'].zero_grad()
        target.eval()
        for p in target.parameters():
            p.trainable = False
        cp = snapshot(source, so, sr)
        source_before = bits(snapshot(source, so, sr))
        cp_before = bits(cp)
        before = bits(snapshot(target, to, tr))
        modes = [m.training for m in target.modules()]
        params = list(target.parameters())
        buffers = list(target.named_buffers(remove_duplicate=False))
        with self.assertRaises(ValueError):
            cp.restore(target, optimizers=to, generator=tr)
        self.assertEqual(before, bits(snapshot(target, to, tr)))
        self.assertEqual(modes, [m.training for m in target.modules()])
        self.assertEqual(params, list(target.parameters()))
        self.assertEqual(buffers, list(target.named_buffers(remove_duplicate=False)))
        self.assertEqual(source_before, bits(snapshot(source, so, sr)))
        self.assertEqual(cp_before, bits(cp))

    def test_inherited_builtin_configuration_rejects_before_mutation(self):
        for cls in [DerivedLinear, Leaf, Nested]:
            with self.subTest(cls=cls):
                source, target = (cls(), cls()) if cls is Nested else (cls(1, 1), cls(1, 1))
                sl = source.child if cls is Nested else source
                tl = target.child if cls is Nested else target
                with fr.no_grad():
                    sl.weight.tensor().copy_(fr.Tensor([1.0], [1, 1]))
                    sl.bias.tensor().copy_(fr.Tensor([0.0], [1]))
                    tl.weight.tensor().copy_(fr.Tensor([1.0], [1, 1]))
                    tl.bias.tensor().copy_(fr.Tensor([0.0], [1]))
                tl.in_features = 2
                x = fr.Tensor([1.0], [1, 1])
                self.assertEqual(sl(x).item(), 1.0)
                self.assertEqual(tl(x).item(), 2.0)
                self.assert_atomic_rejection(source, target)

    def test_buffer_split_and_merge_reject_before_mutation(self):
        for tied_source in [True, False]:
            for saved_b in [1.0, 2.0]:
                with self.subTest(tied_source=tied_source, saved_b=saved_b):
                    source, target = Buffers(), Buffers()
                    split = target if tied_source else source
                    split.b = fr.Tensor([saved_b], [1])
                    self.assert_atomic_rejection(source, target)

    def test_cross_parameter_buffer_alias_split_and_merge(self):
        for reverse in [False, True]:
            with self.subTest(reverse=reverse):
                tied, split = Buffers(), Buffers()
                tied.a = tied.p.tensor()
                tied.b = tied.a
                self.assert_atomic_rejection(split, tied) if reverse else self.assert_atomic_rejection(tied, split)

    def test_distinct_python_wrappers_same_storage_are_tied(self):
        source, target = Buffers(), Buffers()
        source.b = source.a.reshape([1])
        self.assertIsNot(source.a, source.b)
        cp = snapshot(source, {}, fr.Generator(1))
        cp.restore(target, optimizers={}, generator=fr.Generator(2))
        with fr.no_grad():
            target.a.copy_(fr.Tensor([7.0], [1]))
        self.assertEqual(target.b.item(), 7.0)
        self.assertEqual(source.b.item(), 1.0)
        split = Buffers()
        split.b = fr.Tensor([1.0], [1])
        self.assert_atomic_rejection(source, split)

    def test_distinct_aliased_views_rejected_at_snapshot(self):
        for reshape in [False, True]:
            with self.subTest(reshape=reshape):
                source = Buffers()
                source.a = fr.Tensor([1., 2., 3., 4.], [2, 2])
                source.b = source.a.reshape([4]) if reshape else source.a.transpose(0, 1)
                with self.assertRaises(ValueError):
                    snapshot(source, {}, fr.Generator(1))

    def test_matching_nested_aliases_and_config_restore(self):
        class Model(fr.nn.Module):
            def build(self):
                self.child = Leaf(1, 1)
                self.buffers = Buffers()
                self.register_buffer('outer', self.buffers.a)
        source, target = Model(), Model()
        cp = snapshot(source, {}, fr.Generator(1))
        with fr.no_grad():
            source.outer.copy_(fr.Tensor([3.0], [1]))
        cp.restore(target, optimizers={}, generator=fr.Generator(2))
        self.assertEqual(target.outer.item(), 1.0)
        with fr.no_grad():
            target.buffers.a.copy_(fr.Tensor([7.0], [1]))
        self.assertEqual(target.outer.item(), 7.0)
        self.assertEqual(target.buffers.b.item(), 7.0)
        self.assertEqual(source.outer.item(), 3.0)


if __name__ == '__main__':
    unittest.main()
