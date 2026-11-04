"""Actual Torch alpha-dropout arithmetic, adjoints and native stream state."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import functional as F
import test_embedding_options as helpers

native = helpers.native


class AlphaModel(fr.nn.Module):
    probability: float = .3

    def build(self):
        self.elements = fr.nn.AlphaDropout(self.probability, seed=101)
        self.channels = fr.nn.FeatureAlphaDropout(.2, seed=107)
        self.alias = self.elements
        self.projection = fr.nn.Linear(3, 2)

    def forward(self, x):
        return self.projection(self.channels(self.alias(self.elements(x))).transpose(1, 2))


class AlphaDropoutTests(unittest.TestCase):
    same = helpers.EmbeddingOptionsTests.same

    def oracle_seed(self, name, shape, p, dtype, seed=13, layout=None):
        expected = getattr(F, name)(fr.ones(shape, dtype=dtype), p, True, seed=seed)
        expected = torch.tensor(expected.tolist(), dtype=getattr(torch, dtype)).reshape(shape)
        oracle = getattr(torch.nn.functional, name)
        with torch.random.fork_rng():
            for candidate in range(4096):
                torch.manual_seed(candidate)
                sample = torch.ones(shape, dtype=expected.dtype) if layout is None else torch.ones_like(layout)
                actual = oracle(sample, p, True)
                if torch.equal(actual > 0, expected > 0): return candidate
        self.fail('no matching small mask found in bounded actual Torch seed search')

    def test_actual_torch_matched_masks_strides_and_all_input_gradients(self):
        cases = [('alpha_dropout', ()), ('alpha_dropout', (6,)), ('alpha_dropout', (2, 3)),
                 ('feature_alpha_dropout', (2, 3)), ('feature_alpha_dropout', (2, 3, 4)),
                 ('feature_alpha_dropout', (1, 2, 2, 1, 3, 2, 2))]
        for name, shape in cases:
            for dtype in ('float32', 'float64'):
                for p in (.3, .5):
                    with self.subTest(name=name, shape=shape, dtype=dtype, p=p):
                        base = torch.linspace(-1.2, 1.4, math.prod(shape), dtype=getattr(torch, dtype)).reshape(shape)
                        x = (base.transpose(-1, -2) if len(shape) >= 2 else base).detach().requires_grad_(True)
                        nx = (native(base).transpose(-1, -2) if len(shape) >= 2 else native(base)).detach().requires_grad_(True)
                        seed = self.oracle_seed(name, x.shape, p, dtype, layout=x)
                        with torch.random.fork_rng():
                            torch.manual_seed(seed); expected = getattr(torch.nn.functional, name)(x, p, True)
                        actual = getattr(F, name)(nx, p, True, seed=13)
                        self.same(actual, expected)
                        multiplier = torch.linspace(.2, 1.3, x.numel(), dtype=x.dtype).reshape(x.shape)
                        (actual * native(multiplier)).sum().backward(); (expected * multiplier).sum().backward()
                        self.same(nx.grad, x.grad)

    def test_exact_staged_scaling_extreme_probabilities_and_nonfinite_values(self):
        for name in ('alpha_dropout', 'feature_alpha_dropout'):
            for dtype in ('float32', 'float64'):
                for p in (.123456789, .27981547357298786, .5, 1.-2**-24, 1.-2**-40, 1.-2**-53, 1.):
                    with self.subTest(name=name, dtype=dtype, p=p):
                        data = [[-0., 1., -2.], [float('nan'), float('inf'), -float('inf')]]
                        x = torch.tensor(data, dtype=getattr(torch, dtype), requires_grad=True); nx = native(x, True)
                        seed = self.oracle_seed(name, x.shape, p, dtype) if p != 1. else 0
                        with torch.random.fork_rng():
                            torch.manual_seed(seed); expected = getattr(torch.nn.functional, name)(x, p, True)
                        actual = getattr(F, name)(nx, p, True, seed=13)
                        observed = torch.tensor(actual.tolist(), dtype=x.dtype)
                        torch.testing.assert_close(observed, expected.detach(), atol=0., rtol=0., equal_nan=True)
                        actual.sum().backward(); expected.sum().backward()
                        torch.testing.assert_close(torch.tensor(nx.grad.tolist(), dtype=x.dtype), x.grad, atol=0., rtol=0.)
        dropped = F.alpha_dropout(fr.ones(2, 3), 1.-2**-24, True, seed=13)
        self.assertEqual(dropped.tolist(), [[-0.000244140625] * 3] * 2)

    def test_identity_endpoints_rank_empty_and_grad_mode(self):
        for name in ('alpha_dropout', 'feature_alpha_dropout'):
            for shape in ((), (2,), (0,), (2, 0), (0, 3, 4)):
                for p, training in ((0., True), (.3, False), (1., True), (.3, True)):
                    if name == 'feature_alpha_dropout' and len(shape) < 2 and math.prod(shape) and p == .3 and training: continue
                    for enabled in (False, True):
                        x = torch.ones(shape, requires_grad=True); nx = fr.ones(shape, requires_grad=True)
                        with torch.set_grad_enabled(enabled), (fr.enable_grad() if enabled else fr.no_grad()):
                            expected = getattr(torch.nn.functional, name)(x, p, training)
                            actual = getattr(F, name)(nx, p, training, seed=13)
                        self.assertEqual(actual is nx, expected is x)
                        self.assertEqual(actual.requires_grad, expected.requires_grad)
                        if p in (0., 1.) or not training or not x.numel(): self.same(actual, expected)
        empty = fr.zeros(sys.maxsize // 3, 0, 3)
        for name in ('alpha_dropout', 'feature_alpha_dropout'):
            self.assertIs(getattr(F, name)(empty, .4, True), empty)
        for value in (fr.ones([]), fr.ones(2)):
            with self.assertRaises(ValueError): F.feature_alpha_dropout(value, .4, True)
        value = fr.ones(2)
        self.assertIs(F.alpha_dropout(value), value)
        self.assertIs(F.feature_alpha_dropout(value), value)

    def test_functional_native_higher_derivative_and_jvp(self):
        for name in ('alpha_dropout', 'feature_alpha_dropout'):
            x = torch.tensor([[.3, -.8, .2], [.7, 1.1, -.4]], requires_grad=True); nx = native(x, True)
            seed = self.oracle_seed(name, x.shape, .3, 'float32')
            with torch.random.fork_rng():
                torch.manual_seed(seed); expected = getattr(torch.nn.functional, name)(x, .3, True)
            actual = getattr(F, name)(nx, .3, True, seed=13)
            ga = fr.autograd.grad((actual * actual).sum(), (nx,), create_graph=True)[0]
            ge = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[0]
            self.same(ga, ge)
            self.same(fr.autograd.grad(ga.sum(), (nx,))[0], torch.autograd.grad(ge.sum(), x)[0])
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                target = torch.func.jvp(lambda a: getattr(torch.nn.functional, name)(a, .3, True), (x,), (torch.ones_like(x),))
            result = fr.func.jvp(lambda a: getattr(F, name)(a, .3, True, seed=13), (nx,), (fr.ones(nx.shape),))
            self.same(result[0], target[0]); self.same(result[1], target[1])

    def test_module_stream_counts_eval_empty_and_failure_nonmutation(self):
        for kind, name, count in ((fr.nn.AlphaDropout, 'alpha_dropout', 24), (fr.nn.FeatureAlphaDropout, 'feature_alpha_dropout', 6)):
            module = kind(.3, seed=19); x = fr.ones(2, 3, 4, dtype='float64')
            for offset in (0, count):
                expected = getattr(F, name)(x, .3, True, seed=19, offset=offset)
                self.assertEqual(module(x).tolist(), expected.tolist()); self.assertEqual(module._offset, offset + count)
            before = module.rng_state.tolist(); module.eval(); self.assertIs(module(x), x)
            module.train(); module(fr.zeros(2, 3, 0)); self.assertEqual(module.rng_state.tolist(), before)
            with fr.no_grad(): module.rng_state.copy_(module._encode(19, 2**64 - 2))
            before = module.rng_state.tolist()
            with self.assertRaises(ValueError): module(x)
            self.assertEqual(before, module.rng_state.tolist())
            with fr.no_grad(): module.rng_state.copy_(module._encode(19, 0))
            with self.assertRaises(NotImplementedError): fr.capture(lambda: module(x))
            with self.assertRaises(NotImplementedError): fr.func.jvp(module, (x.float(),), (fr.ones(x.shape),))
            self.assertEqual(module._offset, 0)
            for p in (0., 1.):
                fixed = kind(p, seed=19); before = fixed.rng_state.tolist(); fixed(x)
                self.assertEqual(before, fixed.rng_state.tolist())

    def test_validation_and_capacity_precede_masks_or_state(self):
        for kind in (fr.nn.AlphaDropout, fr.nn.FeatureAlphaDropout):
            for options in (dict(p=-1.), dict(p=float('nan')), dict(p=2.), dict(inplace=True), dict(seed=-1)):
                with self.assertRaises((ValueError, TypeError, NotImplementedError)): kind(**options)
            module = kind(.3, seed=17); before = module.rng_state.tolist()
            with self.assertRaises(NotImplementedError): module(fr.ones(2, 3, dtype='int64'))
            huge = fr.ones(1, 1).expand(sys.maxsize // 4 + 1, 1)
            with patch('ferro.nn._alpha_dropout.Tensor.ones', side_effect=AssertionError('mask allocated before validation')):
                with self.assertRaisesRegex(ValueError, 'capacity'): module(huge)
            self.assertEqual(before, module.rng_state.tolist())
        for name in ('alpha_dropout', 'feature_alpha_dropout'):
            for options in (dict(seed=-1), dict(offset=-1), dict(offset=2**64 - 2), dict(training=1), dict(inplace=True)):
                with self.assertRaises((ValueError, TypeError, NotImplementedError)):
                    getattr(F, name)(fr.ones(2, 3), .3, **{'training': True, **options})

    def test_owning_checkpoint_restores_masks_aliases_optimizer_and_generator(self):
        def step(model, optimizer, rng):
            out = model(fr.Tensor.randn([2, 3, 4], generator=rng)); loss = out.square().mean()
            value = out.tolist(), loss.item(); loss.backward(); optimizer.step(); optimizer.zero_grad(); return value
        model, rng = AlphaModel(), fr.Generator(19); optimizer = fr.optim.AdamW(model.parameters(), lr=.01)
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = AlphaModel(), fr.Generator(1); target = fr.optim.AdamW(other.parameters(), lr=.7)
        saved.restore(other, optimizers={'main': target}, generator=other_rng)
        self.assertIs(other.alias, other.elements)
        self.assertEqual(expected, [step(other, target, other_rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'main': target}, generator=other_rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
        wrong = AlphaModel(probability=.4); before = [p.tensor().tolist() for p in wrong.parameters()]
        with self.assertRaises(ValueError): saved.restore(wrong, optimizers={'main': fr.optim.AdamW(wrong.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [p.tensor().tolist() for p in wrong.parameters()])


if __name__ == '__main__': unittest.main(verbosity=2)
