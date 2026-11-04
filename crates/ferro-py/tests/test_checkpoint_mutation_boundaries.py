"""Restore preflight covers immutable storage and active AD before commits."""
import unittest

import ferro as fr
from test_optimizer_groups_schedulers import plain


class Model(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.tensor([1., 2.]))
        self.bias = fr.nn.Parameter(fr.tensor([.5]))
        self.register_buffer('counter', fr.tensor(3, dtype='int64'))
        self.register_buffer('late', fr.tensor([3., 4.]))


class Empty(fr.nn.Module):
    pass


def setup(seed, count):
    model = Model(); optimizer = fr.optim.AdamW([{'params': [model.weight]}, {'params': [model.bias], 'lr': .03}], lr=.02)
    generator = fr.Generator(seed)
    for _ in range(count):
        optimizer.zero_grad(); (model.weight.square().sum() + model.bias.square().sum()).backward()
        optimizer.step(); optimizer.zero_grad(); fr.randn(2, generator=generator)
    return model, optimizer, generator


def snapshot(model, optimizer, generator):
    return fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)


def complete_state(model, optimizer, generator):
    values = {key: (value.dtype, value.shape, value.tolist()) for key, value in snapshot(model, optimizer, generator).tensors().items()}
    identities = [(name, value._id()) for name, value in model.named_parameters()]
    identities += [(name, value._id()) for name, value in model.named_buffers()]
    return values, identities, plain(optimizer.state_dict())


def structural_zero():
    source = fr.tensor([1., 2.], requires_grad=True)
    _, tangent = fr.func.jvp(lambda value: value * 2., (source,), (fr.ones(2),))
    zero = fr.autograd.grad(tangent.sum(), [source])[0]
    if not zero._is_structural_zero(): raise RuntimeError('fixture must use genuinely immutable zero storage')
    return zero


class CheckpointMutationBoundaryTests(unittest.TestCase):
    def test_late_immutable_buffer_rejects_before_all_training_state_and_generator_lock_changes(self):
        source, source_opt, source_rng = setup(7, 1); saved = snapshot(source, source_opt, source_rng)
        target, optimizer, generator = setup(11, 3)
        target.eval(); target.weight.requires_grad_(False); target.bias.trainable = False
        target.counter.fill_(19); target.late = structural_zero()
        before = complete_state(target, optimizer, generator)
        control, control_opt, control_rng = setup(1, 0)
        snapshot(target, optimizer, generator).restore(control, optimizers={'main': control_opt}, generator=control_rng)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            saved.restore(target, optimizers={'main': optimizer}, generator=generator)
        self.assertEqual(complete_state(target, optimizer, generator), before)
        self.assertEqual(fr.randn(7, generator=generator).tolist(), fr.randn(7, generator=control_rng).tolist())
        target.late = target.late.clone()
        saved.restore(target, optimizers={'main': optimizer}, generator=generator)
        self.assertEqual(complete_state(target, optimizer, generator)[0], complete_state(source, source_opt, source_rng)[0])

    def test_module_named_restore_rejects_late_immutable_buffer_without_partial_parameter_copy(self):
        source = Model(); target = Model()
        with fr.no_grad(): target.weight.fill_(9.)
        target.late = structural_zero()
        before = [(name, value._id(), value.tolist()) for name, value in (*target.named_parameters(), *target.named_buffers())]
        with self.assertRaisesRegex((ValueError, RuntimeError), 'immutable'):
            target.load_state_dict(source.state_dict())
        self.assertEqual([(name, value._id(), value.tolist()) for name, value in (*target.named_parameters(), *target.named_buffers())], before)
        target.late = target.late.clone(); target.load_state_dict(source.state_dict())
        self.assertEqual(target.weight.tolist(), source.weight.tolist())

    def test_active_forward_ad_rejects_training_and_optimizer_restore_before_any_mutation(self):
        source, source_opt, source_rng = setup(7, 1); saved = snapshot(source, source_opt, source_rng)
        model, optimizer, generator = setup(11, 3); model.eval()
        before = complete_state(model, optimizer, generator)
        control, control_opt, control_rng = setup(1, 0)
        snapshot(model, optimizer, generator).restore(control, optimizers={'main': control_opt}, generator=control_rng)
        actions = [lambda: saved.restore(model, optimizers={'main': optimizer}, generator=generator),
                   lambda: optimizer.load_state_dict(source_opt.state_dict()),
                   lambda: model.load_state_dict(source.state_dict())]
        for action in actions:
            def transformed(value): action(); return value * 2.
            with self.assertRaisesRegex((ValueError, RuntimeError), 'forward AD'):
                fr.func.jvp(transformed, (fr.tensor([.2, .3]),), (fr.ones(2),))
            self.assertEqual(complete_state(model, optimizer, generator), before)
        self.assertEqual(fr.randn(7, generator=generator).tolist(), fr.randn(7, generator=control_rng).tolist())

    def test_empty_targets_reject_active_forward_ad_and_capture_before_modes_or_rng_change(self):
        source = Empty(); target = Empty(); target.eval()
        saved = fr.checkpoint.snapshot_training(source, optimizers={}, generator=fr.Generator(7))
        generator = fr.Generator(11); control = fr.Generator(11)
        actions = [lambda: saved.restore(target, optimizers={}, generator=generator),
                   lambda: target.load_state_dict(source.state_dict())]
        for action in actions:
            def transformed(value): action(); return value * 2.
            with self.assertRaisesRegex((ValueError, RuntimeError), 'forward AD'):
                fr.func.jvp(transformed, (fr.tensor([.2, .3]),), (fr.ones(2),))
            with self.assertRaises((ValueError, RuntimeError)):
                fr.capture(lambda: transformed(fr.ones(2)))
            self.assertFalse(target.training)
        self.assertEqual(fr.randn(7, generator=generator).tolist(), fr.randn(7, generator=control).tolist())
        saved.restore(target, optimizers={}, generator=generator)
        self.assertTrue(target.training)


if __name__ == '__main__':
    unittest.main()
