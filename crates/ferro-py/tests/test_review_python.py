"""Adversarial CPU regressions for registration, managed AD and iterator ownership."""
import io
import unittest
import ferro as fr
from ferro.nn import Module, Parameter
from ferro.optim import SGD
from ferro.training import Trainer


class Scale(Module):
    def build(self):
        self._weight = Parameter(fr.Tensor([1.], [1]))

    def forward(self, x):
        return self._weight.tensor() * x


class PrivateParent(Module):
    def build(self):
        self._child = Scale()

    def forward(self, x):
        return self._child(x)


class ReviewFixes(unittest.TestCase):
    def test_private_registration_aliases_replacement_deletion_and_cycles(self):
        m = PrivateParent()
        m._alias = m._child
        self.assertEqual(list(dict(m.named_parameters())), ['_child._weight'])
        self.assertEqual(list(dict(m.named_parameters(remove_duplicate=False))), ['_child._weight', '_alias._weight'])
        m.eval()
        self.assertFalse(m._child.training)
        with self.assertRaises(ValueError):
            m._child._parent = m
        replacement = Parameter(fr.Tensor([3.], [1]))
        m._child._weight = replacement
        self.assertIs(list(m.parameters())[0], replacement)
        del m._alias
        del m._child._weight
        self.assertEqual(list(m.parameters()), [])

    def test_private_children_are_locked_and_mixed_modes_restore_on_exception(self):
        m = PrivateParent().eval()
        m._child.train()
        def step(model, batch):
            self.assertTrue(model.training)
            self.assertTrue(model._child.training)
            model._child._weight = Parameter(fr.Tensor.ones([1]))
        with self.assertRaisesRegex(RuntimeError, 'structure'):
            Trainer(m, train_step=step).fit([1])
        self.assertFalse(m.training)
        self.assertTrue(m._child.training)
        self.assertFalse(m._structure_locked)
        self.assertFalse(m._child._structure_locked)

    def test_managed_no_grad_records_updates_and_restores_between_batches(self):
        m = Scale().eval()
        p = fr.Progress(enabled=False)
        def source():
            for _ in range(2):
                self.assertFalse(fr.is_grad_enabled())
                yield fr.Tensor([2.], [1])
        def objective(model, batch):
            self.assertTrue(fr.is_grad_enabled())
            return model(batch).sum()
        with fr.no_grad():
            reports = Trainer(m, optimizer=SGD(m.parameters(), lr=.1), objective=objective, progress=p).fit(source())
            self.assertFalse(fr.is_grad_enabled())
        self.assertTrue(fr.is_grad_enabled())
        self.assertFalse(m.training)
        self.assertAlmostEqual(m._weight.tensor().item(), .6, places=6)
        self.assertEqual(p.updates, 2)
        self.assertTrue(all(r['loss'].requires_grad for r in reports))

    def test_managed_exception_restores_recording_modes_and_locks(self):
        for enabled in (False, True):
            m = PrivateParent().eval()
            m._child.train()
            def objective(model, batch):
                self.assertTrue(fr.is_grad_enabled())
                raise RuntimeError('objective failed')
            with fr.enable_grad() if enabled else fr.no_grad():
                with self.assertRaisesRegex(RuntimeError, 'objective failed'):
                    Trainer(m, optimizer=SGD(m.parameters(), lr=.1), objective=objective).fit([1])
                self.assertEqual(fr.is_grad_enabled(), enabled)
            self.assertFalse(m.training)
            self.assertTrue(m._child.training)
            self.assertFalse(m._child._structure_locked)

    def test_managed_rejects_detached_and_constant_losses_without_update(self):
        for detached in (False, True):
            m = Scale()
            p = fr.Progress(stream=io.StringIO())
            def objective(model, batch):
                return model(batch).sum().detach() if detached else fr.Tensor.ones([1])
            trainer = Trainer(m, optimizer=SGD(m.parameters(), lr=.1, momentum=.9), objective=objective, progress=p)
            with fr.no_grad():
                with self.assertRaisesRegex(ValueError, 'differentiable|requires_grad'):
                    trainer.fit([fr.Tensor.ones([1])])
                self.assertFalse(fr.is_grad_enabled())
            self.assertEqual(m._weight.tensor().item(), 1.)
            self.assertEqual(p.updates, 0)
            self.assertNotIn('complete', p.stream.getvalue())
            self.assertFalse(trainer.evaluate([fr.Tensor.ones([1])])[0]['loss'].requires_grad)

    def test_custom_step_keeps_grad_and_update_ownership(self):
        m = Scale()
        def step(model, batch):
            self.assertFalse(fr.is_grad_enabled())
            return {'loss': model(batch).sum()}
        with fr.no_grad():
            result = Trainer(m, train_step=step).fit([fr.Tensor.ones([1])])
            self.assertFalse(fr.is_grad_enabled())
        self.assertFalse(result[0]['loss'].requires_grad)
        self.assertEqual(m._weight.tensor().item(), 1.)

    def test_retained_iterator_no_consumption_after_exit_or_next_epoch(self):
        for started in (False, True):
            for later in (False, True):
                with self.subTest(started=started, later=later):
                    seen = []
                    def source():
                        for value in range(3):
                            seen.append(value)
                            yield value
                    p = fr.Progress(enabled=False)
                    with p.epoch():
                        old = p.batches(source())
                        if started:
                            self.assertEqual(next(old), 0)
                    before = list(seen)
                    if later:
                        with p.epoch():
                            current = p.batches([10, 11])
                            self.assertEqual(next(current), 10)
                            counts = (p.batch, p.global_batch)
                            with self.assertRaises((RuntimeError, StopIteration)):
                                next(old)
                            self.assertEqual((p.batch, p.global_batch), counts)
                            with self.assertRaises(RuntimeError):
                                next(p.batches([99]))
                            self.assertEqual(list(current), [11])
                    else:
                        counts = (p.batch, p.global_batch)
                        with self.assertRaises((RuntimeError, StopIteration)):
                            next(old)
                        self.assertEqual((p.batch, p.global_batch), counts)
                    self.assertEqual(seen, before)

    def test_delayed_first_next_in_later_epoch_never_starts_source(self):
        seen = []
        def source():
            seen.append('started')
            yield 1
        p = fr.Progress(enabled=False)
        with p.epoch():
            old = p.batches(source())
        with p.epoch():
            with self.assertRaises((RuntimeError, StopIteration)):
                next(old)
            self.assertEqual(seen, [])
            self.assertEqual((p.batch, p.global_batch), (0, 0))
            self.assertEqual(list(p.batches([2])), [2])

    def test_source_exception_releases_iterator_without_completion(self):
        p = fr.Progress(stream=io.StringIO())
        seen = []
        def source():
            seen.append(1)
            yield 1
            seen.append('failed')
            raise RuntimeError('source failed')
        with self.assertRaisesRegex(RuntimeError, 'source failed'):
            with p.epoch():
                iterator = p.batches(source())
                list(iterator)
        self.assertEqual(seen, [1, 'failed'])
        self.assertEqual((p.batch, p.global_batch), (1, 1))
        self.assertNotIn('complete', p.stream.getvalue())
        with p.epoch():
            self.assertEqual(list(p.batches([2])), [2])

    def test_close_old_iterator_does_not_release_new_iterator(self):
        p = fr.Progress(enabled=False)
        with p.epoch():
            old = p.batches([1, 2])
            next(old)
        with p.epoch():
            current = p.batches([3, 4])
            next(current)
            old.close()
            with self.assertRaises(RuntimeError):
                next(p.batches([5]))
            self.assertEqual(list(current), [4])

    def test_close_break_and_exception_do_not_consume_extra_source(self):
        for action in ('close', 'break', 'exception'):
            seen = []
            def source():
                for value in range(3):
                    seen.append(value)
                    yield value
            p = fr.Progress(stream=io.StringIO())
            try:
                with p.epoch():
                    iterator = p.batches(source())
                    for _ in iterator:
                        if action == 'exception':
                            raise RuntimeError('batch failed')
                        if action == 'close':
                            iterator.close()
                        break
            except RuntimeError:
                pass
            with self.assertRaises((RuntimeError, StopIteration)):
                next(iterator)
            self.assertEqual(seen, [0])
            self.assertEqual((p.batch, p.global_batch), (1, 1))
            self.assertNotIn('complete', p.stream.getvalue())

    def test_unstarted_iterator_created_outside_epoch_is_rejected(self):
        p = fr.Progress(enabled=False)
        with self.assertRaises(RuntimeError):
            iterator = p.batches([1])
            with p.epoch():
                next(iterator)

    def test_update_outside_epoch_does_not_change_counters(self):
        p = fr.Progress(enabled=False)
        with self.assertRaises(RuntimeError):
            p.update()
        self.assertEqual(p.updates, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
