"""Deterministic public progress contracts, CPU only."""
import io
import unittest
import ferro as fr


class ProgressTests(unittest.TestCase):
    def test_weighted_epoch_logging_and_counts(self):
        self.assertTrue(hasattr(fr, 'Progress'), 'Progress must be public')
        stream = io.StringIO()
        p = fr.Progress(stream=stream, log_every=2)
        with p.epoch():
            for size, loss, accuracy in p.batches([(3, 2., 1.), (1, 6., 0.)]):
                p.log(loss=loss, accuracy=accuracy, weight=size)
                p.update()
        self.assertEqual((p.epoch_count, p.batch, p.global_batch, p.updates), (1, 2, 2, 2))
        self.assertEqual(p.metrics, {'loss': 3., 'accuracy': .75})
        self.assertEqual(stream.getvalue(),
                         'train epoch=1 batch=2/2 updates=2 loss=3.0000 accuracy=75.00%\n'
                         'train epoch=1 batch=2/2 updates=2 loss=3.0000 accuracy=75.00% complete\n')

    def test_unknown_lengths_phases_silent_and_reset(self):
        p = fr.Progress(enabled=False)
        with p.epoch():
            for _ in p.batches(x for x in range(2)):
                p.log(loss=2.)
        self.assertIsNone(p.total)
        with p.epoch(phase='validation'):
            for _ in p.batches([1]):
                p.log(accuracy=.5)
        self.assertEqual((p.epoch_count, p.batch, p.global_batch), (1, 1, 3))
        self.assertEqual(p.metrics, {'accuracy': .5})
        p.reset()
        self.assertEqual((p.epoch_count, p.global_batch, p.updates), (0, 0, 0))

    def test_failure_and_early_break_are_not_complete(self):
        for failing in (False, True):
            stream = io.StringIO()
            p = fr.Progress(stream=stream)
            try:
                with p.epoch():
                    for _ in p.batches([1, 2]):
                        if failing:
                            raise RuntimeError('failed batch')
                        break
            except RuntimeError:
                pass
            self.assertNotIn('complete', stream.getvalue())
            self.assertEqual(p.updates, 0)

    def test_scalars_only_and_cadence(self):
        p = fr.Progress(stream=io.StringIO(), log_every=2)
        with p.epoch():
            for i in p.batches(range(3)):
                self.assertEqual(p.should_log, i == 1)
                with self.assertRaisesRegex(TypeError, 'explicit Python scalars'):
                    p.log(loss=fr.Tensor([2.], [1]))
                p.log({'perplexity': 4.}, weight=2)
        self.assertEqual(p.metrics, {'perplexity': 4.})

    def test_trainer_explicit_tensor_logging_and_managed_steps(self):
        from ferro.training import Trainer
        stream = io.StringIO()
        p = fr.Progress(stream=stream, log_every=2, tensor_metrics=True)
        model = fr.nn.Linear(1, 1)
        optimizer = fr.optim.SGD(model.parameters(), lr=0.)
        def objective(model, batch):
            return (model(batch) * 0. + 2.).sum()
        trainer = Trainer(model, optimizer=optimizer, objective=objective, progress=p)
        reports = trainer.fit([fr.Tensor([1.], [1, 1])] * 3)
        self.assertEqual(p.updates, 3)
        self.assertEqual(p.metrics, {'loss': 2.})
        self.assertIn('loss=2.0000', stream.getvalue())
        self.assertIsInstance(reports[0]['loss'], fr.Tensor)
        trainer.evaluate([fr.Tensor([1.], [1, 1])] * 2)
        self.assertEqual((p.epoch_count, p.global_batch, p.updates), (1, 5, 3))
        self.assertIn('validation epoch=1', stream.getvalue())

    def test_trainer_scalar_sampling_and_default_no_conversion(self):
        from ferro.training import Trainer
        model = fr.nn.Module()
        tensors = [fr.Tensor([value], [1]) for value in (1., 4., 9.)]
        p = fr.Progress(stream=io.StringIO(), log_every=2, tensor_metrics=True)
        trainer = Trainer(model, train_step=lambda m, b: {'loss': b}, progress=p)
        reports = trainer.fit(tensors)
        self.assertEqual(p.metrics, {'loss': 4.})
        self.assertIs(reports[0]['loss'], tensors[0])
        self.assertEqual(p.updates, 0)
        p = fr.Progress(stream=io.StringIO())
        Trainer(model, train_step=lambda m, b: {'loss': b}, progress=p).fit(tensors)
        self.assertEqual(p.metrics, {})
        p = fr.Progress(stream=io.StringIO(), tensor_metrics=True, enabled=False)
        Trainer(model, train_step=lambda m, b: {'loss': b}, progress=p).fit(tensors)
        self.assertEqual(p.metrics, {})

    def test_trainer_failure_restores_mode_and_does_not_claim_success(self):
        from ferro.training import Trainer
        model = fr.nn.Module().eval()
        stream = io.StringIO()
        p = fr.Progress(stream=stream)
        def step(model, batch):
            raise RuntimeError('step failed')
        with self.assertRaisesRegex(RuntimeError, 'step failed'):
            Trainer(model, train_step=step, progress=p).fit([1])
        self.assertFalse(model.training)
        self.assertEqual(p.updates, 0)
        self.assertNotIn('complete', stream.getvalue())
        with p.epoch():
            list(p.batches([]))
        self.assertIn('complete', stream.getvalue())

    def test_validation_and_explicit_adapters(self):
        from ferro.training import Trainer
        for interval in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                fr.Progress(log_every=interval)
        p = fr.Progress(stream=io.StringIO())
        with self.assertRaises(RuntimeError):
            p.log(loss=1.)
        with p.epoch():
            with self.assertRaises(RuntimeError):
                p.reset()
            for weight in (0, -1, float('inf')):
                with self.assertRaises(ValueError):
                    p.log(loss=1., weight=weight)
            with self.assertRaises(ValueError):
                p.update(-1)
        trainer = Trainer(fr.nn.Module(), train_step=lambda m, b: {'loss': b}, progress=p,
                          progress_metrics=lambda b, r: {'loss': r['loss'].item()})
        trainer.fit([fr.Tensor([7.], [1])])
        self.assertEqual(p.metrics, {'loss': 7.})

    def test_trainer_custom_owns_updates_and_scalar_weighting(self):
        from ferro.training import Trainer, Progress
        self.assertIs(Progress, fr.Progress)
        p = Progress(stream=io.StringIO())
        def step(model, batch):
            p.update(2)
            return {'loss': float(batch), 'accuracy': 1. / batch}
        trainer = Trainer(fr.nn.Module(), train_step=step, progress=p,
                          progress_weight=lambda batch: batch)
        reports = trainer.fit([3, 1], epochs=2)
        self.assertEqual(len(reports), 4)
        self.assertEqual((p.epoch_count, p.global_batch, p.updates), (2, 4, 8))
        self.assertEqual(p.metrics, {'loss': 2.5, 'accuracy': .5})


if __name__ == '__main__':
    unittest.main()
