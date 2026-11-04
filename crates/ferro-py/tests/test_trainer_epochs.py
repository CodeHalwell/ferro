import unittest
from collections.abc import Iterator

import ferro as fr
from ferro.training import Trainer


class Identity(fr.nn.Module):
    def forward(self, value):
        return value


class CountingLoader(fr.data.DataLoader):
    def __iter__(self):
        self.iterations = getattr(self, 'iterations', 0) + 1
        return super().__iter__()


class TrainerEpochTests(unittest.TestCase):
    def test_shuffled_loader_starts_at_epoch_zero_and_iterates_once_per_epoch(self):
        dataset = fr.data.TensorDataset(fr.Tensor(list(range(8)), [8, 1]), fr.Tensor.from_i64(list(range(8)), [8]))
        for workers in (0, 2):
            with self.subTest(workers=workers):
                reference = CountingLoader(dataset, batch_size=3, shuffle=True, seed=7, num_workers=workers)
                expected = [batch[1].tolist() for _ in range(2) for batch in reference]
                loader = CountingLoader(dataset, batch_size=3, shuffle=True, seed=7, num_workers=workers)
                seen = []
                def step(model, batch):
                    seen.append(batch[1].tolist())
                    return {}
                trainer = Trainer(Identity(), train_step=step)
                reports = trainer.fit(loader, epochs=2)
                self.assertEqual(loader.epoch, 2)
                self.assertEqual(loader.iterations, 2)
                self.assertEqual(seen, expected)
                self.assertEqual(len(reports), 6)
                self.assertNotEqual(expected[:3], expected[3:])
                expected.extend(batch[1].tolist() for batch in reference)
                trainer.fit(loader, epochs=1)
                self.assertEqual(loader.epoch, 3)
                self.assertEqual(loader.iterations, 3)
                self.assertEqual(seen, expected)

    def test_reiterable_validation_has_no_discarded_iteration(self):
        class Source:
            def __init__(self):
                self.iterations = 0
            def __iter__(self):
                self.iterations += 1
                return iter([(self.iterations, 0), (self.iterations, 1)])
        source = Source()
        seen = []
        def step(model, batch):
            seen.append(batch)
            return {}
        Trainer(Identity(), train_step=step).fit(source, epochs=2)
        self.assertEqual(source.iterations, 2)
        self.assertEqual(seen, [(1, 0), (1, 1), (2, 0), (2, 1)])

    def test_one_shot_rejection_does_not_call_iter_or_consume(self):
        class OneShot(Iterator):
            def __init__(self):
                self.iterations = self.items = 0
            def __iter__(self):
                self.iterations += 1
                return self
            def __next__(self):
                self.items += 1
                return self.items
        source = OneShot()
        model = Identity()
        model.eval()
        with self.assertRaisesRegex(ValueError, 'reiterable'):
            Trainer(model, train_step=lambda model, batch: {}).fit(source, epochs=2)
        self.assertEqual((source.iterations, source.items), (0, 0))
        self.assertFalse(model.training)


if __name__ == '__main__':
    unittest.main()
