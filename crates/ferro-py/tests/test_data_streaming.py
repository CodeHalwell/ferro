"""Lazy Python data pipelines, typed collation and input gradients."""
from collections import namedtuple
import itertools
import unittest

import torch
import ferro as fr
from ferro import data


class DataStreaming(unittest.TestCase):
    def same(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
            self.assertEqual(actual.shape, list(expected.shape))
            values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
            torch.testing.assert_close(values, expected.detach(), rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.same(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.same(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_nested_collation_types_against_torch(self):
        Row = namedtuple('Row', ('features', 'labels'))
        samples = [{'row': Row(fr.tensor([float(i), i + .5]), i),
                    'id': i + (1 << 53), 'weight': i / 3., 'text': str(i), 'flags': [i % 2 == 0]}
                   for i in range(5)]
        refs = [{'row': Row(torch.tensor([float(i), i + .5]), i),
                 'id': i + (1 << 53), 'weight': i / 3., 'text': str(i), 'flags': [i % 2 == 0]}
                for i in range(5)]
        actual = list(data.DataLoader(samples, batch_size=2))
        expected = list(torch.utils.data.DataLoader(refs, batch_size=2))
        self.assertEqual(len(actual), 3)
        for left, right in zip(actual, expected):
            self.same(left, right)

    def test_custom_map_dataset_is_lazy_and_preserves_input_gradients(self):
        source = fr.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]], requires_grad=True)
        reference = torch.tensor(source.tolist(), requires_grad=True)
        class Samples(data.Dataset):
            def __init__(self):
                self.calls = []
            def __len__(self):
                return 5
            def __getitem__(self, index):
                self.calls.append(index)
                return source[index]
        dataset = Samples()
        loader = data.DataLoader(dataset, 2)
        iterator = iter(loader)
        self.assertEqual(dataset.calls, [])
        first = next(iterator)
        self.assertEqual(dataset.calls, [0, 1])
        first.pow(2).sum().backward()
        reference[:2].square().sum().backward()
        self.same(source.grad, reference.grad)
        second = next(iterator)
        self.assertEqual(dataset.calls, [0, 1, 2, 3])
        self.same(second, reference.detach()[2:4])
        iterator.close()
        self.assertEqual(dataset.calls, [0, 1, 2, 3])

    def test_native_tensor_dataset_lazy_batches_preserve_worker_order(self):
        dataset = data.TensorDataset(fr.arange(30, dtype='float32').reshape([10, 3]), fr.arange(10))
        left = data.DataLoader(dataset, 3, shuffle=True, seed=47)
        right = data.DataLoader(dataset, 3, shuffle=True, seed=47, num_workers=2)
        for epoch in range(2):
            actual, expected = list(left), list(right)
            self.assertTrue(all(type(batch) is tuple for batch in actual + expected))
            self.assertEqual([[v.tolist() for v in batch] for batch in actual], [[v.tolist() for v in batch] for batch in expected])
        from unittest.mock import patch
        with patch('ferro.data._load_epoch', side_effect=AssertionError('materialized whole native epoch')):
            first = next(iter(data.DataLoader(dataset, 2)))
        self.assertEqual(first[1].tolist(), [0, 1])

    def test_infinite_and_finite_iterable_datasets_consume_one_batch(self):
        class Stream(data.IterableDataset):
            def __init__(self, stop=None):
                self.calls, self.stop = 0, stop
            def __iter__(self):
                for i in itertools.count():
                    if self.stop is not None and i == self.stop:
                        return
                    self.calls += 1
                    yield i
        infinite = Stream()
        loader = data.DataLoader(infinite, batch_size=3)
        iterator = iter(loader)
        self.assertEqual(infinite.calls, 0)
        self.assertEqual(next(iterator).tolist(), [0, 1, 2])
        self.assertEqual(infinite.calls, 3)
        self.assertEqual(next(iterator).tolist(), [3, 4, 5])
        self.assertEqual(infinite.calls, 6)
        iterator.close()
        with self.assertRaises(TypeError):
            len(loader)
        self.assertEqual([batch.tolist() for batch in data.DataLoader(Stream(5), 2)], [[0, 1], [2, 3], [4]])
        self.assertEqual([batch.tolist() for batch in data.DataLoader(Stream(5), 2, drop_last=True)], [[0, 1], [2, 3]])

    def test_no_batching_custom_collation_sampler_and_batch_sampler(self):
        dataset = [{'x': i, 'pair': (str(i), i * 2)} for i in range(5)]
        actual = list(data.DataLoader(dataset, batch_size=None))
        expected = list(torch.utils.data.DataLoader(dataset, batch_size=None))
        self.same(actual, expected)
        self.assertEqual(list(data.DataLoader(range(5), 2, collate_fn=tuple)), [(0, 1), (2, 3), (4,)])
        order = [4, 1, 4, 0]
        self.same(list(data.DataLoader(range(5), 3, sampler=order)),
                  list(torch.utils.data.DataLoader(range(5), 3, sampler=order)))
        self.same(list(data.DataLoader(range(5), batch_sampler=[[4, 1], [2], [0, 3]])),
                  list(torch.utils.data.DataLoader(range(5), batch_sampler=[[4, 1], [2], [0, 3]])))

    def test_subset_concat_random_split_and_seeded_epoch_order(self):
        a, b = data.Subset(range(9), [6, 2, 3]), data.Subset(range(8), [5, 1])
        merged = data.ConcatDataset([[], a, [], b, []])
        expected = torch.utils.data.ConcatDataset([[], torch.utils.data.Subset(range(9), [6, 2, 3]), [],
                                                  torch.utils.data.Subset(range(8), [5, 1]), []])
        self.assertEqual([merged[i] for i in range(-len(merged), len(merged))],
                         [expected[i] for i in range(-len(expected), len(expected))])
        with self.assertRaises(IndexError):
            merged[len(merged)]
        self.assertEqual(a[:2], [6, 2])
        self.assertEqual(len(a + b), 5)
        splits = data.random_split(range(11), [.5, .3, .2], seed=7)
        self.assertEqual([len(split) for split in splits], [6, 3, 2])
        indices = [i for split in splits for i in split.indices]
        self.assertEqual(sorted(indices), list(range(11)))
        self.assertEqual([split.indices for split in splits], [split.indices for split in data.random_split(range(11), [.5, .3, .2], seed=7)])
        for seed in [7, 31]:
            left = data.random_split(range(11), [4, 7], generator=fr.Generator(seed))
            right = data.random_split(range(11), [4, 7], generator=fr.Generator(seed))
            self.assertEqual([s.indices for s in left], [s.indices for s in right])
        first, second = data.DataLoader(range(20), 3, shuffle=True, seed=7), data.DataLoader(range(20), 3, shuffle=True, seed=7)
        orders = [[i for batch in first for i in batch.tolist()] for _ in range(2)]
        self.assertNotEqual(*orders)
        self.assertEqual(orders, [[i for batch in second for i in batch.tolist()] for _ in range(2)])

    def test_invalid_requests_and_callback_errors_propagate(self):
        for options in ({'batch_size': 0}, {'batch_size': True}, {'batch_size': None, 'drop_last': True},
                        {'shuffle': True, 'sampler': [0]}, {'batch_sampler': [[0]], 'batch_size': 2}):
            with self.assertRaises((ValueError, TypeError)):
                data.DataLoader([1], **options)
        with self.assertRaises(NotImplementedError):
            data.DataLoader([1], num_workers=1)
        with self.assertRaises(ValueError):
            data.DataLoader(data.IterableDataset(), shuffle=True)
        for batch in ([None, None], [{'x': 1}, {'y': 2}], [[1], [2, 3]], [fr.ones(2), fr.ones(3)]):
            with self.assertRaises((ValueError, TypeError)):
                data.default_collate(batch)
        class Broken(data.Dataset):
            def __len__(self):
                return 3
            def __getitem__(self, index):
                if index == 1:
                    raise RuntimeError('sample failure')
                return index
        iterator = iter(data.DataLoader(Broken(), 2))
        with self.assertRaisesRegex(RuntimeError, 'sample failure'):
            next(iterator)
        for lengths in ([1, 1], [-1, 4], [float('nan'), 1], [2.5, .5]):
            with self.assertRaises(ValueError):
                data.random_split(range(3), lengths)


if __name__ == '__main__':
    unittest.main()
