"""Executable PyG utility references and independent split leakage checks."""
import collections
import random
import tracemalloc
import unittest
import warnings

import torch
from torch_geometric.data import Data as TorchData
from torch_geometric.transforms import RandomLinkSplit as TorchSplit
from torch_geometric.utils import negative_sampling as torch_negative, batched_negative_sampling as torch_batched

import ferro as fr
from ferro._graph_data import Data
from ferro._graph_link import negative_sampling, batched_negative_sampling, RandomLinkSplit


def index(rows):
    value = torch.as_tensor(rows, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def tensor(value, grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


def pairs(edge):
    return list(zip(*edge.tolist()))


def positive(data, undirected=False):
    raw = [p for p, label in zip(pairs(data.edge_label_index), data.edge_label.tolist()) if label > 0]
    return {tuple(sorted(p)) if undirected else p for p in raw}


class LinkUtilityTests(unittest.TestCase):
    def test_complete_complements_match_executable_pyg(self):
        cases = [(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, False),
                 (torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]), 4, True),
                 (torch.tensor([[0, 1], [0, 2]]), (2, 3), False)]
        for edge, size, undirected in cases:
            with self.subTest(size=size, undirected=undirected):
                actual = negative_sampling(index(edge), size, 100, force_undirected=undirected, seed=3)
                reference = torch_negative(edge, size, 100, force_undirected=undirected)
                self.assertEqual(set(pairs(actual)), set(pairs(reference)))
                self.assertEqual(actual.dtype, 'int64')
                self.assertEqual(len(pairs(actual)), len(set(pairs(actual))))

    def test_exact_counts_dense_duplicates_and_large_sparse_population(self):
        edge = index([[0, 0, 0, 1, 2], [1, 1, 0, 2, 2]])
        for method in ('sparse', 'dense'):
            a = negative_sampling(edge, 4, 7, method, seed=17)
            b = negative_sampling(edge, 4, 7, method, seed=17)
            self.assertEqual(a.tolist(), b.tolist())
            self.assertEqual(a.shape, [2, 7])
            self.assertTrue(all(s != d and (s, d) not in set(pairs(edge)) for s, d in pairs(a)))
        all_pairs = [(s, d) for s in range(4) for d in range(4) if s != d and (s, d) != (3, 0)]
        dense = index([[s for s, _ in all_pairs], [d for _, d in all_pairs]])
        self.assertEqual(pairs(negative_sampling(dense, 4, 20)), [(3, 0)])
        tracemalloc.start()
        large = negative_sampling(index([[0], [1]]), 10**10, 6, seed=9)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(large.shape, [2, 6])
        self.assertLess(peak, 2_000_000)
        self.assertTrue(all(0 <= s < 10**10 and 0 <= d < 10**10 and s != d for s, d in pairs(large)))
        self.assertEqual(negative_sampling(index([[], []]), 0).shape, [2, 0])
        self.assertEqual(negative_sampling(index([[0], [0]]), 1, 5).shape, [2, 0])

    def test_complement_rank_sampler_against_exhaustive_small_graphs(self):
        rng = random.Random(103)
        for n in range(7):
            for undirected in (False, True):
                universe = {(s, d) for s in range(n) for d in range(n) if s != d}
                present = [p for p in sorted(universe) if rng.random() < .55]
                if present:
                    present += present[:2]
                edge = index([[s for s, _ in present], [d for _, d in present]])
                forbidden = set(present) | ({(d, s) for s, d in present} if undirected else set())
                expected = universe - forbidden
                actual = negative_sampling(edge, n, 1000, force_undirected=undirected, seed=5)
                self.assertEqual(set(pairs(actual)), expected)
                self.assertEqual(len(pairs(actual)), len(expected))
                bounded = negative_sampling(edge, n, 5, force_undirected=undirected, seed=11)
                self.assertEqual(len(pairs(bounded)), min(4 if undirected else 5, len(expected)))
                self.assertTrue(set(pairs(bounded)) <= expected)
        for ns, nt in ((0, 3), (3, 0), (1, 1), (2, 5), (5, 2)):
            universe = {(s, d) for s in range(ns) for d in range(nt)}
            present = [p for p in sorted(universe) if rng.random() < .4]
            edge = index([[s for s, _ in present], [d for _, d in present]])
            self.assertEqual(set(pairs(negative_sampling(edge, (ns, nt), 100, seed=3))), universe - set(present))

    def test_batched_sampling_matches_pyg_and_handles_noncontiguous_ids(self):
        edge = torch.tensor([[0, 2, 3, 5], [1, 0, 4, 3]])
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        actual = batched_negative_sampling(index(edge), index(batch), 100, seed=2)
        expected = torch_batched(edge, batch, 100)
        self.assertEqual(set(pairs(actual)), set(pairs(expected)))
        batch = [4, 9, 4, 9, 4]
        edge = index([[0, 1], [2, 3]])
        actual = batched_negative_sampling(edge, index(batch), 100, seed=2)
        wanted = {(s, d) for s in range(5) for d in range(5) if s != d and batch[s] == batch[d]} - set(pairs(edge))
        self.assertEqual(set(pairs(actual)), wanted)
        bs, bt = [2, 5, 2], [5, 2, 2, 5]
        edge = index([[0, 1], [1, 0]])
        actual = batched_negative_sampling(edge, (index(bs), index(bt)), 100)
        wanted = {(s, d) for s in range(3) for d in range(4) if bs[s] == bt[d]} - set(pairs(edge))
        self.assertEqual(set(pairs(actual)), wanted)

    def test_split_no_holdout_matches_pyg_values_and_gradients(self):
        edge = torch.tensor([[0, 1, 1, 3], [1, 2, 3, 0]])
        attr = torch.tensor([[.2], [.4], [-.7], [.9]])
        native_attr, reference_attr = tensor(attr, True), attr.clone().requires_grad_(True)
        data = Data(x=tensor(torch.ones(5, 2), True), edge_index=index(edge), edge_attr=native_attr, num_nodes=5)
        actual = RandomLinkSplit(0, 0, neg_sampling_ratio=0, seed=4)(data)[0]
        reference = TorchSplit(0, 0, neg_sampling_ratio=0)(TorchData(x=torch.ones(5, 2), edge_index=edge, edge_attr=reference_attr, num_nodes=5))[0]
        amap = dict(zip(pairs(actual.edge_index), actual.edge_attr.tolist()))
        bmap = dict(zip(pairs(reference.edge_index), reference.edge_attr.tolist()))
        self.assertEqual(amap, bmap)
        self.assertEqual(set(pairs(actual.edge_label_index)), set(pairs(reference.edge_label_index)))
        (actual.edge_attr * actual.edge_attr).sum().backward()
        reference.edge_attr.square().sum().backward()
        torch.testing.assert_close(torch.tensor(native_attr.grad.tolist()), reference_attr.grad)

    def test_duplicate_reverse_groups_disjoint_and_negative_leakage(self):
        raw = [(0, 1), (1, 0), (0, 1), (1, 2), (2, 3), (3, 2), (3, 4), (4, 5), (5, 0), (2, 5)]
        edge = index([[s for s, _ in raw], [d for _, d in raw]])
        attr = tensor([[i + 1.] for i in range(len(raw))], True)
        data = Data(x=tensor(torch.arange(12).reshape(6, 2)), edge_index=edge, edge_attr=attr, num_nodes=6)
        split = RandomLinkSplit(1, 2, is_undirected=True, disjoint_train_ratio=1, neg_sampling_ratio=1., seed=37)
        train, val, test = split(data)
        ps = [positive(d, True) for d in (train, val, test)]
        self.assertTrue(all(not (ps[i] & ps[j]) for i in range(3) for j in range(i)))
        all_positive = {tuple(sorted(p)) for p in raw}
        negatives = []
        for part, heldout in ((train, ps[0] | ps[1] | ps[2]), (val, ps[1] | ps[2]), (test, ps[2])):
            adjacency = set(pairs(part.edge_index))
            self.assertTrue(all((d, s) in adjacency for s, d in adjacency))
            self.assertFalse({tuple(sorted(p)) for p in adjacency} & heldout)
            neg = [tuple(sorted(p)) for p, label in zip(pairs(part.edge_label_index), part.edge_label.tolist()) if label == 0]
            self.assertFalse(set(neg) & all_positive)
            self.assertTrue(all(s != d for s, d in neg))
            negatives.extend(neg)
        self.assertEqual(len(negatives), len(set(negatives)))
        again = RandomLinkSplit(1, 2, is_undirected=True, disjoint_train_ratio=1, seed=37)(data)
        self.assertEqual([p.edge_label_index.tolist() for p in again], [p.edge_label_index.tolist() for p in (train, val, test)])
        counts = collections.Counter(int(row[0]) - 1 for part in (train, val, test) for row in part.edge_attr.tolist())
        loss = sum((part.edge_attr * part.edge_attr).sum() for part in (train, val, test))
        loss.backward()
        self.assertEqual(attr.grad.tolist(), [[2. * (i + 1) * counts[i]] for i in range(len(raw))])

    def test_class_labels_split_labels_and_limited_complements(self):
        edge = index([[0, 1, 2, 3], [1, 2, 3, 0]])
        data = Data(edge_index=edge, edge_label=index([0, 1, 2, 3]), num_nodes=4, field_roles={'edge_label': 'edge'})
        result = RandomLinkSplit(1, 1, split_labels=True, neg_sampling_ratio=1., seed=2)(data)
        labels = {p: v + 1 for p, v in zip(pairs(edge), [0, 1, 2, 3])}
        for part in result:
            self.assertEqual(part.pos_edge_label.dtype, 'int64')
            self.assertEqual(part.neg_edge_label.dtype, 'int64')
            self.assertEqual(part.pos_edge_label.tolist(), [labels[p] for p in pairs(part.pos_edge_label_index)])
            self.assertTrue(all(v == 0 for v in part.neg_edge_label.tolist()))
            self.assertTrue(part.validate())
        dense_pairs = [(s, d) for s in range(3) for d in range(3) if s != d and (s, d) != (2, 1)]
        dense = Data(edge_index=index([[s for s, _ in dense_pairs], [d for _, d in dense_pairs]]), num_nodes=3)
        result = RandomLinkSplit(1, 1, neg_sampling_ratio=5., seed=3)(dense)
        self.assertEqual(sum(part.edge_label.tolist().count(0.) for part in result), 1)

    def test_class_label_shift_is_per_split_like_actual_pyg(self):
        cases = [(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)], False),
                 (3, [(s, d) for s in range(3) for d in range(3) if s != d and (s, d) != (2, 1)], True)]
        for nodes, original, negative_train in cases:
            edge = torch.tensor([[s for s, _ in original], [d for _, d in original]])
            labels = torch.arange(len(original), dtype=torch.int64)
            label_by_pair = dict(zip(original, labels.tolist()))
            for separate in (False, True):
                native = Data(edge_index=index(edge), edge_label=index(labels), num_nodes=nodes, field_roles={'edge_label': 'edge'})
                reference = TorchData(edge_index=edge, edge_label=labels, num_nodes=nodes)
                options = dict(num_val=1, num_test=1, add_negative_train_samples=negative_train, split_labels=separate)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    outputs = (RandomLinkSplit(**options, seed=7)(native), TorchSplit(**options)(reference))
                for implementation, parts in enumerate(outputs):
                    for part in parts:
                        if separate:
                            negative = pairs(part.neg_edge_label_index) if 'neg_edge_label_index' in part else []
                            positive_pairs = pairs(part.pos_edge_label_index)
                            positive_labels = part.pos_edge_label.tolist()
                            if not negative:
                                self.assertNotIn('neg_edge_label', part)
                        else:
                            all_pairs = pairs(part.edge_label_index)
                            negative = [p for p in all_pairs if p not in label_by_pair]
                            positive_pairs = [p for p in all_pairs if p in label_by_pair]
                            positive_labels = [v for p, v in zip(all_pairs, part.edge_label.tolist()) if p in label_by_pair]
                        with self.subTest(implementation=implementation, negative_train=negative_train, separate=separate):
                            self.assertEqual(positive_labels, [label_by_pair[p] + int(bool(negative)) for p in positive_pairs])
                if not negative_train:
                    train = outputs[0][0]
                    labels_out = train.pos_edge_label if separate else train.edge_label
                    pairs_out = train.pos_edge_label_index if separate else train.edge_label_index
                    self.assertEqual(labels_out.tolist(), [label_by_pair[p] for p in pairs(pairs_out)])

    def test_invalid_requests_reject_without_mutating_source(self):
        edge = index([[0, 1], [1, 2]])
        for kwargs in ({'num_nodes': 2}, {'num_neg_samples': -1}, {'num_neg_samples': True}, {'method': 'bogus'}, {'force_undirected': 1}):
            with self.assertRaises((ValueError, TypeError)):
                negative_sampling(edge, **kwargs)
        with self.assertRaises(ValueError):
            negative_sampling(tensor(torch.empty(2, 0)), 3)
        with self.assertRaises(ValueError):
            batched_negative_sampling(edge, index([0, 1, 1]))
        data = Data(edge_index=edge, num_nodes=3)
        before = data.edge_index.tolist()
        with self.assertRaises(ValueError):
            RandomLinkSplit(1, 1)(data)
        with self.assertRaises(NotImplementedError):
            RandomLinkSplit(edge_types=('a', 'r', 'b'))
        with self.assertRaises(ValueError):
            RandomLinkSplit(key='edge')
        with self.assertRaises(ValueError):
            RandomLinkSplit(disjoint_train_ratio=1.)(data)
        self.assertEqual(data.edge_index.tolist(), before)


if __name__ == '__main__':
    unittest.main()
