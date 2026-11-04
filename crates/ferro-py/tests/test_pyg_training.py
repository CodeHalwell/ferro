"""Integrated public graph batching, classifier learning and loader continuation."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import ferro as fr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples'))
import pyg_parity as example


class GraphTrainingTests(unittest.TestCase):
    def test_all_graph_kinds_learn_and_restart_in_fresh_process(self):
        for kind in example.KINDS:
            with self.subTest(kind=kind):
                result = example.demonstrate(kind)
                self.assertEqual(result['restart'], 'bit-exact-at-epoch-boundary')
                self.assertGreaterEqual(result['metrics']['held_out_accuracy'], 0.95)
                self.assertEqual(result['metrics']['edge_erased_accuracy'], 0.5)

    def test_dataset_requires_edges_and_loader_preserves_graph_identity(self):
        data = example.make_dataset()
        for split, graphs in data.items():
            self.assertEqual(len(graphs), 32)
            for left, right in zip(graphs[::2], graphs[1::2]):
                self.assertEqual((left.y.tolist(), right.y.tolist()), ([0], [1]))
                self.assertEqual(left.x.tolist(), right.x.tolist())
                self.assertNotEqual(left.edge_index.tolist(), right.edge_index.tolist())
                self.assertEqual(left.num_nodes, right.num_nodes)
        train_features = {example.canonical(graph.x.tolist()) for graph in data['train']}
        held_features = {example.canonical(graph.x.tolist()) for graph in data['held_out']}
        self.assertTrue(train_features.isdisjoint(held_features))
        loader = fr.graph.DataLoader(data['train'], batch_size=8, shuffle=True, seed=107)
        seen = []
        for batch in loader:
            self.assertIsInstance(batch, fr.graph.Batch)
            self.assertEqual(batch.y.dtype, 'int64')
            self.assertEqual(batch.y.shape, [8])
            self.assertEqual(batch.ptr.tolist()[-1], batch.num_nodes)
            seen.extend(batch.sample_id.tolist())
            for graph in batch.to_data_list():
                original = data['train'][graph.sample_id.item()]
                self.assertEqual(graph.x.tolist(), original.x.tolist())
                self.assertEqual(graph.edge_index.tolist(), original.edge_index.tolist())
                self.assertEqual(graph.y.tolist(), original.y.tolist())
        self.assertEqual(sorted(seen), list(range(32)))
        self.assertNotEqual(seen, list(range(32)))

    def test_saved_data_and_epoch_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            snapshot, first = example.train(epochs=2)
            example.save_run(directory / 'run', snapshot, example.make_dataset(), 'gcn', 7, 2)
            expected_state, expected = example.train(epochs=5)
            with patch.object(example, 'make_dataset', side_effect=AssertionError('regenerated saved graph data')):
                actual_state, actual = example.train(epochs=5, resume=directory / 'run')
            for field in ('losses', 'batch_ids'):
                self.assertEqual(actual[field], expected[field][len(first[field]):])
            for field in ('metrics', 'loader_epoch', 'dataset_digest', 'next_draws'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(example.state_digest(actual_state, directory / 'actual'),
                             example.state_digest(expected_state, directory / 'expected'))
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                example.train(kind='sage', resume=directory / 'run')
            for name, mutate, message in (
                ('epoch', lambda record: record.update(epoch=3), 'loader epoch differ'),
                ('feature', lambda record: record['data']['train'][0]['x'][0].__setitem__(0, 0.5), 'data identity mismatch'),
            ):
                target = directory / name
                shutil.copytree(directory / 'run', target)
                record = json.loads((target / 'run.json').read_bytes())
                mutate(record)
                (target / 'run.json').write_bytes(example.canonical(record))
                with patch.object(example, 'GraphClassifier', side_effect=AssertionError('constructed before data/cursor validation')):
                    with self.assertRaisesRegex(ValueError, message):
                        example.train(resume=target)


if __name__ == '__main__':
    unittest.main()
