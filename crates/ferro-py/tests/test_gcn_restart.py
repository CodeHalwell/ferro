"""Graph learning, explicit topology persistence and fresh-process CPU restart."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples'))
import gcn_restart as harness
import ferro as fr


def saved_run(directory):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(directory, state, data, 7)
    return data, result['dataset_sha256']


class GcnRestartTests(unittest.TestCase):
    def test_learning_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                self.assertEqual(set(result['gradient_max_abs_errors']),
                                 {'input', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'})

    def test_dataset_has_no_node_or_topology_label_leakage(self):
        data = harness.make_dataset()
        train = {harness.canonical(g['features']) for g in data['train']}
        held = {harness.canonical(g['features']) for g in data['held_out']}
        self.assertTrue(train.isdisjoint(held))
        for split in ('train', 'held_out'):
            for left, right in zip(data[split][::2], data[split][1::2]):
                self.assertEqual((left['label'], right['label']), (0, 1))
                for field in ('num_nodes', 'src', 'dst', 'query'):
                    self.assertEqual(left[field], right[field])
                for graph in (left, right):
                    self.assertEqual(graph['features'][graph['query']], [0., 1.])
                    self.assertAlmostEqual(sum(row[0] for row in graph['features']), 0.0)
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % 32, 0)

    def test_resume_loads_saved_topology_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated topology')):
                state, actual = harness.train(7, stop=7, resume=directory / 'run', expected_digest=digest)
            self.assertEqual(actual['losses'], expected['losses'][3:])
            for field in ('metrics', 'next_draws', 'dataset_sha256', 'cursor'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_digest(state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))

    def test_corrupt_bundle_rejects_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            for field, value in (('version', True), ('seed', 11), ('schedule', 'wrong'),
                                 ('batch_size', 4), ('next_graph', 0), ('dataset_sha256', '0' * 64),
                                 ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'GCNModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'topology'
            shutil.copytree(directory / 'original', target)
            data = json.loads((target / 'dataset.json').read_bytes())
            data['train'][0]['src'][0] = data['train'][0]['num_nodes']
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            with patch.object(harness, 'GCNModel', side_effect=AssertionError('model constructed before validation')):
                with self.assertRaises(ValueError):
                    harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'features'
            shutil.copytree(directory / 'original', target)
            data = json.loads((target / 'dataset.json').read_bytes())
            graph = data['train'][0]
            graph['features'][(graph['query'] + 1) % graph['num_nodes']][0] += 0.25
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            with patch.object(harness, 'GCNModel', side_effect=AssertionError('model constructed before validation')):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_graph=16)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_config_mismatch_and_rewritten_digest_do_not_mutate_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            for name, config in (('loops', {'self_loops': False}), ('digest', {'dataset_digest': '0' * 64}),
                                 ('schedule', {'schedule': 'different'})):
                with self.subTest(config=name):
                    target = harness.GCNModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    optimizer, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = fr.checkpoint.snapshot_training(target, optimizers={'main': optimizer}, generator=rng)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                    after = fr.checkpoint.snapshot_training(target, optimizers={'main': optimizer}, generator=rng)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            data['train'][0]['features'][(data['train'][0]['query'] + 1) % data['train'][0]['num_nodes']][0] += 0.25
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class GcnGateSensitivity(unittest.TestCase):
    def test_restart_and_learning_gates(self):
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25],
                    'metrics': {'held_out_accuracy': 0.95, 'edge_removed_accuracy': 0.55, 'edge_removal_accuracy_drop': 0.4},
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS, 'next_graph': 0}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            corrupted = copy.deepcopy(resumed)
            corrupted[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, corrupted, state)
        for metric, bad in (('held_out_accuracy', 0.949999), ('edge_removal_accuracy_drop', 0.399999),
                            ('held_out_accuracy', float('nan')), ('edge_removal_accuracy_drop', float('nan'))):
            altered = copy.deepcopy(expected)
            altered['metrics'][metric] = bad
            altered['metrics']['edge_removed_accuracy'] = altered['metrics']['held_out_accuracy'] - altered['metrics']['edge_removal_accuracy_drop']
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(metric=metric, bad=bad), self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)
        for metric, bad in (('held_out_accuracy', float('inf')), ('held_out_accuracy', 1.01),
                            ('edge_removed_accuracy', -0.1), ('edge_removal_accuracy_drop', 0.5)):
            altered = copy.deepcopy(expected)
            altered['metrics'][metric] = bad
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(metric=metric, bad=bad), self.assertRaisesRegex(AssertionError, 'invalid metrics|inconsistent ablation'):
                harness.check_acceptance(altered, same, state)


class GcnOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GcnGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
