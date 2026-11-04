"""GRU temporal learning and restart at complete optimizer-step boundaries."""
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
import recurrent_restart as harness
import ferro as fr


def saved_run(directory):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(directory, state, data, 7)
    return data, result['dataset_sha256']


def change_pair(data):
    for sequence in data['train'][:2]:
        sign = 2 * sequence['label'] - 1
        sequence['tokens'][0] += sign * 0.01
        sequence['tokens'][1] = -sequence['tokens'][0]


class RecurrentRestartTests(unittest.TestCase):
    def test_learning_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                self.assertEqual(result['resume_scope'], 'between-complete-optimizer-steps')
                self.assertEqual(result['reset_policy'], 'zero-state-per-batch-v1')
                self.assertEqual(set(result['gradient_max_abs_errors']),
                                 {'input', 'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh',
                                  'readout_weight', 'readout_bias'})

    def test_dataset_requires_order_and_disjoint_sequences(self):
        data = harness.make_dataset()
        samples = {split: {tuple(fr.Tensor(s['tokens'], [harness.SEQUENCE_LENGTH]).tolist())
                          for s in data[split]} for split in ('train', 'held_out')}
        self.assertTrue(samples['train'].isdisjoint(samples['held_out']))
        for split in ('train', 'held_out'):
            self.assertEqual(len(samples[split]), 32)
            for left, right in zip(data[split][::2], data[split][1::2]):
                self.assertEqual((left['label'], right['label']), (0, 1))
                self.assertEqual(sorted(left['tokens']), sorted(right['tokens']))
                self.assertEqual(left['tokens'][-1], right['tokens'][-1])
                self.assertEqual(left['tokens'][-1], 0)
        model = harness.GRUModel(seed=7, dataset_digest=hashlib.sha256(harness.canonical(data)).hexdigest())
        tokens, _ = harness.batch(data['held_out'])
        erased = model(tokens, erase_history=True).tolist()
        self.assertTrue(all(row == erased[0] for row in erased))
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % 32, 0)
        invalid = copy.deepcopy(data)
        invalid['held_out'][0]['tokens'] = data['train'][0]['tokens']
        with self.assertRaisesRegex(ValueError, 'unique and disjoint'):
            harness.validate_dataset(invalid)

    def test_resume_loads_saved_data_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated sequences')):
                state, actual = harness.train(7, stop=7, resume=directory / 'run', expected_digest=digest)
            self.assertEqual(actual['losses'], expected['losses'][3:])
            for field in ('metrics', 'next_draws', 'dataset_sha256', 'cursor'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_digest(state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))
            with self.assertRaisesRegex(ValueError, 'expected dataset identity'):
                harness.train(7, resume=directory / 'run')
            for stop in (True, 2, harness.STEPS + 1):
                with self.subTest(stop=stop), self.assertRaisesRegex(ValueError, 'training stop'):
                    harness.train(7, stop=stop, resume=directory / 'run', expected_digest=digest)

    def test_corrupt_bundle_rejects_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            for field, value in (('version', True), ('seed', 11), ('schedule', 'wrong'),
                                 ('batch_size', 4), ('reset_policy', 'carry-state'), ('step', True),
                                 ('next_sequence', 0), ('dataset_sha256', '0' * 64),
                                 ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'GRUModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            for name, corrupt in (('length', lambda data: data['train'][0]['tokens'].pop()),
                                  ('valid-tokens', change_pair)):
                target = directory / name
                shutil.copytree(directory / 'original', target)
                data = json.loads((target / 'dataset.json').read_bytes())
                corrupt(data)
                (target / 'dataset.json').write_bytes(harness.canonical(data))
                with patch.object(harness, 'GRUModel', side_effect=AssertionError('model constructed before validation')):
                    with self.assertRaises(ValueError):
                        harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'step-mismatch'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_sequence=16)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)
            target = directory / 'publication'
            shutil.copytree(directory / 'original', target)
            path = target / 'training' / 'checkpoint.json'
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'checkpoint publication mismatch'):
                harness.load_run(target, digest, 7)

    def test_config_mismatch_and_rewritten_digest_do_not_mutate_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            configs = (('hidden', {'hidden': 9}), ('digest', {'dataset_digest': '0' * 64}),
                       ('schedule', {'schedule': 'different'}), ('length', {'sequence_length': 6}),
                       ('batch', {'batch_size': 4}), ('reset', {'reset_policy': 'carry-state'}))
            for name, config in configs:
                with self.subTest(config=name):
                    target = harness.GRUModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    optimizer, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = fr.checkpoint.snapshot_training(target, optimizers={'main': optimizer}, generator=rng)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                    after = fr.checkpoint.snapshot_training(target, optimizers={'main': optimizer}, generator=rng)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            change_pair(data)
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class RecurrentGateSensitivity(unittest.TestCase):
    def test_restart_and_learning_gates(self):
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25],
                    'metrics': {'held_out_accuracy': 0.95, 'history_erased_accuracy': 0.5,
                                'history_erasure_accuracy_drop': 0.45,
                                'last_token_baseline': 0.5, 'orderless_baseline': 0.5},
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS, 'next_sequence': 0}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            corrupted = copy.deepcopy(resumed)
            corrupted[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, corrupted, state)
        for metric, bad in (('held_out_accuracy', 0.949999), ('history_erasure_accuracy_drop', 0.399999),
                            ('held_out_accuracy', float('nan')), ('history_erasure_accuracy_drop', float('nan'))):
            altered = copy.deepcopy(expected)
            altered['metrics'][metric] = bad
            altered['metrics']['history_erased_accuracy'] = altered['metrics']['held_out_accuracy'] - altered['metrics']['history_erasure_accuracy_drop']
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(metric=metric, bad=bad), self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)
        for metric, bad in (('held_out_accuracy', float('inf')), ('held_out_accuracy', 1.01),
                            ('history_erased_accuracy', -0.1), ('history_erasure_accuracy_drop', 0.5),
                            ('last_token_baseline', 0.75), ('orderless_baseline', 0.75),
                            ('orderless_baseline', float('nan')), ('last_token_baseline', True)):
            altered = copy.deepcopy(expected)
            altered['metrics'][metric] = bad
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(metric=metric, bad=bad), self.assertRaisesRegex(AssertionError, 'invalid metrics|inconsistent ablation|learning gate failed'):
                harness.check_acceptance(altered, same, state)


class RecurrentOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecurrentGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
