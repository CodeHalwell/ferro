"""Nonlinear KAN learning, basis identity and complete-step CPU restart."""
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
import kan_restart as harness
import ferro as fr


def saved_run(path):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(path, state, data, 7)
    return data, result['dataset_sha256']


class KanRestartTests(unittest.TestCase):
    def test_learning_gradients_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                self.assertEqual(set(result['gradients']['max_abs_errors']),
                                 {'input', 'first.coefficients', 'second.coefficients'})
                self.assertGreater(result['gradients']['input_knot_margin'], 1e-3)
                self.assertGreater(result['gradients']['hidden_knot_margin'], 1e-3)
                self.assertGreater(result['initial_metrics']['affine_ratio'], 0.1)

    def test_dataset_and_independent_affine_baseline(self):
        import torch
        data = harness.make_dataset()
        self.assertTrue({tuple(row['x']) for row in data['train']}.isdisjoint(
                        {tuple(row['x']) for row in data['held_out']}))
        x = torch.tensor([[1., *row['x']] for row in data['train']], dtype=torch.float64)
        y = torch.tensor([row['y'] for row in data['train']], dtype=torch.float64)
        coefficients = torch.linalg.lstsq(x, y).solution
        held = torch.tensor([[1., *row['x']] for row in data['held_out']], dtype=torch.float64)
        expected = torch.tensor([row['y'] for row in data['held_out']], dtype=torch.float64)
        baselines = harness.baselines(data)
        self.assertAlmostEqual(baselines['affine_baseline_mse'], float(((held @ coefficients - expected) ** 2).mean()), places=12)
        self.assertAlmostEqual(baselines['constant_baseline_mse'], float(((y.mean() - expected) ** 2).mean()), places=12)
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % 64, 0)

    def test_resume_uses_saved_data_and_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated data')):
                state, actual = harness.train(7, stop=7, resume=directory / 'run', expected_digest=digest)
            self.assertEqual(actual['losses'], expected['losses'][3:])
            for field in ('metrics', 'next_draws', 'cursor', 'dataset_sha256'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_digest(state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))

    def test_corruption_rejects_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            for field, value in (('version', True), ('seed', 11), ('schedule', 'other'),
                                 ('batch_size', 8), ('next_sample', 0), ('dataset_sha256', '0' * 64),
                                 ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'KANModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'data'
            shutil.copytree(directory / 'original', target)
            data = json.loads((target / 'dataset.json').read_bytes())
            data['train'][0]['y'] += 0.25
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            with patch.object(harness, 'KANModel', side_effect=AssertionError('model constructed before validation')):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_sample=32)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_basis_config_and_rewritten_digest_reject_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            knots = list(harness.KNOTS)
            knots[7] = 0.6
            for name, config in (('knots', {'knots': tuple(knots)}), ('outside', {'outside': 'error'}),
                                 ('dataset', {'dataset_digest': '0' * 64}), ('schedule', {'schedule': 'different'})):
                with self.subTest(config=name):
                    target = harness.KANModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    opt, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = fr.checkpoint.snapshot_training(target, optimizers={'main': opt}, generator=rng)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': opt}, generator=rng)
                    after = fr.checkpoint.snapshot_training(target, optimizers={'main': opt}, generator=rng)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            data['train'][0]['y'] += 0.25
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class KanGateSensitivity(unittest.TestCase):
    def test_restart_and_learning_gates(self):
        metrics = {'held_out_mse': 0.05, 'constant_baseline_mse': 1., 'affine_baseline_mse': 0.5,
                   'constant_ratio': 0.05, 'affine_ratio': 0.1}
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25], 'metrics': metrics,
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS, 'next_sample': 0}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            bad = copy.deepcopy(resumed)
            bad[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, bad, state)
        for mse in (0.050001, float('nan'), float('inf'), -0.1):
            altered = copy.deepcopy(expected)
            altered['metrics'].update(held_out_mse=mse, constant_ratio=mse, affine_ratio=mse / 0.5)
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(mse=mse), self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)
        for field, value in (('constant_baseline_mse', 0.), ('affine_ratio', 0.01)):
            altered = copy.deepcopy(expected)
            altered['metrics'][field] = value
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'inconsistent metrics'):
                harness.check_acceptance(altered, same, state)


class KanOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(KanGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
