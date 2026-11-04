"""Strong-form Poisson learning, mixed-gradient gates and complete-step restart."""
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
import pinn_restart as harness
import ferro as fr


def saved_run(path):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(path, state, data, 7)
    return data, result['dataset_sha256']


class PinnRestartTests(unittest.TestCase):
    def test_physics_learning_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                checked = result['derivatives']
                self.assertEqual(set(checked['parameter_max_abs_errors']), {'w1', 'b1', 'w2', 'b2'})
                self.assertTrue(all(v > 1e-6 for v in checked['mixed_gradient_max_abs'].values()))
                self.assertGreater(result['initial_metrics']['residual_rmse'], 0.1)
                self.assertGreater(result['initial_metrics']['solution_rmse'], 0.01)

    def test_saved_collocation_is_disjoint_even_after_jitter(self):
        data = harness.make_dataset()
        self.assertTrue(set(data['train']).isdisjoint(data['held_out']))
        gap = min(abs(a - b) for a in data['train'] for b in data['held_out'])
        self.assertGreater(gap, harness.CONFIG['jitter'])
        self.assertEqual(data['boundary'], [0., 1.])
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % 32, 0)
        invalid = copy.deepcopy(data)
        invalid['held_out'][0] = data['train'][0] + harness.CONFIG['jitter'] / 2
        with self.assertRaisesRegex(ValueError, 'jitter could reach held-out'):
            harness.validate_dataset(invalid)

    def test_resume_uses_saved_points_rng_and_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated collocation')):
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
            for field, value in (('version', True), ('seed', 11), ('config', {**harness.CONFIG, 'jitter': 0.002}),
                                 ('next_point', 0), ('dataset_sha256', '0' * 64), ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'PINNModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'data'
            shutil.copytree(directory / 'original', target)
            data = json.loads((target / 'dataset.json').read_bytes())
            data['train'][0] += 0.00025
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            with patch.object(harness, 'PINNModel', side_effect=AssertionError('model constructed before validation')):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_point=16)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_physics_config_and_rewritten_digest_reject_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            for name, config in (('frequency', {'frequency': 2.}), ('jitter', {'jitter': 0.002}),
                                 ('dataset', {'dataset_digest': '0' * 64}), ('schedule', {'schedule': 'different'}),
                                 ('equation', {'equation': 'different-pde'})):
                with self.subTest(config=name):
                    target = harness.PINNModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    opt, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = fr.checkpoint.snapshot_training(target, optimizers={'main': opt}, generator=rng)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': opt}, generator=rng)
                    after = fr.checkpoint.snapshot_training(target, optimizers={'main': opt}, generator=rng)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            data['train'][0] += 0.00025
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class PinnGateSensitivity(unittest.TestCase):
    def test_detached_second_derivative_cannot_pass(self):
        original = harness.derivatives
        for reenable in (False, True):
            def detached(model, x):
                value, first, second = original(model, x)
                broken = second.detach().requires_grad_(reenable)
                self.assertEqual(broken.tolist(), second.tolist())
                return value, first, broken
            with self.subTest(reenable=reenable), patch.object(harness, 'derivatives', detached):
                with self.assertRaisesRegex(AssertionError, 'second derivative detached|Missing PINN residual parameter gradient'):
                    harness.gradient_check(7, harness.make_dataset())

    def test_restart_and_physics_gates(self):
        metrics = {'solution_rmse': 0.01, 'residual_rmse': 0.1, 'boundary_max_abs': 1e-6,
                   'zero_solution_rmse': 0.7, 'zero_residual_rmse': 7.}
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25], 'metrics': metrics,
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS, 'next_point': 0}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            bad = copy.deepcopy(resumed)
            bad[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, bad, state)
        for field, bad in (('solution_rmse', 0.010001), ('residual_rmse', 0.100001),
                           ('boundary_max_abs', 1.00001e-6), ('solution_rmse', float('nan')),
                           ('residual_rmse', float('inf')), ('boundary_max_abs', -0.1)):
            altered = copy.deepcopy(expected)
            altered['metrics'][field] = bad
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(field=field, bad=bad), self.assertRaisesRegex(AssertionError, 'physics gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)


class PinnOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PinnGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
