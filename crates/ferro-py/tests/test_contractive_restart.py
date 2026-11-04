"""Mixed derivative contractive training, explicit data, and restart gates."""
import copy
import hashlib
import json
import math
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
import contractive_restart as harness
import ferro as fr


def saved_run(directory):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(directory, state, data, 7)
    return data, result['dataset_sha256']


class ContractiveRestartTests(unittest.TestCase):
    def test_learning_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                for errors in result['gradient_check']['max_abs_errors'].values():
                    self.assertEqual(set(errors), {'input', 'encoder.weight', 'encoder.bias', 'decoder.weight', 'decoder.bias'})
                self.assertLessEqual(result['penalty_ratio'], 0.75)

    def test_data_is_disjoint_and_checkpoint_cursor_is_nonzero(self):
        data = harness.make_dataset()
        train = {harness.canonical(item['x']) for item in data['train']}
        held = {harness.canonical(item['x']) for item in data['held_out']}
        self.assertTrue(train.isdisjoint(held))
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % len(data['train']), 0)
        original = {harness.canonical([-1. + 2. * i / 64, math.sin(1.5 * (-1. + 2. * i / 64))]) for i in range(64)}
        self.assertEqual(train, original)
        for offset in range(0, 64, harness.BATCH_SIZE):
            values = [row['x'][0] for row in data['train'][offset:offset + harness.BATCH_SIZE]]
            self.assertLess(min(values), -0.75)
            self.assertGreater(max(values), 0.7)

    def test_resume_loads_saved_samples_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated samples')):
                state, actual = harness.train(7, stop=7, resume=directory / 'run', expected_digest=digest)
            self.assertEqual(actual['losses'], expected['losses'][3:])
            for field in ('metrics', 'next_draws', 'dataset_sha256', 'cursor'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_digest(state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))

    def test_corruption_rejects_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            for field, value in (('version', True), ('seed', 11), ('schedule', 'wrong'), ('batch_size', 4),
                                 ('step', True), ('next_sample', 0), ('penalty_weight', 0.0),
                                 ('dataset_sha256', '0' * 64), ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'ContractiveModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            for name, change in (('shape', lambda data: data['train'][0]['x'].pop()),
                                 ('values', lambda data: data['train'][0]['x'].__setitem__(0, 4.0)),
                                 ('duplicate', lambda data: data['held_out'][0].update(x=data['train'][0]['x']))):
                with self.subTest(name=name):
                    target = directory / name
                    shutil.copytree(directory / 'original', target)
                    data = json.loads((target / 'dataset.json').read_bytes())
                    change(data)
                    (target / 'dataset.json').write_bytes(harness.canonical(data))
                    with patch.object(harness, 'ContractiveModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'valid-cursor-wrong-step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_sample=16)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_config_and_rewritten_identity_reject_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            for name, config in (('penalty', {'penalty_weight': 0.0}), ('digest', {'dataset_digest': '0' * 64}),
                                 ('schedule', {'schedule': 'different'})):
                with self.subTest(config=name):
                    target = harness.ContractiveModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    optimizer, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = harness.snapshot(target, optimizer, rng, 0)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                    after = harness.snapshot(target, optimizer, rng, 0)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            data['train'][0]['x'][0] += 0.25
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class ContractiveGateSensitivity(unittest.TestCase):
    def test_detached_penalty_preserves_value_but_fails_gradient_oracle(self):
        data = harness.make_dataset()
        model = harness.ContractiveModel(seed=7, dataset_digest=hashlib.sha256(harness.canonical(data)).hexdigest())
        inputs = harness.batch(data['train'][:harness.BATCH_SIZE]).requires_grad_(True)
        targets = harness.batch(data['train'][:harness.BATCH_SIZE])
        live = harness.objective(model, inputs, targets)
        detached = harness.objective(model, inputs, targets, detach_penalty=True)
        self.assertEqual([value.item() for value in live], [value.item() for value in detached])
        names, parameters = zip(*[(name, p.tensor()) for name, p in model.named_parameters()])
        live_gradients, detached_gradients = harness.grad(live[2], parameters), harness.grad(detached[2], parameters)
        for name, actual, altered in zip(names, live_gradients, detached_gradients):
            differences = (actual - altered).tolist()
            flat = [v for row in differences for v in row] if isinstance(differences[0], list) else differences
            if name.startswith('encoder.'):
                self.assertGreater(max(abs(v) for v in flat), 1e-6)
            else:
                self.assertEqual(max(abs(v) for v in flat), 0.)
        with self.assertRaisesRegex(AssertionError, 'Contractive penalty gradient mismatch'):
            harness.gradient_check(7, data, detach_penalty=True)

    def test_restart_reconstruction_and_penalty_gates(self):
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25],
                    'metrics': {'held_out_mse': 0.05, 'constant_baseline_mse': 1., 'reconstruction_ratio': 0.05,
                                'jacobian_penalty': 0.7}, 'initial_metrics': {'jacobian_penalty': 1.},
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS, 'next_sample': 0}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            corrupted = copy.deepcopy(resumed)
            corrupted[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, corrupted, state)
        for ratio, penalty in ((0.100001, 0.7), (0.05, 0.750001), (float('nan'), 0.7),
                               (float('inf'), 0.7), (0.05, -0.1), (0.05, float('nan'))):
            altered = copy.deepcopy(expected)
            altered['metrics'].update(held_out_mse=ratio, reconstruction_ratio=ratio, jacobian_penalty=penalty)
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(ratio=ratio, penalty=penalty), self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)
        altered = copy.deepcopy(expected)
        altered['metrics']['reconstruction_ratio'] = 0.01
        same = copy.deepcopy(resumed)
        same['metrics'] = altered['metrics']
        with self.assertRaisesRegex(AssertionError, 'inconsistent reconstruction'):
            harness.check_acceptance(altered, same, state)


class ContractiveOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractiveGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
