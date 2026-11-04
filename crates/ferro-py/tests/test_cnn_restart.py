"""Public convolution learning, saved image identity and CPU restart gates."""
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
import cnn_restart as harness
import ferro as fr


def saved_run(directory):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(directory, state, data, 7)
    return data, result['dataset_sha256']


class CnnRestartTests(unittest.TestCase):
    def test_learning_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                self.assertEqual(set(result['gradient_max_abs_errors']),
                                 {'input', 'conv.weight', 'conv.bias', 'head.weight', 'head.bias'})

    def test_dataset_requires_spatial_information(self):
        data = harness.make_dataset()
        train = {harness.canonical(item['pixels']) for item in data['train']}
        held = {harness.canonical(item['pixels']) for item in data['held_out']}
        self.assertTrue(train.isdisjoint(held))
        for split in ('train', 'held_out'):
            centers = {0: [], 1: []}
            for left, right in zip(data[split][::2], data[split][1::2]):
                self.assertEqual((left['label'], right['label']), (0, 1))
                self.assertEqual(sorted(left['pixels']), sorted(right['pixels']))
                mean_images, _ = harness.batch([left, right], remove_spatial=True)
                self.assertEqual(mean_images.tolist()[0], mean_images.tolist()[1])
                for item in (left, right):
                    bright = [i for i, value in enumerate(item['pixels']) if value == max(item['pixels'])]
                    self.assertEqual(len(bright), 3)
                    rows, columns = [i // 8 for i in bright], [i % 8 for i in bright]
                    self.assertEqual(len(set(rows if item['label'] == 0 else columns)), 1)
                    centers[item['label']].append((sum(rows) // 3, sum(columns) // 3))
            self.assertEqual(sorted(centers[0]), sorted(centers[1]))
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % len(data['train']), 0)

    def test_resume_loads_saved_images_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated images')):
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
            for field, value in (('version', True), ('seed', 11), ('schedule', 'wrong'), ('batch_size', 4),
                                 ('step', True), ('next_image', 0), ('dataset_sha256', '0' * 64),
                                 ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'CNNModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            for name, change in (('shape', lambda data: data['train'][0]['pixels'].pop()),
                                 ('label', lambda data: data['train'][0].update(label=True)),
                                 ('pixels', lambda data: data['train'][0]['pixels'].__setitem__(0, 4.0))):
                with self.subTest(name=name):
                    target = directory / name
                    shutil.copytree(directory / 'original', target)
                    data = json.loads((target / 'dataset.json').read_bytes())
                    change(data)
                    (target / 'dataset.json').write_bytes(harness.canonical(data))
                    with patch.object(harness, 'CNNModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'valid-cursor-wrong-step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_image=16)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_config_mismatch_and_rewritten_digest_do_not_mutate_state(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            for name, config in (('pooling', {'pooling': 'max'}), ('digest', {'dataset_digest': '0' * 64}),
                                 ('schedule', {'schedule': 'different'})):
                with self.subTest(config=name):
                    target = harness.CNNModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    optimizer, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = fr.checkpoint.snapshot_training(target, optimizers={'main': optimizer}, generator=rng)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                    after = fr.checkpoint.snapshot_training(target, optimizers={'main': optimizer}, generator=rng)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            for item in data['train'][:2]:
                item['pixels'] = [v + 0.25 for v in item['pixels']]
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class CnnGateSensitivity(unittest.TestCase):
    def test_restart_and_learning_gates(self):
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25],
                    'metrics': {'held_out_accuracy': 1., 'mean_image_accuracy': 0.5, 'spatial_accuracy_drop': 0.5},
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS, 'next_image': 32}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            corrupted = copy.deepcopy(resumed)
            corrupted[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, corrupted, state)
        for accuracy, ablation in ((0.949999, 0.5), (1., 0.625), (1., 0.49),
                                   (float('nan'), 0.5), (float('inf'), 0.5), (1.01, 0.5), (1., -0.1)):
            altered = copy.deepcopy(expected)
            altered['metrics'] = {'held_out_accuracy': accuracy, 'mean_image_accuracy': ablation,
                                  'spatial_accuracy_drop': accuracy - ablation}
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(accuracy=accuracy, ablation=ablation), self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)
        altered = copy.deepcopy(expected)
        altered['metrics']['spatial_accuracy_drop'] = 0.9
        same = copy.deepcopy(resumed)
        same['metrics'] = altered['metrics']
        with self.assertRaisesRegex(AssertionError, 'inconsistent ablation'):
            harness.check_acceptance(altered, same, state)


class CnnOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(CnnGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
