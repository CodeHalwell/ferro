"""LSTM state across chunks, gradient boundaries, typed bundles and restart."""
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
import streaming_restart as harness
import ferro as fr


def saved_run(directory):
    data = harness.make_dataset()
    checkpoint, carry, result = harness.train(7, stop=2)
    harness.save_run(directory, checkpoint, data, 7, carry)
    return data, result['dataset_sha256']


def change_pair(data):
    for row in data['train'][:2]:
        row['tokens'][0] += (2 * row['label'] - 1) * 0.01


class StreamingRestartTests(unittest.TestCase):
    def test_learning_and_stateful_fresh_process_restart(self):
        parameters = {'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh', 'readout_weight', 'readout_bias'}
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                self.assertEqual(result['resume_scope'], 'between-complete-chunk-optimizer-updates')
                self.assertEqual(result['checkpoint_probe']['cursor']['chunk'], 2)
                self.assertEqual(result['checkpoint_probe']['carried_accuracy'], 1.)
                self.assertEqual(result['checkpoint_probe']['erased_accuracy'], 0.5)
                for mode, check in result['gradient_check'].items():
                    expected = parameters | {'input', 'initial_h', 'initial_c'}
                    if mode.endswith('bridge'):
                        expected.add('suffix_input')
                    self.assertEqual(set(check['max_abs_errors']), expected)
                    if mode in ('detached_bridge', 'truncate'):
                        for name in ('initial_h', 'initial_c'):
                            self.assertEqual(check['gradient_max_abs'][name], 0.)
                    for name in parameters:
                        self.assertGreater(check['gradient_max_abs'][name], 1e-7)

    def test_only_initial_chunk_has_label_information(self):
        data = harness.make_dataset()
        samples = {split: {tuple(fr.Tensor(row['tokens'], [harness.CHUNKS * harness.CHUNK_LENGTH]).tolist())
                           for row in data[split]} for split in ('train', 'held_out')}
        self.assertTrue(samples['train'].isdisjoint(samples['held_out']))
        for split in ('train', 'held_out'):
            self.assertEqual(len(samples[split]), 32)
            for left, right in zip(data[split][::2], data[split][1::2]):
                self.assertEqual((left['label'], right['label']), (0, 1))
                self.assertEqual(left['tokens'][0], -right['tokens'][0])
                self.assertTrue(all(value == 0 for row in (left, right) for value in row['tokens'][1:]))
        position = harness.cursor(data, harness.CHECKPOINT_STEP)
        self.assertEqual(position['chunk'], 2)
        self.assertEqual(position['next_sequence_ids'], position['state_sequence_ids'])
        self.assertNotEqual(position['next_sample'], 0)

    def test_resume_uses_saved_h_and_c_without_recomputing_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            _, _, loaded = harness.load_run(directory / 'run', digest, 7)
            for values in harness.state_values(loaded).values():
                self.assertGreater(max(abs(value) for row in values for value in row), 0.)
            expected_state, expected_carry, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated stream')):
                with patch.object(harness, 'training_chunk', wraps=harness.training_chunk) as chunks:
                    actual_state, actual_carry, actual = harness.train(7, stop=7, resume=directory / 'run', expected_digest=digest)
            self.assertEqual([call.args[1] for call in chunks.call_args_list], list(range(2, 7)))
            self.assertEqual(actual['losses'], expected['losses'][2:])
            for field in ('metrics', 'next_draws', 'dataset_sha256', 'cursor', 'carry_sha256'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_values(actual_carry), harness.state_values(expected_carry))
            self.assertEqual(harness.state_digest(actual_state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))
            _, _, erased = harness.train(7, stop=3, resume=directory / 'run', expected_digest=digest, erase_resume_state=True)
            self.assertNotEqual(erased['losses'][0], expected['losses'][2])
            _, live_carry, live = harness.train(7, stop=3)
            self.assertNotEqual(erased['carry_sha256'], harness.carry_digest(live_carry))
            self.assertNotEqual(erased['losses'], live['losses'][2:])
            with self.assertRaisesRegex(ValueError, 'expected dataset identity'):
                harness.train(7, resume=directory / 'run')

    def test_corrupt_configuration_cursor_and_data_reject_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            changes = (
                ('version', lambda r: r.update(version=True)),
                ('seed', lambda r: r.update(seed=11)),
                ('policy', lambda r: r['configuration'].update(state_policy='never-detach')),
                ('noise', lambda r: r['configuration'].update(noise_scale=0.1)),
                ('hidden', lambda r: r['configuration'].update(hidden=9)),
                ('chunk', lambda r: r['cursor'].update(chunk=0)),
                ('step-type', lambda r: r['cursor'].update(step=2.)),
                ('next-ids', lambda r: r['cursor']['next_sequence_ids'].reverse()),
                ('state-ids', lambda r: r['cursor']['state_sequence_ids'].reverse()),
                ('data-hash', lambda r: r.update(dataset_sha256='0' * 64)),
                ('state-hash', lambda r: r.update(state_sha256='0' * 64)),
                ('checkpoint-hash', lambda r: r.update(checkpoint_index_sha256='0' * 64)),
            )
            for name, change in changes:
                with self.subTest(case=name):
                    target = directory / name
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    change(record)
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'StreamingModel', side_effect=AssertionError('constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            for name, change in (('bad-length', lambda d: d['train'][0]['tokens'].pop()),
                                 ('changed-cue', change_pair),
                                 ('duplicate', lambda d: d['held_out'][0].update(tokens=d['train'][0]['tokens']))):
                target = directory / name
                shutil.copytree(directory / 'original', target)
                altered = copy.deepcopy(data)
                change(altered)
                (target / 'dataset.json').write_bytes(harness.canonical(altered))
                with patch.object(harness, 'StreamingModel', side_effect=AssertionError('constructed before validation')):
                    with self.assertRaises(ValueError):
                        harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'wrong-complete-step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record['cursor'] = harness.cursor(data, 1)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_saved_h_and_c_require_integrity_finite_f32_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            cases = (
                ('missing-c', lambda values: values.pop('c')),
                ('extra', lambda values: values.update(extra=fr.Tensor.zeros([1]))),
                ('h-shape', lambda values: values.update(h=fr.Tensor.zeros([64]))),
                ('c-shape', lambda values: values.update(c=fr.Tensor.zeros([1, 64]))),
                ('h-int', lambda values: values.update(h=fr.Tensor.from_i64([0] * 64, [8, 8]))),
                ('c-int', lambda values: values.update(c=fr.Tensor.from_i64([0] * 64, [8, 8]))),
                ('h-nan', lambda values: values.update(h=fr.Tensor([float('nan')] + [0.] * 63, [8, 8]))),
                ('c-inf', lambda values: values.update(c=fr.Tensor([float('inf')] + [0.] * 63, [8, 8]))),
            )
            for name, change in cases:
                with self.subTest(case=name):
                    target = directory / name
                    shutil.copytree(directory / 'original', target)
                    path = target / 'state.safetensors'
                    values = fr.load_safetensors(str(path))
                    change(values)
                    fr.save_safetensors(str(path), values)
                    record = json.loads((target / 'run.json').read_bytes())
                    record['state_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'StreamingModel', side_effect=AssertionError('constructed before state validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'bit-flip'
            shutil.copytree(directory / 'original', target)
            path = target / 'state.safetensors'
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'h/c integrity mismatch'):
                harness.load_run(target, digest, 7)

    def test_config_rejection_preserves_model_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint, carry = harness.load_run(directory / 'original', digest, 7)
            for name, changed in (('policy', {'state_policy': 'carry-live-graph'}), ('noise', {'noise_scale': 0.1}),
                                  ('chunk', {'chunk_length': 4}), ('digest', {'dataset_digest': '0' * 64})):
                with self.subTest(config=name):
                    target = harness.StreamingModel(**{'seed': 7, 'dataset_digest': digest, **changed})
                    optimizer, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = harness.snapshot(target, optimizer, rng, 0)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                    after = harness.snapshot(target, optimizer, rng, 0)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten-data'
            shutil.copytree(directory / 'original', target)
            change_pair(data)
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'dataset identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)
            live = harness.RecurrentState(carry.h.requires_grad_(True), carry.c)
            with self.assertRaisesRegex(ValueError, 'must be detached'):
                harness.save_run(directory / 'live-graph', checkpoint, harness.make_dataset(), 7, live)


class StreamingGateSensitivity(unittest.TestCase):
    def test_truncation_boundary_and_oracle_are_not_value_only(self):
        data = harness.make_dataset()
        result = harness.gradient_check(7, data)
        self.assertGreater(result['live_bridge']['gradient_max_abs']['input'], 1e-7)
        self.assertEqual(result['detached_bridge']['gradient_max_abs']['input'], 0.)
        original = harness.unroll
        def wrong_truncation(*args, **kwargs):
            kwargs['truncate'] = 1
            return original(*args, **kwargs)
        with patch.object(harness, 'unroll', side_effect=wrong_truncation):
            with self.assertRaises(AssertionError):
                harness.gradient_check(7, data)

    def test_restart_accuracy_loss_and_state_erasure_gates(self):
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [0.25],
                    'metrics': {'chunk_accuracies': [1., 0.95, 0.95], 'chunk_losses': [0.03, 0.04, 0.04],
                                'reset_chunk_accuracies': [1., 0.5, 0.5], 'final_accuracy_drop': 0.45,
                                'blank_suffix_baseline': 0.5}, 'initial_metrics': {'chunk_losses': [0.7, 0.7, 0.7]},
                    'dataset_sha256': 'fixture', 'cursor': {'step': harness.STEPS}, 'carry_sha256': 'saved-h-and-c'}
        state = {'payload_sha256': 'parameters-and-moments', 'metadata': {'rng': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'dataset_sha256', 'cursor', 'carry_sha256', 'state'):
            broken = copy.deepcopy(resumed)
            broken[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, broken, state)
        changes = (
            ('accuracy', lambda m: m['chunk_accuracies'].__setitem__(1, 0.949999)),
            ('loss', lambda m: m['chunk_losses'].__setitem__(-1, 0.050001)),
            ('nan', lambda m: m['chunk_losses'].__setitem__(-1, float('nan'))),
            ('inf', lambda m: m['chunk_accuracies'].__setitem__(0, float('inf'))),
            ('bool', lambda m: m['chunk_accuracies'].__setitem__(0, True)),
            ('range', lambda m: m['chunk_accuracies'].__setitem__(0, 1.1)),
            ('shape', lambda m: m['chunk_losses'].pop()),
            ('baseline', lambda m: m.update(blank_suffix_baseline=0.75)),
            ('inconsistent', lambda m: m.update(final_accuracy_drop=0.4)),
        )
        for name, change in changes:
            altered = copy.deepcopy(expected)
            change(altered['metrics'])
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(case=name), self.assertRaises(AssertionError):
                harness.check_acceptance(altered, same, state)
        altered = copy.deepcopy(expected)
        altered['initial_metrics']['chunk_losses'][-1] = 0.39
        with self.assertRaisesRegex(AssertionError, 'learning gate failed'):
            harness.check_acceptance(altered, resumed, state)


class StreamingOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(StreamingGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
