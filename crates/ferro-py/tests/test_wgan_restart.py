"""Alternating WGAN-GP gradients, distribution learning and mid-cycle restart."""
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
import wgan_restart as harness
import ferro as fr


def saved_run(path):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=9)
    harness.save_run(path, state, data, 7)
    return data, result['dataset_sha256']


class WganRestartTests(unittest.TestCase):
    def test_distribution_learning_and_midcycle_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                self.assertGreater(result['initial_metrics']['w1'], .25)
                self.assertGreater(result['initial_metrics']['std_error'], .25)
                self.assertEqual(result['isolated_phases'], ['critic'] * 5 + ['generator'])
                self.assertEqual(set(result['derivatives']['penalty_mixed_gradient_max_abs']), {'w1', 'b1', 'w2'})
                self.assertTrue(all(v > 1e-6 for v in result['derivatives']['penalty_mixed_gradient_max_abs'].values()))

    def test_saved_real_samples_and_nontrivial_phase(self):
        data = harness.make_dataset()
        self.assertTrue(set(data['train_real']).isdisjoint(data['held_real']))
        self.assertEqual(len(set(data['eval_latent'])), harness.DATA_SIZE)
        self.assertEqual(harness.phase_state(harness.CHECKPOINT_STEP),
                         {'step': 11997, 'cycle': 1999, 'phase': 'critic', 'critic_substep': 3,
                          'critic_updates': 9998, 'generator_updates': 1999, 'next_real': 448})
        self.assertEqual([harness.phase_state(i)['phase'] for i in range(11997, 12000)],
                         ['critic', 'critic', 'generator'])
        invalid = copy.deepcopy(data)
        invalid['held_real'][0] = data['train_real'][0]
        with self.assertRaisesRegex(ValueError, 'samples overlap'):
            harness.validate_dataset(invalid)

    def test_resume_loads_actual_data_phase_rng_and_both_optimizers(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=17)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated real samples')):
                state, actual = harness.train(7, stop=17, resume=directory / 'run', expected_digest=digest)
            self.assertEqual(actual['losses'], expected['losses'][9:])
            self.assertEqual([row['phase'] for row in actual['losses'][:3]], ['critic', 'critic', 'generator'])
            for field in ('metrics', 'target_baseline', 'next_draws', 'phase', 'dataset_sha256'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_digest(state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))

    def test_corruption_rejects_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            corruptions = [('version', True), ('seed', 11), ('config', {**harness.CONFIG, 'gp_weight': 9.}),
                           ('dataset_sha256', '0' * 64), ('checkpoint_index_sha256', '0' * 64)]
            for field in ('phase', 'critic_substep', 'critic_updates', 'generator_updates', 'next_real', 'cycle', 'step'):
                phase = harness.phase_state(9)
                phase[field] = 'generator' if field == 'phase' else phase[field] + 1
                corruptions.append(('phase', phase))
            for index, (field, value) in enumerate(corruptions):
                with self.subTest(field=field, value=value):
                    target = directory / str(index)
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'WGANModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'data'
            shutil.copytree(directory / 'original', target)
            data = json.loads((target / 'dataset.json').read_bytes())
            data['eval_latent'][0] += .0001
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            with patch.object(harness, 'WGANModel', side_effect=AssertionError('model constructed before validation')):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record['phase'] = harness.phase_state(8)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/phase step mismatch'):
                harness.load_run(target, digest, 7)
            target = directory / 'payload'
            shutil.copytree(directory / 'original', target)
            index = json.loads((target / 'training' / 'checkpoint.json').read_bytes())
            payload = target / 'training' / index['generation'] / 'model.safetensors'
            payload.write_bytes(payload.read_bytes()[:8])
            with patch.object(harness, 'WGANModel', side_effect=AssertionError('model constructed before payload validation')):
                with self.assertRaises(ValueError):
                    harness.train(7, resume=target, expected_digest=digest)

    def test_config_and_rewritten_data_identity_reject_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            for name, config in (('gp_weight', {'gp_weight': 8.}), ('epsilon', {'norm_epsilon': 1e-8}),
                                 ('dataset', {'dataset_digest': '0' * 64}), ('schedule', {'critic_steps': 4}),
                                 ('target', {'target_std': .8}), ('critic_lr', {'critic_lr': .002}),
                                 ('generator_lr', {'generator_lr': .002}), ('init', {'critic_init': 'positive-v1'})):
                with self.subTest(config=name):
                    model = harness.WGANModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    opts, rng = harness.optimizers(model, lr=.7), fr.Generator(999)
                    before = fr.checkpoint.snapshot_training(model, optimizers=opts, generator=rng)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(model, optimizers=opts, generator=rng)
                    after = fr.checkpoint.snapshot_training(model, optimizers=opts, generator=rng)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            data['train_real'][0] += .0001
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)

    def test_optimizer_counters_and_inactive_state_are_checked(self):
        data = harness.make_dataset()
        self.assertEqual(harness.isolation_check(7, data), ['critic'] * 5 + ['generator'])
        state, _ = harness.train(7, stop=9)
        for name in ('critic', 'generator'):
            wrong = harness.phase_state(9)
            wrong[name + '_updates'] += 1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'optimizer counter mismatch'):
                harness.check_counters(state, wrong)


class WganGateSensitivity(unittest.TestCase):
    def test_detached_penalty_gradient_cannot_pass(self):
        original = harness.input_gradient
        for reenable in (False, True):
            def detached(critic, x):
                derivative = original(critic, x)
                broken = derivative.detach().requires_grad_(reenable)
                self.assertEqual(broken.tolist(), derivative.tolist())
                return broken
            with self.subTest(reenable=reenable), patch.object(harness, 'input_gradient', detached):
                with self.assertRaisesRegex(AssertionError, 'input gradient detached|Missing WGAN penalty critic gradient'):
                    harness.gradient_check(7, harness.make_dataset())

    def test_restart_and_distribution_gates(self):
        metrics = {'w1': .25, 'mean_error': .25, 'std_error': .25, 'mean': 1., 'std': .6,
                   'held_penalty': .01, 'held_norm_mean': .99, 'held_norm_min': .9, 'held_norm_max': 1.1}
        expected = {'losses': list(range(harness.STEPS)), 'next_draws': [.25], 'metrics': metrics,
                    'initial_metrics': {**metrics, 'w1': 1.5}, 'target_baseline': {**metrics, 'w1': .005},
                    'dataset_sha256': 'fixture', 'phase': harness.phase_state(harness.STEPS)}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizers': ['critic', 'generator']}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_draws', 'metrics', 'target_baseline', 'dataset_sha256', 'phase', 'state'):
            bad = copy.deepcopy(resumed)
            bad[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, bad, state)
        for field, value in (('w1', .250001), ('mean_error', .250001), ('std_error', .250001),
                             ('w1', float('nan')), ('mean', float('inf')), ('std_error', -.1)):
            altered = copy.deepcopy(expected)
            altered['metrics'][field] = value
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(field=field, value=value), self.assertRaisesRegex(AssertionError, 'distribution gate failed|invalid distribution metrics'):
                harness.check_acceptance(altered, same, state)
        altered = copy.deepcopy(expected)
        altered['initial_metrics']['w1'] = .9
        with self.assertRaisesRegex(AssertionError, 'distribution gate failed'):
            harness.check_acceptance(altered, resumed, state)


class WganOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(WganGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
