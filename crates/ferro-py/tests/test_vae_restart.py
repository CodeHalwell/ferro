"""Gaussian VAE likelihood, stochastic gradients, data identity and restart."""
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
import vae_restart as harness
import ferro as fr


def saved_run(directory):
    data = harness.make_dataset()
    state, result = harness.train(7, stop=3)
    harness.save_run(directory, state, data, 7)
    return data, result['dataset_sha256']


class VaeRestartTests(unittest.TestCase):
    def test_learning_and_fresh_process_restart(self):
        for seed in harness.SEEDS:
            with self.subTest(seed=seed):
                result = harness.prove(seed)
                self.assertEqual(result['restart'], 'bit-exact')
                for errors in result['gradient_check']['max_abs_errors'].values():
                    self.assertEqual(set(errors), {'input', 'epsilon', *[f'{layer}.{suffix}' for layer in
                                     ('encoder', 'mu', 'logvar', 'decoder', 'output') for suffix in ('weight', 'bias')]})

    def test_data_is_disjoint_and_evaluation_does_not_advance_training_rng(self):
        data = harness.make_dataset()
        train = {harness.canonical(item['x']) for item in data['train']}
        held = {harness.canonical(item['x']) for item in data['held_out']}
        self.assertTrue(train.isdisjoint(held))
        self.assertNotEqual((harness.CHECKPOINT_STEP * harness.BATCH_SIZE) % len(data['train']), 0)
        model = harness.VAEModel(seed=7, dataset_digest=hashlib.sha256(harness.canonical(data)).hexdigest())
        optimizer, rng = fr.optim.AdamW(model.parameters()), fr.Generator(123)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            before = harness.state_digest(harness.snapshot(model, optimizer, rng, 0), directory / 'before')
            first = harness.measure(model, data)
            with patch.object(fr, 'Tensor', wraps=fr.Tensor) as tensor_class:
                tensor_class.randn.side_effect = AssertionError('evaluation generated noise')
                second = harness.measure(model, data)
            self.assertEqual(first, second)
            after = harness.state_digest(harness.snapshot(model, optimizer, rng, 0), directory / 'after')
            self.assertEqual(before, after)

    def test_resume_loads_saved_data_and_epsilon_without_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'run')
            expected_state, expected = harness.train(7, stop=7)
            with patch.object(harness, 'make_dataset', side_effect=AssertionError('resume regenerated samples/epsilon')):
                state, actual = harness.train(7, stop=7, resume=directory / 'run', expected_digest=digest)
            self.assertEqual(actual['losses'], expected['losses'][3:])
            for field in ('metrics', 'next_latent_draws', 'dataset_sha256', 'cursor'):
                self.assertEqual(actual[field], expected[field])
            self.assertEqual(harness.state_digest(state, directory / 'actual'),
                             harness.state_digest(expected_state, directory / 'expected'))

    def test_corruption_rejects_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, digest = saved_run(directory / 'original')
            for field, value in (('version', True), ('seed', 11), ('schedule', 'wrong'), ('batch_size', 4),
                                 ('step', True), ('next_sample', 0), ('beta', 0.1), ('observation_sigma', 0.5),
                                 ('dataset_sha256', '0' * 64), ('checkpoint_index_sha256', '0' * 64)):
                with self.subTest(field=field):
                    target = directory / field
                    shutil.copytree(directory / 'original', target)
                    record = json.loads((target / 'run.json').read_bytes())
                    record[field] = value
                    (target / 'run.json').write_bytes(harness.canonical(record))
                    with patch.object(harness, 'VAEModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            changes = (('shape', lambda data: data['train'][0]['x'].pop()),
                       ('values', lambda data: data['train'][0]['x'].__setitem__(0, 4.0)),
                       ('noise', lambda data: data['eval_epsilon'][0].__setitem__(0, 4.0)),
                       ('noise-shape', lambda data: data['eval_epsilon'].pop()),
                       ('duplicate', lambda data: data['held_out'][0].update(x=data['train'][0]['x'])))
            for name, change in changes:
                with self.subTest(name=name):
                    target = directory / name
                    shutil.copytree(directory / 'original', target)
                    data = json.loads((target / 'dataset.json').read_bytes())
                    change(data)
                    (target / 'dataset.json').write_bytes(harness.canonical(data))
                    with patch.object(harness, 'VAEModel', side_effect=AssertionError('model constructed before validation')):
                        with self.assertRaises(ValueError):
                            harness.train(7, resume=target, expected_digest=digest)
            target = directory / 'valid-cursor-wrong-step'
            shutil.copytree(directory / 'original', target)
            record = json.loads((target / 'run.json').read_bytes())
            record.update(step=2, next_sample=16)
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'checkpoint/cursor step mismatch'):
                harness.load_run(target, digest, 7)

    def test_config_and_rewritten_epsilon_reject_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, digest = saved_run(directory / 'original')
            _, checkpoint = harness.load_run(directory / 'original', digest, 7)
            configs = (('beta', {'beta': 0.5}), ('sigma', {'observation_sigma': 0.5}),
                       ('digest', {'dataset_digest': '0' * 64}), ('schedule', {'schedule': 'different'}))
            for name, config in configs:
                with self.subTest(config=name):
                    target = harness.VAEModel(**{'seed': 7, 'dataset_digest': digest, **config})
                    optimizer, rng = fr.optim.AdamW(target.parameters(), lr=0.7), fr.Generator(999)
                    before = harness.snapshot(target, optimizer, rng, 0)
                    with self.assertRaises(ValueError):
                        checkpoint.restore(target, optimizers={'main': optimizer}, generator=rng)
                    after = harness.snapshot(target, optimizer, rng, 0)
                    self.assertEqual(harness.state_digest(before, directory / (name + '-before')),
                                     harness.state_digest(after, directory / (name + '-after')))
            target = directory / 'rewritten'
            shutil.copytree(directory / 'original', target)
            data['eval_epsilon'][0][0] += 0.25
            changed = hashlib.sha256(harness.canonical(data)).hexdigest()
            (target / 'dataset.json').write_bytes(harness.canonical(data))
            record = json.loads((target / 'run.json').read_bytes())
            record['dataset_sha256'] = changed
            (target / 'run.json').write_bytes(harness.canonical(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                harness.load_run(target, digest, 7)
            with self.assertRaises(ValueError):
                harness.train(7, resume=target, expected_digest=changed)


class VaeGateSensitivity(unittest.TestCase):
    def test_gaussian_normalizer_feature_sum_and_batch_mean(self):
        for latent_size in (1, 2):
            model = harness.VAEModel(seed=7, dataset_digest='reduction-fixture', latent=latent_size)
            with fr.no_grad():
                for _, parameter in model.named_parameters():
                    parameter.tensor().copy_(fr.Tensor.zeros(list(parameter.tensor().shape)))
                model.mu.bias.tensor().copy_(fr.Tensor([0.5] * latent_size, [latent_size]))
                model.logvar.bias.tensor().copy_(fr.Tensor([math.log(4.)] * latent_size, [latent_size]))
            expected_nll = 30. / (4. * harness.OBSERVATION_SIGMA ** 2) + 2. * math.log(harness.OBSERVATION_SIGMA * math.sqrt(2. * math.pi))
            expected_kl = latent_size * 0.5 * (0.25 + 4. - 1. - math.log(4.))
            for repeats in (1, 2):
                inputs = fr.Tensor([1., 2., 3., 4.] * repeats, [2 * repeats, 2])
                epsilon = fr.Tensor.zeros([2 * repeats, latent_size])
                nll, kl, total = harness.objective(model, inputs, inputs, epsilon)
                self.assertAlmostEqual(nll.item(), expected_nll, delta=2e-5)
                self.assertAlmostEqual(kl.item(), expected_kl, delta=1e-6)
                self.assertAlmostEqual(total.item(), expected_nll + expected_kl, delta=2e-5)

    def test_detached_latent_scale_and_kl_preserve_values_but_fail_oracle(self):
        data = harness.make_dataset()
        model = harness.VAEModel(seed=7, dataset_digest=hashlib.sha256(harness.canonical(data)).hexdigest())
        inputs = harness.batch(data['train'][:harness.BATCH_SIZE]).requires_grad_(True)
        epsilon = fr.Tensor([row[0] for row in data['eval_epsilon'][:harness.BATCH_SIZE]], [harness.BATCH_SIZE, 1])
        live = harness.objective(model, inputs, inputs, epsilon)
        variance_parameters = [model.logvar.weight.tensor(), model.logvar.bias.tensor()]
        for fault in ('latent', 'scale', 'kl'):
            with self.subTest(fault=fault):
                altered = harness.objective(model, inputs, inputs, epsilon, fault=fault)
                self.assertEqual([value.item() for value in live], [value.item() for value in altered])
                component = 1 if fault == 'kl' else 0
                live_gradients = harness.grad(live[component], variance_parameters)
                dead_gradients = altered[component].grad_wrt(variance_parameters)
                for actual, dead in zip(live_gradients, dead_gradients):
                    self.assertGreater((actual * actual).sum().item(), 1e-8)
                    self.assertEqual((dead * dead).sum().item(), 0.)
                with self.assertRaisesRegex(AssertionError, 'VAE .* gradient mismatch'):
                    harness.gradient_check(7, data, fault=fault)

    def test_restart_and_learning_gates(self):
        expected = {'losses': list(range(harness.STEPS)), 'next_latent_draws': [[0.25]],
                    'metrics': {'mc_reconstruction_nll': 1., 'kl': 1., 'mc_negative_elbo': 2.,
                                'decoder_mean_mse': 0.05, 'constant_baseline_mse': 1.,
                                'constant_gaussian_negative_elbo': 8.},
                    'initial_metrics': {'mc_negative_elbo': 10.}, 'dataset_sha256': 'fixture',
                    'cursor': {'step': harness.STEPS, 'next_sample': 0}}
        state = {'payload_sha256': 'expected', 'metadata': {'optimizer': [1, 2, 3]}}
        resumed = {**copy.deepcopy(expected), 'losses': expected['losses'][harness.CHECKPOINT_STEP:], 'state': state}
        harness.check_acceptance(expected, resumed, state)
        for field in ('losses', 'next_latent_draws', 'metrics', 'dataset_sha256', 'cursor', 'state'):
            corrupted = copy.deepcopy(resumed)
            corrupted[field] = 'fault-injected-mismatch'
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'restart differs: ' + field):
                harness.check_acceptance(expected, corrupted, state)
        for field, value in (('mc_negative_elbo', 4.000001), ('decoder_mean_mse', 0.250001),
                             ('kl', 0.1), ('kl', 10.), ('kl', -0.1), ('kl', float('nan')),
                             ('mc_negative_elbo', float('inf')), ('constant_baseline_mse', 0.)):
            altered = copy.deepcopy(expected)
            altered['metrics'][field] = value
            altered['metrics']['mc_reconstruction_nll'] = altered['metrics']['mc_negative_elbo'] - altered['metrics']['kl']
            same = copy.deepcopy(resumed)
            same['metrics'] = altered['metrics']
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, same, state)
        for initial in (3.999999, 0., float('nan')):
            altered = copy.deepcopy(expected)
            altered['initial_metrics']['mc_negative_elbo'] = initial
            with self.assertRaisesRegex(AssertionError, 'learning gate failed|invalid metrics'):
                harness.check_acceptance(altered, resumed, state)
        altered = copy.deepcopy(expected)
        altered['metrics'].update(mc_reconstruction_nll=-2., mc_negative_elbo=-1.)
        same = copy.deepcopy(resumed)
        same['metrics'] = altered['metrics']
        harness.check_acceptance(altered, same, state)
        altered['metrics']['mc_negative_elbo'] = -0.5
        with self.assertRaisesRegex(AssertionError, 'inconsistent negative ELBO'):
            harness.check_acceptance(altered, same, state)


class VaeOptimizedGates(unittest.TestCase):
    def test_gates_survive_optimized_python(self):
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        result = subprocess.run([sys.executable, '-B', '-O', str(Path(__file__).resolve()), '--sensitivity'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(VaeGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
