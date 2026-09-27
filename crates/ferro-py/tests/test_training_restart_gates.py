"""Acceptance sensitivity with real CPU training and fresh worker processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'examples'))
import training_restart as harness


class RestartGateSensitivity(unittest.TestCase):
    def test_worker_mismatches_are_rejected(self):
        real_run = subprocess.run
        messages = {'losses': 'Restart loss trajectory differs', 'next_draws': 'Restart RNG differs',
                    'state': 'Restart model/optimizer/config/RNG state differs', 'metrics': 'Restart held-out result differs'}
        for field, message in messages.items():
            with self.subTest(field=field):
                def corrupted(command, **kwargs):
                    result = real_run(command, **kwargs)
                    path = Path(command[command.index('--worker-output') + 1])
                    output = json.loads(path.read_text())
                    if field in ('losses', 'next_draws'):
                        output[field][0] += 1.0
                    elif field == 'state':
                        output['state']['payload_sha256'] = 'fault-injected-not-a-hash'
                    else:
                        output['metrics']['ratio'] = 1000.0
                    path.write_text(json.dumps(output))
                    return result
                with patch.object(harness.subprocess, 'run', corrupted):
                    with self.assertRaisesRegex(AssertionError, message):
                        harness.prove('regression', 7)

    def test_learning_thresholds_remain_enforced(self):
        real_train, real_loads = harness.train, json.loads
        for task, key, boundary, bad in (('regression', 'ratio', 0.25, 0.250001),
                                         ('classifier', 'held_out_accuracy', 0.95, 0.949999)):
            for value in (boundary, bad, float('nan')):
                with self.subTest(task=task, value=value):
                    metrics = {key: value}

                    def changed_train(*args, **kwargs):
                        state, output = real_train(*args, **kwargs)
                        output['metrics'] = metrics
                        return state, output

                    def changed_metrics(*args, **kwargs):
                        output = real_loads(*args, **kwargs)
                        if isinstance(output, dict) and 'losses' in output:
                            # Equal injected metrics isolate learning from restart gates.
                            # Shared NaN identity makes dictionary equality pass too.
                            output['metrics'] = metrics
                        return output

                    with patch.object(harness, 'train', changed_train), patch.object(harness.json, 'loads', changed_metrics):
                        if value == boundary:
                            self.assertEqual(harness.prove(task, 7)['restart'], 'bit-exact')
                        else:
                            with self.assertRaises(AssertionError) as caught:
                                harness.prove(task, 7)
                            self.assertIs(caught.exception.args[0], metrics)


class TrainingRestartGateProcesses(unittest.TestCase):
    def test_acceptance_in_normal_and_optimized_processes(self):
        for mode, flags, optimize in (('normal', [], '0'), ('optimized', ['-O'], '0'),
                                      ('environment', [], '1')):
            with self.subTest(mode=mode):
                env = {**os.environ, 'PYTHONOPTIMIZE': optimize, 'PYTHONDONTWRITEBYTECODE': '1',
                       'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                result = subprocess.run([sys.executable, '-B', *flags, str(Path(__file__).resolve()),
                                         '--sensitivity'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--sensitivity' in sys.argv:
        sys.argv.remove('--sensitivity')
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RestartGateSensitivity)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
