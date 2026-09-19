"""CPU f32 public training proofs. No engine code or private bindings.

Fixed gates: 400 steps, seeds 7/11/43, gradient atol=1e-5/rtol=1e-4.
Restart comparisons require bit-exact continuation in the same environment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import ferro as fr

STEPS = 400
SEEDS = (7, 11, 43)


class Model(fr.nn.Module):
    task: str
    seed: int
    hidden: int = 16

    def build(self):
        inputs, outputs = (1, 1) if self.task == 'regression' else (2, 2)
        self.w1 = fr.nn.Parameter(fr.Tensor.randn([inputs, self.hidden], seed=self.seed) * inputs ** -0.5)
        self.b1 = fr.nn.Parameter(fr.Tensor.zeros([self.hidden]))
        self.w2 = fr.nn.Parameter(fr.Tensor.randn([self.hidden, outputs], seed=self.seed + 1) * self.hidden ** -0.5)
        self.b2 = fr.nn.Parameter(fr.Tensor.zeros([outputs]))

    def forward(self, x):
        return (x @ self.w1.tensor() + self.b1.tensor()).tanh() @ self.w2.tensor() + self.b2.tensor()


def dataset(task, held_out=False):
    n = 128
    offset = 0.5 if held_out else 0.0
    if task == 'regression':
        x = [-1.0 + 2.0 * (i + offset) / n for i in range(n)]
        y = [math.sin(2.5 * v) + 0.3 * v * v for v in x]
        return fr.Tensor(x, [n, 1]), fr.Tensor(y, [n, 1]), y
    x, y = [], []
    for i in range(n // 2):
        angle = 2 * math.pi * (i + offset) / (n // 2)
        for label, radius in ((0, 0.45), (1, 1.15)):
            x.extend((radius * math.cos(angle), radius * math.sin(angle)))
            y.append(label)
    return fr.Tensor(x, [n, 2]), fr.Tensor.from_i64(y, [n]), y


def objective(task, logits, targets):
    if task == 'regression':
        return fr.nn.functional.mse_loss(logits, targets)
    return fr.nn.functional.cross_entropy(logits, targets)


def gradient_check(task, seed):
    import torch
    torch.set_num_threads(1)
    model = Model(task=task, seed=seed)
    x, y, _ = dataset(task)
    ids = [0, 3, 9, 22, 35, 61, 87, 110]
    x = x.index_select(0, ids).requires_grad_(True)
    y = y.index_select(0, ids)
    tx = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    logits = torch.tanh(tx @ weights['w1'] + weights['b1']) @ weights['w2'] + weights['b2']
    target = torch.tensor(y.tolist(), dtype=torch.float32 if task == 'regression' else torch.int64)
    reference = (torch.nn.functional.mse_loss(logits, target) if task == 'regression'
                 else torch.nn.functional.cross_entropy(logits, target))
    loss = objective(task, model(x), y)
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=1e-5, rtol=1e-4)
    loss.backward()
    reference.backward()
    pairs = [('input', x.grad, tx.grad)] + [
        (name, p.tensor().grad, weights[name].grad) for name, p in model.named_parameters()]
    errors = {}
    for name, actual, expected in pairs:
        if actual is None or expected is None:
            raise AssertionError('Missing gradient: ' + name)
        got = torch.tensor(actual.tolist())
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        errors[name] = float((got - expected).abs().max())
    return errors


def snapshot(model, optimizer, generator, step):
    return fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator, step=step)


def state_digest(checkpoint, directory):
    directory = Path(directory)
    checkpoint.save(directory)
    # Only publication directory names vary. Payload includes model, moments,
    # counters, schema/config and RNG state; compare the actual serialized bytes.
    metadata = json.loads((directory / 'checkpoint.json').read_text())
    generation = metadata.pop('generation')
    payload = (directory / generation / 'model.safetensors').read_bytes()
    return {'metadata': metadata, 'payload_sha256': hashlib.sha256(payload).hexdigest()}


def train(task, seed, stop=STEPS, resume=None, midpoint=None):
    model = Model(task=task, seed=seed)
    # Deliberately different starting RNG and optimizer settings on restore.
    rng = fr.Generator(seed + 100 if resume is None else seed + 999)
    opt = fr.optim.AdamW(model.parameters(), lr=0.025 if resume is None else 0.7, weight_decay=0.001)
    start = 0 if resume is None else fr.checkpoint.load_training(
        resume, model, optimizers={'main': opt}, generator=rng)
    x, y, _ = dataset(task)
    losses = []
    for step in range(start, stop):
        ids = [(step * 32 + j) % 128 for j in range(32)]
        batch = x.index_select(0, ids)
        # Stochastic input perturbation exercises the explicit Generator.
        batch = batch + fr.Tensor.randn(list(batch.shape), generator=rng) * 0.005
        target = y.index_select(0, ids)
        opt.zero_grad()
        loss = objective(task, model(batch), target)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite training loss')
        losses.append(value)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == STEPS // 2:
            snapshot(model, opt, rng, step + 1).save(midpoint)
    state = snapshot(model, opt, rng, stop)
    with fr.no_grad():
        hx, hy, labels = dataset(task, held_out=True)
        output = model(hx)
        if task == 'regression':
            mse = objective(task, output, hy).item()
            train_labels = dataset(task)[2]
            mean = sum(train_labels) / len(train_labels)
            baseline = sum((v - mean) ** 2 for v in labels) / len(labels)
            metrics = {'held_out_mse': mse, 'constant_baseline_mse': baseline, 'ratio': mse / baseline}
        else:
            predictions = [max(range(2), key=lambda j: row[j]) for row in output.tolist()]
            accuracy = sum(a == b for a, b in zip(predictions, labels)) / len(labels)
            metrics = {'held_out_accuracy': accuracy, 'class_prior_baseline': 0.5}
    next_draws = fr.Tensor.randn([8], generator=rng).tolist()
    return state, {'losses': losses, 'metrics': metrics, 'next_draws': next_draws}


def prove(task, seed):
    gradients = gradient_check(task, seed)
    with tempfile.TemporaryDirectory(prefix='ferro-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(task, seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_' + ('regression' if task == 'regression' else 'classifier') + '.py')
        command = [sys.executable, '-B', str(script), '--seed', str(seed), '--resume', str(midpoint),
                   '--worker-output', str(result_path)]
        subprocess.run(command, check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_text())
        assert resumed['losses'] == expected['losses'][STEPS // 2:], 'Restart loss trajectory differs'
        assert resumed['next_draws'] == expected['next_draws'], 'Restart RNG differs'
        assert resumed['state'] == expected_state, 'Restart model/optimizer/config/RNG state differs'
        assert resumed['metrics'] == expected['metrics'], 'Restart held-out result differs'
    metrics = expected['metrics']
    if task == 'regression':
        assert metrics['ratio'] <= 0.25, metrics
    else:
        assert metrics['held_out_accuracy'] >= 0.95, metrics
    return {'task': task, 'seed': seed, 'steps': STEPS, 'metrics': metrics,
            'gradient_max_abs_errors': gradients, 'restart': 'bit-exact',
            'initial_loss': expected['losses'][0], 'final_loss': expected['losses'][-1]}


def main(task):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    if args.worker_output is not None:
        if args.resume is None or args.seed is None:
            parser.error('worker output requires seed and resume')
        state, result = train(task, args.seed, resume=args.resume)
        result['state'] = state_digest(state, args.worker_output.with_suffix('.state'))
        args.worker_output.write_text(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        if args.resume is not None:
            parser.error('resume is reserved for the restart worker')
        for seed in SEEDS if args.seed is None else (args.seed,):
            print(json.dumps(prove(task, seed), sort_keys=True, allow_nan=False), flush=True)
