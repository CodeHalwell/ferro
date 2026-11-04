"""Two-layer fixed-knot CPU KAN learning and fresh-process restart proof.

Declared gates: seeds 7/11/43, 400 AdamW steps, held-out MSE <= 10% of
both training-fit constant and affine baselines, gradient atol=1e-5/rtol=1e-4,
and exact complete-step restart in the same binary/environment. Knots are
fixed, inputs/coefficients are f32, and only first-order AD is certified.
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
from ferro.basis import BSplineBasis, KanLayer
from training_restart import state_digest

SEEDS = (7, 11, 43)
STEPS = 400
CHECKPOINT_STEP = 197
BATCH_SIZE = 16
SCHEDULE = 'cyclic-spline-samples-v1'
TASK = 'two-layer-spline-v1'
KNOTS = (-2., -2., -2., -2., -1., -0.5, 0., 0.5, 1., 2., 2., 2., 2.)
DEGREE = 3


class KANModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    knots: tuple = KNOTS
    degree: int = DEGREE
    outside: str = 'zero'
    schedule: str = SCHEDULE

    def build(self):
        basis = BSplineBasis(list(self.knots), self.degree, outside=self.outside)
        greville = [sum(self.knots[i + 1:i + self.degree + 1]) / self.degree for i in range(basis.num_basis)]
        projections = ((1., 0.4), (-0.4, 0.8), (0.5, -0.7))
        initial = [weight * point for row in projections for weight in row for point in greville]
        first = fr.Tensor(initial, [3, 2, basis.num_basis]) + fr.Tensor.randn([3, 2, basis.num_basis], seed=self.seed) * 0.01
        second = fr.Tensor.randn([1, 3, basis.num_basis], seed=self.seed + 1) * 0.1
        self.first = KanLayer(basis, first)
        self.second = KanLayer(basis, second)

    def forward(self, x):
        return self.second(self.first(x))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK}
    for split, phase in (('train', 0.25), ('held_out', 0.75)):
        rows = []
        for i in range(8):
            for j in range(8):
                x = [-0.9 + 1.8 * (i + phase) / 8, -0.9 + 1.8 * (j + phase) / 8]
                y = math.sin(2 * x[0] + 0.8 * x[1]) + 0.3 * x[0] * x[1]
                rows.append({'id': f'{split}-{i}-{j}', 'x': x, 'y': y})
        data[split] = rows
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported KAN dataset schema')
    identities, inputs = set(), set()
    for split in ('train', 'held_out'):
        rows = data[split]
        if type(rows) is not list or len(rows) != 64:
            raise ValueError('Expected 64 KAN samples per split')
        for row in rows:
            if type(row) is not dict or set(row) != {'id', 'x', 'y'} or type(row['id']) is not str:
                raise ValueError('Invalid KAN sample schema')
            if not row['id'].startswith(split + '-') or row['id'] in identities:
                raise ValueError('KAN sample identities must be unique and split-specific')
            identities.add(row['id'])
            if type(row['x']) is not list or len(row['x']) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not -0.9 < v < 0.9 for v in row['x']):
                raise ValueError('Invalid KAN inputs')
            if type(row['y']) not in (int, float) or not math.isfinite(row['y']):
                raise ValueError('Invalid KAN target')
            if tuple(row['x']) in inputs:
                raise ValueError('KAN train/held-out inputs must be disjoint')
            inputs.add(tuple(row['x']))


def tensors(rows):
    return fr.Tensor([v for row in rows for v in row['x']], [len(rows), 2]), fr.Tensor([row['y'] for row in rows], [len(rows), 1])


def baselines(data):
    rows = data['train']
    mean_x = [sum(row['x'][j] for row in rows) / len(rows) for j in range(2)]
    mean_y = sum(row['y'] for row in rows) / len(rows)
    # On the Cartesian grid the centered feature columns are orthogonal.
    slopes = [sum((row['x'][j] - mean_x[j]) * (row['y'] - mean_y) for row in rows) /
              sum((row['x'][j] - mean_x[j]) ** 2 for row in rows) for j in range(2)]
    cross = sum((row['x'][0] - mean_x[0]) * (row['x'][1] - mean_x[1]) for row in rows)
    if abs(cross) > 1e-10:
        raise ValueError('Affine baseline expects the declared Cartesian grid')
    held = data['held_out']
    constant = sum((row['y'] - mean_y) ** 2 for row in held) / len(held)
    affine = sum((row['y'] - mean_y - sum(slopes[j] * (row['x'][j] - mean_x[j]) for j in range(2))) ** 2 for row in held) / len(held)
    return {'constant_baseline_mse': constant, 'affine_baseline_mse': affine}


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE, 'batch_size': BATCH_SIZE,
              'step': checkpoint.step, 'next_sample': (checkpoint.step * BATCH_SIZE) % 64,
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'schedule', 'batch_size', 'step', 'next_sample',
            'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported KAN run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE:
        raise ValueError('KAN run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('KAN dataset identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_sample']) is not int or record['next_sample'] != (step * BATCH_SIZE) % 64:
        raise ValueError('KAN cursor mismatch')
    if hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest() != record['checkpoint_index_sha256']:
        raise ValueError('KAN checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('KAN checkpoint/cursor step mismatch')
    return data, checkpoint


def torch_basis(x, knots, degree):
    import torch
    value = x.to(torch.float64)
    t = torch.tensor(knots, dtype=torch.float64)
    level = ((value[..., None] >= t[:-1]) & (value[..., None] < t[1:])).to(torch.float64)
    for d in range(1, degree + 1):
        next_level = []
        for i in range(level.shape[-1] - 1):
            left, right = knots[i + d] - knots[i], knots[i + d + 1] - knots[i + 1]
            a = (value - knots[i]) / left * level[..., i] if left else torch.zeros_like(value)
            b = (knots[i + d + 1] - value) / right * level[..., i + 1] if right else torch.zeros_like(value)
            next_level.append(a + b)
        level = torch.stack(next_level, dim=-1)
    return level.to(torch.float32)


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = KANModel(seed=seed, dataset_digest=digest)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    candidates = torch.tensor([row['x'] for row in data['train']], dtype=torch.float32)
    hidden = torch.einsum('nik,oik->no', torch_basis(candidates, KNOTS, DEGREE), weights['first.coefficients'])
    knots = torch.tensor(sorted(set(KNOTS)), dtype=torch.float32)
    input_margin = (candidates[..., None] - knots).abs().amin(dim=(1, 2))
    hidden_margin = (hidden[..., None] - knots).abs().amin(dim=(1, 2))
    ids = [i for i in range(len(candidates)) if input_margin[i] > 1e-3 and hidden_margin[i] > 1e-3
           and hidden[i].abs().max() < 1.999][:8]
    if len(ids) != 8:
        raise AssertionError('Insufficient differentiable KAN oracle points')
    x, y = tensors([data['train'][i] for i in ids])
    x.requires_grad_(True)
    tx = candidates[ids].clone().requires_grad_(True)
    hidden = torch.einsum('nik,oik->no', torch_basis(tx, KNOTS, DEGREE), weights['first.coefficients'])
    output = torch.einsum('nik,oik->no', torch_basis(hidden, KNOTS, DEGREE), weights['second.coefficients'])
    target = torch.tensor(y.tolist(), dtype=torch.float32)
    reference = torch.nn.functional.mse_loss(output, target)
    actual = model(x)
    torch.testing.assert_close(torch.tensor(actual.tolist()), output.detach(), atol=1e-5, rtol=1e-4)
    loss = fr.nn.functional.mse_loss(actual, y)
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=1e-5, rtol=1e-4)
    loss.backward()
    reference.backward()
    errors = {}
    for name, got, want in [('input', x.grad, tx.grad)] + [(name, p.grad, weights[name].grad) for name, p in model.named_parameters()]:
        if got is None or want is None:
            raise AssertionError('Missing KAN gradient: ' + name)
        actual = torch.tensor(got.tolist())
        torch.testing.assert_close(actual, want, atol=1e-5, rtol=1e-4)
        errors[name] = float((actual - want).abs().max())
    return {'max_abs_errors': errors, 'input_knot_margin': float(input_margin[ids].min()),
            'hidden_knot_margin': float(hidden_margin[ids].detach().min()), 'sample_ids': ids}


def measure(model, data):
    x, y = tensors(data['held_out'])
    with fr.no_grad():
        mse = fr.nn.functional.mse_loss(model(x), y).item()
    values = baselines(data)
    return {'held_out_mse': mse, **values, 'constant_ratio': mse / values['constant_baseline_mse'],
            'affine_ratio': mse / values['affine_baseline_mse']}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = KANModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.02 if restored is None else 0.7, weight_decay=0.001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid KAN training stop')
    initial = measure(model, data)
    x, y = tensors(data['train'])
    losses = []
    for step in range(start, stop):
        ids = [(step * BATCH_SIZE + j) % 64 for j in range(BATCH_SIZE)]
        features = x.index_select(0, ids) + fr.Tensor.randn([BATCH_SIZE, 2], generator=rng) * 0.003
        targets = y.index_select(0, ids)
        opt.zero_grad()
        loss = fr.nn.functional.mse_loss(model(features), targets)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite KAN training loss')
        losses.append(value)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=step + 1), data, seed)
    state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=stop)
    result = {'losses': losses, 'initial_metrics': initial, 'metrics': measure(model, data),
              'next_draws': fr.Tensor.randn([8], generator=rng).tolist(), 'dataset_sha256': digest,
              'cursor': {'step': stop, 'next_sample': (stop * BATCH_SIZE) % 64}}
    return state, result


def check_acceptance(expected, resumed, state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', state)):
        if resumed[field] != wanted:
            raise AssertionError('KAN restart differs: ' + field)
    metrics = expected['metrics']
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in metrics.values()):
        raise AssertionError('KAN invalid metrics: ' + repr(metrics))
    for name in ('constant', 'affine'):
        baseline = metrics[name + '_baseline_mse']
        ratio = metrics[name + '_ratio']
        if baseline <= 0 or not math.isclose(ratio, metrics['held_out_mse'] / baseline, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError('KAN inconsistent metrics: ' + repr(metrics))
        if not ratio <= 0.1:
            raise AssertionError('KAN learning gate failed: ' + repr(metrics))


def prove(seed):
    data = make_dataset()
    gradients = gradient_check(seed, data)
    with tempfile.TemporaryDirectory(prefix='ferro-kan-restart-') as directory:
        directory = Path(directory)
        state, expected = train(seed, midpoint=directory / 'midpoint')
        expected_state = state_digest(state, directory / 'expected')
        path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_kan.py')
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(directory / 'midpoint'),
                   '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(path)]
        subprocess.run(command, check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        check_acceptance(expected, json.loads(path.read_bytes()), expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'checkpoint_step': CHECKPOINT_STEP,
            'metrics': expected['metrics'], 'initial_metrics': expected['initial_metrics'], 'gradients': gradients,
            'restart': 'bit-exact', 'dataset_sha256': expected['dataset_sha256'],
            'initial_loss': expected['losses'][0], 'final_loss': expected['losses'][-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dataset-digest')
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    if args.worker_output is not None:
        if args.resume is None or args.seed is None or args.dataset_digest is None:
            parser.error('worker output requires seed, resume and dataset digest')
        state, result = train(args.seed, resume=args.resume, expected_digest=args.dataset_digest)
        result['state'] = state_digest(state, args.worker_output.with_suffix('.state'))
        args.worker_output.write_bytes(canonical(result))
    else:
        if args.resume is not None or args.dataset_digest is not None:
            parser.error('resume/dataset digest are reserved for the worker')
        for seed in SEEDS if args.seed is None else (args.seed,):
            print(json.dumps(prove(seed), sort_keys=True, allow_nan=False), flush=True)
