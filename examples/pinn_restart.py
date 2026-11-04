"""Strong-form CPU Poisson PINN with second coordinate and mixed gradients.

u'' = -pi^2 sin(pi*x), u(0)=u(1)=0. Boundary conditions are enforced by
u=x(1-x)*MLP(2x-1); analytic solution values are used only for evaluation.
Declared gates: seeds 7/11/43, 1500 AdamW steps, held-out solution RMSE <= 0.01,
residual RMSE<=0.1, boundary maxabs<=1e-6, and exact fresh-process restart.
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
from ferro.autograd import grad
from training_restart import state_digest

SEEDS = (7, 11, 43)
STEPS = 1500
CHECKPOINT_STEP = 749
BATCH_SIZE = 8
SCHEDULE = 'cyclic-collocation-v1'
TASK = 'strong-poisson-hard-dirichlet-v1'
CONFIG = {'frequency': math.pi, 'hidden': 12, 'jitter': 0.003, 'steps': STEPS,
          'batch_size': BATCH_SIZE, 'schedule': SCHEDULE, 'boundary_ansatz': 'x*(1-x)'}


class PINNModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 12
    frequency: float = math.pi
    jitter: float = 0.003
    schedule: str = SCHEDULE
    equation: str = TASK
    steps: int = STEPS
    batch_size: int = BATCH_SIZE

    def build(self):
        self.w1 = fr.nn.Parameter(fr.Tensor.randn([1, self.hidden], seed=self.seed) * 0.5)
        self.b1 = fr.nn.Parameter(fr.Tensor.randn([self.hidden], seed=self.seed + 1) * 0.1)
        self.w2 = fr.nn.Parameter(fr.Tensor.randn([self.hidden, 1], seed=self.seed + 2) * 0.3)
        self.b2 = fr.nn.Parameter(fr.Tensor.zeros([1]))

    def forward(self, x):
        network = ((2 * x - 1) @ self.w1.tensor() + self.b1.tensor()).tanh() @ self.w2.tensor() + self.b2.tensor()
        return x * (1 - x) * network


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK, 'train': [(i + 0.5) / 32 for i in range(32)],
            'held_out': [(i + 0.25) / 64 for i in range(64)], 'boundary': [0., 1.]}
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out', 'boundary'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported PINN dataset schema')
    for split, count in (('train', 32), ('held_out', 64)):
        points = data[split]
        if type(points) is not list or len(points) != count or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in points) or len(set(points)) != count:
            raise ValueError('Invalid PINN collocation points')
    if set(data['train']) & set(data['held_out']) or type(data['boundary']) is not list or data['boundary'] != [0., 1.] or any(type(v) not in (int, float) for v in data['boundary']):
        raise ValueError('PINN held-out points must be disjoint with fixed endpoints')
    if min(data['train']) <= CONFIG['jitter'] or max(data['train']) >= 1 - CONFIG['jitter']:
        raise ValueError('PINN jitter could leave the domain')
    if min(abs(a - b) for a in data['train'] for b in data['held_out']) <= CONFIG['jitter']:
        raise ValueError('PINN jitter could reach held-out coordinates')


def derivatives(model, x):
    value = model(x)
    first, = grad(value.sum(), [x], create_graph=True)
    if not first.requires_grad:
        raise AssertionError('PINN first derivative detached')
    second, = grad(first.sum(), [x], create_graph=True)
    if not second.requires_grad:
        raise AssertionError('PINN second derivative detached')
    return value, first, second


def physics(model, x):
    value, first, second = derivatives(model, x)
    if not second.requires_grad:
        raise AssertionError('PINN second derivative detached')
    forcing = fr.Tensor([-model.frequency ** 2 * math.sin(model.frequency * row[0]) for row in x.tolist()], list(x.shape))
    residual = second - forcing
    return (residual * residual).mean(), (value, first, second)


def torch_model(x, weights):
    return x * (1 - x) * (((2 * x - 1) @ weights['w1'] + weights['b1']).tanh() @ weights['w2'] + weights['b2'])


def torch_physics(x, weights):
    import torch
    value = torch_model(x, weights)
    first, = torch.autograd.grad(value.sum(), x, create_graph=True)
    second, = torch.autograd.grad(first.sum(), x, create_graph=True)
    return ((second + math.pi ** 2 * torch.sin(math.pi * x)) ** 2).mean(), (value, first, second)


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = PINNModel(seed=seed, dataset_digest=digest)
    points = [data['train'][i] for i in (0, 4, 9, 13, 18, 22, 27, 31)]
    x = fr.Tensor(points, [len(points), 1]).requires_grad_(True)
    tx = torch.tensor(points, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    loss, actual = physics(model, x)
    reference, expected = torch_physics(tx, weights)
    errors = {}
    for name, got, want in zip(('u', 'ux', 'uxx'), actual, expected):
        values = torch.tensor(got.tolist())
        torch.testing.assert_close(values, want.detach(), atol=5e-5, rtol=2e-4)
        errors[name] = float((values - want.detach()).abs().max())
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=5e-5, rtol=2e-4)
    loss.backward()
    reference.backward()
    norms, parameter_errors = {}, {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or weights[name].grad is None:
            raise AssertionError('Missing PINN residual parameter gradient: ' + name)
        got, want = torch.tensor(parameter.grad.tolist()), weights[name].grad
        torch.testing.assert_close(got, want, atol=5e-5, rtol=2e-4)
        norms[name] = float(got.abs().max())
        if not norms[name] > 1e-6:
            raise AssertionError('Zero PINN mixed parameter gradient: ' + name)
        parameter_errors[name] = float((got - want).abs().max())
    double_weights = {name: value.detach().to(torch.float64) for name, value in weights.items()}
    dx = tx.detach().to(torch.float64)
    h = 1e-4
    numeric_second = (torch_model(dx + h, double_weights) - 2 * torch_model(dx, double_weights) + torch_model(dx - h, double_weights)) / h ** 2
    actual_second = torch.tensor(actual[2].tolist(), dtype=torch.float64)
    torch.testing.assert_close(actual_second, numeric_second, atol=2e-3, rtol=2e-3)
    fd_errors = {}
    for name, parameter in model.named_parameters():
        got = torch.tensor(parameter.grad.tolist(), dtype=torch.float64).reshape(-1)
        index = int(got.abs().argmax())
        pair = []
        for sign in (-1, 1):
            changed = {key: value.clone() for key, value in double_weights.items()}
            changed[name].reshape(-1)[index] += sign * 1e-5
            result, _ = torch_physics(dx.clone().requires_grad_(True), changed)
            pair.append(float(result.detach()))
        numeric = (pair[1] - pair[0]) / 2e-5
        if not math.isclose(float(got[index]), numeric, rel_tol=2e-3, abs_tol=2e-3):
            raise AssertionError('PINN mixed gradient finite difference mismatch: ' + name)
        fd_errors[name] = abs(float(got[index]) - numeric)
    return {'derivative_max_abs_errors': errors, 'parameter_max_abs_errors': parameter_errors,
            'mixed_gradient_max_abs': norms, 'finite_difference_parameter_errors': fd_errors,
            'finite_difference_uxx_max_abs_error': float((actual_second - numeric_second).abs().max())}


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'config': CONFIG,
              'step': checkpoint.step, 'next_point': (checkpoint.step * BATCH_SIZE) % 32,
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'config', 'step', 'next_point', 'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported PINN run schema')
    if type(record['seed']) is not int or record['seed'] != seed or canonical(record['config']) != canonical(CONFIG):
        raise ValueError('PINN run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('PINN dataset identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_point']) is not int or record['next_point'] != (step * BATCH_SIZE) % 32:
        raise ValueError('PINN cursor mismatch')
    if hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest() != record['checkpoint_index_sha256']:
        raise ValueError('PINN checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('PINN checkpoint/cursor step mismatch')
    return data, checkpoint


def measure(model, data):
    x = fr.Tensor(data['held_out'], [len(data['held_out']), 1]).requires_grad_(True)
    _, (values, _, second) = physics(model, x)
    output, curvature = values.tolist(), second.tolist()
    solution = [math.sin(model.frequency * point) for point in data['held_out']]
    forcing = [-model.frequency ** 2 * value for value in solution]
    solution_rmse = math.sqrt(sum((row[0] - target) ** 2 for row, target in zip(output, solution)) / len(solution))
    residual_rmse = math.sqrt(sum((row[0] - target) ** 2 for row, target in zip(curvature, forcing)) / len(forcing))
    with fr.no_grad():
        boundary = model(fr.Tensor(data['boundary'], [2, 1])).tolist()
    return {'solution_rmse': solution_rmse, 'residual_rmse': residual_rmse,
            'boundary_max_abs': max(abs(row[0]) for row in boundary),
            'zero_solution_rmse': math.sqrt(sum(v * v for v in solution) / len(solution)),
            'zero_residual_rmse': math.sqrt(sum(v * v for v in forcing) / len(forcing))}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = PINNModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.01 if restored is None else 0.7, weight_decay=0.0)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid PINN training stop')
    initial = measure(model, data)
    base = fr.Tensor(data['train'], [32, 1])
    losses = []
    for step in range(start, stop):
        ids = [(step * BATCH_SIZE + j) % 32 for j in range(BATCH_SIZE)]
        noise = (fr.Tensor.rand([BATCH_SIZE, 1], generator=rng) * 2 - 1) * model.jitter
        x = (base.index_select(0, ids) + noise).requires_grad_(True)
        opt.zero_grad()
        loss, _ = physics(model, x)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite PINN physics loss')
        losses.append(value)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=step + 1), data, seed)
    state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=stop)
    result = {'losses': losses, 'initial_metrics': initial, 'metrics': measure(model, data),
              'next_draws': fr.Tensor.rand([8], generator=rng).tolist(), 'dataset_sha256': digest,
              'cursor': {'step': stop, 'next_point': (stop * BATCH_SIZE) % 32}}
    return state, result


def check_acceptance(expected, resumed, state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', state)):
        if resumed[field] != wanted:
            raise AssertionError('PINN restart differs: ' + field)
    metrics = expected['metrics']
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in metrics.values()):
        raise AssertionError('PINN invalid metrics: ' + repr(metrics))
    if not metrics['solution_rmse'] <= 0.01 or not metrics['residual_rmse'] <= 0.1 or not metrics['boundary_max_abs'] <= 1e-6:
        raise AssertionError('PINN physics gate failed: ' + repr(metrics))


def prove(seed):
    data = make_dataset()
    derivatives_checked = gradient_check(seed, data)
    with tempfile.TemporaryDirectory(prefix='ferro-pinn-restart-') as directory:
        directory = Path(directory)
        state, expected = train(seed, midpoint=directory / 'midpoint')
        expected_state = state_digest(state, directory / 'expected')
        path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_pinn.py')
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(directory / 'midpoint'),
                   '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(path)]
        subprocess.run(command, check=True, timeout=180, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        check_acceptance(expected, json.loads(path.read_bytes()), expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'checkpoint_step': CHECKPOINT_STEP,
            'metrics': expected['metrics'], 'initial_metrics': expected['initial_metrics'], 'derivatives': derivatives_checked,
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
