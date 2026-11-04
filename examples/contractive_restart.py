"""Public CPU f32 contractive autoencoder and fresh-process restart proof.

Predeclared gates: seeds 7/11/43, 400 steps, held-out reconstruction MSE <=
10% of the training-mean baseline, final encoder Jacobian penalty <= 75%
of initialization, and independent analytic-Jacobian gradient parity at
atol=1e-5/rtol=1e-4. This does not establish adversarial robustness or
latent identifiability: decoder rescaling can compensate encoder contraction.
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
from ferro.autograd import grad, vjp
from training_restart import snapshot, state_digest

SEEDS = (7, 11, 43)
STEPS = 400
CHECKPOINT_STEP = 197
BATCH_SIZE = 8
PENALTY_WEIGHT = 0.05
SCHEDULE = 'interleaved-cyclic-contractive-v2'
TASK = 'contractive-smooth-curve-v1'


class ContractiveModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 3
    penalty_weight: float = PENALTY_WEIGHT
    schedule: str = SCHEDULE

    def build(self):
        self.encoder = fr.nn.Linear(2, self.hidden)
        self.decoder = fr.nn.Linear(self.hidden, 2)
        with fr.no_grad():
            self.encoder.weight.tensor().copy_(fr.Tensor.randn([self.hidden, 2], seed=self.seed) * 0.5)
            self.encoder.bias.tensor().copy_(fr.Tensor.zeros([self.hidden]))
            self.decoder.weight.tensor().copy_(fr.Tensor.randn([2, self.hidden], seed=self.seed + 1) * self.hidden ** -0.5)
            self.decoder.bias.tensor().copy_(fr.Tensor.zeros([2]))

    def forward(self, inputs):
        hidden = self.encoder(inputs).tanh()
        return self.decoder(hidden), hidden


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK}
    for split, offset in (('train', 0.), ('held_out', 0.5)):
        data[split] = []
        for i in range(64):
            index = (i % BATCH_SIZE) * 8 + i // BATCH_SIZE if split == 'train' else i
            t = -1. + 2. * (index + offset) / 64
            data[split].append({'id': f'{split}-{index:02d}', 'x': [t, math.sin(1.5 * t)]})
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported contractive dataset schema')
    identities, contents = set(), set()
    for split in ('train', 'held_out'):
        rows = data[split]
        if type(rows) is not list or len(rows) != 64:
            raise ValueError('Expected 64 contractive samples per split')
        for row in rows:
            if type(row) is not dict or set(row) != {'id', 'x'}:
                raise ValueError('Invalid contractive sample schema')
            if type(row['id']) is not str or not row['id'].startswith(split + '-') or row['id'] in identities:
                raise ValueError('Sample identities must be unique and split-specific')
            identities.add(row['id'])
            values = row['x']
            if type(values) is not list or len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError('Invalid contractive sample values')
            encoded = canonical(values)
            if encoded in contents:
                raise ValueError('Sample contents must be disjoint, including held-out data')
            contents.add(encoded)


def batch(rows):
    return fr.Tensor([v for row in rows for v in row['x']], [len(rows), 2])


def jacobian_penalty(hidden, inputs):
    count, width = hidden.shape
    terms = []
    # Samples are independent. Each VJP selects one Jacobian row per sample;
    # the result is (1/B) sum_{sample,hidden,input} J^2, with a live graph.
    for channel in range(width):
        seed = fr.Tensor([float(j == channel) for _ in range(count) for j in range(width)], [count, width])
        row, = vjp(hidden, [inputs], seed, create_graph=True)
        terms.append((row * row).sum())
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total / count


def objective(model, inputs, targets, detach_penalty=False):
    output, hidden = model(inputs)
    reconstruction = fr.nn.functional.mse_loss(output, targets)
    penalty = jacobian_penalty(hidden, inputs)
    if detach_penalty:
        penalty = penalty.detach()
    return reconstruction, penalty, reconstruction + model.penalty_weight * penalty


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE,
              'batch_size': BATCH_SIZE, 'penalty_weight': PENALTY_WEIGHT, 'step': checkpoint.step,
              'next_sample': (checkpoint.step * BATCH_SIZE) % len(data['train']),
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'schedule', 'batch_size', 'penalty_weight', 'step',
            'next_sample', 'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported contractive run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE or type(record['penalty_weight']) is not float or record['penalty_weight'] != PENALTY_WEIGHT:
        raise ValueError('Contractive run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('Contractive dataset identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_sample']) is not int or record['next_sample'] != (step * BATCH_SIZE) % len(data['train']):
        raise ValueError('Contractive cursor mismatch')
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('Contractive checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('Contractive checkpoint/cursor step mismatch')
    return data, checkpoint


def gradient_check(seed, data, detach_penalty=False):
    import torch
    torch.set_num_threads(1)
    model = ContractiveModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    targets = batch(data['train'][:BATCH_SIZE])
    inputs = batch(data['train'][:BATCH_SIZE]).requires_grad_(True)
    tx = torch.tensor(inputs.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    th = torch.tanh(tx @ weights['encoder.weight'].T + weights['encoder.bias'])
    output = th @ weights['decoder.weight'].T + weights['decoder.bias']
    reconstruction = torch.nn.functional.mse_loss(output, torch.tensor(targets.tolist()))
    analytic_jacobian = (1 - th * th)[:, :, None] * weights['encoder.weight'][None, :, :]
    penalty = (analytic_jacobian * analytic_jacobian).sum() / len(tx)
    reference = (reconstruction, penalty, reconstruction + PENALTY_WEIGHT * penalty)
    actual = objective(model, inputs, targets, detach_penalty=detach_penalty)
    names = ['input', *weights]
    parameters = [inputs, *(p.tensor() for _, p in model.named_parameters())]
    reference_parameters = [tx, *weights.values()]
    errors, penalty_gradients = {}, {}
    for component, got_loss, expected_loss in zip(('reconstruction', 'penalty', 'total'), actual, reference):
        torch.testing.assert_close(torch.tensor(got_loss.item()), expected_loss.detach(), atol=1e-5, rtol=1e-4)
        # Fault controls intentionally compare even fully detached outputs as zeros.
        got_grads = got_loss.grad_wrt(parameters)
        expected_grads = torch.autograd.grad(expected_loss, reference_parameters, retain_graph=True, allow_unused=True)
        errors[component] = {}
        for name, got, expected, parameter in zip(names, got_grads, expected_grads, reference_parameters):
            if expected is None:
                expected = torch.zeros_like(parameter)
            actual_gradient = torch.tensor(got.tolist())
            try:
                torch.testing.assert_close(actual_gradient, expected, atol=1e-5, rtol=1e-4)
            except AssertionError as error:
                raise AssertionError(f'Contractive {component} gradient mismatch: {name}') from error
            errors[component][name] = float((actual_gradient - expected).abs().max())
            if component == 'penalty' and name.startswith('encoder.'):
                penalty_gradients[name] = float(actual_gradient.abs().max())
    if any(value <= 1e-4 for value in penalty_gradients.values()) or set(penalty_gradients) != {'encoder.weight', 'encoder.bias'}:
        raise AssertionError('Contractive penalty must have nonzero encoder gradients')
    return {'max_abs_errors': errors, 'penalty_encoder_gradient_max_abs': penalty_gradients,
            'values': {name: item.item() for name, item in zip(('reconstruction', 'penalty', 'total'), actual)}}


def measure(model, data):
    inputs = batch(data['held_out']).requires_grad_(True)
    reconstruction, penalty, _ = objective(model, inputs, batch(data['held_out']))
    mean = [sum(row['x'][j] for row in data['train']) / len(data['train']) for j in range(2)]
    baseline = sum((row['x'][j] - mean[j]) ** 2 for row in data['held_out'] for j in range(2)) / (2 * len(data['held_out']))
    return {'held_out_mse': reconstruction.item(), 'constant_baseline_mse': baseline,
            'reconstruction_ratio': reconstruction.item() / baseline, 'jacobian_penalty': penalty.item()}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = ContractiveModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.025 if restored is None else 0.7, weight_decay=0.001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid contractive training stop')
    initial_metrics = measure(model, data)
    batches = [batch(data['train'][i:i + BATCH_SIZE]) for i in range(0, len(data['train']), BATCH_SIZE)]
    losses = []
    for step in range(start, stop):
        targets = batches[step % len(batches)]
        inputs = (targets + fr.Tensor.randn(list(targets.shape), generator=rng) * 0.005).requires_grad_(True)
        opt.zero_grad()
        reconstruction, penalty, loss = objective(model, inputs, targets)
        values = [item.item() for item in (reconstruction, penalty, loss)]
        if any(not math.isfinite(value) for value in values):
            raise AssertionError('Nonfinite contractive training loss')
        losses.append(values)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, snapshot(model, opt, rng, step + 1), data, seed)
    state = snapshot(model, opt, rng, stop)
    result = {'losses': losses, 'metrics': measure(model, data), 'initial_metrics': initial_metrics,
              'next_draws': fr.Tensor.randn([8], generator=rng).tolist(), 'dataset_sha256': digest,
              'cursor': {'step': stop, 'next_sample': (stop * BATCH_SIZE) % len(data['train'])}}
    return state, result


def check_acceptance(expected, resumed, expected_state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', expected_state)):
        if resumed[field] != wanted:
            raise AssertionError('Contractive restart differs: ' + field)
    metrics, initial = expected['metrics'], expected['initial_metrics']
    values = [metrics[key] for key in ('held_out_mse', 'constant_baseline_mse', 'reconstruction_ratio', 'jacobian_penalty')]
    values.append(initial['jacobian_penalty'])
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values) or values[1] <= 0 or values[4] <= 0:
        raise AssertionError('Contractive invalid metrics: ' + repr(metrics))
    if not math.isclose(values[2], values[0] / values[1], rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError('Contractive inconsistent reconstruction ratio')
    if values[2] > 0.1 or values[3] / values[4] > 0.75:
        raise AssertionError('Contractive learning gate failed: ' + repr(metrics))


def prove(seed):
    gradients = gradient_check(seed, make_dataset())
    with tempfile.TemporaryDirectory(prefix='ferro-contractive-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_contractive.py')
        flags = ['-O'] if sys.flags.optimize else []
        subprocess.run([sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(midpoint),
                        '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(result_path)],
                       check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'schedule': SCHEDULE, 'seed': seed, 'steps': STEPS,
            'penalty_weight': PENALTY_WEIGHT, 'metrics': expected['metrics'],
            'initial_metrics': expected['initial_metrics'], 'gradient_check': gradients, 'restart': 'bit-exact',
            'dataset_sha256': expected['dataset_sha256'], 'checkpoint_step': CHECKPOINT_STEP,
            'penalty_ratio': expected['metrics']['jacobian_penalty'] / expected['initial_metrics']['jacobian_penalty'],
            'initial_losses': expected['losses'][0], 'final_losses': expected['losses'][-1]}


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
            parser.error('resume/dataset digest are reserved for the restart worker')
        for seed in SEEDS if args.seed is None else (args.seed,):
            print(json.dumps(prove(seed), sort_keys=True, allow_nan=False), flush=True)
