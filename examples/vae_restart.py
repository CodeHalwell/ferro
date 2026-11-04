"""Public CPU Gaussian VAE learning and exact fresh-process restart.

Declared standard ELBO (beta=1): feature-summed Gaussian NLL with fixed
observation sigma=0.2 plus latent-summed Gaussian KL, both batch means.
Seeds 7/11/43, 600 steps, fixed-bank held-out MC negative ELBO <= half its
initial value and constant-Gaussian baseline, decoder(mu) MSE <= 25% of
the constant baseline, and 0.1 < KL < 10. Initial/baseline negative ELBO
must be positive; final Gaussian NLL/negative ELBO may be negative.
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
from training_restart import snapshot, state_digest

SEEDS = (7, 11, 43)
STEPS = 600
CHECKPOINT_STEP = 293
BATCH_SIZE = 8
EVAL_DRAWS = 4
OBSERVATION_SIGMA = 0.2
BETA = 1.0
SCHEDULE = 'interleaved-cyclic-vae-v1'
TASK = 'gaussian-vae-curve-v1'


class VAEModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 8
    latent: int = 1
    observation_sigma: float = OBSERVATION_SIGMA
    beta: float = BETA
    schedule: str = SCHEDULE

    def build(self):
        self.encoder = fr.nn.Linear(2, self.hidden)
        self.mu = fr.nn.Linear(self.hidden, self.latent)
        self.logvar = fr.nn.Linear(self.hidden, self.latent)
        self.decoder = fr.nn.Linear(self.latent, self.hidden)
        self.output = fr.nn.Linear(self.hidden, 2)
        with fr.no_grad():
            for index, (name, parameter) in enumerate(self.named_parameters()):
                shape = list(parameter.tensor().shape)
                if name.endswith('weight'):
                    scale = shape[1] ** -0.5 * (0.1 if name == 'logvar.weight' else 1.)
                    value = fr.Tensor.randn(shape, seed=self.seed + index) * scale
                else:
                    value = fr.Tensor.zeros(shape) - (1. if name == 'logvar.bias' else 0.)
                parameter.tensor().copy_(value)

    def encode(self, inputs):
        hidden = self.encoder(inputs).tanh()
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, latent):
        return self.output(self.decoder(latent).tanh())

    def forward(self, inputs, epsilon, fault=None):
        mu, logvar = self.encode(inputs)
        scale = (logvar * 0.5).exp()
        if fault == 'scale':
            scale = scale.detach()
        latent = mu + scale * epsilon
        if fault == 'latent':
            latent = latent.detach()
        return self.decode(latent), mu, logvar


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK, 'eval_draws': EVAL_DRAWS}
    for split, offset in (('train', 0.), ('held_out', 0.5)):
        data[split] = []
        for i in range(64):
            index = (i % BATCH_SIZE) * 8 + i // BATCH_SIZE if split == 'train' else i
            t = -1. + 2. * (index + offset) / 64
            data[split].append({'id': f'{split}-{index:02d}', 'x': [t, math.sin(1.5 * t)]})
    # This bank is saved as data, never drawn from the training stream.
    data['eval_epsilon'] = fr.Tensor.randn([64 * EVAL_DRAWS, 1], seed=20261103).tolist()
    validate_dataset(data)
    return data


def validate_dataset(data):
    keys = {'version', 'task', 'eval_draws', 'eval_epsilon', 'train', 'held_out'}
    if type(data) is not dict or set(data) != keys or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported VAE dataset schema')
    if type(data['eval_draws']) is not int or data['eval_draws'] != EVAL_DRAWS:
        raise ValueError('Invalid VAE evaluation draw count')
    bank = data['eval_epsilon']
    if type(bank) is not list or len(bank) != 64 * EVAL_DRAWS or any(type(row) is not list or len(row) != 1 or type(row[0]) not in (int, float) or not math.isfinite(row[0]) for row in bank):
        raise ValueError('Invalid VAE evaluation epsilon bank')
    identities, contents = set(), set()
    for split in ('train', 'held_out'):
        rows = data[split]
        if type(rows) is not list or len(rows) != 64:
            raise ValueError('Expected 64 VAE samples per split')
        for row in rows:
            if type(row) is not dict or set(row) != {'id', 'x'}:
                raise ValueError('Invalid VAE sample schema')
            if type(row['id']) is not str or not row['id'].startswith(split + '-') or row['id'] in identities:
                raise ValueError('Sample identities must be unique and split-specific')
            identities.add(row['id'])
            values = row['x']
            if type(values) is not list or len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError('Invalid VAE sample values')
            encoded = canonical(values)
            if encoded in contents:
                raise ValueError('Sample contents must be disjoint, including held-out data')
            contents.add(encoded)


def batch(rows, copies=1):
    return fr.Tensor([v for row in rows for _ in range(copies) for v in row['x']], [len(rows) * copies, 2])


def objective(model, inputs, targets, epsilon, fault=None):
    output, mu, logvar = model(inputs, epsilon, fault=fault)
    count, features = inputs.shape
    difference = output - targets
    reconstruction = (difference * difference).sum() / (2. * model.observation_sigma ** 2 * count)
    reconstruction = reconstruction + features * math.log(model.observation_sigma * math.sqrt(2. * math.pi))
    kl = (mu * mu + logvar.exp() - 1. - logvar).sum() / (2. * count)
    if fault == 'kl':
        kl = kl.detach()
    return reconstruction, kl, reconstruction + model.beta * kl


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE,
              'batch_size': BATCH_SIZE, 'beta': BETA, 'observation_sigma': OBSERVATION_SIGMA,
              'step': checkpoint.step, 'next_sample': (checkpoint.step * BATCH_SIZE) % len(data['train']),
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'schedule', 'batch_size', 'beta', 'observation_sigma',
            'step', 'next_sample', 'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported VAE run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE or type(record['beta']) is not float or record['beta'] != BETA or type(record['observation_sigma']) is not float or record['observation_sigma'] != OBSERVATION_SIGMA:
        raise ValueError('VAE run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('VAE dataset/evaluation-noise identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_sample']) is not int or record['next_sample'] != (step * BATCH_SIZE) % len(data['train']):
        raise ValueError('VAE cursor mismatch')
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('VAE checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('VAE checkpoint/cursor step mismatch')
    return data, checkpoint


def gradient_check(seed, data, fault=None):
    import torch
    torch.set_num_threads(1)
    model = VAEModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    targets = batch(data['train'][:BATCH_SIZE])
    inputs = batch(data['train'][:BATCH_SIZE]).requires_grad_(True)
    epsilon = fr.Tensor([row[0] for row in data['eval_epsilon'][:BATCH_SIZE]], [BATCH_SIZE, 1]).requires_grad_(True)
    tx = torch.tensor(inputs.tolist(), dtype=torch.float32, requires_grad=True)
    te = torch.tensor(epsilon.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    hidden = torch.tanh(tx @ weights['encoder.weight'].T + weights['encoder.bias'])
    mu = hidden @ weights['mu.weight'].T + weights['mu.bias']
    logvar = hidden @ weights['logvar.weight'].T + weights['logvar.bias']
    latent = mu + torch.exp(logvar / 2.) * te
    decoded = torch.tanh(latent @ weights['decoder.weight'].T + weights['decoder.bias'])
    output = decoded @ weights['output.weight'].T + weights['output.bias']
    difference = output - torch.tensor(targets.tolist())
    reconstruction = (0.5 * (difference / OBSERVATION_SIGMA) ** 2 + math.log(OBSERVATION_SIGMA) + 0.5 * math.log(2. * math.pi)).sum(1).mean()
    kl = (0.5 * (mu ** 2 + torch.exp(logvar) - 1. - logvar)).sum(1).mean()
    reference = (reconstruction, kl, reconstruction + kl)
    actual = objective(model, inputs, targets, epsilon, fault=fault)
    names = ['input', 'epsilon', *weights]
    parameters = [inputs, epsilon, *(p.tensor() for _, p in model.named_parameters())]
    reference_parameters = [tx, te, *weights.values()]
    errors = {}
    for component, got_loss, expected_loss in zip(('reconstruction_nll', 'kl', 'negative_elbo'), actual, reference):
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
                raise AssertionError(f'VAE {component} gradient mismatch: {name}') from error
            errors[component][name] = float((actual_gradient - expected).abs().max())
    return {'max_abs_errors': errors,
            'values': {name: item.item() for name, item in zip(('reconstruction_nll', 'kl', 'negative_elbo'), actual)}}


def measure(model, data):
    with fr.no_grad():
        inputs = batch(data['held_out'], copies=EVAL_DRAWS)
        epsilon = fr.Tensor([row[0] for row in data['eval_epsilon']], [len(data['eval_epsilon']), 1])
        nll, kl, negative_elbo = objective(model, inputs, inputs, epsilon)
        clean = batch(data['held_out'])
        mu, _ = model.encode(clean)
        mean_mse = fr.nn.functional.mse_loss(model.decode(mu), clean).item()
    mean = [sum(row['x'][j] for row in data['train']) / len(data['train']) for j in range(2)]
    baseline_mse = sum((row['x'][j] - mean[j]) ** 2 for row in data['held_out'] for j in range(2)) / (2 * len(data['held_out']))
    # A constant decoder at the training mean with q(z|x)=p(z) has zero KL.
    baseline_nll = baseline_mse / model.observation_sigma ** 2 + 2 * math.log(model.observation_sigma * math.sqrt(2 * math.pi))
    return {'mc_reconstruction_nll': nll.item(), 'kl': kl.item(), 'mc_negative_elbo': negative_elbo.item(),
            'decoder_mean_mse': mean_mse, 'constant_baseline_mse': baseline_mse,
            'constant_gaussian_negative_elbo': baseline_nll}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = VAEModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.015 if restored is None else 0.7, weight_decay=0.0001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid VAE training stop')
    initial_metrics = measure(model, data)
    batches = [batch(data['train'][i:i + BATCH_SIZE]) for i in range(0, len(data['train']), BATCH_SIZE)]
    losses = []
    for step in range(start, stop):
        inputs = batches[step % len(batches)]
        epsilon = fr.Tensor.randn([BATCH_SIZE, model.latent], generator=rng)
        opt.zero_grad()
        nll, kl, loss = objective(model, inputs, inputs, epsilon)
        values = [item.item() for item in (nll, kl, loss)]
        if any(not math.isfinite(value) for value in values):
            raise AssertionError('Nonfinite VAE training loss')
        losses.append(values)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, snapshot(model, opt, rng, step + 1), data, seed)
    state = snapshot(model, opt, rng, stop)
    result = {'losses': losses, 'metrics': measure(model, data), 'initial_metrics': initial_metrics,
              'next_latent_draws': fr.Tensor.randn([8, model.latent], generator=rng).tolist(), 'dataset_sha256': digest,
              'cursor': {'step': stop, 'next_sample': (stop * BATCH_SIZE) % len(data['train'])}}
    return state, result


def check_acceptance(expected, resumed, expected_state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_latent_draws', expected['next_latent_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', expected_state)):
        if resumed[field] != wanted:
            raise AssertionError('VAE restart differs: ' + field)
    metrics, initial = expected['metrics'], expected['initial_metrics']
    keys = ('mc_reconstruction_nll', 'kl', 'mc_negative_elbo', 'decoder_mean_mse',
            'constant_baseline_mse', 'constant_gaussian_negative_elbo')
    values = [metrics[key] for key in keys] + [initial['mc_negative_elbo']]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or values[1] < 0 or values[3] < 0 or min(values[4:]) <= 0:
        raise AssertionError('VAE invalid metrics: ' + repr(metrics))
    if not math.isclose(values[2], values[0] + values[1], rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError('VAE inconsistent negative ELBO components')
    if values[2] > 0.5 * values[5] or values[2] > 0.5 * values[6] or values[3] > 0.25 * values[4] or not 0.1 < values[1] < 10.:
        raise AssertionError('VAE learning gate failed: ' + repr(metrics))


def prove(seed):
    gradients = gradient_check(seed, make_dataset())
    with tempfile.TemporaryDirectory(prefix='ferro-vae-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_vae.py')
        flags = ['-O'] if sys.flags.optimize else []
        subprocess.run([sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(midpoint),
                        '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(result_path)],
                       check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'schedule': SCHEDULE, 'seed': seed, 'steps': STEPS, 'beta': BETA,
            'observation_sigma': OBSERVATION_SIGMA, 'eval_draws': EVAL_DRAWS,
            'metrics': expected['metrics'], 'initial_metrics': expected['initial_metrics'],
            'gradient_check': gradients, 'restart': 'bit-exact', 'dataset_sha256': expected['dataset_sha256'],
            'checkpoint_step': CHECKPOINT_STEP, 'initial_losses': expected['losses'][0], 'final_losses': expected['losses'][-1]}


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
