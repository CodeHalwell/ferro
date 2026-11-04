"""Bounded CPU WGAN-GP: learn N(1, .6^2) with an affine generator.

Revision 2, declared before training: seeds 7/11/43, 4000 cycles of 5 critic +
1 generator updates, Adam D lr=.005/G lr=.001, betas=(0,.9), batch32, GP
weight .1, centered mixed-sign critic initialization. Held-out empirical
W1/mean error/std error <=.25, W1 <=25% of initial, and bit-exact restart
after critic substep3 of5. This tests location/scale learning, not general GANs.
Revision 1 (1500 cycles, both lr=.002, GP10, positive critic) failed seed7
with W1=1.2431; independent Torch agreed. Gates are unchanged in revision 2.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
import tempfile

import ferro as fr
from ferro.autograd import grad
from training_restart import state_digest

SEEDS = (7, 11, 43)
CYCLES = 4000
CRITIC_STEPS = 5
STEPS = CYCLES * (CRITIC_STEPS + 1)
CHECKPOINT_STEP = 11997
BATCH_SIZE = 32
DATA_SIZE = 512
TASK = 'gaussian-location-scale-wgan-gp-v2'
CONFIG = {'target_mean': 1., 'target_std': .6, 'hidden': 16, 'cycles': CYCLES,
          'critic_steps': CRITIC_STEPS, 'batch_size': BATCH_SIZE, 'gp_weight': .1,
          'norm_epsilon': 1e-12, 'critic_lr': .005, 'generator_lr': .001, 'betas': [0., .9],
          'critic_init': 'centered-normal-v2',
          'schedule': 'cyclic-interleaved-real-and-generator-latent-v1'}


class GeneratorModel(fr.nn.Module):
    seed: int

    def build(self):
        self.shift = fr.nn.Parameter(fr.Tensor.randn([1], seed=self.seed) * .05 - .75)
        self.log_scale = fr.nn.Parameter(fr.Tensor.randn([1], seed=self.seed + 1) * .02 + math.log(1.1))

    def forward(self, z):
        return z * self.log_scale.tensor().exp() + self.shift.tensor()


class CriticModel(fr.nn.Module):
    seed: int
    hidden: int = 16

    def build(self):
        self.w1 = fr.nn.Parameter(fr.Tensor.randn([1, self.hidden], seed=self.seed + 2) * .5)
        self.b1 = fr.nn.Parameter(fr.Tensor.randn([self.hidden], seed=self.seed + 3) * .1)
        self.w2 = fr.nn.Parameter(fr.Tensor.randn([self.hidden, 1], seed=self.seed + 4) * .3)

    def forward(self, x, frozen=False):
        # Parameter.trainable freezes only optimizer updates. Detached reads here
        # additionally exclude critic parameter gradients while preserving dx.
        weights = [p.tensor().detach() if frozen else p.tensor() for p in (self.w1, self.b1, self.w2)]
        return (x @ weights[0] + weights[1]).tanh() @ weights[2]


class WGANModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    target_mean: float = 1.
    target_std: float = .6
    hidden: int = 16
    cycles: int = CYCLES
    critic_steps: int = CRITIC_STEPS
    batch_size: int = BATCH_SIZE
    gp_weight: float = .1
    norm_epsilon: float = 1e-12
    critic_lr: float = .005
    generator_lr: float = .001
    critic_init: str = 'centered-normal-v2'
    schedule: str = CONFIG['schedule']
    task: str = TASK

    def build(self):
        self.generator = GeneratorModel(seed=self.seed)
        self.critic = CriticModel(seed=self.seed, hidden=self.hidden)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    normal = NormalDist()
    train = [1 + .6 * normal.inv_cdf((i + .5) / DATA_SIZE) for i in range(DATA_SIZE)]
    data = {'version': 1, 'task': TASK,
            'train_real': [train[(i * 73) % DATA_SIZE] for i in range(DATA_SIZE)],
            'held_real': [1 + .6 * normal.inv_cdf((i + .25) / DATA_SIZE) for i in range(DATA_SIZE)],
            'eval_latent': [normal.inv_cdf((i + .75) / DATA_SIZE) for i in range(DATA_SIZE)]}
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train_real', 'held_real', 'eval_latent'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported WGAN dataset schema')
    for name in ('train_real', 'held_real', 'eval_latent'):
        values = data[name]
        if type(values) is not list or len(values) != DATA_SIZE or any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or len(set(values)) != DATA_SIZE:
            raise ValueError('Invalid WGAN dataset values')
    train32 = [row[0] for row in fr.Tensor(data['train_real'], [DATA_SIZE, 1]).tolist()]
    held32 = [row[0] for row in fr.Tensor(data['held_real'], [DATA_SIZE, 1]).tolist()]
    if set(train32) & set(held32):
        raise ValueError('WGAN train and held-out real samples overlap')


def phase_state(step):
    cycle, position = divmod(step, CRITIC_STEPS + 1)
    d_count = cycle * CRITIC_STEPS + min(position, CRITIC_STEPS)
    return {'step': step, 'cycle': cycle, 'phase': 'critic' if position < CRITIC_STEPS else 'generator',
            'critic_substep': position, 'critic_updates': d_count, 'generator_updates': cycle,
            'next_real': (d_count * BATCH_SIZE) % DATA_SIZE}


def optimizers(model, lr=None):
    return {'critic': fr.optim.Adam(model.critic.parameters(), lr=model.critic_lr if lr is None else lr, betas=(0., .9)),
            'generator': fr.optim.Adam(model.generator.parameters(), lr=model.generator_lr if lr is None else lr, betas=(0., .9))}


def clear(opts):
    for optimizer in opts.values():
        optimizer.zero_grad()


def input_gradient(critic, interpolated):
    derivative, = grad(critic(interpolated).sum(), [interpolated], create_graph=True)
    return derivative


def critic_losses(model, real, fake, alpha):
    fake = fake.detach()
    interpolated = (real * alpha + fake * (1 - alpha)).requires_grad_(True)
    derivative = input_gradient(model.critic, interpolated)
    if not derivative.requires_grad:
        raise AssertionError('WGAN penalty input gradient detached')
    norm = (derivative * derivative + model.norm_epsilon).sqrt()
    penalty = ((norm - 1) * (norm - 1)).mean()
    transport = model.critic(fake).mean() - model.critic(real).mean()
    return transport + model.gp_weight * penalty, transport, penalty, derivative


def update(model, opts, rng, data, step):
    clear(opts)
    phase = phase_state(step)
    z = fr.Tensor.randn([BATCH_SIZE, 1], generator=rng)
    if phase['phase'] == 'critic':
        ids = [(phase['next_real'] + j) % DATA_SIZE for j in range(BATCH_SIZE)]
        real = fr.Tensor([data['train_real'][i] for i in ids], [BATCH_SIZE, 1])
        alpha = fr.Tensor.rand([BATCH_SIZE, 1], generator=rng)
        loss, transport, penalty, _ = critic_losses(model, real, model.generator(z), alpha)
        record = {'phase': 'critic', 'transport': transport.item(), 'penalty': penalty.item(), 'loss': loss.item()}
        loss.backward()
        if any(p.tensor().grad is not None for p in model.generator.parameters()):
            raise AssertionError('Critic phase reached generator parameters')
        opts['critic'].step()
    else:
        for p in model.critic.parameters():
            p.trainable = False
        try:
            loss = -model.critic(model.generator(z), frozen=True).mean()
            record = {'phase': 'generator', 'loss': loss.item()}
            loss.backward()
            if any(p.tensor().grad is not None for p in model.critic.parameters()):
                raise AssertionError('Frozen critic graph reached critic parameters')
            if any(p.grad is None for p in model.generator.parameters()):
                raise AssertionError('Frozen critic blocked generator input gradient')
            opts['generator'].step()
        finally:
            for p in model.critic.parameters():
                p.trainable = True
    clear(opts)
    if any(not math.isfinite(value) for name, value in record.items() if name != 'phase'):
        raise AssertionError('Nonfinite WGAN loss')
    return record


def torch_critic(x, weights, frozen=False):
    w = {name: value.detach() if frozen else value for name, value in weights.items()}
    return (x @ w['w1'] + w['b1']).tanh() @ w['w2']


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    model = WGANModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    opts = optimizers(model)
    real_values, z_values, alpha_values = [-.2, .35, .8, 1.3, 1.9], [-1.4, -.65, .1, .55, 1.2], [.12, .31, .48, .69, .88]
    real, z, alpha = [fr.Tensor(values, [5, 1]) for values in (real_values, z_values, alpha_values)]
    tr, tz, ta = [torch.tensor(values, dtype=torch.float32).reshape(5, 1) for values in (real_values, z_values, alpha_values)]
    dw = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True) for name, p in model.critic.named_parameters()}
    gw = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True) for name, p in model.generator.named_parameters()}
    tfake = tz * gw['log_scale'].exp() + gw['shift']
    ti = (tr * ta + tfake.detach() * (1 - ta)).requires_grad_(True)
    td, = torch.autograd.grad(torch_critic(ti, dw).sum(), ti, create_graph=True)
    tp = ((torch.sqrt(td * td + model.norm_epsilon) - 1) ** 2).mean()
    tt = torch_critic(tfake.detach(), dw).mean() - torch_critic(tr, dw).mean()
    tl = tt + model.gp_weight * tp
    if any(value is not None for value in torch.autograd.grad(tl, list(gw.values()), allow_unused=True, retain_graph=True)):
        raise AssertionError('Torch critic oracle reached inactive generator')
    loss, transport, penalty, derivative = critic_losses(model, real, model.generator(z), alpha)
    errors = {}
    def compare(name, actual, wanted):
        got = torch.tensor(actual.tolist())
        torch.testing.assert_close(got, wanted.detach(), atol=5e-5, rtol=3e-4)
        errors[name] = float((got - wanted.detach()).abs().max())
    compare('generator_forward', model.generator(z), tfake)
    compare('critic_real', model.critic(real), torch_critic(tr, dw))
    compare('critic_fake', model.critic(model.generator(z)), torch_critic(tfake, dw))
    for name, actual, wanted in (('input_gradient', derivative, td), ('transport', transport, tt), ('penalty', penalty, tp), ('critic_total', loss, tl)):
        compare(name, actual, wanted)
    mixed = {}
    for objective, reference in (('transport', tt), ('penalty', tp), ('critic_total', tl)):
        clear(opts)
        current = critic_losses(model, real, model.generator(z), alpha)
        current[{'critic_total': 0, 'transport': 1, 'penalty': 2}[objective]].backward()
        wanted = torch.autograd.grad(reference, list(dw.values()), retain_graph=True)
        for (name, p), expected in zip(model.critic.named_parameters(), wanted):
            if p.tensor().grad is None:
                raise AssertionError('Missing WGAN ' + objective + ' critic gradient: ' + name)
            compare(objective + '.' + name, p.tensor().grad, expected)
            if objective == 'penalty':
                mixed[name] = float(expected.abs().max())
                if not mixed[name] > 1e-6:
                    raise AssertionError('Zero WGAN mixed penalty gradient: ' + name)
                if not max(abs(v) for v in p.tensor().grad.flatten().tolist()) > 1e-6:
                    raise AssertionError('Zero Ferro WGAN mixed penalty gradient: ' + name)
        if any(p.tensor().grad is not None for p in model.generator.parameters()):
            raise AssertionError('Critic oracle reached inactive generator')
    clear(opts)
    for p in model.critic.parameters():
        p.trainable = False
    try:
        gl = -model.critic(model.generator(z), frozen=True).mean()
        tgl = -torch_critic(tfake, dw, frozen=True).mean()
        if any(value is not None for value in torch.autograd.grad(tgl, list(dw.values()), allow_unused=True, retain_graph=True)):
            raise AssertionError('Torch generator oracle reached inactive critic')
        compare('generator_loss', gl, tgl)
        gl.backward()
        wanted = torch.autograd.grad(tgl, list(gw.values()))
        for (name, p), expected in zip(model.generator.named_parameters(), wanted):
            if p.grad is None or not float(expected.abs().max()) > 1e-6:
                raise AssertionError('Missing WGAN generator gradient: ' + name)
            compare('generator.' + name, p.grad, expected)
        if any(p.tensor().grad is not None for p in model.critic.parameters()):
            raise AssertionError('Generator oracle reached frozen critic')
    finally:
        for p in model.critic.parameters():
            p.trainable = True
        clear(opts)
    return {'max_abs_errors': errors, 'penalty_mixed_gradient_max_abs': mixed}


def check_counters(checkpoint, phase):
    tensors = checkpoint.tensors()
    for name in ('critic', 'generator'):
        counters = [value.item() for key, value in tensors.items() if key.startswith('optim.' + name + '.t.')]
        count = phase[name + '_updates']
        if count and (not counters or any(v != count for v in counters)):
            raise ValueError('WGAN native optimizer counter mismatch: ' + name)
        if not count and any(v != 0 for v in counters):
            raise ValueError('WGAN native optimizer counter mismatch: ' + name)


def isolation_check(seed, data):
    model = WGANModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    opts, rng = optimizers(model), fr.Generator(seed + 100)
    for step in range(6):
        update(model, opts, rng, data, step)
    checked = []
    for step in range(6, 12):
        inactive = 'generator' if phase_state(step)['phase'] == 'critic' else 'critic'
        before = fr.checkpoint.snapshot_training(model, optimizers=opts, generator=rng, step=step).tensors()
        update(model, opts, rng, data, step)
        snapshot = fr.checkpoint.snapshot_training(model, optimizers=opts, generator=rng, step=step + 1)
        after = snapshot.tensors()
        prefixes = ('model.' + inactive + '.', 'optim.' + inactive + '.')
        # Canonical finite scalar encodings also distinguish signed zero.
        left = {k: (tuple(v.shape), canonical(v.tolist())) for k, v in before.items() if k.startswith(prefixes)}
        right = {k: (tuple(v.shape), canonical(v.tolist())) for k, v in after.items() if k.startswith(prefixes)}
        if not left or left != right:
            raise AssertionError('Inactive WGAN state changed: ' + inactive)
        if any(p.tensor().grad is not None or not p.trainable for p in model.parameters()):
            raise AssertionError('WGAN phase did not restore paused trainable state')
        check_counters(snapshot, phase_state(step + 1))
        checked.append(phase_state(step)['phase'])
    return checked


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    phase = phase_state(checkpoint.step)
    check_counters(checkpoint, phase)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'config': CONFIG, 'phase': phase,
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'config', 'phase', 'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported WGAN run schema')
    if type(record['seed']) is not int or record['seed'] != seed or canonical(record['config']) != canonical(CONFIG):
        raise ValueError('WGAN run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('WGAN dataset identity mismatch')
    phase = record['phase']
    if type(phase) is not dict or type(phase.get('step')) is not int or not 0 <= phase['step'] <= STEPS or canonical(phase) != canonical(phase_state(phase['step'])):
        raise ValueError('WGAN phase/counter/cursor mismatch')
    if hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest() != record['checkpoint_index_sha256']:
        raise ValueError('WGAN checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != phase['step']:
        raise ValueError('WGAN checkpoint/phase step mismatch')
    check_counters(checkpoint, phase)
    return data, checkpoint


def distribution_metrics(values, target):
    mean, wanted = sum(values) / len(values), sum(target) / len(target)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    target_std = math.sqrt(sum((v - wanted) ** 2 for v in target) / len(target))
    return {'w1': sum(abs(a - b) for a, b in zip(sorted(values), sorted(target))) / len(values),
            'mean_error': abs(mean - wanted), 'std_error': abs(std - target_std), 'mean': mean, 'std': std}


def measure(model, data):
    with fr.no_grad():
        output = model.generator(fr.Tensor(data['eval_latent'], [DATA_SIZE, 1])).tolist()
    result = distribution_metrics([row[0] for row in output], data['held_real'])
    # Fixed held-out interpolation pairs and coefficients, independent of RNG.
    real = fr.Tensor(list(reversed(data['held_real'])), [DATA_SIZE, 1])
    fake = fr.Tensor([row[0] for row in output], [DATA_SIZE, 1])
    alpha = fr.Tensor([(i + .5) / DATA_SIZE for i in range(DATA_SIZE)], [DATA_SIZE, 1])
    _, _, penalty, derivative = critic_losses(model, real, fake, alpha)
    norms = [math.sqrt(row[0] ** 2 + model.norm_epsilon) for row in derivative.tolist()]
    result.update(held_penalty=penalty.item(), held_norm_mean=sum(norms) / len(norms),
                  held_norm_min=min(norms), held_norm_max=max(norms))
    return result


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = WGANModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opts = optimizers(model, lr=None if restored is None else .7)
    start = 0 if restored is None else restored.restore(model, optimizers=opts, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid WGAN training stop')
    initial = measure(model, data)
    losses = []
    for step in range(start, stop):
        losses.append(update(model, opts, rng, data, step))
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, fr.checkpoint.snapshot_training(model, optimizers=opts, generator=rng, step=step + 1), data, seed)
    state = fr.checkpoint.snapshot_training(model, optimizers=opts, generator=rng, step=stop)
    check_counters(state, phase_state(stop))
    result = {'losses': losses, 'initial_metrics': initial, 'metrics': measure(model, data),
              'target_baseline': distribution_metrics([1 + .6 * z for z in data['eval_latent']], data['held_real']),
              'next_draws': fr.Tensor.rand([8], generator=rng).tolist(), 'dataset_sha256': digest, 'phase': phase_state(stop)}
    return state, result


def check_acceptance(expected, resumed, state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('target_baseline', expected['target_baseline']),
                          ('dataset_sha256', expected['dataset_sha256']), ('phase', expected['phase']), ('state', state)):
        if resumed[field] != wanted:
            raise AssertionError('WGAN restart differs: ' + field)
    metrics, initial = expected['metrics'], expected['initial_metrics']
    for values in (metrics, initial, expected['target_baseline']):
        if any(type(v) not in (int, float) or not math.isfinite(v) or (key != 'mean' and v < 0) for key, v in values.items()):
            raise AssertionError('WGAN invalid distribution metrics')
    if not all(metrics[key] <= .25 for key in ('w1', 'mean_error', 'std_error')) or not metrics['w1'] <= .25 * initial['w1']:
        raise AssertionError('WGAN distribution gate failed: ' + repr(metrics))


def prove(seed):
    data = make_dataset()
    derivatives_checked, isolated = gradient_check(seed, data), isolation_check(seed, data)
    with tempfile.TemporaryDirectory(prefix='ferro-wgan-restart-') as directory:
        directory = Path(directory)
        state, expected = train(seed, midpoint=directory / 'midpoint')
        expected_state = state_digest(state, directory / 'expected')
        path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_wgan.py')
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(directory / 'midpoint'),
                   '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(path)]
        subprocess.run(command, check=True, timeout=180, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        check_acceptance(expected, json.loads(path.read_bytes()), expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'checkpoint_phase': phase_state(CHECKPOINT_STEP),
            'metrics': expected['metrics'], 'initial_metrics': expected['initial_metrics'], 'target_baseline': expected['target_baseline'],
            'derivatives': derivatives_checked, 'isolated_phases': isolated, 'restart': 'bit-exact',
            'dataset_sha256': expected['dataset_sha256'], 'final_phase': expected['phase'],
            'initial_critic': expected['losses'][0], 'final_critic': expected['losses'][-2], 'final_generator': expected['losses'][-1]}


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
