"""CPU f32 GRU learning and restart between complete optimizer steps.

Declared gates: seeds 7/11/43, 300 AdamW steps, held-out accuracy >= 95%,
history-erasure accuracy drop >= 40 points, gradient atol=1e-5/rtol=1e-4,
and bit-exact fresh-process continuation in the same environment.
Each batch starts with zero hidden state. No mid-sequence resume is claimed.
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
from ferro.recurrent import RecurrentState, unroll
from training_restart import state_digest

SEEDS = (7, 11, 43)
STEPS = 300
CHECKPOINT_STEP = 137
BATCH_SIZE = 8
SEQUENCE_LENGTH = 5
SCHEDULE = 'cyclic-sequences-v1'
RESET_POLICY = 'zero-state-per-batch-v1'
TASK = 'paired-token-order-v1'


class GRUModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 8
    sequence_length: int = SEQUENCE_LENGTH
    batch_size: int = BATCH_SIZE
    schedule: str = SCHEDULE
    reset_policy: str = RESET_POLICY
    task: str = TASK

    def build(self):
        h = self.hidden
        self.weight_ih = fr.nn.Parameter(fr.Tensor.randn([3 * h, 1], seed=self.seed) * 0.5)
        self.weight_hh = fr.nn.Parameter(fr.Tensor.randn([3 * h, h], seed=self.seed + 1) * h ** -0.5)
        self.bias_ih = fr.nn.Parameter(fr.Tensor.randn([3 * h], seed=self.seed + 2) * 0.05)
        self.bias_hh = fr.nn.Parameter(fr.Tensor.randn([3 * h], seed=self.seed + 3) * 0.05)
        self.readout_weight = fr.nn.Parameter(fr.Tensor.randn([h, 2], seed=self.seed + 4) * h ** -0.5)
        self.readout_bias = fr.nn.Parameter(fr.Tensor.zeros([2]))

    def forward(self, x, erase_history=False):
        time, count, _ = x.shape
        initial = RecurrentState(fr.Tensor.zeros([count, self.hidden]))
        reset = [False] * ((time - 1) * count) + [True] * count if erase_history else None
        result = unroll('gru', x, initial, self.weight_ih.tensor(), self.weight_hh.tensor(),
                        self.bias_ih.tensor(), self.bias_hh.tensor(), lengths=[time] * count, reset=reset)
        return result.state.h @ self.readout_weight.tensor() + self.readout_bias.tensor()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK}
    for split, phase in (('train', 0.25), ('held_out', 0.75)):
        sequences = []
        for pair in range(16):
            a = 0.8 + 0.3 * math.sin(pair + phase) ** 2
            b = 0.15 + 0.15 * math.sin(2 * pair + phase) ** 2
            for label in (0, 1):
                sign = 2 * label - 1
                sequences.append({'id': f'{split}-{pair:02d}-{label}', 'label': label,
                                  'tokens': [sign * a, -sign * a, b, -b, 0.]})
        data[split] = sequences
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported GRU dataset schema')
    identities, samples = set(), set()
    for split in ('train', 'held_out'):
        sequences = data[split]
        if type(sequences) is not list or len(sequences) != 32:
            raise ValueError('Expected 32 sequences in each split')
        for sequence in sequences:
            if type(sequence) is not dict or set(sequence) != {'id', 'tokens', 'label'}:
                raise ValueError('Invalid sequence schema')
            identity, tokens, label = sequence['id'], sequence['tokens'], sequence['label']
            if type(identity) is not str or not identity.startswith(split + '-') or identity in identities:
                raise ValueError('Sequence identities must be unique and split-specific')
            identities.add(identity)
            if type(label) is not int or label not in (0, 1):
                raise ValueError('Invalid sequence label')
            if type(tokens) is not list or len(tokens) != SEQUENCE_LENGTH or any(type(v) not in (int, float) or not math.isfinite(v) for v in tokens):
                raise ValueError('Invalid sequence tokens')
            if not 0.8 <= abs(tokens[0]) <= 1.1 or tokens[1] != -tokens[0] or not 0.15 <= tokens[2] <= 0.3 or tokens[3] != -tokens[2] or tokens[4] != 0 or (tokens[0] > 0) != bool(label):
                raise ValueError('Invalid delayed-signal sequence')
            # Compare the actual CPU f32 values consumed by training, not just JSON decimals.
            sample = tuple(fr.Tensor(tokens, [SEQUENCE_LENGTH]).tolist())
            if sample in samples:
                raise ValueError('Sequence samples must be unique and disjoint')
            samples.add(sample)
        for left, right in zip(sequences[::2], sequences[1::2]):
            if (left['label'], right['label']) != (0, 1) or left['tokens'][:2] != right['tokens'][1::-1] or left['tokens'][2:] != right['tokens'][2:]:
                raise ValueError('Token multisets must be paired across labels')


def batch(sequences):
    count = len(sequences)
    values = [sequence['tokens'][t] for t in range(SEQUENCE_LENGTH) for sequence in sequences]
    return (fr.Tensor(values, [SEQUENCE_LENGTH, count, 1]),
            fr.Tensor.from_i64([sequence['label'] for sequence in sequences], [count]))


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE,
              'batch_size': BATCH_SIZE, 'reset_policy': RESET_POLICY, 'step': checkpoint.step,
              'next_sequence': (checkpoint.step * BATCH_SIZE) % len(data['train']),
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'schedule', 'batch_size', 'reset_policy', 'step', 'next_sequence',
            'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported GRU run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE or record['reset_policy'] != RESET_POLICY:
        raise ValueError('GRU run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('GRU dataset identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_sequence']) is not int or record['next_sequence'] != (step * BATCH_SIZE) % len(data['train']):
        raise ValueError('GRU cursor mismatch')
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('GRU checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('GRU checkpoint/cursor step mismatch')
    return data, checkpoint


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    model = GRUModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    x, labels = batch(data['train'][:BATCH_SIZE])
    x.requires_grad_(True)
    tx = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    hidden = torch.zeros(BATCH_SIZE, model.hidden, dtype=torch.float32)
    for token in tx.unbind(0):
        input_gates = token @ weights['weight_ih'].T + weights['bias_ih']
        hidden_gates = hidden @ weights['weight_hh'].T + weights['bias_hh']
        ir, iz, inn = input_gates.chunk(3, dim=1)
        hr, hz, hn = hidden_gates.chunk(3, dim=1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        new = torch.tanh(inn + reset * hn)
        hidden = (1 - update) * new + update * hidden
    logits = hidden @ weights['readout_weight'] + weights['readout_bias']
    actual_logits = model(x)
    torch.testing.assert_close(torch.tensor(actual_logits.tolist()), logits.detach(), atol=1e-5, rtol=1e-4)
    reference = torch.nn.functional.cross_entropy(logits, torch.tensor(labels.tolist(), dtype=torch.int64))
    loss = fr.nn.functional.cross_entropy(actual_logits, labels)
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=1e-5, rtol=1e-4)
    loss.backward()
    reference.backward()
    errors = {}
    pairs = [('input', x.grad, tx.grad)] + [(name, p.grad, weights[name].grad) for name, p in model.named_parameters()]
    for name, actual, expected in pairs:
        if actual is None or expected is None:
            raise AssertionError('Missing GRU gradient: ' + name)
        got = torch.tensor(actual.tolist())
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        errors[name] = float((got - expected).abs().max())
    return errors


def measure(model, data):
    with fr.no_grad():
        x, labels = batch(data['held_out'])
        outputs, erased = model(x).tolist(), model(x, erase_history=True).tolist()
        wanted = labels.tolist()
        accuracy = sum(max(range(2), key=lambda j: row[j]) == label for row, label in zip(outputs, wanted)) / len(wanted)
        ablated = sum(max(range(2), key=lambda j: row[j]) == label for row, label in zip(erased, wanted)) / len(wanted)
    return {'held_out_accuracy': accuracy, 'history_erased_accuracy': ablated,
            'history_erasure_accuracy_drop': accuracy - ablated,
            'last_token_baseline': 0.5, 'orderless_baseline': 0.5}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = GRUModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.025 if restored is None else 0.7, weight_decay=0.001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid GRU training stop')
    initial_metrics = measure(model, data)
    batches = [batch(data['train'][i:i + BATCH_SIZE]) for i in range(0, 32, BATCH_SIZE)]
    losses = []
    for step in range(start, stop):
        x, labels = batches[step % len(batches)]
        tokens = x + fr.Tensor.randn(list(x.shape), generator=rng) * 0.005
        opt.zero_grad()
        loss = fr.nn.functional.cross_entropy(model(tokens), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite GRU training loss')
        losses.append(value)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=step + 1), data, seed)
    state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=stop)
    result = {'losses': losses, 'metrics': measure(model, data),
              'next_draws': fr.Tensor.randn([8], generator=rng).tolist(), 'dataset_sha256': digest,
              'initial_metrics': initial_metrics,
              'cursor': {'step': stop, 'next_sequence': (stop * BATCH_SIZE) % 32}}
    return state, result


def check_acceptance(expected, resumed, expected_state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', expected_state)):
        if resumed[field] != wanted:
            raise AssertionError('GRU restart differs: ' + field)
    metrics = expected['metrics']
    values = [metrics[k] for k in ('held_out_accuracy', 'history_erased_accuracy', 'history_erasure_accuracy_drop',
                                  'last_token_baseline', 'orderless_baseline')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise AssertionError('GRU invalid metrics: ' + repr(metrics))
    if not math.isclose(values[2], values[0] - values[1], rel_tol=0, abs_tol=1e-12):
        raise AssertionError('GRU inconsistent ablation metrics: ' + repr(metrics))
    if not values[0] >= 0.95 or not values[2] >= 0.4 or values[1] != 0.5 or any(value != 0.5 for value in values[3:]):
        raise AssertionError('GRU learning gate failed: ' + repr(metrics))


def prove(seed):
    gradients = gradient_check(seed, make_dataset())
    with tempfile.TemporaryDirectory(prefix='ferro-gru-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(Path(__file__).with_name('train_restart_recurrent.py')),
                   '--seed', str(seed), '--resume', str(midpoint), '--dataset-digest', expected['dataset_sha256'],
                   '--worker-output', str(result_path)]
        subprocess.run(command, check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'metrics': expected['metrics'],
            'gradient_max_abs_errors': gradients, 'restart': 'bit-exact', 'reset_policy': RESET_POLICY,
            'resume_scope': 'between-complete-optimizer-steps', 'checkpoint_step': CHECKPOINT_STEP,
            'dataset_sha256': expected['dataset_sha256'],
            'initial_held_out_accuracy': expected['initial_metrics']['held_out_accuracy'],
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
            parser.error('resume/dataset digest are reserved for the restart worker')
        for seed in SEEDS if args.seed is None else (args.seed,):
            print(json.dumps(prove(seed), sort_keys=True, allow_nan=False), flush=True)
