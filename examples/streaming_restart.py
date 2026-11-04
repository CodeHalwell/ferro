"""CPU LSTM learning with explicit state across truncated-BPTT chunks.

Declared: seeds 7/11/43, 600 chunk updates, checkpoint after update 293.
Continuation chunks must achieve >=95% accuracy; final CE <=0.05 and <=10%
of initialization. Erasing carried state must lose >=40 accuracy points.
Checkpoints are between complete chunk/optimizer updates, never mid-backward.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import ferro as fr
from ferro.autograd import grad
from ferro.recurrent import RecurrentState, unroll
from training_restart import snapshot, state_digest

SEEDS = (7, 11, 43)
STEPS = 600
CHECKPOINT_STEP = 293
BATCH_SIZE = 8
HIDDEN = 8
CHUNK_LENGTH = 3
CHUNKS = 3
NOISE_SCALE = 0.002
TASK = 'streaming-delayed-sign-v1'
SCHEDULE = 'cyclic-episodes-three-chunks-v1'
STATE_POLICY = 'zero-at-episode-start-detach-after-each-chunk-v1'


class StreamingModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = HIDDEN
    batch_size: int = BATCH_SIZE
    chunk_length: int = CHUNK_LENGTH
    chunks: int = CHUNKS
    noise_scale: float = NOISE_SCALE
    schedule: str = SCHEDULE
    state_policy: str = STATE_POLICY

    def build(self):
        h = self.hidden
        self.weight_ih = fr.nn.Parameter(fr.Tensor.randn([4 * h, 1], seed=self.seed) * 0.5)
        self.weight_hh = fr.nn.Parameter(fr.Tensor.randn([4 * h, h], seed=self.seed + 1) * h ** -0.5)
        self.bias_ih = fr.nn.Parameter(fr.Tensor([float(h <= i < 2 * h) for i in range(4 * h)], [4 * h]))
        self.bias_hh = fr.nn.Parameter(fr.Tensor.zeros([4 * h]))
        self.readout_weight = fr.nn.Parameter(fr.Tensor.randn([h, 2], seed=self.seed + 2) * h ** -0.5)
        self.readout_bias = fr.nn.Parameter(fr.Tensor.zeros([2]))

    def forward(self, inputs, initial, truncate=None):
        output = unroll('lstm', inputs, initial, self.weight_ih.tensor(), self.weight_hh.tensor(),
                        self.bias_ih.tensor(), self.bias_hh.tensor(), lengths=[inputs.shape[0]] * inputs.shape[1],
                        truncate=truncate)
        return output.state.h @ self.readout_weight.tensor() + self.readout_bias.tensor(), output.state


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK}
    for split, phase in (('train', 0.25), ('held_out', 0.75)):
        data[split] = []
        for pair in range(16):
            amplitude = 0.8 + 0.3 * math.sin(pair + phase) ** 2
            for label in (0, 1):
                data[split].append({'id': f'{split}-{pair:02d}-{label}', 'label': label,
                                    'tokens': [(2 * label - 1) * amplitude] + [0.] * (CHUNK_LENGTH * CHUNKS - 1)})
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported streaming dataset schema')
    identities, samples = set(), set()
    for split in ('train', 'held_out'):
        rows = data[split]
        if type(rows) is not list or len(rows) != 32:
            raise ValueError('Expected 32 streaming sequences per split')
        for row in rows:
            if type(row) is not dict or set(row) != {'id', 'label', 'tokens'}:
                raise ValueError('Invalid streaming sequence schema')
            if type(row['id']) is not str or not row['id'].startswith(split + '-') or row['id'] in identities:
                raise ValueError('Invalid streaming sequence identity')
            identities.add(row['id'])
            tokens, label = row['tokens'], row['label']
            if type(label) is not int or label not in (0, 1) or type(tokens) is not list or len(tokens) != CHUNK_LENGTH * CHUNKS or any(type(v) not in (int, float) or not math.isfinite(v) for v in tokens):
                raise ValueError('Invalid streaming sequence values')
            if not 0.8 <= abs(tokens[0]) <= 1.1 or (tokens[0] > 0) != bool(label) or any(tokens[1:]):
                raise ValueError('Only the first token may reveal the delayed label')
            sample = tuple(fr.Tensor(tokens, [len(tokens)]).tolist())
            if sample in samples:
                raise ValueError('Streaming samples must be unique and disjoint')
            samples.add(sample)
        for left, right in zip(rows[::2], rows[1::2]):
            if (left['label'], right['label']) != (0, 1) or left['tokens'][0] != -right['tokens'][0]:
                raise ValueError('Streaming labels must form balanced amplitude pairs')


def chunk_batch(rows, chunk):
    begin = chunk * CHUNK_LENGTH
    values = [row['tokens'][t] for t in range(begin, begin + CHUNK_LENGTH) for row in rows]
    return fr.Tensor(values, [CHUNK_LENGTH, len(rows), 1]), fr.Tensor.from_i64([row['label'] for row in rows], [len(rows)])


def zero_state(count=BATCH_SIZE, hidden=HIDDEN):
    return RecurrentState(fr.Tensor.zeros([count, hidden]), fr.Tensor.zeros([count, hidden]))


def cursor(data, step):
    def ids(episode):
        start = (episode * BATCH_SIZE) % len(data['train'])
        return [row['id'] for row in data['train'][start:start + BATCH_SIZE]]
    return {'step': step, 'episode': step // CHUNKS, 'chunk': step % CHUNKS,
            'next_sample': (step // CHUNKS * BATCH_SIZE) % len(data['train']),
            'next_sequence_ids': ids(step // CHUNKS),
            'state_sequence_ids': ids((step - 1) // CHUNKS if step else 0)}


def training_chunk(data, step):
    position = cursor(data, step)
    rows = data['train'][position['next_sample']:position['next_sample'] + BATCH_SIZE]
    return chunk_batch(rows, position['chunk'])


def state_values(state):
    if state.c is None or state.h.shape != [BATCH_SIZE, HIDDEN] or state.c.shape != [BATCH_SIZE, HIDDEN]:
        raise ValueError('Streaming h/c must both have shape [8,8]')
    values = {'h': state.h.tolist(), 'c': state.c.tolist()}
    if any(not math.isfinite(value) for matrix in values.values() for row in matrix for value in row):
        raise ValueError('Streaming h/c must be finite')
    if state.h.requires_grad or state.c.requires_grad:
        raise ValueError('Streaming h/c must be detached at the saved boundary')
    return values


def carry_digest(state):
    values = state_values(state)
    raw = b''.join(struct.pack('<f', value) for name in ('h', 'c') for row in values[name] for value in row)
    return hashlib.sha256(raw).hexdigest()


def read_state(path, expected_hash):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Streaming h/c integrity mismatch')
    if len(raw) < 8:
        raise ValueError('Invalid streaming state header')
    length = int.from_bytes(raw[:8], 'little')
    if length > len(raw) - 8:
        raise ValueError('Invalid streaming state header length')
    header = json.loads(raw[8:8 + length])
    if type(header) is not dict or set(header) != {'h', 'c'}:
        raise ValueError('Streaming state requires exactly h and c')
    for item in header.values():
        if type(item) is not dict or set(item) != {'dtype', 'shape', 'data_offsets'} or item['dtype'] != 'F32' or item['shape'] != [BATCH_SIZE, HIDDEN] or any(type(n) is not int for n in item['shape']):
            raise ValueError('Streaming h/c must be typed F32 [8,8]')
    tensors = fr.load_safetensors(str(path))
    state = RecurrentState(tensors['h'], tensors['c'])
    state_values(state)
    return state


def configuration():
    return {'batch_size': BATCH_SIZE, 'hidden': HIDDEN, 'chunk_length': CHUNK_LENGTH, 'chunks': CHUNKS,
            'noise_scale': NOISE_SCALE, 'budget': STEPS, 'schedule': SCHEDULE, 'state_policy': STATE_POLICY}


def save_run(directory, checkpoint, data, seed, carry):
    validate_dataset(data)
    state_values(carry)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    state_path = directory / 'state.safetensors'
    fr.save_safetensors(str(state_path), {'h': carry.h, 'c': carry.c})
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    read_state(state_path, state_hash)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'configuration': configuration(),
              'cursor': cursor(data, checkpoint.step), 'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'state_sha256': state_hash,
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'configuration', 'cursor', 'dataset_sha256', 'state_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported streaming run schema')
    if type(record['seed']) is not int or record['seed'] != seed or canonical(record['configuration']) != canonical(configuration()):
        raise ValueError('Streaming configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('Streaming dataset identity mismatch')
    position = record['cursor']
    step = position.get('step') if type(position) is dict else None
    if type(step) is not int or not 0 <= step <= STEPS or canonical(position) != canonical(cursor(data, step)):
        raise ValueError('Streaming cursor/sequence identity mismatch')
    carry = read_state(directory / 'state.safetensors', record['state_sha256'])
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('Streaming checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('Streaming checkpoint/cursor step mismatch')
    return data, checkpoint, carry


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    results = {}
    for mode in ('single', 'live_bridge', 'detached_bridge', 'truncate'):
        model = StreamingModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
        x, labels = chunk_batch(data['train'][:4], 0)
        x = (x + fr.Tensor.randn(list(x.shape), seed=seed + 10) * 0.05).requires_grad_(True)
        suffix = (fr.Tensor.randn([CHUNK_LENGTH, 4, 1], seed=seed + 11) * 0.05).requires_grad_(True)
        h = (fr.Tensor.randn([4, HIDDEN], seed=seed + 12) * 0.2).requires_grad_(True)
        c = (fr.Tensor.randn([4, HIDDEN], seed=seed + 13) * 0.3).requires_grad_(True)
        if mode == 'truncate':
            x = fr.Tensor([v for block in (x.tolist(), suffix.tolist()) for time in block for row in time for v in row],
                          [2 * CHUNK_LENGTH, 4, 1]).requires_grad_(True)
        tx, ts, th, tc = [torch.tensor(item.tolist(), dtype=torch.float32, requires_grad=True) for item in (x, suffix, h, c)]
        weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
                   for name, p in model.named_parameters()}

        def reference(inputs, hidden, cell, truncate=False):
            for index, token in enumerate(inputs.unbind(0)):
                if truncate and index == CHUNK_LENGTH:
                    hidden, cell = hidden.detach(), cell.detach()
                gates = token @ weights['weight_ih'].T + weights['bias_ih'] + hidden @ weights['weight_hh'].T + weights['bias_hh']
                i, f, g, o = gates.chunk(4, dim=1)
                cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
                hidden = torch.sigmoid(o) * torch.tanh(cell)
            return hidden @ weights['readout_weight'] + weights['readout_bias'], hidden, cell

        logits, outgoing = model(x, RecurrentState(h, c), truncate=CHUNK_LENGTH if mode == 'truncate' else None)
        wanted, wh, wc = reference(tx, th, tc, truncate=mode == 'truncate')
        names, actual_inputs, reference_inputs = ['input', 'initial_h', 'initial_c'], [x, h, c], [tx, th, tc]
        if mode.endswith('bridge'):
            bridge = outgoing.detach() if mode == 'detached_bridge' else outgoing
            if mode == 'detached_bridge':
                if bridge.h.requires_grad or bridge.c.requires_grad:
                    raise AssertionError('LSTM state detach retained a derivative graph')
                wh, wc = wh.detach(), wc.detach()
            logits, outgoing = model(suffix, bridge)
            wanted, wh, wc = reference(ts, wh, wc)
            names.append('suffix_input')
            actual_inputs.append(suffix)
            reference_inputs.append(ts)
        for actual, expected in ((logits, wanted), (outgoing.h, wh), (outgoing.c, wc)):
            torch.testing.assert_close(torch.tensor(actual.tolist()), expected.detach(), atol=1e-5, rtol=1e-4)
        loss = fr.nn.functional.cross_entropy(logits, labels)
        expected_loss = torch.nn.functional.cross_entropy(wanted, torch.tensor(labels.tolist(), dtype=torch.int64))
        torch.testing.assert_close(torch.tensor(loss.item()), expected_loss.detach(), atol=1e-5, rtol=1e-4)
        names.extend(weights)
        actual_inputs.extend(p.tensor() for _, p in model.named_parameters())
        reference_inputs.extend(weights.values())
        actual_grads = grad(loss, actual_inputs, materialize_grads=True)
        expected_grads = torch.autograd.grad(expected_loss, reference_inputs, allow_unused=True)
        errors, norms = {}, {}
        for name, actual, expected, parameter in zip(names, actual_grads, expected_grads, reference_inputs):
            wanted_gradient = torch.zeros_like(parameter) if expected is None else expected
            got = torch.tensor(actual.tolist())
            torch.testing.assert_close(got, wanted_gradient, atol=1e-5, rtol=1e-4)
            errors[name], norms[name] = float((got - wanted_gradient).abs().max()), float(got.abs().max())
        if mode == 'live_bridge' and any(norms[name] <= 1e-7 for name in ('input', 'initial_h', 'initial_c')):
            raise AssertionError('Live LSTM bridge must preserve earlier-state gradients')
        if mode == 'detached_bridge' and any(norms[name] != 0 for name in ('input', 'initial_h', 'initial_c')):
            raise AssertionError('Detached LSTM bridge leaked earlier gradients')
        if mode == 'truncate':
            if any(value != 0 for time in actual_grads[0].tolist()[:CHUNK_LENGTH] for row in time for value in row) or norms['initial_h'] != 0 or norms['initial_c'] != 0:
                raise AssertionError('LSTM truncate leaked gradients across the chunk boundary')
        if any(norms[name] <= 1e-7 for name in weights):
            raise AssertionError('Shared LSTM parameter gradients must survive truncation')
        if any(item.grad is not None for item in actual_inputs):
            raise AssertionError('Functional LSTM oracle mutated persistent gradient slots')
        results[mode] = {'max_abs_errors': errors, 'gradient_max_abs': norms}
    return results


def accuracy(logits, labels):
    return sum(max(range(2), key=lambda j: row[j]) == label for row, label in zip(logits.tolist(), labels.tolist())) / labels.shape[0]


def measure(model, data):
    accuracies, losses, erased = [], [], []
    with fr.no_grad():
        carry = zero_state(len(data['held_out']))
        for chunk in range(CHUNKS):
            inputs, labels = chunk_batch(data['held_out'], chunk)
            logits, carry = model(inputs, carry)
            reset_logits, _ = model(inputs, zero_state(len(data['held_out'])))
            accuracies.append(accuracy(logits, labels))
            losses.append(fr.nn.functional.cross_entropy(logits, labels).item())
            erased.append(accuracy(reset_logits, labels))
    return {'chunk_accuracies': accuracies, 'chunk_losses': losses, 'reset_chunk_accuracies': erased,
            'final_accuracy_drop': accuracies[-1] - erased[-1], 'blank_suffix_baseline': 0.5}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None, erase_resume_state=False):
    if resume is None:
        data, restored, carry = make_dataset(), None, zero_state()
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored, carry = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = StreamingModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    optimizer = fr.optim.AdamW(model.parameters(), lr=0.02 if restored is None else 0.7, weight_decay=0.0001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': optimizer}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid streaming training stop')
    if erase_resume_state:
        carry = zero_state()
    initial_metrics = measure(model, data)
    losses = []
    for step in range(start, stop):
        if step % CHUNKS == 0:
            carry = zero_state()
        inputs, labels = training_chunk(data, step)
        inputs = inputs + fr.Tensor.randn(list(inputs.shape), generator=rng) * model.noise_scale
        optimizer.zero_grad()
        logits, outgoing = model(inputs, carry)
        loss = fr.nn.functional.cross_entropy(logits, labels)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite streaming training loss')
        losses.append(value)
        loss.backward()
        # Carry the numeric state computed with this chunk's parameter values.
        carry = outgoing.detach()
        optimizer.step()
        optimizer.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, snapshot(model, optimizer, rng, step + 1), data, seed, carry)
    checkpoint = snapshot(model, optimizer, rng, stop)
    result = {'losses': losses, 'metrics': measure(model, data), 'initial_metrics': initial_metrics,
              'next_draws': fr.Tensor.randn([8], generator=rng).tolist(), 'dataset_sha256': digest,
              'cursor': cursor(data, stop), 'carry_sha256': carry_digest(carry)}
    return checkpoint, carry, result


def checkpoint_probe(directory, digest, seed):
    data, checkpoint, carry = load_run(directory, digest, seed)
    model = StreamingModel(seed=seed, dataset_digest=digest)
    optimizer, rng = fr.optim.AdamW(model.parameters(), lr=0.7), fr.Generator(999)
    checkpoint.restore(model, optimizers={'main': optimizer}, generator=rng)
    if checkpoint.step % CHUNKS == 0:
        raise AssertionError('Streaming checkpoint must be inside an episode')
    with fr.no_grad():
        inputs, labels = training_chunk(data, checkpoint.step)
        logits, _ = model(inputs, carry)
        erased, _ = model(inputs, zero_state())
        kept, reset = accuracy(logits, labels), accuracy(erased, labels)
    norms = {name: max(abs(v) for row in matrix for v in row) for name, matrix in state_values(carry).items()}
    if kept < 0.95 or reset != 0.5 or any(value <= 0 for value in norms.values()):
        raise AssertionError('Saved LSTM state must carry the delayed label: ' + repr((kept, reset, norms)))
    return {'carried_accuracy': kept, 'erased_accuracy': reset, 'state_max_abs': norms,
            'cursor': cursor(data, checkpoint.step), 'carry_sha256': carry_digest(carry)}


def check_acceptance(expected, resumed, expected_state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('carry_sha256', expected['carry_sha256']), ('state', expected_state)):
        if resumed[field] != wanted:
            raise AssertionError('Streaming restart differs: ' + field)
    metrics, initial = expected['metrics'], expected['initial_metrics']
    arrays = [metrics[key] for key in ('chunk_accuracies', 'chunk_losses', 'reset_chunk_accuracies')]
    if any(type(values) is not list or len(values) != CHUNKS for values in arrays):
        raise AssertionError('Invalid streaming metric shape')
    values = [v for array in arrays for v in array] + [metrics['final_accuracy_drop'], metrics['blank_suffix_baseline'], initial['chunk_losses'][-1]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values) or any(v > 1 for v in arrays[0] + arrays[2]) or values[-1] <= 0:
        raise AssertionError('Invalid streaming metrics')
    if not math.isclose(metrics['final_accuracy_drop'], arrays[0][-1] - arrays[2][-1], abs_tol=1e-12, rel_tol=0):
        raise AssertionError('Inconsistent streaming ablation metrics')
    if min(arrays[0][1:]) < 0.95 or arrays[1][-1] > 0.05 or arrays[1][-1] > 0.1 * initial['chunk_losses'][-1] or metrics['final_accuracy_drop'] < 0.4 or arrays[2][-1] != 0.5 or metrics['blank_suffix_baseline'] != 0.5:
        raise AssertionError('Streaming learning gate failed: ' + repr(metrics))


def prove(seed):
    gradients = gradient_check(seed, make_dataset())
    with tempfile.TemporaryDirectory(prefix='ferro-streaming-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        checkpoint, _, expected = train(seed, midpoint=midpoint)
        boundary = checkpoint_probe(midpoint, expected['dataset_sha256'], seed)
        expected_state = state_digest(checkpoint, directory / 'expected')
        result_path = directory / 'resumed.json'
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(Path(__file__).with_name('train_restart_streaming.py')),
                   '--seed', str(seed), '--resume', str(midpoint), '--dataset-digest', expected['dataset_sha256'],
                   '--worker-output', str(result_path)]
        subprocess.run(command, check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'chunk_length': CHUNK_LENGTH, 'chunks_per_episode': CHUNKS,
            'state_policy': STATE_POLICY, 'resume_scope': 'between-complete-chunk-optimizer-updates',
            'metrics': expected['metrics'], 'initial_metrics': expected['initial_metrics'],
            'gradient_check': gradients, 'restart': 'bit-exact', 'checkpoint_probe': boundary,
            'dataset_sha256': expected['dataset_sha256'], 'initial_loss': expected['losses'][0], 'final_loss': expected['losses'][-1]}


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
        checkpoint, _, result = train(args.seed, resume=args.resume, expected_digest=args.dataset_digest)
        result['state'] = state_digest(checkpoint, args.worker_output.with_suffix('.state'))
        args.worker_output.write_bytes(canonical(result))
    else:
        if args.resume is not None or args.dataset_digest is not None:
            parser.error('resume/dataset digest are reserved for the restart worker')
        for seed in SEEDS if args.seed is None else (args.seed,):
            print(json.dumps(prove(seed), sort_keys=True, allow_nan=False), flush=True)
