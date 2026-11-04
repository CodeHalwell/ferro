"""CPU f32 CNN learning with saved image data and fresh-process restart.

Declared gates: seeds 7/11/43, 300 AdamW steps, held-out accuracy >= 95%,
mean-image ablation accuracy 50%, gradient atol=1e-5/rtol=1e-4, and bit-exact
continuation in the same environment. Held-out amplitudes are disjoint;
spatial templates repeat. This is a small synthetic orientation task.
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
from training_restart import snapshot, state_digest

SEEDS = (7, 11, 43)
STEPS = 300
CHECKPOINT_STEP = 147
BATCH_SIZE = 8
IMAGE_SIZE = 8
SCHEDULE = 'cyclic-images-v1'
TASK = 'paired-bar-orientation-v1'


class CNNModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 8
    schedule: str = SCHEDULE
    pooling: str = 'mean'

    def build(self):
        if self.pooling not in ('mean', 'max'):
            raise ValueError('Unsupported CNN pooling')
        self.conv = fr.nn.Conv2d(1, self.hidden, 3)
        self.pool = (fr.nn.AvgPool2d if self.pooling == 'mean' else fr.nn.MaxPool2d)(6)
        self.flatten = fr.nn.Flatten()
        self.head = fr.nn.Linear(self.hidden, 2)
        with fr.no_grad():
            self.conv.weight.tensor().copy_(fr.Tensor.randn([self.hidden, 1, 3, 3], seed=self.seed) / 3)
            self.conv.bias.tensor().copy_(fr.Tensor.zeros([self.hidden]))
            self.head.weight.tensor().copy_(fr.Tensor.randn([2, self.hidden], seed=self.seed + 1) * self.hidden ** -0.5)
            self.head.bias.tensor().copy_(fr.Tensor.zeros([2]))

    def forward(self, x):
        return self.head(self.flatten(self.pool(self.conv(x).tanh())))


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK, 'image_size': IMAGE_SIZE}
    for split, offset in (('train', 0.0), ('held_out', 0.005)):
        images = []
        for pair in range(32):
            row, column = 2 + (pair // 4) % 4, 2 + pair % 4
            background = 0.05 * math.sin(pair)
            amplitude = 0.8 + pair * 0.01 + offset
            horizontal = [background] * IMAGE_SIZE ** 2
            for dx in (-1, 0, 1):
                horizontal[row * IMAGE_SIZE + column + dx] = background + amplitude
            vertical = [horizontal[c * IMAGE_SIZE + r] for r in range(IMAGE_SIZE) for c in range(IMAGE_SIZE)]
            for label, pixels in enumerate((horizontal, vertical)):
                images.append({'id': f'{split}-{pair:02d}-{label}', 'pixels': pixels, 'label': label})
        data[split] = images
    validate_dataset(data)
    return data


def validate_dataset(data):
    keys = {'version', 'task', 'image_size', 'train', 'held_out'}
    if type(data) is not dict or set(data) != keys or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported CNN dataset schema')
    if type(data['image_size']) is not int or data['image_size'] != IMAGE_SIZE:
        raise ValueError('Invalid CNN image size')
    identities, contents = set(), set()
    for split in ('train', 'held_out'):
        images = data[split]
        if type(images) is not list or len(images) != 64:
            raise ValueError('Expected 64 images in each split')
        for item in images:
            if type(item) is not dict or set(item) != {'id', 'pixels', 'label'}:
                raise ValueError('Invalid image schema')
            if type(item['id']) is not str or not item['id'].startswith(split + '-') or item['id'] in identities:
                raise ValueError('Image identities must be unique and split-specific')
            identities.add(item['id'])
            pixels = item['pixels']
            if type(pixels) is not list or len(pixels) != IMAGE_SIZE ** 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in pixels):
                raise ValueError('Invalid image pixels')
            encoded = canonical(pixels)
            if encoded in contents:
                raise ValueError('Image contents must be disjoint, including held-out data')
            contents.add(encoded)
            if type(item['label']) is not int or item['label'] not in (0, 1):
                raise ValueError('Invalid image label')
        for left, right in zip(images[::2], images[1::2]):
            transposed = [left['pixels'][c * IMAGE_SIZE + r] for r in range(IMAGE_SIZE) for c in range(IMAGE_SIZE)]
            if (left['label'], right['label']) != (0, 1) or right['pixels'] != transposed:
                raise ValueError('Each label pair must have transposed pixels and identical intensity histogram')


def batch(images, remove_spatial=False):
    pixels = []
    for item in images:
        values = item['pixels']
        pixels.extend([math.fsum(values) / len(values)] * len(values) if remove_spatial else values)
    return (fr.Tensor(pixels, [len(images), 1, IMAGE_SIZE, IMAGE_SIZE]),
            fr.Tensor.from_i64([item['label'] for item in images], [len(images)]))


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE,
              'batch_size': BATCH_SIZE, 'step': checkpoint.step,
              'next_image': (checkpoint.step * BATCH_SIZE) % len(data['train']),
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'schedule', 'batch_size', 'step', 'next_image',
            'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported CNN run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE:
        raise ValueError('CNN run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('CNN dataset identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_image']) is not int or record['next_image'] != (step * BATCH_SIZE) % len(data['train']):
        raise ValueError('CNN cursor mismatch')
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('CNN checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('CNN checkpoint/cursor step mismatch')
    return data, checkpoint


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    model = CNNModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    x, labels = batch(data['train'][:BATCH_SIZE])
    x.requires_grad_(True)
    tx = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    hidden = torch.nn.functional.conv2d(tx, weights['conv.weight'], weights['conv.bias']).tanh()
    pooled = torch.nn.functional.avg_pool2d(hidden, 6).flatten(1)
    reference_logits = torch.nn.functional.linear(pooled, weights['head.weight'], weights['head.bias'])
    logits = model(x)
    torch.testing.assert_close(torch.tensor(logits.tolist()), reference_logits.detach(), atol=1e-5, rtol=1e-4)
    reference = torch.nn.functional.cross_entropy(reference_logits, torch.tensor(labels.tolist(), dtype=torch.int64))
    loss = fr.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=1e-5, rtol=1e-4)
    loss.backward()
    reference.backward()
    errors = {}
    pairs = [('input', x.grad, tx.grad)] + [(name, p.grad, weights[name].grad) for name, p in model.named_parameters()]
    for name, actual, expected in pairs:
        if actual is None or expected is None:
            raise AssertionError('Missing CNN gradient: ' + name)
        got = torch.tensor(actual.tolist())
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        errors[name] = float((got - expected).abs().max())
    return errors


def measure(model, data):
    accuracies = []
    with fr.no_grad():
        for remove_spatial in (False, True):
            x, labels = batch(data['held_out'], remove_spatial=remove_spatial)
            predictions = [max(range(2), key=lambda j: row[j]) for row in model(x).tolist()]
            accuracies.append(sum(a == b for a, b in zip(predictions, labels.tolist())) / len(predictions))
    return {'held_out_accuracy': accuracies[0], 'mean_image_accuracy': accuracies[1],
            'spatial_accuracy_drop': accuracies[0] - accuracies[1], 'class_prior_baseline': 0.5,
            'global_intensity_baseline': 0.5}


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = CNNModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.04 if restored is None else 0.7, weight_decay=0.001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid CNN training stop')
    initial_metrics = measure(model, data)
    batches = [batch(data['train'][i:i + BATCH_SIZE]) for i in range(0, len(data['train']), BATCH_SIZE)]
    losses = []
    for step in range(start, stop):
        x, labels = batches[step % len(batches)]
        inputs = x + fr.Tensor.randn(list(x.shape), generator=rng) * 0.005
        opt.zero_grad()
        loss = fr.nn.functional.cross_entropy(model(inputs), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite CNN training loss')
        losses.append(value)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, snapshot(model, opt, rng, step + 1), data, seed)
    state = snapshot(model, opt, rng, stop)
    result = {'losses': losses, 'metrics': measure(model, data), 'initial_metrics': initial_metrics,
              'next_draws': fr.Tensor.randn([8], generator=rng).tolist(), 'dataset_sha256': digest,
              'cursor': {'step': stop, 'next_image': (stop * BATCH_SIZE) % len(data['train'])}}
    return state, result


def check_acceptance(expected, resumed, expected_state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', expected_state)):
        if resumed[field] != wanted:
            raise AssertionError('CNN restart differs: ' + field)
    metrics = expected['metrics']
    values = [metrics[k] for k in ('held_out_accuracy', 'mean_image_accuracy', 'spatial_accuracy_drop')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise AssertionError('CNN invalid metrics: ' + repr(metrics))
    if not math.isclose(values[2], values[0] - values[1], rel_tol=0, abs_tol=1e-12):
        raise AssertionError('CNN inconsistent ablation metrics: ' + repr(metrics))
    if values[0] < 0.95 or values[1] != 0.5 or values[2] < 0.4:
        raise AssertionError('CNN learning gate failed: ' + repr(metrics))


def prove(seed):
    gradients = gradient_check(seed, make_dataset())
    with tempfile.TemporaryDirectory(prefix='ferro-cnn-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_cnn.py')
        flags = ['-O'] if sys.flags.optimize else []
        subprocess.run([sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(midpoint),
                        '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(result_path)],
                       check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'metrics': expected['metrics'],
            'gradient_max_abs_errors': gradients, 'restart': 'bit-exact',
            'dataset_sha256': expected['dataset_sha256'], 'checkpoint_step': CHECKPOINT_STEP,
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
