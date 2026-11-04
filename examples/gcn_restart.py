"""CPU f32 GCN learning with an explicit graph/data restart bundle.

Declared gates: seeds 7/11/43, 240 AdamW steps, held-out accuracy >= 95%,
edge-removal accuracy drop >= 40 points, gradient atol=1e-5/rtol=1e-4,
and bit-exact continuation in a fresh process in the same environment.
Held-out feature samples are disjoint; graph topology templates may repeat.
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
from ferro.graph import EdgeIndex, GcnConv
from training_restart import state_digest

SEEDS = (7, 11, 43)
STEPS = 240
CHECKPOINT_STEP = 117
BATCH_SIZE = 8
SCHEDULE = 'cyclic-graphs-v1'
TASK = 'paired-neighborhood-v1'


class GCNModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 8
    self_loops: bool = True
    schedule: str = SCHEDULE

    def build(self):
        self.conv1 = GcnConv(in_features=2, out_features=self.hidden, self_loops=self.self_loops)
        self.conv2 = GcnConv(in_features=self.hidden, out_features=2, self_loops=self.self_loops)
        with fr.no_grad():
            self.conv1.weight.tensor().copy_(fr.Tensor.randn([2, self.hidden], seed=self.seed) * 0.5)
            self.conv2.weight.tensor().copy_(fr.Tensor.randn([self.hidden, 2], seed=self.seed + 1) * self.hidden ** -0.5)
            self.conv1.bias.tensor().copy_(fr.Tensor.zeros([self.hidden]))
            self.conv2.bias.tensor().copy_(fr.Tensor.zeros([2]))

    def forward(self, x, edges, queries):
        hidden = self.conv1(x, edges).tanh()
        return self.conv2(hidden, edges).index_select(0, queries)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK}
    for split, phase in (('train', 0.25), ('held_out', 0.75)):
        graphs = []
        for pair in range(16):
            neighbors = 1 + pair % 3
            count = neighbors + 2
            rotation = pair % count
            remap = lambda i: (i + rotation) % count
            src = [i for i in range(1, neighbors + 1)] + [0] * neighbors
            dst = [0] * neighbors + list(range(1, neighbors + 1))
            if pair % 4 == 0:
                src.append(1)
                dst.append(0)
            for label in (0, 1):
                sign = 2 * label - 1
                signals = [sign * (0.8 + 0.3 * math.sin(pair + j + phase) ** 2) for j in range(neighbors)]
                features = [[0., 0.] for _ in range(count)]
                features[remap(0)] = [0., 1.]
                for i, value in enumerate(signals, 1):
                    features[remap(i)] = [value, 0.]
                features[remap(count - 1)] = [-sum(signals), 0.]
                graphs.append({'id': f'{split}-{pair:02d}-{label}', 'num_nodes': count,
                               'src': [remap(i) for i in src], 'dst': [remap(i) for i in dst],
                               'features': features, 'query': remap(0), 'label': label})
        data[split] = graphs
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported GCN dataset schema')
    identities = set()
    for split in ('train', 'held_out'):
        graphs = data[split]
        if type(graphs) is not list or len(graphs) != 32:
            raise ValueError('Expected 32 graphs in each split')
        for graph in graphs:
            if type(graph) is not dict or set(graph) != {'id', 'num_nodes', 'src', 'dst', 'features', 'query', 'label'}:
                raise ValueError('Invalid graph schema')
            identity, count = graph['id'], graph['num_nodes']
            if type(identity) is not str or not identity.startswith(split + '-') or identity in identities:
                raise ValueError('Graph identities must be unique and split-specific')
            identities.add(identity)
            if type(count) is not int or not 3 <= count <= 5:
                raise ValueError('Invalid graph node count')
            if type(graph['query']) is not int or not 0 <= graph['query'] < count:
                raise ValueError('Invalid query node')
            if type(graph['label']) is not int or graph['label'] not in (0, 1):
                raise ValueError('Invalid graph label')
            for side in ('src', 'dst'):
                if type(graph[side]) is not list or any(type(i) is not int or not 0 <= i < count for i in graph[side]):
                    raise ValueError('Invalid topology indices')
            EdgeIndex(count, graph['src'], graph['dst'])
            features = graph['features']
            if type(features) is not list or len(features) != count or any(
                    type(row) is not list or len(row) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                    for row in features):
                raise ValueError('Invalid node features')
            if features[graph['query']] != [0., 1.]:
                raise ValueError('Query features must be uninformative')
        for left, right in zip(graphs[::2], graphs[1::2]):
            if left['label'] != 0 or right['label'] != 1 or any(left[k] != right[k] for k in ('num_nodes', 'src', 'dst', 'query')):
                raise ValueError('Topology must be paired across labels')


def batch(graphs):
    edges, _ = EdgeIndex.batch([EdgeIndex(g['num_nodes'], g['src'], g['dst']) for g in graphs])
    features, queries, offset = [], [], 0
    for graph in graphs:
        features.extend(v for row in graph['features'] for v in row)
        queries.append(offset + graph['query'])
        offset += graph['num_nodes']
    return (fr.Tensor(features, [offset, 2]), edges, queries,
            fr.Tensor.from_i64([g['label'] for g in graphs], [len(graphs)]))


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE,
              'batch_size': BATCH_SIZE, 'step': checkpoint.step,
              'next_graph': (checkpoint.step * BATCH_SIZE) % len(data['train']),
              'dataset_sha256': hashlib.sha256(encoded).hexdigest(),
              'checkpoint_index_sha256': hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()}
    temporary = directory / 'run.json.tmp'
    temporary.write_bytes(canonical(record))
    os.replace(temporary, directory / 'run.json')


def load_run(directory, expected_digest, seed):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_bytes())
    keys = {'version', 'task', 'seed', 'schedule', 'batch_size', 'step', 'next_graph',
            'dataset_sha256', 'checkpoint_index_sha256'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported GCN run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE:
        raise ValueError('GCN run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('GCN dataset/topology identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_graph']) is not int or record['next_graph'] != (step * BATCH_SIZE) % len(data['train']):
        raise ValueError('GCN cursor mismatch')
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('GCN checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('GCN checkpoint/cursor step mismatch')
    return data, checkpoint


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = GCNModel(seed=seed, dataset_digest=digest)
    x, edges, queries, labels = batch(data['train'][:BATCH_SIZE])
    x.requires_grad_(True)
    tx = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    adjacency = torch.zeros(edges.num_nodes, edges.num_nodes)
    for source, target in zip(edges.src, edges.dst):
        if source != target:
            adjacency[target, source] += 1
    adjacency.fill_diagonal_(1)
    inv = adjacency.sum(1).rsqrt()
    norm = inv[:, None] * adjacency * inv[None, :]
    hidden = torch.tanh(norm @ (tx @ weights['conv1.weight']) + weights['conv1.bias'])
    logits = (norm @ (hidden @ weights['conv2.weight']) + weights['conv2.bias'])[queries]
    reference = torch.nn.functional.cross_entropy(logits, torch.tensor(labels.tolist(), dtype=torch.int64))
    loss = fr.nn.functional.cross_entropy(model(x, edges, queries), labels)
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=1e-5, rtol=1e-4)
    loss.backward()
    reference.backward()
    errors = {}
    pairs = [('input', x.grad, tx.grad)] + [(name, p.grad, weights[name].grad) for name, p in model.named_parameters()]
    for name, actual, expected in pairs:
        if actual is None or expected is None:
            raise AssertionError('Missing GCN gradient: ' + name)
        got = torch.tensor(actual.tolist())
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        errors[name] = float((got - expected).abs().max())
    return errors


def measure(model, data):
    with fr.no_grad():
        x, edges, queries, labels = batch(data['held_out'])
        outputs = model(x, edges, queries).tolist()
        removed = model(x, EdgeIndex(edges.num_nodes, [], []), queries).tolist()
        wanted = labels.tolist()
        accuracy = sum(max(range(2), key=lambda j: row[j]) == label for row, label in zip(outputs, wanted)) / len(wanted)
        ablated = sum(max(range(2), key=lambda j: row[j]) == label for row, label in zip(removed, wanted)) / len(wanted)
    metrics = {'held_out_accuracy': accuracy, 'edge_removed_accuracy': ablated,
               'edge_removal_accuracy_drop': accuracy - ablated, 'class_prior_baseline': 0.5,
               'node_local_baseline': 0.5}
    return metrics


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = GCNModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=0.025 if restored is None else 0.7, weight_decay=0.001)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid GCN training stop')
    initial_metrics = measure(model, data)
    batches = [batch(data['train'][i:i + BATCH_SIZE]) for i in range(0, 32, BATCH_SIZE)]
    losses = []
    for step in range(start, stop):
        x, edges, queries, labels = batches[step % len(batches)]
        features = x + fr.Tensor.randn(list(x.shape), generator=rng) * 0.005
        opt.zero_grad()
        loss = fr.nn.functional.cross_entropy(model(features, edges, queries), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite GCN training loss')
        losses.append(value)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if midpoint is not None and step + 1 == CHECKPOINT_STEP:
            save_run(midpoint, fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=step + 1), data, seed)
    state = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=stop)
    metrics = measure(model, data)
    result = {'losses': losses, 'metrics': metrics, 'next_draws': fr.Tensor.randn([8], generator=rng).tolist(),
              'dataset_sha256': digest, 'initial_metrics': initial_metrics, 'cursor': {'step': stop, 'next_graph': (stop * BATCH_SIZE) % 32}}
    return state, result


def check_acceptance(expected, resumed, expected_state):
    for field, wanted in (('losses', expected['losses'][CHECKPOINT_STEP:]), ('next_draws', expected['next_draws']),
                          ('metrics', expected['metrics']), ('dataset_sha256', expected['dataset_sha256']),
                          ('cursor', expected['cursor']), ('state', expected_state)):
        if resumed[field] != wanted:
            raise AssertionError('GCN restart differs: ' + field)
    metrics = expected['metrics']
    values = [metrics[k] for k in ('held_out_accuracy', 'edge_removed_accuracy', 'edge_removal_accuracy_drop')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise AssertionError('GCN invalid metrics: ' + repr(metrics))
    if not math.isclose(values[2], values[0] - values[1], rel_tol=0, abs_tol=1e-12):
        raise AssertionError('GCN inconsistent ablation metrics: ' + repr(metrics))
    if not metrics['held_out_accuracy'] >= 0.95 or not metrics['edge_removal_accuracy_drop'] >= 0.4:
        raise AssertionError('GCN learning gate failed: ' + repr(metrics))


def prove(seed):
    data = make_dataset()
    gradients = gradient_check(seed, data)
    with tempfile.TemporaryDirectory(prefix='ferro-gcn-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_gcn.py')
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(midpoint),
                   '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(result_path)]
        subprocess.run(command, check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'metrics': expected['metrics'],
            'gradient_max_abs_errors': gradients, 'restart': 'bit-exact',
            'dataset_sha256': expected['dataset_sha256'], 'checkpoint_step': CHECKPOINT_STEP,
            'initial_held_out_accuracy': expected['initial_metrics']['held_out_accuracy'], 'initial_loss': expected['losses'][0],
            'final_loss': expected['losses'][-1]}


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
