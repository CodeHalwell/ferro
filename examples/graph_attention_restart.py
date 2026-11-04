"""CPU f32 sparse attention learns crossed query/key and edge-attribute routing.

Declared gates: seeds7/11/43,400 AdamW steps, held-out accuracy>=95%; removing
query, edge bias, all edges, rewiring attribute endpoints or using uniform
attention gives accuracy<=60% and drop>=35 points. Gradient atol2e-5/rtol2e-4;
held-out cross-entropy<=.05 and<=25% of its initial value;
bit-exact fresh-process restart197. Disjoint continuous feature/attribute samples
use repeated topology templates; this is not universal graph generalization.
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
from ferro.graph import EdgeIndex, edge_attention
from training_restart import state_digest

SEEDS = (7, 11, 43)
STEPS = 400
CHECKPOINT_STEP = 197
BATCH_SIZE = 8
SCHEDULE = 'cyclic-graphs-v1'
TASK = 'crossed-key-edge-attribute-routing-v1'
CONFIG = {'hidden': 2, 'lr': .03, 'weight_decay': 0., 'jitter': .002, 'steps': STEPS,
          'batch_size': BATCH_SIZE, 'schedule': SCHEDULE, 'score': 'scaled-dot-plus-edge-bias'}
ABLATIONS = ('query_removed', 'edge_attribute_removed', 'rewired', 'edge_removed', 'uniform')


class AttentionModel(fr.nn.Module):
    seed: int
    dataset_digest: str
    hidden: int = 2
    lr: float = .03
    weight_decay: float = 0.
    jitter: float = .002
    steps: int = STEPS
    batch_size: int = BATCH_SIZE
    score: str = 'scaled-dot-plus-edge-bias'
    task: str = TASK
    schedule: str = SCHEDULE

    def build(self):
        self.wq = fr.nn.Parameter(fr.Tensor.randn([3, self.hidden], seed=self.seed) * .4)
        self.wk = fr.nn.Parameter(fr.Tensor.randn([3, self.hidden], seed=self.seed + 1) * .4)
        self.wv = fr.nn.Parameter(fr.Tensor.randn([3, self.hidden], seed=self.seed + 2) * .4)
        self.we = fr.nn.Parameter(fr.Tensor.randn([1, 1], seed=self.seed + 3) * .4)
        self.wo = fr.nn.Parameter(fr.Tensor.randn([self.hidden, 2], seed=self.seed + 4) * .4)
        self.bo = fr.nn.Parameter(fr.Tensor.zeros([2]))

    def forward(self, x, attributes, edges, queries, mode='normal'):
        q, k, v = x @ self.wq.tensor(), x @ self.wk.tensor(), x @ self.wv.tensor()
        bias = (attributes @ self.we.tensor()).reshape([len(edges.src)])
        if mode in ('query_removed', 'uniform'):
            q = fr.Tensor.zeros(list(q.shape))
        if mode in ('edge_attribute_removed', 'uniform'):
            bias = fr.Tensor.zeros(list(bias.shape))
        attended = edge_attention(q, k, v, edges, edge_bias=bias)
        return attended.index_select(0, queries) @ self.wo.tensor() + self.bo.tensor()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def make_dataset():
    data = {'version': 1, 'task': TASK}
    for split, phase in (('train', .25), ('held_out', .75)):
        graphs = []
        for pair in range(16):
            count = 5 + pair % 4
            rotation = pair % count
            remap = lambda node: (node + rotation) % count
            query_sign = 1 if pair % 2 else -1
            qm = .9 + .15 * math.sin(pair + phase) ** 2
            km = .9 + .15 * math.cos(pair + phase) ** 2
            vm = .8 + .3 * math.sin(pair * .7 + phase) ** 2
            am = .9 + .15 * math.cos(pair * .8 + phase) ** 2
            repeat = 2 if pair % 3 == 0 else 1
            for label in (0, 1):
                signal = 2 * label - 1
                features = [[0., 0., 0.] for _ in range(count)]
                features[remap(0)] = [query_sign * qm, 0., 0.]
                attributes = []
                for node, (key_sign, attribute_sign) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)), 1):
                    features[remap(node)] = [0., key_sign * km, signal * query_sign * key_sign * attribute_sign * vm]
                    attributes.append([attribute_sign * am])
                graphs.append({'id': f'{split}-{pair:02d}-{label}', 'num_nodes': count,
                               'src': [remap(i) for i in (1, 2, 3, 4)] * repeat,
                               'rewired_src': [remap(i) for i in (2, 1, 4, 3)] * repeat,
                               'dst': [remap(0)] * (4 * repeat), 'attributes': attributes * repeat,
                               'features': features, 'query': remap(0), 'label': label})
        data[split] = graphs
    validate_dataset(data)
    return data


def validate_dataset(data):
    if type(data) is not dict or set(data) != {'version', 'task', 'train', 'held_out'} or type(data['version']) is not int or data['version'] != 1 or data['task'] != TASK:
        raise ValueError('Unsupported attention dataset schema')
    identities = set()
    for split in ('train', 'held_out'):
        graphs = data[split]
        if type(graphs) is not list or len(graphs) != 32:
            raise ValueError('Expected32 attention graphs per split')
        for graph in graphs:
            fields = {'id', 'num_nodes', 'src', 'dst', 'rewired_src', 'attributes', 'features', 'query', 'label'}
            if type(graph) is not dict or set(graph) != fields:
                raise ValueError('Invalid attention graph schema')
            identity, count = graph['id'], graph['num_nodes']
            if type(identity) is not str or not identity.startswith(split + '-') or identity in identities:
                raise ValueError('Attention graph identities must be unique and split-specific')
            identities.add(identity)
            if type(count) is not int or not 5 <= count <= 8:
                raise ValueError('Invalid attention node count')
            if type(graph['query']) is not int or not 0 <= graph['query'] < count:
                raise ValueError('Invalid attention query node')
            if type(graph['label']) is not int or graph['label'] not in (0, 1):
                raise ValueError('Invalid attention label')
            for side in ('src', 'dst', 'rewired_src'):
                if type(graph[side]) is not list or len(graph[side]) not in (4, 8) or any(type(i) is not int or not 0 <= i < count for i in graph[side]):
                    raise ValueError('Invalid attention topology indices')
            EdgeIndex(count, graph['src'], graph['dst'])
            EdgeIndex(count, graph['rewired_src'], graph['dst'])
            for name, rows, width in (('features', count, 3), ('attributes', len(graph['src']), 1)):
                values = graph[name]
                if type(values) is not list or len(values) != rows or any(type(row) is not list or len(row) != width or any(type(v) not in (int, float) or not math.isfinite(v) for v in row) for row in values):
                    raise ValueError('Invalid attention ' + name)
        for left, right in zip(graphs[::2], graphs[1::2]):
            if (left['label'], right['label']) != (0, 1) or any(left[k] != right[k] for k in ('num_nodes', 'src', 'dst', 'rewired_src', 'attributes', 'query')):
                raise ValueError('Attention topology and attributes must be paired across labels')
            if left['features'][left['query']] != right['features'][right['query']] or sorted(left['features']) != sorted(right['features']):
                raise ValueError('Attention paired query/features leak the label')
    if {canonical(g['features']) for g in data['train']} & {canonical(g['features']) for g in data['held_out']}:
        raise ValueError('Attention train and held-out features overlap')


def batch(graphs, rewired=False):
    sources = 'rewired_src' if rewired else 'src'
    edges, _ = EdgeIndex.batch([EdgeIndex(g['num_nodes'], g[sources], g['dst']) for g in graphs])
    features, attributes, queries, offset = [], [], [], 0
    for graph in graphs:
        features.extend(v for row in graph['features'] for v in row)
        attributes.extend(v for row in graph['attributes'] for v in row)
        queries.append(offset + graph['query'])
        offset += graph['num_nodes']
    return (fr.Tensor(features, [offset, 3]), fr.Tensor(attributes, [len(edges.src), 1]), edges, queries,
            fr.Tensor.from_i64([g['label'] for g in graphs], [len(graphs)]))


def save_run(directory, checkpoint, data, seed):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoded = canonical(data)
    (directory / 'dataset.json').write_bytes(encoded)
    checkpoint.save(directory / 'training')
    record = {'version': 1, 'task': TASK, 'seed': seed, 'schedule': SCHEDULE,
              'batch_size': BATCH_SIZE, 'config': CONFIG, 'step': checkpoint.step,
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
            'dataset_sha256', 'checkpoint_index_sha256', 'config'}
    if type(record) is not dict or set(record) != keys or type(record['version']) is not int or record['version'] != 1 or record['task'] != TASK:
        raise ValueError('Unsupported graph attention run schema')
    if type(record['seed']) is not int or record['seed'] != seed or record['schedule'] != SCHEDULE or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE or canonical(record['config']) != canonical(CONFIG):
        raise ValueError('graph attention run configuration mismatch')
    data = json.loads((directory / 'dataset.json').read_bytes())
    validate_dataset(data)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    if digest != record['dataset_sha256'] or digest != expected_digest:
        raise ValueError('graph attention dataset/topology identity mismatch')
    step = record['step']
    if type(step) is not int or not 0 <= step <= STEPS or type(record['next_graph']) is not int or record['next_graph'] != (step * BATCH_SIZE) % len(data['train']):
        raise ValueError('graph attention cursor mismatch')
    actual = hashlib.sha256((directory / 'training' / 'checkpoint.json').read_bytes()).hexdigest()
    if actual != record['checkpoint_index_sha256']:
        raise ValueError('graph attention checkpoint publication mismatch')
    checkpoint = fr.checkpoint.load(directory / 'training')
    if checkpoint.step != step:
        raise ValueError('graph attention checkpoint/cursor step mismatch')
    return data, checkpoint


def gradient_check(seed, data):
    import torch
    torch.set_num_threads(1)
    model = AttentionModel(seed=seed, dataset_digest=hashlib.sha256(canonical(data)).hexdigest())
    x, attributes, edges, queries, labels = batch(data['train'][:BATCH_SIZE])
    x.requires_grad_(True)
    attributes.requires_grad_(True)
    tx = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)
    ta = torch.tensor(attributes.tolist(), dtype=torch.float32, requires_grad=True)
    weights = {name: torch.tensor(p.tensor().tolist(), dtype=torch.float32, requires_grad=True)
               for name, p in model.named_parameters()}
    q, k, v = [tx @ weights[name] for name in ('wq', 'wk', 'wv')]
    scores = q @ k.T / math.sqrt(model.hidden)
    bias = (ta @ weights['we']).reshape(-1)
    rows = []
    for destination in range(edges.num_nodes):
        ids = [i for i, node in enumerate(edges.dst) if node == destination]
        if ids:
            sources = [edges.src[i] for i in ids]
            probabilities = torch.softmax(scores[destination, sources] + bias[ids], dim=0)
            rows.append((probabilities[:, None] * v[sources]).sum(0))
        else:
            rows.append(v.sum(0) * 0)
    attended = torch.stack(rows)
    logits = attended[queries] @ weights['wo'] + weights['bo']
    reference = torch.nn.functional.cross_entropy(logits, torch.tensor(labels.tolist(), dtype=torch.int64))
    actual_logits = model(x, attributes, edges, queries)
    torch.testing.assert_close(torch.tensor(actual_logits.tolist()), logits.detach(), atol=2e-5, rtol=2e-4)
    loss = fr.nn.functional.cross_entropy(actual_logits, labels)
    torch.testing.assert_close(torch.tensor(loss.item()), reference.detach(), atol=2e-5, rtol=2e-4)
    loss.backward()
    reference.backward()
    errors, norms = {}, {}
    pairs = [('input', x.grad, tx.grad), ('edge_attributes', attributes.grad, ta.grad)] + [(name, p.grad, weights[name].grad) for name, p in model.named_parameters()]
    for name, actual, expected in pairs:
        if actual is None or expected is None:
            raise AssertionError('Missing attention gradient: ' + name)
        got = torch.tensor(actual.tolist())
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)
        errors[name] = float((got - expected).abs().max())
        norms[name] = float(got.abs().max())
    for name in ('wq', 'wk', 'we', 'edge_attributes'):
        if not norms[name] > 1e-6:
            raise AssertionError('Zero attention score/edge gradient: ' + name)
    return {'max_abs_errors': errors, 'max_abs_gradients': norms}


def measure(model, data):
    with fr.no_grad():
        x, attributes, edges, queries, labels = batch(data['held_out'])
        wanted = labels.tolist()
        def accuracy(output):
            return sum(max(range(2), key=lambda j: row[j]) == label for row, label in zip(output.tolist(), wanted)) / len(wanted)
        logits = model(x, attributes, edges, queries)
        metrics = {'held_out_accuracy': accuracy(logits), 'held_out_loss': fr.nn.functional.cross_entropy(logits, labels).item(), 'class_prior_baseline': .5,
                   'query_local_baseline': .5}
        for ablation in ABLATIONS:
            if ablation == 'rewired':
                rx, ra, re, rq, _ = batch(data['held_out'], rewired=True)
                output = model(rx, ra, re, rq)
            elif ablation == 'edge_removed':
                output = model(x, fr.Tensor.zeros([0, 1]), EdgeIndex(edges.num_nodes, [], []), queries)
            else:
                output = model(x, attributes, edges, queries, mode=ablation)
            metrics[ablation + '_accuracy'] = accuracy(output)
            metrics[ablation + '_drop'] = metrics['held_out_accuracy'] - metrics[ablation + '_accuracy']
    return metrics


def train(seed, stop=STEPS, resume=None, midpoint=None, expected_digest=None):
    if resume is None:
        data, restored = make_dataset(), None
    else:
        if expected_digest is None:
            raise ValueError('Resume requires the expected dataset identity')
        data, restored = load_run(resume, expected_digest, seed)
    digest = hashlib.sha256(canonical(data)).hexdigest()
    model = AttentionModel(seed=seed, dataset_digest=digest)
    rng = fr.Generator(seed + (100 if restored is None else 999))
    opt = fr.optim.AdamW(model.parameters(), lr=model.lr if restored is None else .7, weight_decay=model.weight_decay)
    start = 0 if restored is None else restored.restore(model, optimizers={'main': opt}, generator=rng)
    if type(stop) is not int or not start <= stop <= STEPS:
        raise ValueError('Invalid graph attention training stop')
    initial_metrics = measure(model, data)
    batches = [batch(data['train'][i:i + BATCH_SIZE]) for i in range(0, 32, BATCH_SIZE)]
    losses = []
    for step in range(start, stop):
        x, attributes, edges, queries, labels = batches[step % len(batches)]
        features = x + fr.Tensor.randn(list(x.shape), generator=rng) * model.jitter
        edge_attributes = attributes + fr.Tensor.randn(list(attributes.shape), generator=rng) * model.jitter
        opt.zero_grad()
        loss = fr.nn.functional.cross_entropy(model(features, edge_attributes, edges, queries), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise AssertionError('Nonfinite graph attention training loss')
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
            raise AssertionError('Attention restart differs: ' + field)
    metrics = expected['metrics']
    if any(type(v) not in (int, float) or not math.isfinite(v) or not (-1 if key.endswith('_drop') else 0) <= v <= (float('inf') if key == 'held_out_loss' else 1) for key, v in metrics.items()):
        raise AssertionError('Attention invalid metrics: ' + repr(metrics))
    if not metrics['held_out_accuracy'] >= .95:
        raise AssertionError('Attention learning gate failed: ' + repr(metrics))
    initial_loss = expected['initial_metrics']['held_out_loss']
    if type(initial_loss) not in (int, float) or not math.isfinite(initial_loss) or initial_loss <= 0:
        raise AssertionError('Attention invalid initial loss')
    if not metrics['held_out_loss'] <= .05 or not metrics['held_out_loss'] <= .25 * initial_loss:
        raise AssertionError('Attention loss gate failed: ' + repr(metrics))
    for ablation in ABLATIONS:
        accuracy, drop = metrics[ablation + '_accuracy'], metrics[ablation + '_drop']
        if not math.isclose(drop, metrics['held_out_accuracy'] - accuracy, rel_tol=0, abs_tol=1e-12):
            raise AssertionError('Attention inconsistent ablation metrics: ' + ablation)
        if not accuracy <= .6 or not drop >= .35:
            raise AssertionError('Attention ablation gate failed: ' + ablation)


def prove(seed):
    data = make_dataset()
    gradients = gradient_check(seed, data)
    with tempfile.TemporaryDirectory(prefix='ferro-graph_attention-restart-') as directory:
        directory = Path(directory)
        midpoint = directory / 'midpoint'
        state, expected = train(seed, midpoint=midpoint)
        expected_state = state_digest(state, directory / 'expected')
        result_path = directory / 'resumed.json'
        script = Path(__file__).with_name('train_restart_graph_attention.py')
        flags = ['-O'] if sys.flags.optimize else []
        command = [sys.executable, '-B', *flags, str(script), '--seed', str(seed), '--resume', str(midpoint),
                   '--dataset-digest', expected['dataset_sha256'], '--worker-output', str(result_path)]
        subprocess.run(command, check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(result_path.read_bytes())
        check_acceptance(expected, resumed, expected_state)
    return {'task': TASK, 'seed': seed, 'steps': STEPS, 'metrics': expected['metrics'],
            'gradients': gradients, 'restart': 'bit-exact',
            'dataset_sha256': expected['dataset_sha256'], 'checkpoint_step': CHECKPOINT_STEP,
            'initial_metrics': expected['initial_metrics'], 'initial_loss': expected['losses'][0],
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
