"""PyG-shaped graph classification using only Ferro and the standard library.

Run: python examples/pyg_parity.py  (or select --kind gcn / sage / gatv2).
Seed 7, 60 epochs: >=95% held-out accuracy, CE <=25% of initialization,
and 50% accuracy after edge removal. Checkpoint continuation is exact within
one environment at a completed epoch boundary; loader iterators are not saved.
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
from ferro.graph import Data, Batch, DataLoader, GCNConv, SAGEConv, global_mean_pool
from training_restart import state_digest

EPOCHS = 60
CHECKPOINT_EPOCH = 13
BATCH_SIZE = 8
KINDS = ('gcn', 'sage', 'gatv2')


class GraphClassifier(fr.nn.Module):
    dataset_digest: str
    kind: str = 'gcn'
    seed: int = 7
    hidden: int = 8
    batch_size: int = BATCH_SIZE
    noise_scale: float = 0.002

    def build(self):
        if self.kind == 'gcn':
            self.conv = GCNConv(2, self.hidden)
        elif self.kind == 'sage':
            self.conv = SAGEConv(2, self.hidden)
        elif self.kind == 'gatv2':
            self.conv = fr.graph.GATv2Conv(2, self.hidden // 2, heads=2)
        else:
            raise ValueError('kind must be gcn, sage or gatv2')
        self.readout = fr.nn.Linear(self.hidden, 2)
        with fr.no_grad():
            for index, (name, parameter) in enumerate(self.named_parameters()):
                shape = list(parameter.tensor().shape)
                value = fr.Tensor.zeros(shape) if 'bias' in name else fr.Tensor.randn(shape, seed=self.seed + index) * shape[-1] ** -0.5
                parameter.tensor().copy_(value)

    def forward(self, batch, noise=None, erase_edges=False):
        features = batch.x if noise is None else batch.x + noise
        edges = fr.Tensor.from_i64([], [2, 0]) if erase_edges else batch.edge_index
        hidden = self.conv(features, edges).tanh()
        return self.readout(global_mean_pool(hidden, batch.batch, size=batch.num_graphs))


def make_dataset():
    data = {}
    for split, phase, offset in (('train', 0.25, 0), ('held_out', 0.75, 1000)):
        data[split] = []
        for pair in range(16):
            count, amplitude = 4 + pair % 3, 0.8 + 0.3 * math.sin(pair + phase) ** 2
            values = [[0., 1.], [amplitude, 0.], [-amplitude, 0.], [0., -1.]] + [[0., 0.]] * (count - 4)
            for label in (0, 1):
                neighbor = 1 if label else 2
                graph = Data(x=fr.Tensor([v for row in values for v in row], [count, 2]),
                             edge_index=fr.Tensor.from_i64([0, neighbor, neighbor, 0], [2, 2]),
                             y=fr.Tensor.from_i64([label], [1]), num_nodes=count,
                             sample_id=fr.Tensor.from_i64([offset + 2 * pair + label], [1]),
                             field_roles={'sample_id': 'graph'})
                graph.validate()
                data[split].append(graph)
    return data


def encode(data):
    return {split: [{'num_nodes': graph.num_nodes, **{name: value.tolist() for name, value in graph.items()}}
                    for graph in rows] for split, rows in data.items()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def identity(data):
    return hashlib.sha256(canonical(encode(data))).hexdigest()


def decode(data):
    if type(data) is not dict or set(data) != {'train', 'held_out'}:
        raise ValueError('invalid graph dataset splits')
    result = {}
    for split, rows in data.items():
        if type(rows) is not list or len(rows) != 32:
            raise ValueError('expected 32 graphs per split')
        result[split] = []
        for row in rows:
            if type(row) is not dict or set(row) != {'num_nodes', 'x', 'edge_index', 'y', 'sample_id'}:
                raise ValueError('invalid saved graph fields')
            graph = Data(x=fr.Tensor([v for item in row['x'] for v in item], [row['num_nodes'], 2]),
                         edge_index=fr.Tensor.from_i64([v for side in row['edge_index'] for v in side], [2, len(row['edge_index'][0])]),
                         y=fr.Tensor.from_i64(row['y'], [1]), num_nodes=row['num_nodes'],
                         sample_id=fr.Tensor.from_i64(row['sample_id'], [1]), field_roles={'sample_id': 'graph'})
            graph.validate()
            result[split].append(graph)
    return result


def measure(model, data):
    batch = Batch.from_data_list(data['held_out'])
    with fr.no_grad():
        output, erased = model(batch), model(batch, erase_edges=True)
        labels = batch.y.tolist()
        accuracy = sum(a == b for a, b in zip(output.argmax(1).tolist(), labels)) / len(labels)
        ablated = sum(a == b for a, b in zip(erased.argmax(1).tolist(), labels)) / len(labels)
        loss = fr.nn.CrossEntropyLoss()(output, batch.y).item()
    return {'held_out_accuracy': accuracy, 'held_out_loss': loss,
            'edge_erased_accuracy': ablated, 'feature_only_baseline': 0.5}


def save_run(path, checkpoint, data, kind, seed, epoch):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    checkpoint.save(path / 'training')
    record = {'version': 1, 'kind': kind, 'seed': seed, 'epoch': epoch, 'batch_size': BATCH_SIZE,
              'data': encode(data), 'dataset_digest': identity(data)}
    (path / 'run.json.tmp').write_bytes(canonical(record))
    os.replace(path / 'run.json.tmp', path / 'run.json')


def train(kind='gcn', seed=7, epochs=EPOCHS, resume=None, midpoint=None):
    start, checkpoint = 0, None
    if resume is None:
        data = make_dataset()
    else:
        record = json.loads((Path(resume) / 'run.json').read_bytes())
        if set(record) != {'version', 'kind', 'seed', 'epoch', 'batch_size', 'data', 'dataset_digest'} or type(record['version']) is not int or record['version'] != 1 or record['kind'] != kind or type(record['seed']) is not int or record['seed'] != seed or type(record['batch_size']) is not int or record['batch_size'] != BATCH_SIZE:
            raise ValueError('graph run configuration mismatch')
        start = record['epoch']
        if type(start) is not int or not 0 <= start <= EPOCHS:
            raise ValueError('invalid saved loader epoch')
        data = decode(record['data'])
        if identity(data) != record['dataset_digest']:
            raise ValueError('saved graph data identity mismatch')
        checkpoint = fr.checkpoint.load(Path(resume) / 'training')
    if type(epochs) is not int or not start <= epochs <= EPOCHS:
        raise ValueError('invalid training epoch limit')
    loader = DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True, seed=seed + 100)
    loader.epoch = start
    if checkpoint is not None and checkpoint.step != start * len(loader):
        raise ValueError('checkpoint step and loader epoch differ')
    model = GraphClassifier(kind=kind, seed=seed, dataset_digest=identity(data))
    optimizer = fr.optim.AdamW(model.parameters(), lr=0.025 if checkpoint is None else 0.7, weight_decay=0.0001)
    rng = fr.Generator(seed + (200 if checkpoint is None else 999))
    if checkpoint is not None:
        checkpoint.restore(model, optimizers={'main': optimizer}, generator=rng)
    criterion, losses, batches = fr.nn.CrossEntropyLoss(), [], []
    initial = measure(model, data)
    for epoch in range(start, epochs):
        for batch in loader:
            optimizer.zero_grad()
            noise = fr.Tensor.randn(list(batch.x.shape), generator=rng) * model.noise_scale
            loss = criterion(model(batch, noise), batch.y)
            value = loss.item()
            if not math.isfinite(value):
                raise AssertionError('graph training loss is not finite')
            losses.append(value)
            batches.append(batch.sample_id.tolist())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if midpoint is not None and epoch + 1 == CHECKPOINT_EPOCH:
            snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=(epoch + 1) * len(loader))
            save_run(midpoint, snapshot, data, kind, seed, epoch + 1)
    snapshot = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=epochs * len(loader))
    return snapshot, {'losses': losses, 'batch_ids': batches, 'initial_metrics': initial, 'metrics': measure(model, data),
                      'loader_epoch': loader.epoch, 'dataset_digest': identity(data), 'next_draws': fr.Tensor.randn([7], generator=rng).tolist()}


def demonstrate(kind='gcn', seed=7):
    with tempfile.TemporaryDirectory(prefix='ferro-pyg-training-') as directory:
        directory = Path(directory)
        snapshot, expected = train(kind, seed, midpoint=directory / 'checkpoint')
        expected_state = state_digest(snapshot, directory / 'expected')
        output = directory / 'resumed.json'
        flags = ['-O'] if sys.flags.optimize else []
        subprocess.run([sys.executable, '-B', *flags, str(Path(__file__).resolve()), '--kind', kind, '--seed', str(seed),
                        '--resume', str(directory / 'checkpoint'), '--worker-output', str(output)],
                       check=True, timeout=120, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        resumed = json.loads(output.read_bytes())
        offset = CHECKPOINT_EPOCH * (32 // BATCH_SIZE)
        for field, wanted in (('losses', expected['losses'][offset:]), ('batch_ids', expected['batch_ids'][offset:]),
                              ('metrics', expected['metrics']), ('loader_epoch', expected['loader_epoch']),
                              ('dataset_digest', expected['dataset_digest']), ('next_draws', expected['next_draws']), ('state', expected_state)):
            if resumed[field] != wanted:
                raise AssertionError('graph continuation differs: ' + field)
    metrics, initial = expected['metrics'], expected['initial_metrics']
    if not metrics['held_out_accuracy'] >= 0.95 or not metrics['held_out_loss'] <= 0.25 * initial['held_out_loss'] or metrics['edge_erased_accuracy'] != 0.5 or metrics['held_out_accuracy'] - metrics['edge_erased_accuracy'] < 0.4:
        raise AssertionError('graph learning gate failed: ' + repr(metrics))
    return {'kind': kind, 'seed': seed, 'epochs': EPOCHS, 'checkpoint_epoch': CHECKPOINT_EPOCH,
            'initial_metrics': initial, 'metrics': metrics, 'restart': 'bit-exact-at-epoch-boundary',
            'dataset_digest': expected['dataset_digest']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('all', *KINDS), default='all')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    if args.worker_output is not None:
        if args.resume is None or args.kind == 'all':
            parser.error('worker output requires a checkpoint and one graph kind')
        snapshot, result = train(args.kind, args.seed, resume=args.resume)
        result['state'] = state_digest(snapshot, args.worker_output.with_suffix('.state'))
        args.worker_output.write_bytes(canonical(result))
    else:
        if args.resume is not None:
            parser.error('resume is reserved for the continuation worker')
        for kind in KINDS if args.kind == 'all' else (args.kind,):
            print(json.dumps(demonstrate(kind, args.seed), sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    # Checkpoint class names stay identical when imported or run as a script.
    from pyg_parity import main as entrypoint
    entrypoint()
