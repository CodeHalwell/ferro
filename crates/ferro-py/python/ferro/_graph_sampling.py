"""Synchronous CPU neighborhood loaders, without optional backends.

Incoming edges follow source-to-target flow. Each newly discovered frontier is
expanded once; -1 takes all incoming edges and zero takes none. Node IDs start
with the requested seeds, followed by first-discovery order. Sampling uses a
local Python RNG seeded by seed+epoch, not PyG's random sequence. Topology and
field references are fixed when constructing a loader; gradients remain live.
"""
import math
import random

from ._native import Tensor, cat
from ._graph_data import Data
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, _edge_tensor, _index_ids
from ._graph_link import _graph_fields, _select_data, _sample_negative, _rng


class _SampledData(Data):
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_sampled_nodes(self):
        return list(self._sampled_nodes)

    @property
    def num_sampled_edges(self):
        return list(self._sampled_edges)


class _Neighborhood:
    def __init__(self, data, num_neighbors, replace, subgraph_type, drop_fields=()):
        roles = _graph_fields(data)
        if any(key in data for key in ('n_id', 'e_id', 'input_id', 'batch_size', 'num_sampled_nodes', 'num_sampled_edges')):
            raise ValueError('input already contains reserved sampler metadata')
        if not isinstance(num_neighbors, (list, tuple)) or any(type(v) is not int or v < -1 for v in num_neighbors):
            raise ValueError('num_neighbors must be a sequence of nonnegative fanouts or -1')
        for fanout in num_neighbors:
            if fanout >= 0:
                _capacity((fanout,), 8)
        if type(replace) is not bool:
            raise TypeError('replace must be bool')
        if subgraph_type not in ('directional', 'induced'):
            raise NotImplementedError('subgraph_type must be directional or induced')
        self.src, self.dst = ([], []) if data.edge_index is None else _edge_parts(data.edge_index, num_nodes=data.num_nodes)[:2]
        self.roles = {k: role for k, role in roles.items() if k not in drop_fields}
        self.data = Data(num_nodes=data.num_nodes, field_roles=self.roles)
        self.data._store = {k: v for k, v in data.items() if k == 'edge_index' or k in self.roles}
        self.fanouts, self.replace, self.subgraph_type = tuple(num_neighbors), replace, subgraph_type
        self.incoming = {}
        for i, node in enumerate(self.dst):
            self.incoming.setdefault(node, []).append(i)

    def sample(self, seeds, input_ids, rng, batch_size, excluded=()):
        nodes = list(seeds)
        mapping = {node: i for i, node in enumerate(nodes)}
        frontier, edge_ids = list(seeds), []
        node_counts, edge_counts = [len(nodes)], []
        excluded = set(excluded)
        for fanout in self.fanouts:
            if self.replace and fanout >= 0:
                _capacity((len(frontier), fanout), 8)
            next_frontier, sampled = [], []
            for target in frontier:
                available = [i for i in self.incoming.get(target, ()) if (self.src[i], target) not in excluded]
                if fanout == -1:
                    picked = available
                elif self.replace and available:
                    picked = rng.choices(available, k=fanout)
                else:
                    picked = available if fanout >= len(available) else rng.sample(available, fanout)
                sampled.extend(picked)
                for i in picked:
                    source = self.src[i]
                    if source not in mapping:
                        mapping[source] = len(nodes)
                        nodes.append(source)
                        next_frontier.append(source)
            edge_ids.extend(sampled)
            node_counts.append(len(next_frontier))
            edge_counts.append(len(sampled))
            frontier = next_frontier
        if self.subgraph_type == 'induced':
            edge_ids = [i for i, (s, d) in enumerate(zip(self.src, self.dst)) if s in mapping and d in mapping and (s, d) not in excluded]
        _capacity((len(nodes),), 8)
        _capacity((2, len(edge_ids)), 8)
        src = [mapping[self.src[i]] for i in edge_ids]
        dst = [mapping[self.dst[i]] for i in edge_ids]
        selected = _select_data(self.data, self.roles, edge_ids, src, dst, nodes)
        result = _SampledData(num_nodes=len(nodes), field_roles=selected._field_roles)
        result._store = selected._store
        for key, values, role in (('n_id', nodes, 'node'), ('e_id', edge_ids, 'edge'), ('input_id', input_ids, 'graph')):
            result._store[key] = Tensor.from_i64(values, [len(values)])
            result._field_roles[key] = role
        result._batch_size = batch_size
        result._sampled_nodes, result._sampled_edges = node_counts, edge_counts
        return result, mapping


def _loader_options(batch_size, shuffle, drop_last, num_workers, seed):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if type(shuffle) is not bool or type(drop_last) is not bool:
        raise TypeError('shuffle and drop_last must be bool')
    if type(num_workers) is not int or num_workers != 0:
        raise NotImplementedError('neighborhood loaders support num_workers=0 only')
    _rng(seed)
    return random.SystemRandom().randrange(2**64) if seed is None else seed


def _unsupported(options):
    if options:
        raise NotImplementedError('unsupported neighborhood loader options: ' + ', '.join(sorted(options)))


def _batches(loader, count):
    order = list(range(count))
    rng = random.Random(loader.seed + loader.epoch)
    loader.epoch += 1
    if loader.shuffle:
        rng.shuffle(order)
    def iterate():
        for start in range(0, count, loader.batch_size):
            ids = order[start:start + loader.batch_size]
            if len(ids) < loader.batch_size and loader.drop_last:
                break
            yield ids, rng
    return iterate()


class NeighborLoader:
    """Sample incoming neighborhoods for CPU node minibatches.

    input_nodes is an int64 vector/sequence, a bool mask, or None (all nodes). n_id maps local
    nodes to source nodes; e_id maps each retained edge to its original row.
    input_id indexes input_nodes, and the first batch_size n_id entries are
    supervision seeds. Node labels require field_roles={'y': 'node'}.

    Supports directional sampled edges or the induced graph on sampled nodes.
    HeteroData requires input_nodes='type' or ('type', IDs/Bool mask/None), and
    accepts shared fanouts or an exact relation-to-fanouts mapping of equal depth.
    Every declared node/relation store is retained, including empty stores.
    Typed n_id/e_id fields remain aligned; input-store input_id/batch_size and
    per-store hop counts are read-only properties, not persisted graph fields.
    Per-hop counts describe expansion before induced-edge completion. Temporal,
    weighted, disjoint, bidirectional and worker sampling reject.
    Finite fanouts use Ferro's once-discovered frontier sampling; neither its
    random draws nor every optional PyG sampling-backend behavior is reproduced.
    """
    def __init__(self, data, num_neighbors, input_nodes=None, *, batch_size=1, shuffle=False, drop_last=False, replace=False, subgraph_type='directional', num_workers=0, seed=None, **kwargs):
        _unsupported(kwargs)
        self.seed = _loader_options(batch_size, shuffle, drop_last, num_workers, seed)
        from ._graph_hetero import HeteroData
        if isinstance(data, HeteroData):
            from ._graph_hetero_sampling import _HeteroNeighborhood
            self._graph = _HeteroNeighborhood(data, num_neighbors, replace, subgraph_type, input_nodes)
            self.input_nodes = self._graph.input_nodes
            self.batch_size, self.shuffle, self.drop_last, self.epoch = batch_size, shuffle, drop_last, 0
            return
        self._graph = _Neighborhood(data, num_neighbors, replace, subgraph_type)
        _capacity((data.num_nodes,), 8)
        if isinstance(input_nodes, Tensor) and input_nodes.dtype == 'bool':
            _cpu_tensor(input_nodes, 'input_nodes', 'bool')
            if input_nodes.shape != [data.num_nodes]:
                raise ValueError('boolean input_nodes must have shape [num_nodes]')
            self.input_nodes = [i for i, take in enumerate(input_nodes.tolist()) if take]
        else:
            self.input_nodes = list(range(data.num_nodes)) if input_nodes is None else _index_ids(input_nodes, 'input_nodes')
        if len(set(self.input_nodes)) != len(self.input_nodes) or any(n >= data.num_nodes for n in self.input_nodes):
            raise ValueError('input_nodes must be unique in-bounds node IDs')
        self.batch_size, self.shuffle, self.drop_last, self.epoch = batch_size, shuffle, drop_last, 0

    def __len__(self):
        n = len(self.input_nodes)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = _batches(self, len(self.input_nodes))
        return (self._graph.sample([self.input_nodes[i] for i in ids], ids, rng, len(ids))[0] for ids, rng in batches)


class LinkNeighborLoader:
    """Endpoint-seeded incoming neighborhoods with local edge_label_index.

    Binary negative sampling excludes every message edge and supplied supervision
    pair, but cannot know positives absent from both inputs. Explicit labels are
    copied; when generating negatives, labels containing zero shift by one.
    No label is added when edge_label=None and no negatives are requested.

    As in PyG, supervision edges remain in message adjacency by default.
    exclude_seed_edges=True removes all current supervision pairs; also set
    exclude_reverse_edges=True to remove their reverses. RandomLinkSplit's
    disjoint mode can provide a permanently separate supervision graph.
    """
    def __init__(self, data, num_neighbors, edge_label_index=None, edge_label=None, *, batch_size=1, shuffle=False, drop_last=False, replace=False, subgraph_type='directional', neg_sampling_ratio=0., neg_sampling=None, exclude_seed_edges=False, exclude_reverse_edges=False, num_workers=0, seed=None, **kwargs):
        _unsupported(kwargs)
        if neg_sampling is not None:
            raise NotImplementedError('use binary neg_sampling_ratio; custom/triplet negative sampling is unsupported')
        if neg_sampling_ratio is None:
            neg_sampling_ratio = 0.
        if type(neg_sampling_ratio) not in (int, float) or not math.isfinite(neg_sampling_ratio) or neg_sampling_ratio < 0:
            raise ValueError('neg_sampling_ratio must be finite and nonnegative')
        if type(exclude_seed_edges) is not bool or type(exclude_reverse_edges) is not bool:
            raise TypeError('edge exclusion flags must be bool')
        if exclude_reverse_edges and not exclude_seed_edges:
            raise ValueError('exclude_reverse_edges requires exclude_seed_edges=True')
        self.seed = _loader_options(batch_size, shuffle, drop_last, num_workers, seed)
        self._graph = _Neighborhood(data, num_neighbors, replace, subgraph_type, ('edge_label', 'edge_label_index'))
        self.src, self.dst = (list(self._graph.src), list(self._graph.dst)) if edge_label_index is None else _edge_parts(edge_label_index, num_nodes=data.num_nodes)[:2]
        if edge_label is not None:
            _cpu_tensor(edge_label, 'edge_label', dtype=None)
            if edge_label.dtype not in ('float32', 'float64', 'int64') or edge_label.shape != [len(self.src)]:
                raise ValueError('edge_label must be float32, float64 or int64 with shape [supervision edges]')
            if neg_sampling_ratio and any(v < 0 or not math.isfinite(v) for v in edge_label.tolist()):
                raise ValueError('automatic negative sampling requires finite nonnegative labels')
        self.edge_label = edge_label
        self.neg_sampling_ratio = float(neg_sampling_ratio)
        self.exclude_seed_edges, self.exclude_reverse_edges = exclude_seed_edges, exclude_reverse_edges
        self.batch_size, self.shuffle, self.drop_last, self.epoch = batch_size, shuffle, drop_last, 0

    def __len__(self):
        n = len(self.src)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = _batches(self, len(self.src))
        def iterate():
            for ids, rng in batches:
                pairs = [(self.src[i], self.dst[i]) for i in ids]
                negatives = _sample_negative(self._graph.src + self.src, self._graph.dst + self.dst, (self._graph.data.num_nodes,) * 2, math.ceil(len(ids) * self.neg_sampling_ratio), False, False, rng)
                endpoints = pairs + negatives
                seeds = list(dict.fromkeys([s for s, _ in endpoints] + [d for _, d in endpoints]))
                excluded = set(pairs) if self.exclude_seed_edges else set()
                if self.exclude_reverse_edges:
                    excluded.update((d, s) for s, d in pairs)
                result, mapping = self._graph.sample(seeds, ids, rng, len(ids), excluded)
                result._store['edge_label_index'] = _edge_tensor([mapping[s] for s, _ in endpoints], [mapping[d] for _, d in endpoints])
                result._field_roles['edge_label_index'] = 'graph'
                if self.edge_label is not None or self.neg_sampling_ratio:
                    if self.edge_label is None:
                        labels = Tensor([1.] * len(ids) + [0.] * len(negatives), [len(endpoints)])
                    elif self.edge_label.dtype == 'int64':
                        values = self.edge_label.tolist()
                        shift = int(bool(self.neg_sampling_ratio) and min(values, default=1) == 0)
                        labels = Tensor.from_i64([values[i] + shift for i in ids] + [0] * len(negatives), [len(endpoints)])
                    else:
                        labels = self.edge_label.index_select(0, ids)
                        if self.neg_sampling_ratio and min(self.edge_label.tolist(), default=1) == 0:
                            labels = labels + 1.
                        zeros = (Tensor.from_f64([0.] * len(negatives), [len(negatives)])
                                 if labels.dtype == 'float64' else Tensor.zeros([len(negatives)]))
                        labels = cat([labels, zeros], 0)
                    result._store['edge_label'] = labels
                    result._field_roles['edge_label'] = 'graph'
                yield result
        return iterate()
