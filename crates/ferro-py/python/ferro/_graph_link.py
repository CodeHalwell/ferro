"""CPU link prediction helpers with exact exclusion and Ferro-specific RNG.

Negative sampling uses complement ranks, never a dense node-by-node matrix.
RandomLinkSplit groups duplicate positives and, for undirected graphs, reverse
pairs before partitioning. This prevents cross-split leakage; its random draws
and duplicate handling intentionally need not reproduce PyG's edge ordering.
"""
import bisect
import math
import random

from ._native import Tensor, cat
from ._graph_data import Data, Batch
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor, _index_ids


def _rng(seed):
    if seed is not None and type(seed) is not int:
        raise TypeError('seed must be an integer or None')
    return random.Random(seed)


def _requested(value, edges):
    if value is None:
        return edges
    if type(value) is float:
        if not math.isfinite(value) or value < 0:
            raise ValueError('negative sample ratio must be finite and nonnegative')
        value = int(edges * value)
    return _count(value, 'num_neg_samples')


def _sample_negative(src, dst, size, count, undirected, bipartite, rng):
    ns, nt = size
    if undirected and bipartite:
        undirected = False
    if undirected:
        population = ns * (ns - 1) // 2
        def encode(s, d):
            s, d = min(s, d), max(s, d)
            return s * (2 * ns - s - 1) // 2 + d - s - 1
        def decode(value):
            low, high = 0, ns - 1
            while low < high:
                mid = (low + high + 1) // 2
                if mid * (2 * ns - mid - 1) // 2 <= value:
                    low = mid
                else:
                    high = mid - 1
            return low, low + 1 + value - low * (2 * ns - low - 1) // 2
    elif bipartite:
        population = ns * nt
        def encode(s, d): return s * nt + d
        def decode(value): return divmod(value, nt)
    else:
        population = ns * (ns - 1)
        def encode(s, d): return s * (ns - 1) + d - int(d > s)
        def decode(value):
            s, d = divmod(value, ns - 1)
            return s, d + int(d >= s)
    occupied = sorted({encode(s, d) for s, d in zip(src, dst) if bipartite or s != d})
    available = population - len(occupied)
    count = min(count, available)
    _capacity((2, count), 8)
    # Floyd sampling supports populations larger than sys.maxsize without
    # materializing the complement or relying on len(range(population)).
    selected, ranks = set(), []
    for upper in range(available - count, available):
        candidate = rng.randrange(upper + 1)
        rank = upper if candidate in selected else candidate
        selected.add(rank)
        ranks.append(rank)
    rng.shuffle(ranks)
    output = []
    for rank in ranks:
        low, high = rank, rank + len(occupied)
        while low < high:
            mid = (low + high) // 2
            if mid - bisect.bisect_right(occupied, mid) < rank:
                low = mid + 1
            else:
                high = mid
        output.append(decode(low))
    return output


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None, method='sparse', force_undirected=False, *, seed=None):
    """Unique nonedges, excluding self loops for homogeneous graphs.

    num_nodes=(Ns, Nt) selects bipartite semantics, including equal endpoint
    IDs. A float num_neg_samples is a positive-edge-count ratio. Undirected
    output contains both orientations and rounds requested edge count down to
    an even number. Returns fewer samples only when the complement is exhausted.
    Both accepted method names use the same bounded sparse complement algorithm.
    """
    if method not in ('sparse', 'dense'):
        raise ValueError('method must be sparse or dense')
    if type(force_undirected) is not bool:
        raise TypeError('force_undirected must be bool')
    bipartite = isinstance(num_nodes, (tuple, list))
    src, dst, size = _edge_parts(edge_index, **({'size': num_nodes} if bipartite else {'num_nodes': num_nodes}))
    count = _requested(num_neg_samples, len(src))
    undirected = force_undirected and not bipartite
    pairs = _sample_negative(src, dst, size, count // 2 if undirected else count, undirected, bipartite, _rng(seed))
    if undirected:
        pairs += [(d, s) for s, d in pairs]
    return _edge_tensor([s for s, _ in pairs], [d for _, d in pairs])


def batched_negative_sampling(edge_index, batch, num_neg_samples=None, method='sparse', force_undirected=False, *, seed=None):
    """Sample within graph IDs; tuple batch vectors support bipartite graphs."""
    if method not in ('sparse', 'dense') or type(force_undirected) is not bool:
        raise ValueError('invalid method or force_undirected')
    bipartite = isinstance(batch, tuple)
    if bipartite:
        if len(batch) != 2:
            raise ValueError('bipartite batch must be a pair of index vectors')
        bs, bt = _index_ids(batch[0], 'source batch'), _index_ids(batch[1], 'target batch')
    else:
        bs = bt = _index_ids(batch, 'batch')
    src, dst, _ = _edge_parts(edge_index, size=(len(bs), len(bt)))
    if any(bs[s] != bt[d] for s, d in zip(src, dst)):
        raise ValueError('positive edges must stay within their graph IDs')
    nodes_s, nodes_t, grouped = {}, {}, {}
    for i, g in enumerate(bs): nodes_s.setdefault(g, []).append(i)
    for i, g in enumerate(bt): nodes_t.setdefault(g, []).append(i)
    for s, d in zip(src, dst): grouped.setdefault(bs[s], []).append((s, d))
    rng, result = _rng(seed), []
    for group in sorted(nodes_s.keys() | nodes_t.keys()):
        ns, nt = nodes_s.get(group, []), nodes_t.get(group, [])
        ms, mt = {n: i for i, n in enumerate(ns)}, {n: i for i, n in enumerate(nt)}
        pairs = grouped.get(group, [])
        count = _requested(num_neg_samples, len(pairs))
        undirected = force_undirected and not bipartite
        sampled = _sample_negative([ms[s] for s, _ in pairs], [mt[d] for _, d in pairs], (len(ns), len(nt)), count // 2 if undirected else count, undirected, bipartite, rng)
        if undirected:
            sampled += [(d, s) for s, d in sampled]
        result.extend((ns[s], nt[d]) for s, d in sampled)
    _capacity((2, len(result)), 8)
    return _edge_tensor([s for s, _ in result], [d for _, d in result])


def _graph_fields(data):
    if not isinstance(data, Data) or isinstance(data, Batch):
        raise TypeError('expected one homogeneous Data graph, not a Batch')
    data.validate()
    n, e = data.num_nodes, data.num_edges
    return {key: data._role(key, n, e) for key in data.keys() if key != 'edge_index'}


def _select_data(data, roles, edge_ids, src, dst, nodes=None):
    result = Data(num_nodes=data.num_nodes if nodes is None else len(nodes), field_roles=roles)
    result._store['edge_index'] = _edge_tensor(src, dst)
    for key, role in roles.items():
        value = data[key]
        result._store[key] = value.index_select(0, edge_ids) if role == 'edge' else value.index_select(0, nodes) if role == 'node' and nodes is not None else value
    return result


def _split_count(value, total, name):
    if type(value) is float:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + ' ratio must be between zero and one')
        return int(total * value)
    value = _count(value, name)
    if value > total:
        raise ValueError(name + ' exceeds the number of positive groups')
    return value


class RandomLinkSplit:
    """Split homogeneous positive pairs without validation/test adjacency leaks.

    Duplicate pairs stay together. Undirected reverse pairs stay together and
    missing reverse message edges are added with the matching attribute row.
    One representative per positive pair is supervised. Train adjacency includes
    train supervision unless disjoint_train_ratio is positive (as in PyG).
    Validation sees training edges; test sees training plus validation edges.
    Negative pairs are distinct across splits and exclude all original positives
    and their reverses when undirected. A finite complement can reduce counts.
    Supplied integer class labels shift only in splits that contain negatives.
    Standard edge_label_index and pos_/neg_ variants can be collated by Batch;
    custom label keys ending in _index require explicit sampling instead.
    """
    def __init__(self, num_val=.1, num_test=.2, is_undirected=False, key='edge_label', split_labels=False, add_negative_train_samples=True, neg_sampling_ratio=1., disjoint_train_ratio=0., edge_types=None, rev_edge_types=None, *, seed=None):
        if edge_types is not None or rev_edge_types is not None:
            raise NotImplementedError('RandomLinkSplit supports homogeneous graphs only')
        if any(type(v) is not bool for v in (is_undirected, split_labels, add_negative_train_samples)):
            raise TypeError('split flags must be bool')
        if not isinstance(key, str) or not key or key.startswith('_') or key == 'edge_index':
            raise ValueError('key must be a non-topology public field name')
        names = [f'{prefix}{key}{suffix}' for prefix in ('', 'pos_', 'neg_') for suffix in ('', '_index')]
        if any(name in ('edge_index', 'batch', 'ptr', 'num_graphs') or hasattr(Data, name) for name in names):
            raise ValueError('key would overwrite reserved graph metadata')
        for value, name in ((num_val, 'num_val'), (num_test, 'num_test'), (disjoint_train_ratio, 'disjoint_train_ratio')):
            if type(value) not in (int, float) or value < 0 or not math.isfinite(value) or (type(value) is float and value > 1):
                raise ValueError(name + ' must be a nonnegative count or ratio in [0,1]')
        if type(neg_sampling_ratio) not in (int, float) or not math.isfinite(neg_sampling_ratio) or neg_sampling_ratio < 0:
            raise ValueError('neg_sampling_ratio must be finite and nonnegative')
        _rng(seed)
        self.num_val, self.num_test, self.is_undirected = num_val, num_test, is_undirected
        self.key, self.split_labels = key, split_labels
        self.add_negative_train_samples, self.neg_sampling_ratio = add_negative_train_samples, float(neg_sampling_ratio)
        self.disjoint_train_ratio, self.seed, self.calls = disjoint_train_ratio, seed, 0

    def __call__(self, data):
        roles = _graph_fields(data)
        src, dst = ([], []) if data.edge_index is None else _edge_parts(data.edge_index, num_nodes=data.num_nodes)[:2]
        groups = {}
        for i, (s, d) in enumerate(zip(src, dst)):
            pair = (min(s, d), max(s, d)) if self.is_undirected else (s, d)
            groups.setdefault(pair, []).append(i)
        count = len(groups)
        nv = _split_count(self.num_val, count, 'num_val')
        nt = _split_count(self.num_test, count, 'num_test')
        train_count = count - nv - nt
        if train_count <= 0:
            raise ValueError('split requires at least one training positive pair')
        nd = _split_count(self.disjoint_train_ratio, train_count, 'disjoint_train_ratio')
        if nd >= train_count:
            raise ValueError('disjoint split must leave training message edges')
        label = data._store.get(self.key)
        if label is not None:
            _cpu_tensor(label, self.key, dtype=None)
            if label.dtype not in ('int64', 'float32', 'float64') or label.shape != [len(src)]:
                raise ValueError('existing edge labels must have shape [E] and dtype int64, float32 or float64')
            values = label.tolist()
            if any(any(values[i] != values[ids[0]] for i in ids) for ids in groups.values()):
                raise ValueError('duplicate/reverse positive pairs must have equal labels')
        generated = [f'{prefix}{self.key}{suffix}' for prefix in ('', 'pos_', 'neg_') for suffix in ('', '_index')]
        if any(key in roles and key != self.key for key in generated):
            raise ValueError('input contains reserved split output fields')
        rng = _rng(None if self.seed is None else self.seed + self.calls)
        pairs = list(groups)
        rng.shuffle(pairs)
        train, val, test = pairs[:train_count], pairs[train_count:train_count + nv], pairs[train_count + nv:]
        supervised = [train[:nd] if nd else train, val, test]
        message = [train[nd:] if nd else train, train, train + val]
        wanted = [int(len(part) * self.neg_sampling_ratio) if i or self.add_negative_train_samples else 0 for i, part in enumerate(supervised)]
        negatives = _sample_negative(src, dst, (data.num_nodes, data.num_nodes), sum(wanted), self.is_undirected, False, rng)
        if label is not None and negatives and (label.dtype != 'int64' or any(v < 0 for v in values)):
            raise ValueError('negative sampling with supplied labels requires nonnegative int64 class labels')
        total = sum(wanted)
        if len(negatives) < total:
            counts = [n * len(negatives) // total for n in wanted]
            for i in range(len(negatives) - sum(counts)):
                choices = [j for j in range(3) if counts[j] < wanted[j]]
                counts[choices[i % len(choices)]] += 1
        else:
            counts = wanted
        output, offset = [], 0
        for positive, adjacency, negative_count in zip(supervised, message, counts):
            edge_ids, es, ed = [], [], []
            for pair in adjacency:
                ids = groups[pair]
                for i in ids:
                    edge_ids.append(i); es.append(src[i]); ed.append(dst[i])
                if self.is_undirected and pair[0] != pair[1]:
                    present = {(src[i], dst[i]) for i in ids}
                    for orientation in (pair, pair[::-1]):
                        if orientation not in present:
                            edge_ids.append(ids[0]); es.append(orientation[0]); ed.append(orientation[1])
            result = _select_data(data, {k: v for k, v in roles.items() if k != self.key}, edge_ids, es, ed)
            neg = negatives[offset:offset + negative_count]
            offset += negative_count
            pos_ids = [groups[p][0] for p in positive]
            if label is None:
                pos_labels = Tensor.ones([len(positive)])
                neg_labels = Tensor.zeros([len(neg)])
            elif label.dtype == 'int64':
                pos_labels = Tensor.from_i64([values[i] + int(bool(neg)) for i in pos_ids], [len(pos_ids)])
                neg_labels = Tensor.from_i64([0] * len(neg), [len(neg)])
            else:
                pos_labels = label.index_select(0, pos_ids)
                neg_labels = (Tensor.from_f64([0.] * len(neg), [len(neg)])
                              if label.dtype == 'float64' else Tensor.zeros([len(neg)]))
            if self.split_labels:
                parts = [('pos_' + self.key, positive, pos_labels)]
                if neg:
                    parts.append(('neg_' + self.key, neg, neg_labels))
            else:
                parts = [(self.key, positive + neg, cat([pos_labels, neg_labels], 0))]
            for name, endpoints, labels in parts:
                result._store[name + '_index'] = _edge_tensor([s for s, _ in endpoints], [d for _, d in endpoints])
                result._store[name] = labels
                result._field_roles[name + '_index'] = result._field_roles[name] = 'graph'
            output.append(result)
        self.calls += 1
        return tuple(output)
