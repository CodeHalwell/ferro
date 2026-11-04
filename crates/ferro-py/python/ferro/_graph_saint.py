"""CPU GraphSAINT induced-subgraph sampling with empirical normalization.

Node sampling draws source endpoints of uniformly sampled edges (so duplicate
edges contribute their multiplicity). Random walks choose outgoing edges
uniformly and remain at sinks. Returned nodes and COO edges use sorted node/CSR
order; n_id/e_id retain original identities. Field selection uses native tensor
operations and keeps gradients. Graph-level fields are shared references.

The local seed+epoch RNG differs from PyG. Normalization uses a separate seeded
calibration stream, completes whole num_steps rounds, and does not consume a
training epoch. These are Monte Carlo weights, not exact inclusion probabilities.
Workers, normalization caches, heterogeneous graphs and custom index remapping
are unsupported. Induced extraction scans the edge list, without dense N*N data.
"""
import bisect
import random

from ._native import Tensor
from ._graph_data import Data
from ._graph_link import _graph_fields, _rng, _select_data
from ._graph_utils import _capacity, _count, _edge_parts


class _GraphSAINTSampler:
    def __init__(self, data, batch_size, num_steps=1, sample_coverage=0,
                 save_dir=None, log=True, *, seed=None, num_workers=0, **kwargs):
        if kwargs:
            raise NotImplementedError('unsupported GraphSAINT options: ' + ', '.join(sorted(kwargs)))
        for value, name in ((batch_size, 'batch_size'), (num_steps, 'num_steps'),
                            (sample_coverage, 'sample_coverage')):
            _count(value, name)
        if batch_size == 0 or num_steps == 0:
            raise ValueError('batch_size and num_steps must be positive')
        if type(num_workers) is not int or num_workers != 0:
            raise NotImplementedError('GraphSAINT supports num_workers=0 only')
        if save_dir is not None:
            raise NotImplementedError('normalization caches are unsupported')
        if type(log) is not bool:
            raise TypeError('log must be bool')
        _rng(seed)
        roles = _graph_fields(data)
        if data.num_nodes == 0 or data.edge_index is None:
            raise ValueError('GraphSAINT requires nodes and an explicit edge_index')
        if any(key in data for key in ('n_id', 'e_id', 'node_norm', 'edge_norm')):
            raise ValueError('input contains reserved GraphSAINT metadata')
        if any(key.endswith('_index') for key in roles):
            raise NotImplementedError('GraphSAINT cannot remap custom or supervision index fields')
        for count in (batch_size, data.num_nodes, data.num_edges):
            _capacity((count,), 8)
        _capacity((data.num_nodes, sample_coverage), 8)
        self.N, self.E = data.num_nodes, data.num_edges
        self.src, self.dst, _ = _edge_parts(data.edge_index, num_nodes=self.N)
        self._edge_order = sorted(range(self.E), key=lambda i: (self.src[i], self.dst[i], i))
        self._outgoing = {}
        for i in self._edge_order:
            self._outgoing.setdefault(self.src[i], []).append(self.dst[i])
        self.data = Data(num_nodes=self.N, field_roles=roles)
        self.data._store = dict(data.items())
        self._roles = roles
        self.batch_size, self.num_steps, self.sample_coverage = batch_size, num_steps, sample_coverage
        self.seed = random.SystemRandom().randrange(2**64) if seed is None else seed
        self.epoch, self.log = 0, log
        self._validate_sampler()
        if sample_coverage:
            self.node_norm, self.edge_norm = self._compute_norm()

    def _validate_sampler(self):
        pass

    def _sample_nodes(self, rng):
        raise NotImplementedError

    def _sample(self, rng):
        nodes = sorted(set(self._sample_nodes(rng)))
        mapping = {node: i for i, node in enumerate(nodes)}
        edges = [i for i in self._edge_order if self.src[i] in mapping and self.dst[i] in mapping]
        return nodes, edges, mapping

    def _compute_norm(self):
        rng = random.Random(self.seed ^ 0x6A09E667F3BCC909)
        node_count, edge_count = [0] * self.N, [0] * self.E
        samples = total = 0
        while total < self.N * self.sample_coverage:
            for _ in range(self.num_steps):
                nodes, edges, _ = self._sample(rng)
                for node in nodes:
                    node_count[node] += 1
                for edge in edges:
                    edge_count[edge] += 1
                total += len(nodes)
                samples += 1
        self.normalization_samples = samples
        node_norm = [samples / (count if count else .1) / self.N for count in node_count]
        edge_norm = [min(node_count[self.src[i]] / count, 1e4) if count else
                     (1e4 if node_count[self.src[i]] else .1) for i, count in enumerate(edge_count)]
        return Tensor(node_norm, [self.N]), Tensor(edge_norm, [self.E])

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        def iterate():
            for _ in range(self.num_steps):
                nodes, edges, mapping = self._sample(rng)
                result = _select_data(self.data, self._roles, edges,
                                      [mapping[self.src[i]] for i in edges],
                                      [mapping[self.dst[i]] for i in edges], nodes)
                result._store['n_id'] = Tensor.from_i64(nodes, [len(nodes)])
                result._store['e_id'] = Tensor.from_i64(edges, [len(edges)])
                result._field_roles.update(n_id='node', e_id='edge')
                if self.sample_coverage:
                    result._store['node_norm'] = self.node_norm.index_select(0, nodes)
                    result._store['edge_norm'] = self.edge_norm.index_select(0, edges)
                    result._field_roles.update(node_norm='node', edge_norm='edge')
                yield result
        return iterate()


class GraphSAINTNodeSampler(_GraphSAINTSampler):
    """Draw batch_size source endpoints with replacement, then induce their graph.

    Sources are weighted by outgoing edge multiplicity. Nodes without outgoing
    edges cannot be drawn; use random-walk sampling to include isolated nodes.
    sample_coverage=0 omits normalization fields, matching PyG's default.
    """
    def _validate_sampler(self):
        if not self.E:
            raise ValueError('GraphSAINT node sampling requires at least one edge')

    def _sample_nodes(self, rng):
        return [self.src[self._edge_order[rng.randrange(self.E)]] for _ in range(self.batch_size)]


class GraphSAINTEdgeSampler(_GraphSAINTSampler):
    """Draw weighted edges with replacement and induce their endpoint union.

    Matches PyG 2.8.1's implemented edge mass: indegree[source] plus
    outdegree[target], with a 1e-10 floor term. Degrees count duplicates.
    Each draw is independent, including parallel edges and self-loops. This
    preserves that version's distribution, not its RNG sequence or a different
    inverse-degree proposal. Cumulative masses avoid a batch_size by E matrix.
    """
    def __init__(self, data, batch_size, num_steps=1, sample_coverage=0,
                 save_dir=None, log=True, *, seed=None, num_workers=0, **kwargs):
        _capacity((batch_size, 2), 8)
        super().__init__(data, batch_size, num_steps, sample_coverage, save_dir,
                         log, seed=seed, num_workers=num_workers, **kwargs)

    def _validate_sampler(self):
        if not self.E:
            raise ValueError('GraphSAINT edge sampling requires at least one edge')
        incoming = {}
        for target in self.dst:
            incoming[target] = incoming.get(target, 0) + 1
        self._edge_cdf = []
        total = 0.
        for edge in self._edge_order:
            mass = incoming.get(self.src[edge], 0) + len(self._outgoing.get(self.dst[edge], ())) + 1e-10
            total += mass
            self._edge_cdf.append(total)

    def _sample_nodes(self, rng):
        nodes = []
        for _ in range(self.batch_size):
            position = bisect.bisect_right(self._edge_cdf, rng.random() * self._edge_cdf[-1], hi=self.E - 1)
            edge = self._edge_order[position]
            nodes.extend((self.dst[edge], self.src[edge]))
        return nodes


class GraphSAINTRandomWalkSampler(_GraphSAINTSampler):
    """Start batch_size uniform nodes and take walk_length outgoing edge steps.

    All visited nodes form an induced subgraph, including edges not traversed.
    Duplicate edges retain their sampling multiplicity. A sink stays in place;
    walk_length=0 samples uniform nodes, including isolates and edgeless graphs.
    """
    def __init__(self, data, batch_size, walk_length, num_steps=1, sample_coverage=0,
                 save_dir=None, log=True, *, seed=None, num_workers=0, **kwargs):
        _count(walk_length, 'walk_length')
        _capacity((batch_size, walk_length + 1), 8)
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage, save_dir,
                         log, seed=seed, num_workers=num_workers, **kwargs)

    def _sample_nodes(self, rng):
        roots = [rng.randrange(self.N) for _ in range(self.batch_size)]
        nodes = []
        for node in roots:
            nodes.append(node)
            for _ in range(self.walk_length):
                neighbors = self._outgoing.get(node)
                if neighbors:
                    node = neighbors[rng.randrange(len(neighbors))]
                nodes.append(node)
        return nodes
