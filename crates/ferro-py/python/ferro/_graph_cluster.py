"""CPU cluster minibatches with explicit, non-METIS partitioning.

The default balanced BFS orders each connected component of the undirected
topology, then divides that traversal into parts whose sizes differ by at most
one. It is a deterministic heuristic, not METIS or an edge-cut optimum. Supply
partition assignments to reuse partitions from an external partitioner. Nodes
within a part use increasing original IDs; duplicate edges remain distinct.
"""
from collections import deque
from dataclasses import dataclass
import random

from ._native import Tensor
from ._graph_data import Data
from ._graph_link import _graph_fields, _select_data
from ._graph_utils import _capacity, _count, _edge_parts, _index_ids


@dataclass(frozen=True)
class _Partition:
    indptr: Tensor
    index: Tensor
    partptr: Tensor
    node_perm: Tensor
    edge_perm: Tensor
    sparse_format: str


class ClusterData:
    """Partition one homogeneous graph for synchronous cluster minibatches.

    ``partition`` is an int64 [N] Tensor or integer sequence assigning each
    original node to [0, num_parts). Otherwise method='bfs' uses the balanced
    heuristic above. CSR and CSC determine edge order; all selections preserve
    tensor gradients and int64/bool fields. ``data`` retains the original graph
    field order, while ``partition`` exposes permutation/adjacency metadata.
    Metadata is a snapshot; changing its tensors does not change partitions.

    Individual items contain only internal edges. ClusterLoader also retains
    edges between parts in the same minibatch. METIS, recursive partitioning,
    disk caches and dangling inter-partition edges in individual items are
    unsupported. Add node/edge fields n_id/e_id before partitioning to retain
    original identities in returned Data objects.
    """
    def __init__(self, data, num_parts, recursive=False, save_dir=None, filename=None, log=True,
                 keep_inter_cluster_edges=False, sparse_format='csr', *, method='bfs', partition=None):
        num_parts = _count(num_parts, 'num_parts')
        if not num_parts:
            raise ValueError('num_parts must be positive')
        if type(recursive) is not bool or type(log) is not bool or type(keep_inter_cluster_edges) is not bool:
            raise TypeError('recursive, log and keep_inter_cluster_edges must be bool')
        if recursive or method != 'bfs':
            raise NotImplementedError("only deterministic method='bfs' or supplied partition assignments are supported; METIS is unavailable")
        if save_dir is not None or filename is not None:
            raise NotImplementedError('partition disk caching is unsupported')
        if keep_inter_cluster_edges:
            raise NotImplementedError('individual partitions cannot expose dangling edges; use ClusterLoader to merge parts')
        if sparse_format not in ('csr', 'csc'):
            raise ValueError("sparse_format must be 'csr' or 'csc'")
        roles = _graph_fields(data)
        if data.edge_index is None:
            raise ValueError('ClusterData requires edge_index, including for an edgeless graph')
        n = data.num_nodes
        src, dst, _ = _edge_parts(data.edge_index, num_nodes=n)
        _capacity((n + 1,), 8)
        _capacity((num_parts + 1,), 8)
        _capacity((len(src),), 8)
        if partition is None:
            adjacency = {}
            for source, target in zip(src, dst):
                adjacency.setdefault(source, set()).add(target)
                adjacency.setdefault(target, set()).add(source)
            adjacency = {node: sorted(neighbors) for node, neighbors in adjacency.items()}
            visited, order = bytearray(n), []
            for root in range(n):
                if visited[root]:
                    continue
                visited[root] = 1
                queue = deque([root])
                while queue:
                    node = queue.popleft()
                    order.append(node)
                    for neighbor in adjacency.get(node, ()):
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            queue.append(neighbor)
            assignments, offset = [0] * n, 0
            for part in range(num_parts):
                count = n // num_parts + int(part < n % num_parts)
                for node in order[offset:offset + count]:
                    assignments[node] = part
                offset += count
        else:
            assignments = _index_ids(partition, 'partition', n)
            if any(part >= num_parts for part in assignments):
                raise ValueError('partition assignments must be smaller than num_parts')
        parts = [[] for _ in range(num_parts)]
        for node, part in enumerate(assignments):
            parts[part].append(node)
        node_perm, partptr = [], [0]
        for nodes in parts:
            node_perm.extend(nodes)
            partptr.append(len(node_perm))
        inverse = [0] * n
        for position, node in enumerate(node_perm):
            inverse[node] = position
        primary, secondary = (src, dst) if sparse_format == 'csr' else (dst, src)
        edge_perm = sorted(range(len(src)), key=lambda edge: (inverse[primary[edge]], inverse[secondary[edge]], edge))
        part_edges, indptr = [[] for _ in range(num_parts)], [0] * (n + 1)
        for edge in edge_perm:
            part_edges[assignments[primary[edge]]].append(edge)
            indptr[inverse[primary[edge]] + 1] += 1
        for node in range(n):
            indptr[node + 1] += indptr[node]
        self.data = Data(num_nodes=n, field_roles=roles, **data.to_dict())
        self.num_parts, self.sparse_format, self.method = num_parts, sparse_format, method
        self.recursive, self.keep_inter_cluster_edges = False, False
        self._roles, self._src, self._dst = roles, tuple(src), tuple(dst)
        self._parts = tuple(tuple(nodes) for nodes in parts)
        self._part_edges = tuple(tuple(edges) for edges in part_edges)
        self.partition = _Partition(Tensor.from_i64(indptr, [n + 1]),
            Tensor.from_i64([inverse[secondary[edge]] for edge in edge_perm], [len(src)]),
            Tensor.from_i64(partptr, [num_parts + 1]), Tensor.from_i64(node_perm, [n]),
            Tensor.from_i64(edge_perm, [len(src)]), sparse_format)

    def __len__(self):
        return self.num_parts

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError('partition index must be an integer')
        if index < 0:
            index += self.num_parts
        if not 0 <= index < self.num_parts:
            raise IndexError('partition index is out of range')
        return self._collate([index])

    def _collate(self, indices):
        nodes = [node for part in indices for node in self._parts[part]]
        remap = {node: index for index, node in enumerate(nodes)}
        edges = [edge for part in indices for edge in self._part_edges[part]
                 if self._src[edge] in remap and self._dst[edge] in remap]
        return _select_data(self.data, self._roles, edges,
                            [remap[self._src[edge]] for edge in edges],
                            [remap[self._dst[edge]] for edge in edges], nodes)


class ClusterLoader:
    """Yield induced unions of graph partitions, retaining cross-part edges.

    Shuffle uses Ferro's local seed+epoch RNG, not Torch's random sequence.
    Partition order is decided at iterator creation; field selection is lazy.
    Worker processes, asynchronous prefetch and iterator checkpointing are
    unsupported. Returned values are individual Data graphs, not Batch objects.
    """
    def __init__(self, cluster_data, batch_size=1, shuffle=False, drop_last=False, num_workers=0, seed=None):
        if not isinstance(cluster_data, ClusterData):
            raise TypeError('cluster_data must be a ClusterData instance')
        batch_size = _count(batch_size, 'batch_size')
        if not batch_size:
            raise ValueError('batch_size must be positive')
        if type(shuffle) is not bool or type(drop_last) is not bool:
            raise TypeError('shuffle and drop_last must be bool')
        if type(num_workers) is not int or num_workers != 0:
            raise NotImplementedError('ClusterLoader supports num_workers=0 only')
        if seed is not None and type(seed) is not int:
            raise TypeError('seed must be an integer or None')
        self.cluster_data, self.dataset = cluster_data, cluster_data
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.seed = random.SystemRandom().randrange(2**64) if seed is None else seed
        self.epoch = 0

    def __len__(self):
        return len(self.dataset) // self.batch_size if self.drop_last else (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        def batches():
            for start in range(0, len(order), self.batch_size):
                parts = order[start:start + self.batch_size]
                if len(parts) < self.batch_size and self.drop_last:
                    break
                yield self.cluster_data._collate(parts)
        return batches()
