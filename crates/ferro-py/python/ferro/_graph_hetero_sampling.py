"""Incoming CPU heterogeneous sampling without an optional graph backend.

Finite fanouts follow Ferro's once-discovered per-type frontier policy. Random
draws are deterministic for seed+epoch, not identical to PyG backend draws.
"""
from collections.abc import Mapping

from ._native import Tensor
from ._graph_hetero import HeteroData, _Store
from ._graph_utils import _capacity, _cpu_tensor, _edge_parts, _edge_tensor, _index_ids


_METADATA = {'n_id', 'e_id', 'input_id', 'batch_size', 'num_sampled_nodes', 'num_sampled_edges'}


class _SampledStore(_Store):
    @property
    def input_id(self):
        return self._input_id

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_sampled_nodes(self):
        return list(self._sampled_nodes)

    @property
    def num_sampled_edges(self):
        return list(self._sampled_edges)


class _HeteroNeighborhood:
    def __init__(self, data, num_neighbors, replace, subgraph_type, input_nodes):
        data.validate()
        if any(_METADATA.intersection(store.keys()) for store in data.stores):
            raise ValueError('input already contains reserved sampler metadata')
        if type(input_nodes) is str:
            input_type, selection = input_nodes, None
        elif type(input_nodes) is tuple and len(input_nodes) == 2 and type(input_nodes[0]) is str:
            input_type, selection = input_nodes
        else:
            raise ValueError('heterogeneous input_nodes must be a node type or (node type, IDs/Bool mask/None)')
        if input_type not in data.node_types:
            raise ValueError('input_nodes references an unknown node type')
        count = data[input_type].num_nodes
        _capacity((count,), 8)
        if isinstance(selection, Tensor) and selection.dtype == 'bool':
            _cpu_tensor(selection, 'input_nodes', 'bool')
            if selection.shape != [count]:
                raise ValueError('boolean input_nodes must have shape [input type num_nodes]')
            seeds = [i for i, take in enumerate(selection.tolist()) if take]
        else:
            seeds = list(range(count)) if selection is None else _index_ids(selection, 'input_nodes')
        if len(set(seeds)) != len(seeds) or any(node >= count for node in seeds):
            raise ValueError('input_nodes must be unique in-bounds node IDs')
        if type(replace) is not bool:
            raise TypeError('replace must be bool')
        if subgraph_type not in ('directional', 'induced'):
            raise NotImplementedError('subgraph_type must be directional or induced')
        if isinstance(num_neighbors, Mapping):
            if set(num_neighbors) != set(data.edge_types):
                raise ValueError('heterogeneous fanout mapping must cover every declared relation exactly')
            fanouts = dict(num_neighbors)
        else:
            fanouts = {relation: num_neighbors for relation in data.edge_types}
            if not fanouts and (not isinstance(num_neighbors, (list, tuple)) or any(type(n) is not int or n < -1 for n in num_neighbors)):
                raise ValueError('num_neighbors must contain nonnegative fanouts or -1')
        for values in fanouts.values():
            if not isinstance(values, (list, tuple)) or any(type(n) is not int or n < -1 for n in values):
                raise ValueError('num_neighbors must contain nonnegative fanouts or -1')
            for value in values:
                if value >= 0:
                    _capacity((value,), 8)
        depths = {len(values) for values in fanouts.values()}
        if len(depths) > 1:
            raise ValueError('all relation fanout lists must have equal length')
        self.depth = next(iter(depths), 0 if isinstance(num_neighbors, Mapping) else len(num_neighbors))
        self.data = HeteroData(data.to_dict())
        self.input_type, self.input_nodes = input_type, seeds
        self.replace, self.subgraph_type = replace, subgraph_type
        self.fanouts = {relation: tuple(values) for relation, values in fanouts.items()}
        self.topology, self.incoming = {}, {}
        for relation, store in self.data._edges.items():
            source, _, target = relation
            if 'edge_index' not in store and store.num_edges:
                raise ValueError('sampling an edge store requires edge_index')
            src, dst = ([], []) if 'edge_index' not in store else _edge_parts(store.edge_index, size=(data[source].num_nodes, data[target].num_nodes))[:2]
            self.topology[relation] = src, dst
            incoming = {}
            for edge, node in enumerate(dst):
                incoming.setdefault(node, []).append(edge)
            self.incoming[relation] = incoming

    def sample(self, seeds, input_ids, rng, batch_size):
        nodes = {name: [] for name in self.data.node_types}
        nodes[self.input_type] = list(seeds)
        mapping = {name: {node: i for i, node in enumerate(ids)} for name, ids in nodes.items()}
        frontier = {name: list(ids) for name, ids in nodes.items()}
        edge_ids = {relation: [] for relation in self.data.edge_types}
        node_counts = {name: [len(ids)] for name, ids in nodes.items()}
        edge_counts = {relation: [] for relation in edge_ids}
        for hop in range(self.depth):
            following = {name: [] for name in nodes}
            for relation in edge_ids:
                source, _, target = relation
                src, _ = self.topology[relation]
                fanout = self.fanouts[relation][hop]
                if self.replace and fanout >= 0:
                    _capacity((len(frontier[target]), fanout), 8)
                sampled = []
                for node in frontier[target]:
                    available = self.incoming[relation].get(node, ())
                    if fanout == -1 or not self.replace and fanout >= len(available):
                        picked = available
                    elif self.replace and available:
                        picked = rng.choices(available, k=fanout)
                    else:
                        picked = rng.sample(available, min(fanout, len(available)))
                    sampled.extend(picked)
                    for edge in picked:
                        neighbor = src[edge]
                        if neighbor not in mapping[source]:
                            mapping[source][neighbor] = len(nodes[source])
                            nodes[source].append(neighbor)
                            following[source].append(neighbor)
                edge_ids[relation].extend(sampled)
                edge_counts[relation].append(len(sampled))
            for name in nodes:
                node_counts[name].append(len(following[name]))
            frontier = following
        if self.subgraph_type == 'induced':
            for relation, (src, dst) in self.topology.items():
                source, _, target = relation
                edge_ids[relation] = [i for i, (s, d) in enumerate(zip(src, dst)) if s in mapping[source] and d in mapping[target]]
        result = HeteroData()
        result._global.update(self.data._global)
        for name, original in self.data._nodes.items():
            ids = nodes[name]
            _capacity((len(ids),), 8)
            store = _SampledStore(result, name)
            store._mapping = {key: value.index_select(0, ids) if isinstance(value, Tensor) and value.shape else value
                              for key, value in original.items() if key != 'num_nodes'}
            store._mapping.update(num_nodes=len(ids), n_id=Tensor.from_i64(ids, [len(ids)]))
            store._sampled_nodes = node_counts[name]
            if name == self.input_type:
                store._input_id = Tensor.from_i64(input_ids, [len(input_ids)])
                store._batch_size = batch_size
            result._nodes[name] = store
        for relation, original in self.data._edges.items():
            source, _, target = relation
            src, dst = self.topology[relation]
            ids = edge_ids[relation]
            _capacity((2, len(ids)), 8)
            store = _SampledStore(result, relation)
            store._mapping = {key: value.index_select(0, ids) if value.shape else value
                              for key, value in original.items() if key != 'edge_index'}
            store._mapping.update(edge_index=_edge_tensor([mapping[source][src[i]] for i in ids], [mapping[target][dst[i]] for i in ids]),
                                  e_id=Tensor.from_i64(ids, [len(ids)]))
            store._sampled_edges = edge_counts[relation]
            result._edges[relation] = store
        return result, mapping
