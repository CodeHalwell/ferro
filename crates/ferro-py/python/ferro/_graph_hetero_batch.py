"""Typed heterogeneous disjoint batches over native concatenation and views.

Store sets and retained tensor field names must match across examples. Empty
types are explicit stores. Scalars stack and recover their original scalar
shape; row fields concatenate. Relation COO rows receive their own source and
target node offsets. Custom index hooks, sparse stores and nested batches are
unsupported. Metadata and original field slices belong to this batch only.
"""
from ._native import Tensor, cat
from ._graph_data import _field_names
from ._graph_hetero import HeteroData, _Store, _field
from ._graph_utils import _capacity, _count, _edge_parts, _edge_tensor


def _mappings(data):
    return {None: data._global, **{key: store._mapping for key, store in data._nodes.items()},
            **{key: store._mapping for key, store in data._edges.items()}}


class _BatchStore(_Store):
    @property
    def num_edges(self):
        if isinstance(self._key, tuple):
            return self._parent._edge_totals[self._key]
        return super().num_edges


class HeteroBatch(HeteroData):
    """Disjoint HeteroData examples with per-type int64 batch and ptr.

    follow_batch/exclude_keys apply to matching names in every store. Scalars
    recover their original shape on extraction, including store metadata.
    A relation with edges requires explicit edge_index. Excluding topology
    also requires excluding its nonempty row fields. Additional fields added
    after collation are not reconstructed by to_data_list. CPU f32/f64/int64/bool
    fields and their existing first-order autograd paths are preserved.
    """
    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None):
        items = list(data_list)
        follow = _field_names(follow_batch, 'follow_batch')
        excluded = set(_field_names(exclude_keys, 'exclude_keys'))
        if not items:
            raise ValueError('from_data_list requires at least one heterogeneous graph')
        if any(not isinstance(data, HeteroData) or isinstance(data, HeteroBatch) for data in items):
            raise TypeError('heterogeneous batches require HeteroData instances, not mixed or nested batches')
        if any(hasattr(type(data), hook) for data in items for hook in ('__inc__', '__cat_dim__')):
            raise NotImplementedError('custom heterogeneous batching hooks are unsupported')
        for data in items:
            data.validate()
        nodes, edges = items[0].metadata()
        if any(set(data.node_types) != set(nodes) or set(data.edge_types) != set(edges) for data in items):
            raise ValueError('heterogeneous batches require matching declared node and relation types; retain empty stores')
        mappings = [_mappings(data) for data in items]
        node_counts = {key: tuple(data[key].num_nodes for data in items) for key in nodes}
        edge_counts = {key: tuple(data[key].num_edges for data in items) for key in edges}
        nodeptr = {}
        for key, counts in node_counts.items():
            offsets = [0]
            for count in counts:
                offsets.append(_count(offsets[-1] + count, 'total typed nodes'))
            _capacity((offsets[-1],), 8)
            nodeptr[key] = tuple(offsets)
        _capacity((len(items) + 1,), 8)
        keys, slices, shapes, generated, edge_totals = {}, {}, {}, {}, {}
        for store_key in (None, *nodes, *edges):
            source = [mapping[store_key] for mapping in mappings]
            retained = [key for key in source[0] if key not in excluded and key != 'num_nodes']
            if any(set(mapping) - excluded - {'num_nodes'} != set(retained) for mapping in source):
                raise ValueError('retained heterogeneous tensor field names must match per store')
            if isinstance(store_key, tuple) and any('index' in key and key != 'edge_index' for key in retained):
                raise NotImplementedError('custom relation index fields are unsupported')
            if isinstance(store_key, str) and any(key in retained for key in ('batch', 'ptr')):
                raise ValueError('input node stores already contain batch metadata')
            names = [key + suffix for key in retained if key in follow for suffix in ('_batch', '_ptr')]
            if any(name in retained for name in names) or len(set(names)) != len(names):
                raise ValueError('follow_batch generated names conflict with heterogeneous fields')
            if store_key is None and any(key in nodes or hasattr(cls, key) for key in (*retained, *names)):
                raise ValueError('global batch fields conflict with node types or batch metadata')
            keys[store_key], slices[store_key], shapes[store_key] = tuple(retained), {}, {}
            generated[store_key] = {}
            for key in retained:
                values = [mapping[key] for mapping in source]
                if key == 'edge_index' and isinstance(store_key, tuple):
                    offsets = [0]
                    for graph, value in zip(items, values):
                        src, _, _ = _edge_parts(value, size=(graph[store_key[0]].num_nodes, graph[store_key[2]].num_nodes))
                        offsets.append(_count(offsets[-1] + len(src), 'total typed edges'))
                    _capacity((2, offsets[-1]), 8)
                else:
                    normalized = [value.shape if value.shape else [1] for value in values]
                    if any(value.dtype != values[0].dtype or shape[1:] != normalized[0][1:] for value, shape in zip(values, normalized)):
                        raise ValueError('heterogeneous fields need matching dtype and trailing shape')
                    offsets = [0]
                    for shape in normalized:
                        offsets.append(_count(offsets[-1] + shape[0], 'total field rows'))
                    _capacity([offsets[-1]] + normalized[0][1:], 8 if values[0].dtype in ('int64', 'float64') else 4)
                slices[store_key][key] = tuple(offsets)
                shapes[store_key][key] = tuple(tuple(value.shape) for value in values)
                if key in follow:
                    _capacity((offsets[-1],), 8)
                    generated[store_key][key + '_batch'] = ('batch', tuple(offsets))
                    generated[store_key][key + '_ptr'] = ('ptr', tuple(offsets))
            if isinstance(store_key, tuple):
                if any(count and 'edge_index' not in mapping for count, mapping in zip(edge_counts[store_key], source)):
                    raise NotImplementedError('nonempty relation stores require explicit edge_index for batching')
                if 'edge_index' not in retained and any(edge_counts[store_key]) and any(any(mapping[key].shape for mapping in source) for key in retained):
                    raise NotImplementedError('excluding relation edge_index also requires excluding its nonempty row fields')
                edge_totals[store_key] = sum(edge_counts[store_key]) if 'edge_index' in retained else 0
            if isinstance(store_key, str):
                generated[store_key]['batch'] = ('batch', nodeptr[store_key])
                generated[store_key]['ptr'] = ('ptr', nodeptr[store_key])
        result = cls()
        result._node_counts, result._nodeptr, result._edge_counts = node_counts, nodeptr, edge_counts
        result._edge_totals, result._field_keys, result._slices, result._shapes = edge_totals, keys, slices, shapes
        result._generated, result._num_graphs, result._follow = generated, len(items), tuple(follow)
        for key in nodes:
            result._nodes[key] = _BatchStore(result, key)
            result._nodes[key]._mapping['num_nodes'] = nodeptr[key][-1]
        for key in edges:
            result._edges[key] = _BatchStore(result, key)
        output = _mappings(result)
        for store_key in keys:
            for key in keys[store_key]:
                if key == 'edge_index' and isinstance(store_key, tuple):
                    src, dst = [], []
                    for number, data in enumerate(items):
                        s, t, _ = _edge_parts(data[store_key].edge_index, size=data[store_key].size())
                        src.extend(value + nodeptr[store_key[0]][number] for value in s)
                        dst.extend(value + nodeptr[store_key[2]][number] for value in t)
                    output[store_key][key] = _edge_tensor(src, dst)
                else:
                    values = [mapping[store_key][key] for mapping in mappings]
                    output[store_key][key] = cat([value if value.shape else value.unsqueeze(0) for value in values], 0)
            for key, (kind, offsets) in generated[store_key].items():
                values = list(offsets) if kind == 'ptr' else [i for i, (a, b) in enumerate(zip(offsets, offsets[1:])) for _ in range(b - a)]
                output[store_key][key] = Tensor.from_i64(values, [len(values)])
        return result

    @property
    def num_graphs(self):
        return self._num_graphs

    def validate(self, raise_on_error=True):
        if type(raise_on_error) is not bool:
            raise TypeError('raise_on_error must be bool')
        if not hasattr(self, '_field_keys'):
            raise ValueError('HeteroBatch must be constructed by from_data_list')
        try:
            if set(self.node_types) != set(self._node_counts) or set(self.edge_types) != set(self._edge_counts):
                raise ValueError('batched store types changed after collation')
            mappings = _mappings(self)
            for store_key, keys in self._field_keys.items():
                mapping = mappings[store_key]
                if isinstance(store_key, str) and self[store_key].num_nodes != self._nodeptr[store_key][-1]:
                    raise ValueError('batched node count changed after collation')
                for key in keys:
                    value = mapping[key]
                    _field(key, value)
                    if key == 'edge_index' and isinstance(store_key, tuple):
                        src, dst, _ = _edge_parts(value, size=self[store_key].size())
                        extent = value.shape[1]
                        for number in range(self.num_graphs):
                            start, stop = self._slices[store_key][key][number:number + 2]
                            source_start, source_stop = self._nodeptr[store_key[0]][number:number + 2]
                            target_start, target_stop = self._nodeptr[store_key[2]][number:number + 2]
                            if any(not source_start <= index < source_stop for index in src[start:stop]) or any(not target_start <= index < target_stop for index in dst[start:stop]):
                                raise ValueError('relation edges cross original graph boundaries')
                    else:
                        if not value.shape:
                            raise ValueError('batched scalar fields require the graph axis')
                        extent = value.shape[0]
                        if all(not shape for shape in self._shapes[store_key][key]) and len(value.shape) != 1:
                            raise ValueError('batched scalar metadata shape changed')
                    if extent != self._slices[store_key][key][-1]:
                        raise ValueError('batched field concatenation length changed')
                for key, (kind, offsets) in self._generated[store_key].items():
                    value = mapping[key]
                    expected = list(offsets) if kind == 'ptr' else [i for i, (a, b) in enumerate(zip(offsets, offsets[1:])) for _ in range(b - a)]
                    if value.dtype != 'int64' or value.device != 'cpu' or value.shape != [len(expected)] or value.tolist() != expected:
                        raise ValueError('generated batch metadata changed after collation')
            return True
        except (KeyError, TypeError, ValueError):
            if raise_on_error:
                raise
            return False

    def __getitem__(self, idx):
        from ._graph_batch_select import getitem
        return getitem(self, idx)

    def get_example(self, idx):
        """Reconstruct one graph from current values and saved slice/offset metadata."""
        from ._graph_batch_select import get_example
        return get_example(self, idx)

    def index_select(self, idx):
        """Return selected graphs; negative IDs normalize and Boolean masks are exact-length."""
        from ._graph_batch_select import index_select
        return index_select(self, idx)

    def to_data_list(self):
        """Split current values through the same inverse used for graph selection."""
        from ._graph_batch_select import to_data_list
        return to_data_list(self)

    def clone(self, *args):
        """Clone selected tensor fields, retaining stores and isolated batch metadata."""
        from ._graph_clone import clone
        return clone(self, *args)

