"""Homogeneous in-memory graph data and synchronous, gradient-preserving batches.

Tensor fields concatenate along axis zero; scalar graph fields are stacked.
Known node fields are x/pos/node_attr/node_type, known edge fields are
edge_attr/edge_weight, and y is a graph field by default. Set field_roles to
{'y': 'node'} for node labels or to disambiguate custom node/edge/graph fields.
CPU float32/float64 fields retain their dtype and autograd history; integer
labels retain int64 storage and masks retain bool. Graph convolution layers
can impose a narrower feature dtype contract. Homogeneous edge_label_index and its
pos_/neg_ variants concatenate columns with node offsets. Nested values,
sparse layouts, non-integer custom increments, worker processes
and loader checkpointing are unsupported.
"""
import random

from ._native import Tensor, EdgeIndex, cat
from ._graph_utils import _capacity, _count, _cpu_tensor, _edge_parts, _edge_tensor


_NODE_FIELDS = {'x', 'pos', 'node_attr', 'node_type', 'train_mask', 'val_mask', 'test_mask'}
_EDGE_FIELDS = {'edge_attr', 'edge_weight'}
_INDEX_FIELDS = {'edge_label_index', 'pos_edge_label_index', 'neg_edge_label_index'}
_RESERVED = {'batch', 'ptr', 'num_graphs', 'field_roles'}


def _custom_hooks(data):
    return any(hasattr(type(data), hook) and getattr(type(data), hook) is not getattr(Data, hook)
               for hook in ('__inc__', '__cat_dim__'))


def _needs_hook_collation(data, excluded=()):
    return _custom_hooks(data) or any(key != 'edge_index' and key not in _INDEX_FIELDS
                                      and ('index' in key or key == 'face' or 'batch' in key)
                                      for key in data.keys() if key not in excluded)


def _has_node_count(data):
    return data._num_nodes is not None or any(key in data for key in ('x', 'pos', 'node_attr', 'node_type', 'edge_index', 'face'))


def _field_names(value, name):
    if value is None:
        return ()
    if not isinstance(value, (tuple, list, set, frozenset)) or any(not isinstance(key, str) for key in value):
        raise TypeError(f'{name} must be a sequence of field names')
    return tuple(dict.fromkeys(value))


def _label_parts(data, key):
    _cpu_tensor(data[key], key, 'int64')
    if data._role(key, data.num_nodes, data.num_edges) != 'graph':
        raise ValueError('link-label index fields must have graph role')
    return _edge_parts(data[key], num_nodes=data.num_nodes)


class Data:
    """A homogeneous CPU graph with tensor fields and optional explicit node count.

    Set num_nodes when x/pos are absent to retain trailing isolated nodes.
    Custom tensor roles can be 'node', 'edge' or 'graph'; ambiguous inferred
    roles raise during batching instead of silently corrupting field boundaries.
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, num_nodes=None, *, field_roles=None, **kwargs):
        object.__setattr__(self, '_store', {})
        object.__setattr__(self, '_num_nodes', None if num_nodes is None else _count(num_nodes, 'num_nodes'))
        roles = {} if field_roles is None else dict(field_roles)
        if any(not isinstance(k, str) or v not in ('node', 'edge', 'graph') for k, v in roles.items()):
            raise ValueError('field_roles maps names to node, edge or graph')
        object.__setattr__(self, '_field_roles', roles)
        for key, value in dict(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, **kwargs).items():
            if value is not None:
                self[key] = value

    def __getattr__(self, key):
        store = object.__getattribute__(self, '_store')
        if key in store:
            return store[key]
        if key in ('x', 'edge_index', 'edge_attr', 'y', 'pos'):
            return None
        raise AttributeError(key)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        elif key == 'num_nodes':
            object.__setattr__(self, '_num_nodes', None if value is None else _count(value, 'num_nodes'))
        else:
            self[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str) or key.startswith('_') or key in _RESERVED or hasattr(type(self), key):
            raise ValueError(f'unsupported or reserved graph field {key!r}')
        if value is None:
            self._store.pop(key, None)
            return
        if key == 'edge_index':
            _edge_parts(value)
        else:
            _cpu_tensor(value, key, dtype=None)
            if value.dtype not in ('float32', 'float64', 'int64', 'bool'):
                raise ValueError('graph fields support CPU float32, float64, int64 or bool only')
        self._store[key] = value

    def __contains__(self, key):
        return key in self._store

    def keys(self):
        return list(self._store)

    def items(self):
        return self._store.items()

    def to_dict(self):
        return dict(self._store)

    def subgraph(self, subset):
        """Select nodes in order, aligning fields and relabeling induced edges.

        Repeated nodes keep repeated feature rows; edges use the last selected
        occurrence. Eager unbatched CPU graphs use int64 IDs or bool masks.
        """
        from ._graph_subgraph import subgraph
        return subgraph(self, subset)

    def edge_subgraph(self, subset):
        """Select ordered edges and aligned fields while retaining every node."""
        from ._graph_subgraph import edge_subgraph
        return edge_subgraph(self, subset)

    def clone(self, *args):
        """Clone selected native tensor fields while preserving class and batch state."""
        from ._graph_clone import clone
        return clone(self, *args)

    def apply(self, func, *args):
        """Replace selected tensor fields sequentially with supported Tensor results.

        Counts, native EdgeIndex objects and private batch metadata stay fixed.
        Callbacks own alignment and partial side effects if a later call fails.
        """
        from ._graph_apply import apply
        return apply(self, func, *args)

    def detach(self, *args):
        """Replace selected slots with detached views; external handles stay live."""
        from ._graph_apply import detach
        return detach(self, *args)

    def detach_(self, *args):
        """True in-place Tensor detachment is unsupported."""
        from ._graph_apply import detach_
        return detach_(self, *args)

    def contiguous(self, *args):
        """Make selected tensor fields contiguous using native ownership rules."""
        from ._graph_apply import contiguous
        return contiguous(self, *args)

    def cpu(self, *args):
        """Return this CPU graph without changing selected tensor handles."""
        from ._graph_apply import cpu
        return cpu(self, *args)

    def to(self, device, *args, non_blocking=False):
        """Accept an identity CPU conversion; device migration is unsupported."""
        from ._graph_apply import to
        return to(self, device, *args, non_blocking=non_blocking)

    def __cat_dim__(self, key, value, *args, **kwargs):
        """Concatenate index/face columns, other dense fields along axis zero.

        Subclasses may return another axis or None to stack a graph field.
        """
        return -1 if 'index' in key or key == 'face' else 0

    def __inc__(self, key, value, *args, **kwargs):
        """Return the next example's scalar or broadcast int64 ID increment."""
        if 'batch' in key:
            _cpu_tensor(value, key, 'int64')
            _capacity(value.shape, 8)
            ids = value.reshape([-1]).tolist()
            if not ids:
                raise ValueError('cannot infer an increment from an empty batch field')
            if any(item < 0 for item in ids):
                raise ValueError('batch IDs must be nonnegative')
            return _count(max(ids) + 1, 'batch increment')
        if 'index' in key or key == 'face':
            if not _has_node_count(self):
                raise ValueError('default index increments require an inferable num_nodes; set it explicitly or override __inc__')
            return self.num_nodes
        return 0

    def to_heterogeneous(self, node_type=None, edge_type=None, node_type_names=None, edge_type_names=None):
        """Split typed CPU fields and COO endpoints using int64 type vectors.

        Explicit field_roles resolve custom node/edge/graph fields. In-memory
        to_homogeneous results retain type names, including empty types. Custom
        index fields and batched graphs require separate conversion per graph.
        """
        from ._graph_convert import to_heterogeneous
        return to_heterogeneous(self, node_type, edge_type, node_type_names, edge_type_names)

    @property
    def num_nodes(self):
        if self._num_nodes is not None:
            return self._num_nodes
        for key in ('x', 'pos', 'node_attr', 'node_type'):
            value = self._store.get(key)
            if value is not None and value.shape:
                return value.shape[0]
        if self.edge_index is not None:
            return _edge_parts(self.edge_index)[2][0]
        if 'face' in self:
            value = _cpu_tensor(self.face, 'face', 'int64')
            _capacity(value.shape, 8)
            return _count(max(value.reshape([-1]).tolist(), default=-1) + 1, 'num_nodes')
        return 0

    @property
    def num_edges(self):
        return 0 if self.edge_index is None else len(_edge_parts(self.edge_index)[0])

    @property
    def num_node_features(self):
        if self.x is not None and not self.x.shape:
            raise ValueError('x must have a node axis')
        return 0 if self.x is None else (1 if len(self.x.shape) == 1 else self.x.shape[-1])

    @property
    def num_edge_features(self):
        if self.edge_attr is not None and not self.edge_attr.shape:
            raise ValueError('edge_attr must have an edge axis')
        return 0 if self.edge_attr is None else (1 if len(self.edge_attr.shape) == 1 else self.edge_attr.shape[-1])

    def _role(self, key, n, e):
        if key in self._field_roles:
            return self._field_roles[key]
        if key in _NODE_FIELDS:
            return 'node'
        if key in _EDGE_FIELDS:
            return 'edge'
        if key == 'y' or key in _INDEX_FIELDS or not self[key].shape:
            return 'graph'
        if _custom_hooks(self) or 'index' in key or key == 'face' or 'batch' in key:
            return 'graph'
        row = self[key].shape[0]
        if row == n and row == e:
            raise ValueError(f'ambiguous field {key!r}; specify field_roles')
        return 'node' if row == n else 'edge' if row == e else 'graph'

    def validate(self, raise_on_error=True):
        """Check topology bounds and field lengths without materializing features."""
        try:
            n = self.num_nodes
            e = 0 if self.edge_index is None else len(_edge_parts(self.edge_index, num_nodes=n)[0])
            if any(key not in self._store or key == 'edge_index' for key in self._field_roles):
                raise ValueError('field_roles must name existing non-topology fields')
            for key, value in self.items():
                if key == 'edge_index' or key in ('batch', 'ptr'):
                    continue
                role = self._role(key, n, e)
                count = n if role == 'node' else e if role == 'edge' else None
                if count is not None and (not value.shape or value.shape[0] != count):
                    raise ValueError(f'{role} field {key!r} has the wrong row count')
            return True
        except (TypeError, ValueError):
            if raise_on_error:
                raise
            return False


class Batch(Data):
    """Disjoint graphs with int64 node-to-graph batch IDs and cumulative ptr.

    follow_batch adds <field>_batch and <field>_ptr for each requested retained
    field's concatenation axis. exclude_keys omits fields from the batch and
    reconstructed examples. The three standard homogeneous link-label index
    fields use column concatenation and node offsets. Dense custom hooks support
    independent node spaces, alternate concatenation axes and stacked fields.
    HeteroData inputs dispatch to the dedicated HeteroBatch implementation.
    """
    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None):
        follow = _field_names(follow_batch, 'follow_batch')
        excluded = set(_field_names(exclude_keys, 'exclude_keys'))
        data_list = list(data_list)
        if not data_list:
            raise ValueError('from_data_list requires at least one graph')
        from ._graph_hetero import HeteroData
        if isinstance(data_list[0], HeteroData):
            from ._graph_hetero_batch import HeteroBatch
            return HeteroBatch.from_data_list(data_list, follow_batch=follow, exclude_keys=excluded)
        if any(not isinstance(d, Data) or isinstance(d, Batch) for d in data_list):
            raise TypeError('from_data_list requires Data instances, not nested batches')
        if any(_needs_hook_collation(d, excluded) for d in data_list):
            from ._graph_batch_hooks import collate
            return collate(cls, data_list, follow, excluded)
        for d in data_list:
            d.validate()
        keys = [key for key in data_list[0].keys() if key not in excluded]
        if any(set(d.keys()) - excluded != set(keys) for d in data_list):
            raise ValueError('all graphs must have the same tensor field names')
        if any(key != 'edge_index' and key not in _INDEX_FIELDS and key.endswith('_index') for key in keys):
            raise NotImplementedError('Batch custom index offsets are unsupported')
        follow = tuple(key for key in follow if key in keys)
        generated = [key + suffix for key in follow for suffix in ('_batch', '_ptr')]
        if any(key in keys or key in ('batch', 'ptr') for key in generated) or len(set(generated)) != len(generated):
            raise ValueError('follow_batch generated names conflict with graph fields')
        nodes = [d.num_nodes for d in data_list]
        edges = [d.num_edges for d in data_list]
        ptr = [0]
        for n in nodes:
            ptr.append(_count(ptr[-1] + n, 'total nodes'))
        _capacity((ptr[-1],), 8)
        _capacity((2, sum(edges)), 8)
        roles, slices, shapes = {}, {}, {}
        edgeptr = [0]
        for e in edges:
            edgeptr.append(edgeptr[-1] + e)
        for key in keys:
            if key == 'edge_index':
                continue
            tensors = [d[key] for d in data_list]
            field_roles = [d._role(key, n, e) for d, n, e in zip(data_list, nodes, edges)]
            if len(set(field_roles)) != 1:
                raise ValueError(f'field {key!r} has inconsistent roles; specify field_roles')
            roles[key] = field_roles[0]
            if key in _INDEX_FIELDS:
                offsets = [0]
                for d in data_list:
                    src, _, _ = _label_parts(d, key)
                    offsets.append(_count(offsets[-1] + len(src), 'total labeled edges'))
                _capacity((2, offsets[-1]), 8)
                slices[key], shapes[key] = offsets, [t.shape for t in tensors]
                continue
            normalized = [t.shape if t.shape else [1] for t in tensors]
            if any(t.dtype != tensors[0].dtype or s[1:] != normalized[0][1:] for t, s in zip(tensors, normalized)):
                raise ValueError(f'field {key!r} must have matching dtype and trailing shape')
            offsets = [0]
            for s in normalized:
                offsets.append(offsets[-1] + s[0])
            _capacity([offsets[-1]] + normalized[0][1:], 8 if tensors[0].dtype in ('int64', 'float64') else 4)
            slices[key], shapes[key] = offsets, [t.shape for t in tensors]
        if 'edge_index' not in keys and any(role == 'edge' for role in roles.values()) and any(edges):
            raise NotImplementedError('excluding edge_index also requires excluding nonempty edge fields')
        for key in follow:
            _capacity(((edgeptr if key == 'edge_index' else slices[key])[-1],), 8)
        _capacity((len(data_list) + 1,), 8)
        result = cls(num_nodes=ptr[-1], field_roles=roles)
        for key in keys:
            if key == 'edge_index' or key in _INDEX_FIELDS:
                src, dst = [], []
                for d, off in zip(data_list, ptr):
                    s, t, _ = _edge_parts(d[key], num_nodes=d.num_nodes)
                    src.extend(i + off for i in s)
                    dst.extend(i + off for i in t)
                result._store[key] = _edge_tensor(src, dst)
            else:
                result._store[key] = cat([d[key] if d[key].shape else d[key].unsqueeze(0) for d in data_list], 0)
        result._store['batch'] = Tensor.from_i64([i for i, n in enumerate(nodes) for _ in range(n)], [ptr[-1]])
        result._store['ptr'] = Tensor.from_i64(ptr, [len(ptr)])
        for key in follow:
            offsets = edgeptr if key == 'edge_index' else slices[key]
            result._store[key + '_batch'] = Tensor.from_i64([i for i, (a, b) in enumerate(zip(offsets, offsets[1:])) for _ in range(b - a)], [offsets[-1]])
            result._store[key + '_ptr'] = Tensor.from_i64(offsets, [len(offsets)])
            result._field_roles[key + '_batch'] = result._field_roles[key + '_ptr'] = 'graph'
        result._num_graphs = len(data_list)
        result._nodes, result._edges = nodes, edges
        result._slices, result._shapes = slices, shapes
        result._original_keys = keys
        return result

    @property
    def num_graphs(self):
        return self._num_graphs

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



class DataLoader:
    """Synchronous graph minibatches, with deterministic seed+epoch shuffling.

    Each batch contains either Data or HeteroData examples. It is collated only when yielded;
    no worker processes, asynchronous prefetch or resumable iterator are provided.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, seed=None, follow_batch=None, exclude_keys=None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if type(num_workers) is not int or num_workers != 0:
            raise NotImplementedError('graph DataLoader supports num_workers=0 only')
        if type(shuffle) is not bool or type(drop_last) is not bool:
            raise TypeError('shuffle and drop_last must be bool')
        if seed is not None and type(seed) is not int:
            raise TypeError('seed must be an integer or None')
        if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
            raise TypeError('dataset must support len and integer indexing')
        self.dataset, self.batch_size = dataset, batch_size
        self.follow_batch = _field_names(follow_batch, 'follow_batch')
        self.exclude_keys = _field_names(exclude_keys, 'exclude_keys')
        self.shuffle, self.drop_last = shuffle, drop_last
        self.seed = random.SystemRandom().randrange(2**64) if seed is None else seed
        self.epoch = 0

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        def batches():
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                if len(ids) < self.batch_size and self.drop_last:
                    break
                yield Batch.from_data_list([self.dataset[i] for i in ids], self.follow_batch, self.exclude_keys)
        return batches()
