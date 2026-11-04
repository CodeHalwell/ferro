"""First-order graph, segment and sparse algebra; capture/replay is unsupported.

EdgeIndex and GcnConv expose the core graph layer on CPU f32. Edges flow from
source to destination; duplicate edges send independent messages. GcnConv's
weight is [in_features, out_features], matching the Rust layer. Construct it as
``GcnConv(in_features=3, out_features=4)`` and call ``layer(x, edges)``.
``scatter`` supports CPU sum/mean/max/min with zero-filled missing rows; extremum
ties share gradients. Use the segment APIs below for supported CUDA reductions.
Save ``edges.to_tensor()`` and ``edges.num_nodes`` separately from layer state;
topology is an explicit input and is not included in a training checkpoint.

``edge_attention(q, k, v, edges, edge_bias=None)`` performs CPU f32 single-head
attention over the supplied edges, without a dense node-by-node matrix. Queries
and keys have shape [N, D] with D > 0, values [N, Dv], and additive bias [E]. Each
destination query attends source keys/values using dot products scaled by sqrt(D).
Softmax normalizes incoming edges; duplicates remain independent entries and
isolated destinations return zeros. Logits must be finite. Empty graphs and
zero-width values are supported. Gradients reach q, k, v and edge_bias through
the existing first-order autograd engine; higher derivatives and capture reject.

Sum/softmax support CPU and resident CUDA; mean/max and sparse algebra are CPU-only.
Initialize CUDA with ``ferro.cuda_init()`` before preparing a CUDA topology::

    plan = PreparedSegments([2, 0, 2], num_segments=4, device="cuda:0")
    y = plan.sum(x)       # x has shape [3, ...] on cuda:0
    z = plan.softmax(x)   # reuse with fresh inputs, without repeating IDs

IDs are copied as nonnegative i64 integers and validated at construction. CUDA
uploads topology once; the plan and recorded backward retain its original backend
and allocation owners. Inputs must match that device (foreign contexts reject,
never silently copy to CPU). Free functions prepare topology on every invocation.
CUDA softmax rejects nonfinite logits via one synchronizing four-byte status read
per nonempty forward, not a feature download. For resident backward use a contiguous
seed on the output device: the general backward API can copy strided seeds via CPU.
"""
from ._native import COO, CSR, PreparedSegments
from ._native import segment_sum, segment_mean, segment_max, segment_softmax
from ._native import EdgeIndex, scatter, edge_attention, _gcn_conv, Tensor
from .nn.module import Module, Parameter


class GcnConv(Module):
    """CPU f32 GCN with Xavier weights, bias and optional remaining self loops.

Configuration is immutable and participates in checkpoint compatibility. Forward
uses the existing Rust tensor operations and first-order autograd. This layer
does not provide resident CUDA execution, capture/replay or higher derivatives.
"""
    in_features: int
    out_features: int
    self_loops: bool = True

    def build(self):
        if any(type(n) is not int or n <= 0 for n in (self.in_features, self.out_features)):
            raise ValueError('GcnConv dimensions must be positive integers')
        if type(self.self_loops) is not bool:
            raise TypeError('self_loops must be a bool')
        bound = (6.0 / (self.in_features + self.out_features)) ** 0.5
        self.weight = Parameter((Tensor.rand([self.in_features, self.out_features]) * 2 - 1) * bound)
        self.bias = Parameter(Tensor.zeros([self.out_features]))

    def forward(self, x, edges):
        return _gcn_conv(x, self.weight.tensor(), self.bias.tensor(), edges, self.self_loops)


from ._graph_utils import (degree, softmax, global_add_pool, global_mean_pool, global_max_pool,
                           add_self_loops, remove_self_loops, coalesce, to_undirected, subgraph)
from ._graph_data import Data, Batch, DataLoader
from ._graph_conv import GCNConv, SAGEConv, GINConv
from ._graph_attention import GATConv, GATv2Conv
from ._graph_extra_conv import GINEConv, APPNP, ChebConv, TransformerConv
from ._graph_hetero import HeteroData, HeteroConv, RGCNConv
from ._graph_sampling import NeighborLoader, LinkNeighborLoader
from ._graph_link import negative_sampling, batched_negative_sampling, RandomLinkSplit
from ._graph_advanced import PNAConv, HGTConv, GCN2Conv
from ._graph_spectral import SGConv, TAGConv, ARMAConv
from ._graph_norm import GraphNorm, PairNorm, MeanSubtractionNorm, GraphSizeNorm, LayerNorm
from ._graph_conditioned import NNConv, ECConv, EdgeConv, GatedGraphConv
from ._graph_residual import ResGatedGraphConv, AGNNConv, LEConv
from ._graph_point import PointNetConv, PointTransformerConv
from ._graph_dense import DenseGCNConv, DenseSAGEConv, DenseGINConv, DenseGraphConv
from ._graph_modulated import FiLMConv, CGConv, FAConv
from ._graph_mixture import MFConv, FeaStConv, SSGConv
from ._graph_kernels import GMMConv, PDNConv
from ._graph_composition import JumpingKnowledge, GeneralConv
from ._graph_aggregation import (SumAggregation, MeanAggregation, MaxAggregation, MinAggregation,
                                 VarAggregation, StdAggregation, SoftmaxAggregation,
                                 PowerMeanAggregation, MultiAggregation)
from ._graph_deep import MessageNorm
from ._graph_hyper import HypergraphConv
from ._graph_gen import GENConv
from ._graph_signed import SignedConv
from ._graph_simple import LGConv, SimpleConv
from ._graph_efficient import EGConv
from ._graph_recurrent_aggr import LSTMAggregation, GRUAggregation
from ._graph_sort_aggr import SortAggregation
from ._graph_set_transformer import SetTransformerAggregation
from ._graph_degree_aggr import DegreeScalerAggregation
from ._graph_meta import MetaLayer
from ._graph_stateful_norm import BatchNorm, DiffGroupNorm
from ._graph_deep_sets import DeepSetsAggregation
from ._graph_multiset import GraphMultisetTransformer
from ._graph_dense_pool import dense_mincut_pool, DMoNPooling
from ._graph_gps import GPSConv
from ._graph_quantile import QuantileAggregation, MedianAggregation
from ._graph_hetero_linear import HeteroLinear, HeteroDictLinear
from ._graph_antisymmetric import AntiSymmetricConv
from ._graph_variance_aggr import VariancePreservingAggregation
from ._graph_mixhop import MixHopConv
from ._graph_han import HANConv
from ._graph_dense_gat import DenseGATConv
from ._graph_aggr_base import Aggregation
from ._graph_hetero_batch import HeteroBatch
from ._graph_saint import GraphSAINTNodeSampler, GraphSAINTRandomWalkSampler, GraphSAINTEdgeSampler
from ._graph_advanced import (GraphConv, TopKPooling, SAGPooling, AttentionalAggregation,
                            Set2Set, dense_diff_pool)
from ._graph_dataset import Dataset, InMemoryDataset
from ._graph_cluster import ClusterData, ClusterLoader
from ._message_passing import MessagePassing
from ._graph_transforms import (Compose, NormalizeFeatures, AddSelfLoops, RemoveSelfLoops,
                               ToUndirected, RandomNodeSplit, dropout_node, dropout_edge,
                               k_hop_subgraph, bipartite_subgraph)

from ._graph_typed_norm import InstanceNorm, HeteroBatchNorm, HeteroLayerNorm

from ._graph_rgat import RGATConv

from ._graph_mlp import MLP
