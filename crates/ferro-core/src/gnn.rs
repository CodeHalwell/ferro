//! Graph neural networks in the PyTorch Geometric style: a validated
//! `EdgeIndex` topology, `scatter` aggregation over destination nodes, the
//! `MessagePassing` gather -> message -> aggregate -> update template, and
//! `GcnConv`. Edges flow source -> target (PyG's default `flow`), so node i
//! aggregates messages from every edge whose dst is i. Aggregation is along
//! axis zero over CPU f32 (sum also runs resident via `segment`); there is no
//! capture/replay support, matching `segment`.

use crate::error::{Error, Result};
use crate::nn::Init;
use crate::params::Param;
use crate::rng::Rng;
use crate::segment;
use crate::tensor::Tensor;

fn invalid(op: &'static str, msg: String) -> Error { Error::InvalidShape { op, msg } }

/// Directed edges `src[e] -> dst[e]` over `num_nodes` nodes. Edge order is
/// retained and duplicate edges stay independent (each sends its own message).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeIndex {
    num_nodes: usize,
    src: Vec<usize>,
    dst: Vec<usize>,
}

impl EdgeIndex {
    pub fn new(num_nodes: usize, src: Vec<usize>, dst: Vec<usize>) -> Result<Self> {
        if src.len() != dst.len() {
            return Err(invalid("edge_index", format!("{} sources but {} targets", src.len(), dst.len())));
        }
        if let Some(&bad) = src.iter().chain(&dst).find(|&&n| n >= num_nodes) {
            return Err(invalid("edge_index", format!("node {bad} out of range for {num_nodes} nodes")));
        }
        Ok(Self { num_nodes, src, dst })
    }

    /// From a PyG-layout `[2, E]` I64 tensor (row 0 sources, row 1 targets).
    /// `num_nodes` is explicit so trailing isolated nodes are not lost.
    pub fn from_tensor(edge_index: &Tensor, num_nodes: usize) -> Result<Self> {
        if edge_index.ndim() != 2 || edge_index.shape()[0] != 2 {
            return Err(invalid("edge_index", format!("expected [2, E], got {:?}", edge_index.shape())));
        }
        if edge_index.dtype() != crate::DType::I64 {
            return Err(Error::DtypeMismatch { op: "edge_index", expected: crate::DType::I64, got: edge_index.dtype() });
        }
        let ids = edge_index.to_vec_i64().into_iter().map(|i| usize::try_from(i)
            .map_err(|_| invalid("edge_index", format!("negative node id {i}")))).collect::<Result<Vec<_>>>()?;
        let e = edge_index.shape()[1];
        Self::new(num_nodes, ids[..e].to_vec(), ids[e..].to_vec())
    }

    /// The `[2, E]` I64 form, for saving topology alongside model state.
    pub fn to_tensor(&self) -> Tensor {
        let ids = self.src.iter().chain(&self.dst).map(|&n| n as i64).collect();
        Tensor::from_vec_i64(ids, &[2, self.src.len()]).expect("edge index shape is valid")
    }

    pub fn num_nodes(&self) -> usize { self.num_nodes }
    pub fn num_edges(&self) -> usize { self.src.len() }
    pub fn src(&self) -> &[usize] { &self.src }
    pub fn dst(&self) -> &[usize] { &self.dst }

    /// PyG `add_remaining_self_loops`: existing self loops are dropped and
    /// exactly one loop per node is appended, so repeated loops collapse.
    pub fn add_remaining_self_loops(&self) -> Self {
        let (mut src, mut dst): (Vec<_>, Vec<_>) = self.src.iter().zip(&self.dst).filter(|(s, d)| s != d).unzip();
        src.extend(0..self.num_nodes);
        dst.extend(0..self.num_nodes);
        Self { num_nodes: self.num_nodes, src, dst }
    }

    /// Number of incoming edges per node, duplicates counted.
    pub fn in_degree(&self) -> Vec<usize> {
        let mut deg = vec![0; self.num_nodes];
        for &d in &self.dst { deg[d] += 1; }
        deg
    }

    /// Symmetric GCN edge weights `deg(src)^-1/2 * deg(dst)^-1/2` as `[E]`,
    /// with degree taken over incoming edges (PyG `gcn_norm`, unit weights).
    /// A zero-degree endpoint contributes weight zero rather than infinity.
    pub fn gcn_norm(&self) -> Tensor {
        let inv: Vec<f32> = self.in_degree().into_iter().map(|d| if d == 0 { 0. } else { (d as f32).powf(-0.5) }).collect();
        let w = self.src.iter().zip(&self.dst).map(|(&s, &d)| inv[s] * inv[d]).collect();
        Tensor::from_vec(w, &[self.src.len()]).expect("norm shape is valid")
    }

    /// Disjoint union for mini-batching: node ids of graph k are offset by the
    /// node counts of graphs 0..k. Returns the union and the node -> graph
    /// vector, which `scatter` takes directly for global pooling.
    pub fn batch(graphs: &[EdgeIndex]) -> (EdgeIndex, Vec<usize>) {
        let (mut src, mut dst, mut owner) = (Vec::new(), Vec::new(), Vec::new());
        for (k, g) in graphs.iter().enumerate() {
            let off = owner.len();
            src.extend(g.src.iter().map(|&n| n + off));
            dst.extend(g.dst.iter().map(|&n| n + off));
            owner.extend(std::iter::repeat(k).take(g.num_nodes));
        }
        (EdgeIndex { num_nodes: owner.len(), src, dst }, owner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggr { Sum, Mean, Max, Min }

/// Reduce rows of `src` ([n, ...]) into `dim_size` rows by `index` ([n]),
/// like `torch_geometric.utils.scatter(src, index, 0, dim_size, reduce)`.
/// Rows nobody scatters into are zero for every reduction (PyG semantics),
/// so isolated nodes never see -inf. Max/min ties share the adjoint equally.
pub fn scatter(src: &Tensor, index: &[usize], dim_size: usize, reduce: Aggr) -> Result<Tensor> {
    match reduce {
        Aggr::Sum => segment::sum(src, index, dim_size),
        Aggr::Mean => segment::mean(src, index, dim_size),
        Aggr::Max | Aggr::Min => {
            let flip = reduce == Aggr::Min;
            let y = segment::max(&if flip { src.neg() } else { src.clone() }, index, dim_size)?;
            let y = if flip { y.neg() } else { y };
            let mut hit = vec![0.; dim_size];
            for &i in index { hit[i] = 1.; }
            let mut mask_shape = vec![1; y.ndim()];
            mask_shape[0] = dim_size;
            Tensor::where_cond(&Tensor::from_vec(hit, &mask_shape)?, &y, &Tensor::zeros(&[1]))
        }
    }
}

/// The PyG message-passing template. `propagate` builds one message per edge
/// with `message`, reduces them onto destination nodes with `aggr`, then
/// post-processes with `update`. The default message is the source row x_j;
/// layers needing the target row x_i gather `edges.dst()` themselves, so the
/// common case pays for one gather, not two.
pub trait MessagePassing {
    fn aggr(&self) -> Aggr { Aggr::Sum }

    fn message(&self, x: &Tensor, edges: &EdgeIndex) -> Result<Tensor> { x.index_select(0, edges.src()) }

    fn update(&self, aggregated: Tensor, _x: &Tensor) -> Result<Tensor> { Ok(aggregated) }

    fn propagate(&self, x: &Tensor, edges: &EdgeIndex) -> Result<Tensor> {
        if x.ndim() == 0 || x.shape()[0] != edges.num_nodes() {
            return Err(invalid("propagate", format!("expected [{}, ...] node features, got {:?}", edges.num_nodes(), x.shape())));
        }
        let messages = self.message(x, edges)?;
        self.update(scatter(&messages, edges.dst(), edges.num_nodes(), self.aggr())?, x)
    }
}

/// A layer whose forward needs the graph as well as node features.
pub trait GraphModule {
    fn forward(&self, x: &Tensor, edges: &EdgeIndex) -> Result<Tensor>;

    fn named_parameters(&self) -> Vec<(String, Param)>;

    fn parameters(&self) -> Vec<Param> { self.named_parameters().into_iter().map(|(_, p)| p).collect() }
}

/// Kipf & Welling graph convolution, `D^-1/2 (A + I) D^-1/2 X W + b`, matching
/// PyG `GCNConv` defaults (remaining self loops, symmetric norm, bias).
/// Weights are Xavier-initialized `[in, out]`, so `X @ W` needs no transpose.
pub struct GcnConv {
    weight: Param,
    bias: Param,
    self_loops: bool,
}

impl GcnConv {
    pub fn new(in_features: usize, out_features: usize, rng: &Rng) -> GcnConv {
        let w = Init::Xavier.fill(rng, &[in_features, out_features], in_features, out_features);
        GcnConv { weight: Param::new(w), bias: Param::new(Tensor::zeros(&[out_features])), self_loops: true }
    }

    pub fn with_self_loops(mut self, on: bool) -> GcnConv {
        self.self_loops = on;
        self
    }
}

impl MessagePassing for GcnConv {
    fn message(&self, x: &Tensor, edges: &EdgeIndex) -> Result<Tensor> {
        x.index_select(0, edges.src())?.mul(&edges.gcn_norm().reshape(&[edges.num_edges(), 1])?)
    }
}

impl GraphModule for GcnConv {
    fn forward(&self, x: &Tensor, edges: &EdgeIndex) -> Result<Tensor> {
        let w = self.weight.tensor();
        if x.ndim() != 2 || x.shape()[1] != w.shape()[0] {
            return Err(invalid("gcn_conv", format!("expected [nodes, {}] features, got {:?}", w.shape()[0], x.shape())));
        }
        let h = x.matmul(&w)?;
        let out = if self.self_loops { self.propagate(&h, &edges.add_remaining_self_loops())? } else { self.propagate(&h, edges)? };
        out.add(&self.bias.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("weight".into(), self.weight.clone()), ("bias".into(), self.bias.clone())]
    }
}
