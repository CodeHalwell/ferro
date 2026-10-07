//! Loss and functional bindings over core's existing implementations. Core
//! losses are mean-reduced; Python applies other reductions by rescaling.
use super::*;

#[pyfunction]
fn cross_entropy(logits: &PyTensor, targets: &PyTensor) -> PyResult<PyTensor> {
    use ferro_core::DType;
    let x = &logits.inner;
    let y = &targets.inner;
    if x.device() != Device::Cpu || y.device() != Device::Cpu {
        return Err(PyValueError::new_err("cross_entropy supports CPU tensors only"));
    }
    if x.dtype() != DType::F32 || !matches!(y.dtype(), DType::I64 | DType::F32) {
        return Err(PyValueError::new_err("cross_entropy requires f32 logits and i64 class or f32 probability targets"));
    }
    if x.shape().len() != 2 || x.shape()[0] == 0 || x.shape()[1] == 0 {
        return Err(PyValueError::new_err("cross_entropy requires nonempty [batch, classes] logits"));
    }
    if y.dtype() == DType::F32 {
        return ok(ferro_core::nn::cross_entropy(x, y));
    }
    if y.shape() != [x.shape()[0]] {
        return Err(PyValueError::new_err("cross_entropy requires [batch] class targets"));
    }
    ok(ferro_core::nn::cross_entropy_indices(x, y))
}

#[pyfunction]
fn l1_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> { ok(input.inner.l1_loss(&target.inner)) }
#[pyfunction]
fn bce_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> { ok(input.inner.bce_loss(&target.inner)) }
#[pyfunction]
fn bce_with_logits_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> { ok(input.inner.bce_with_logits_loss(&target.inner)) }
#[pyfunction]
fn huber_loss(input: &PyTensor, target: &PyTensor, delta: f32) -> PyResult<PyTensor> { ok(input.inner.huber_loss(&target.inner, delta)) }
#[pyfunction]
fn smooth_l1_loss(input: &PyTensor, target: &PyTensor, beta: f32) -> PyResult<PyTensor> { ok(input.inner.smooth_l1_loss(&target.inner, beta)) }
#[pyfunction]
fn kl_div_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> { ok(input.inner.kl_div_loss(&target.inner)) }
#[pyfunction]
fn poisson_nll_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> { ok(input.inner.poisson_nll_loss(&target.inner)) }
#[pyfunction]
fn soft_margin_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> { ok(input.inner.soft_margin_loss(&target.inner)) }
#[pyfunction]
fn hinge_embedding_loss(input: &PyTensor, target: &PyTensor, margin: f32) -> PyResult<PyTensor> { ok(input.inner.hinge_embedding_loss(&target.inner, margin)) }
#[pyfunction]
fn margin_ranking_loss(input1: &PyTensor, input2: &PyTensor, target: &PyTensor, margin: f32) -> PyResult<PyTensor> { ok(input1.inner.margin_ranking_loss(&input2.inner, &target.inner, margin)) }
#[pyfunction]
fn cosine_embedding_loss(input1: &PyTensor, input2: &PyTensor, target: &PyTensor, margin: f32, eps: f32) -> PyResult<PyTensor> { ok(input1.inner.cosine_embedding_loss(&input2.inner, &target.inner, margin, eps)) }
#[pyfunction]
fn triplet_margin_loss(anchor: &PyTensor, positive: &PyTensor, negative: &PyTensor, margin: f32, p: f32, eps: f32) -> PyResult<PyTensor> { ok(anchor.inner.triplet_margin_loss(&positive.inner, &negative.inner, margin, p, eps)) }

/// Row lookup of 1-D I64 `ids` into a [num_embeddings, dim] weight; duplicate ids accumulate grad.
#[pyfunction]
fn embedding(weight: &PyTensor, ids: &PyTensor) -> PyResult<PyTensor> { ok(ferro_core::ops_ext::embedding(&weight.inner, &ids.inner)) }
#[pyfunction]
fn one_hot(ids: &PyTensor, num_classes: usize) -> PyResult<PyTensor> { ok(ferro_core::nn::one_hot(&ids.inner, num_classes)) }
/// [batch, seq, head_dim] attention; heads must already be folded into batch.
#[pyfunction]
#[pyo3(signature = (q, k, v, is_causal=false))]
fn scaled_dot_product_attention(q: &PyTensor, k: &PyTensor, v: &PyTensor, is_causal: bool) -> PyResult<PyTensor> {
    ok(ferro_core::nn::scaled_dot_product_attention(&q.inner, &k.inner, &v.inner, is_causal))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(bce_loss, m)?)?;
    m.add_function(wrap_pyfunction!(bce_with_logits_loss, m)?)?;
    m.add_function(wrap_pyfunction!(huber_loss, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(kl_div_loss, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(soft_margin_loss, m)?)?;
    m.add_function(wrap_pyfunction!(hinge_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(margin_ranking_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_margin_loss, m)?)?;
    m.add_function(wrap_pyfunction!(embedding, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot, m)?)?;
    m.add_function(wrap_pyfunction!(scaled_dot_product_attention, m)?)?;
    Ok(())
}
