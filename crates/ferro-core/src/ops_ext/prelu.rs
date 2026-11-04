//! Typed CPU PReLU with branch-wise first-order adjoints and a genuine local JVP.
use crate::{DType, Device, Error, Result, Tensor};

impl Tensor {
    pub fn prelu(&self, weight: &Tensor) -> Result<Tensor> {
        const OP: &str = "prelu";
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: OP, msg: "PReLU does not support inference capture".into() });
        }
        if self.device() != Device::Cpu || weight.device() != Device::Cpu {
            return Err(Error::Unsupported { op: OP, msg: "PReLU currently requires CPU storage".into() });
        }
        if !matches!(self.dtype(), DType::F32 | DType::F64) {
            return Err(Error::Unsupported { op: OP, msg: "PReLU requires float32 or float64".into() });
        }
        if weight.dtype() != self.dtype() {
            return Err(Error::DtypeMismatch { op: OP, expected: self.dtype(), got: weight.dtype() });
        }
        let channels = if self.ndim() < 2 { 1 } else { self.shape()[1] };
        if weight.ndim() > 1 || (weight.numel() != 1 && weight.numel() != channels) {
            return Err(Error::InvalidShape { op: OP, msg: "weight must be scalar or rank one with one value or one value per input channel".into() });
        }
        crate::typed_math::check_shape(OP, self.shape())?;
        crate::typed_math::check_shape(OP, weight.shape())?;
        let mut expanded = vec![1; self.ndim()];
        if self.ndim() >= 2 { expanded[1] = weight.numel(); }
        let weights = weight.view(&expanded)?;
        let output = crate::typed_math::pointwise_binary_real(OP, self, &weights,
            |x, a| if x > 0. { x } else { x * a },
            |x, a| if x > 0. { x } else { x * a })?;
        let x = self.clone();
        let w = weight.clone();
        let dx = crate::forward_ad::records_grad(self);
        let dw = crate::forward_ad::records_grad(weight);
        let inner = if self.ndim() < 2 || self.numel() == 0 { 1 } else { self.shape()[2..].iter().product() };
        Ok(output.record_fn_forward_optional(vec![self.clone(), weight.clone()], crate::autograd::ForwardOp::Prelu, move |g| {
            macro_rules! backward { ($read:ident, $zero:expr, $ctor:ident) => {{
                let values = x.$read();
                let slopes = if values.is_empty() { Vec::new() } else { w.$read() };
                let incoming = g.$read();
                let mut gx = if dx { crate::typed_math::filled(OP, x.shape(), $zero).expect("validated PReLU input gradient allocation") } else { Vec::new() };
                let mut gw = if dw { crate::typed_math::filled(OP, w.shape(), $zero).expect("validated PReLU weight gradient allocation") } else { Vec::new() };
                for (i, (&value, &gradient)) in values.iter().zip(&incoming).enumerate() {
                    let channel = if w.numel() == 1 { 0 } else { (i / inner) % channels };
                    if dx { gx[i] = if value > 0. { gradient } else { gradient * slopes[channel] }; }
                    if dw && !(value > 0.) { gw[channel] += gradient * value; }
                }
                let gx = dx.then(|| {
                    let raw = Tensor::$ctor(gx, x.shape()).expect("validated PReLU input gradient");
                    let stride = crate::typed_math::binary_pointwise_layout(&x, g, x.shape(), x.dtype());
                    if raw.strides() == stride { raw } else { raw.copy_grad_layout(stride) }
                });
                let gw = dw.then(|| Tensor::$ctor(gw, w.shape()).expect("validated PReLU weight gradient"));
                vec![gx, gw]
            }}; }
            if x.dtype() == DType::F64 { backward!(to_vec_f64, 0.0f64, from_vec_f64) }
            else { backward!(to_vec, 0.0f32, from_vec) }
        }))
    }
}

// Torch uses >= here, while the ordinary kernel uses >. In particular,
// the input tangent at either signed zero is passed through unchanged.
pub(crate) fn jvp(input: &Tensor, weight: &Tensor, tangents: &[Option<Tensor>]) -> Result<Tensor> {
    let tangent = |index: usize, primal: &Tensor| -> Result<Tensor> {
        match &tangents[index] {
            Some(value) => Ok(value.clone()),
            // A missing direction is metadata-only even for virtual slopes
            // on an empty input. Preserve the engine's structural-zero marker.
            None => Tensor::structural_zero(&[], primal.dtype(), primal.device())?.expand(primal.shape()),
        }
    };
    let dx = tangent(0, input)?;
    let dw = tangent(1, weight)?;
    let mut shape = vec![1; input.ndim()];
    if input.ndim() >= 2 { shape[1] = weight.numel(); }
    let slope = weight.view(&shape)?;
    let slope_tangent = dw.view(&shape)?;
    // The condition is the first concrete TensorIterator operand in Torch's
    // where rule. Select in its dense physical order, then restore logical axes.
    // Every permutation is an ordinary recorded view; no gradient is patched.
    let stride = crate::typed_math::unary_pointwise_layout(input);
    let mut axes: Vec<_> = (0..input.ndim()).collect();
    if stride != crate::shape::default_strides(input.shape()) {
        axes.sort_by(|&a, &b| stride[b].cmp(&stride[a]).then_with(|| b.cmp(&a)));
    }
    let physical_shape = axes.iter().map(|&axis| input.shape()[axis]).collect::<Vec<_>>();
    let physical_stride = crate::shape::default_strides(&physical_shape);
    if axes.iter().zip(&physical_stride).any(|(&axis, &value)| stride[axis] != value) {
        return Err(Error::Unsupported { op: "prelu JVP", msg: "this singleton stride order is not representable by the supported dense permutation".into() });
    }
    // Arithmetic keeps logical axes so structural-zero reverse contributions
    // have the original input layout policy before the final selection.
    let negative = slope.mul(&dx)?.add(&slope_tangent.mul(input)?)?;
    let dtype = crate::typed_math::result_type(&dx, &negative)?;
    let left = dx.to_dtype_autograd(dtype, false)?;
    let right = negative.to_dtype_autograd(dtype, false)?;
    let input = input.permute(&axes)?;
    let left = left.permute(&axes)?;
    let right = right.permute(&axes)?;
    let zero = if input.dtype() == DType::F64 { Tensor::from_vec_f64(vec![0.0], &[])? } else { Tensor::scalar(0.0) };
    let physical = Tensor::where_cond(&input.ge_tensor(&zero)?, &left, &right)?;
    let mut inverse = vec![0; axes.len()];
    for (physical_axis, &logical_axis) in axes.iter().enumerate() { inverse[logical_axis] = physical_axis; }
    physical.permute(&inverse)
}
