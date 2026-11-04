//! Existing product and PReLU rules with an explicit leading direction axis.
use crate::{DType, Device, Error, Result, Tensor};

pub(super) fn product(input: &Tensor, output: &Tensor, direction: &Tensor, dim: Option<usize>, keepdim: bool, batch: usize) -> Result<Tensor> {
    let one = if output.dtype() == DType::F64 { Tensor::from_vec_f64(vec![1.], &[])? } else { Tensor::scalar(1.) };
    let consumed = input.to_dtype_autograd(output.dtype(), false)?;
    let coefficient = crate::ops_ext::prod::graph_vjp(&consumed, Some(output), &one.expand(output.shape())?, dim, keepdim, crate::forward_ad::is_functional())?;
    let coefficient = if input.ndim() == 0 { coefficient.expand(direction.shape())? } else { coefficient };
    let weighted = coefficient.mul(direction)?;
    if input.ndim() == 0 { weighted.unsqueeze(1)?.sum_dim(1, false) }
    else if let Some(dim) = dim { weighted.sum_dim(dim + 1, keepdim) }
    else { weighted.reshape(&[batch, input.numel()])?.sum_dim(1, false) }
}

pub(super) fn prelu(inputs: &[Tensor], tangents: &[Option<Tensor>], batch: usize) -> Result<Tensor> {
    let (input, weight) = (&inputs[0], &inputs[1]);
    let stride = crate::typed_math::unary_pointwise_layout(input);
    let mut axes: Vec<_> = (0..input.ndim()).collect();
    if stride != crate::shape::default_strides(input.shape()) {
        axes.sort_by(|&a, &b| stride[b].cmp(&stride[a]).then_with(|| b.cmp(&a)));
    }
    let physical_shape = axes.iter().map(|&axis| input.shape()[axis]).collect::<Vec<_>>();
    if axes.iter().zip(crate::shape::default_strides(&physical_shape)).any(|(&axis, expected)| stride[axis] != expected) {
        return Err(Error::Unsupported { op: "physical prelu JVP", msg: "this singleton stride order is not representable by a dense permutation".into() });
    }
    let mut logical_weight = vec![1; input.ndim()];
    if input.ndim() >= 2 { logical_weight[1] = weight.numel(); }
    let mut physical_axes = vec![0]; physical_axes.extend(axes.iter().map(|axis| axis + 1));
    let direction = |index: usize| -> Result<Tensor> {
        match &tangents[index] {
            Some(value) => {
                let value = if index == 1 {
                    let mut shape = vec![batch]; shape.extend_from_slice(&logical_weight);
                    value.view(&shape)?
                } else { value.clone() };
                Ok(value)
            }
            None => {
                let shape = if index == 1 { logical_weight.as_slice() } else { input.shape() };
                Tensor::structural_zero(&[], inputs[index].dtype(), Device::Cpu)?.expand(shape)
            }
        }
    };
    let slope = weight.view(&logical_weight)?;
    let dx = direction(0)?;
    let dw = direction(1)?;
    let negative = slope.mul(&dx)?.add(&dw.mul(input)?)?;
    let dtype = crate::typed_math::result_type(&dx, &negative)?;
    let positive = input.ge_tensor(&Tensor::scalar(0.))?.permute(&axes)?;
    let left = dx.to_dtype_autograd(dtype, false)?.permute(if tangents[0].is_some() {&physical_axes}else{&axes})?;
    let right = negative.to_dtype_autograd(dtype, false)?.permute(&physical_axes)?;
    let output = Tensor::where_cond(&positive, &left, &right)?;
    let mut inverse = vec![0; physical_axes.len()];
    for (physical, &logical) in physical_axes.iter().enumerate() { inverse[logical] = physical; }
    output.permute(&inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_ad::DualLevel;

    #[test]
    fn scalar_and_width_one_product_materialize_real_zero_directions() {
        for shape in [&[][..], &[1][..]] { for output_dtype in [DType::F32,DType::F64] { for seed_dtype in [DType::F32,DType::F64] { for value in [2.,f32::INFINITY] {
            let x=Tensor::full(shape,value).requires_grad_(true).unwrap();
            let mut direction_shape=vec![2];direction_shape.extend_from_slice(shape);
            let direction=Tensor::structural_zero(&direction_shape,seed_dtype,Device::Cpu).unwrap();
            let level=DualLevel::functional_batched(&[&x],&[&direction],2).unwrap();
            let y=x.product(None,false,Some(output_dtype)).unwrap();
            let tangent=level.batched_tangent(&y).unwrap().unwrap();drop(level);
            assert_eq!(tangent.shape(),[2]);assert_eq!(tangent.strides(),[1]);
            assert_eq!(tangent.dtype(),if output_dtype==DType::F64||seed_dtype==DType::F64 {DType::F64}else{DType::F32});
            assert!(!tangent.is_structural_zero()&&!tangent.requires_grad());
            assert_ne!(tangent._storage_ptr(),direction._storage_ptr());
            assert!(tangent.to_vec_f64().iter().all(|value|*value==0.&&!value.is_sign_negative()));
        } } } }
    }
}
