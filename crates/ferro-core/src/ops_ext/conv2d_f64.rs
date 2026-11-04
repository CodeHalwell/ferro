//! Direct CPU double convolution and its first-order adjoint. F32 dispatch stays separate.
use super::window::{invalid, Window2d};
use crate::{typed_math, DType, Error, Result, Tensor};

#[derive(Clone, Copy)]
struct Geometry {
    input: [usize; 4],
    weight: [usize; 4],
    output: [usize; 4],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl Geometry {
    fn visit(self, mut element: impl FnMut(Option<usize>, usize, usize)) {
        let [batch, channels, height, width] = self.input;
        let [outputs, inputs, kh, kw] = self.weight;
        let [_, _, oh, ow] = self.output;
        for n in 0..batch {
            for oc in 0..outputs {
                let group = oc / (outputs / self.groups);
                for y in 0..oh {
                    for x in 0..ow {
                        let out = ((n * outputs + oc) * oh + y) * ow + x;
                        for ci in 0..inputs {
                            for ky in 0..kh {
                                let iy = (y * self.stride[0] + ky * self.dilation[0])
                                    .checked_sub(self.padding[0]).filter(|&v| v < height);
                                for kx in 0..kw {
                                    let ix = (x * self.stride[1] + kx * self.dilation[1])
                                        .checked_sub(self.padding[1]).filter(|&v| v < width);
                                    let source = iy.zip(ix).map(|(iy, ix)|
                                        ((n * channels + group * inputs + ci) * height + iy) * width + ix);
                                    let weight = ((oc * inputs + ci) * kh + ky) * kw + kx;
                                    element(source, weight, out);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn convolution(input: &Tensor, weight: &Tensor, stride: [usize; 2], padding: [usize; 2],
                          dilation: [usize; 2], groups: usize) -> Result<Tensor> {
    const OP: &str = "conv2d";
    typed_math::check(OP, &[input, weight], DType::F64)?;
    if crate::forward_ad::is_active() {
        return Err(Error::Unsupported { op: OP, msg: "CPU float64 convolution supports ordinary reverse gradients; forward AD is unsupported".into() });
    }
    if input.ndim() != 4 || weight.ndim() != 4 {
        return Err(invalid(OP, "expected NCHW input and OIHW weight"));
    }
    let xs: [usize; 4] = input.shape().try_into().unwrap();
    let ws: [usize; 4] = weight.shape().try_into().unwrap();
    if groups == 0 || xs[1] == 0 || ws[0] == 0 || xs[1] % groups != 0 || ws[0] % groups != 0
        || ws[1] != xs[1] / groups {
        return Err(invalid(OP, "positive groups must divide compatible positive input/output channels"));
    }
    let window = Window2d::new([xs[2], xs[3]], [ws[2], ws[3]], stride, padding, dilation)?;
    let [oh, ow] = window.output_size();
    let shape = [xs[0], ws[0], oh, ow];
    typed_math::check_shape(OP, &shape)?;
    typed_math::check_shape(OP, &xs[1..])?;
    typed_math::check_shape(OP, &shape[1..])?;
    let geometry = Geometry { input: xs, weight: ws, output: shape, stride, padding, dilation, groups };
    // No image or weight data is consumed for an empty batch.
    let x = if xs[0] == 0 { Vec::new() } else { input.to_vec_f64() };
    let w = if xs[0] == 0 { Vec::new() } else { weight.to_vec_f64() };
    let mut values = typed_math::filled(OP, &shape, 0f64)?;
    geometry.visit(|source, wi, oi| {
        // Keep the genuine padded zero product, including IEEE 0*Inf behavior.
        values[oi] += source.map_or(0., |xi| x[xi]) * w[wi];
    });
    Ok(Tensor::from_vec_f64(values, &shape)?.record_fn(vec![input.clone(), weight.clone()], move |seed| {
        let g = seed.to_vec_f64();
        let mut dx = typed_math::filled(OP, &xs, 0f64).expect("validated input gradient allocation");
        let mut dw = typed_math::filled(OP, &ws, 0f64).expect("validated weight gradient allocation");
        geometry.visit(|source, wi, oi| {
            if let Some(xi) = source { dx[xi] += g[oi] * w[wi]; }
            dw[wi] += g[oi] * source.map_or(0., |xi| x[xi]);
        });
        vec![Tensor::from_vec_f64(dx, &xs).unwrap(), Tensor::from_vec_f64(dw, &ws).unwrap()]
    }))
}
