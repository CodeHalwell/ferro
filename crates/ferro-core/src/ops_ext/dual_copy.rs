//! Owning seed copies required by the primal's forward-gradient metadata.
use crate::{Tensor,DType,Device,Error,Result};
use crate::autograd::ForwardOp;

pub(crate) fn normalize(primal: &Tensor, tangent: &Tensor) -> Result<Tensor> {
    let same = primal.storage_len() == tangent.storage_len()
        && (primal.numel() == 0 || (primal.storage_offset() == tangent.storage_offset()
            && primal.shape().iter().zip(primal.strides().iter().zip(tangent.strides()))
                .all(|(&size,(&left,&right))| size <= 1 || left == right)));
    if same { return Ok(tangent.clone()); }
    tangent.copy_dual_layout(primal.strides(),primal.storage_offset(),primal.storage_len())
}

impl Tensor {
    pub(crate) fn copy_dual_layout(&self, strides: &[usize], offset: usize, storage_len: usize) -> Result<Tensor> {
        const OP: &str = "forward AD seed copy";
        if crate::capture::is_recording() { return Err(Error::Unsupported { op:OP,msg:"seed layout copying is unavailable during inference capture".into() }); }
        if self.device()!=Device::Cpu || !matches!(self.dtype(),DType::F32|DType::F64) {
            return Err(Error::Unsupported { op:OP,msg:"seed layout copying requires CPU floating storage".into() });
        }
        if strides.len()!=self.ndim() { return Err(Error::InvalidShape {op:OP,msg:"stride rank differs from the seed shape".into()}); }
        let count=crate::shape::checked_numel(OP,self.shape())?;
        if count>isize::MAX as usize/if self.dtype()==DType::F64 {8}else{4} {
            return Err(Error::InvalidShape {op:OP,msg:"logical seed materialization exceeds allocation capacity".into()});
        }
        if self.numel()!=0 {
            if self.shape().iter().zip(strides).any(|(&size,&stride)|size>1&&stride==0) {
                return Err(Error::Unsupported {op:OP,msg:"seed destination has an overlapping broadcast dimension".into()});
            }
            let mut end=offset;
            for (&size,&stride) in self.shape().iter().zip(strides) {
                end=end.checked_add((size-1).checked_mul(stride).ok_or_else(||Error::InvalidShape {op:OP,msg:"seed layout offset overflows".into()})?)
                    .ok_or_else(||Error::InvalidShape {op:OP,msg:"seed layout offset overflows".into()})?;
            }
            if end>=storage_len { return Err(Error::InvalidShape {op:OP,msg:"seed layout exceeds destination storage".into()}); }
        }
        let mut values=crate::typed_math::filled(OP,&[storage_len],0.0f32)?;
        // The output participates first in TensorIterator's axis ordering.
        // This matters for the nonzero-stride overlaps that Torch permits.
        let mut axes=(0..self.ndim()).rev().collect::<Vec<_>>();
        for n in 1..axes.len() {
            let mut current=n;
            for previous in (0..n).rev() {
                let (left,right)=(axes[previous],axes[current]);let mut comparison=0;
                for layout in [strides,self.strides()] {
                    let (ls,rs)=(layout[left],layout[right]);if ls==0||rs==0 {continue;}
                    if ls<rs {comparison=-1;break;}
                    if ls>rs||self.shape()[left]>self.shape()[right] {comparison=1;break;}
                }
                if comparison<0 {break;}
                if comparison>0 {axes.swap(previous,current);current=previous;}
            }
        }
        let data=self.to_vec();let logical=crate::shape::default_strides(self.shape());
        for index in 0..self.numel() {
            let mut position=index;let mut source=0;let mut destination=offset;
            for &axis in &axes {
                let coordinate=position%self.shape()[axis];position/=self.shape()[axis];
                source+=coordinate*logical[axis];destination+=coordinate*strides[axis];
            }
            values[destination]=data[source];
        }
        let storage=Tensor::from_vec(values,&[storage_len])?;
        let output=Tensor::from_parts(storage.0.storage.clone(),self.shape().to_vec(),strides.to_vec(),offset,Device::Cpu,false,None);
        let dtype=self.dtype();
        Ok(output.record_fn_forward(vec![self.clone()],ForwardOp::CopyDualLayout {strides:strides.to_vec(),offset,storage_len},move|g| {
            vec![g.to_dtype_autograd(dtype,false).expect("validated seed copy gradient dtype")]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{with_grad_enabled,is_grad_enabled};
    use crate::forward_ad::{DualLevel,is_active};
    struct Golden { xs:&'static[usize],xo:usize,xn:usize,vs:&'static[usize],vo:usize,vn:usize,wide:bool,functional:bool,error:bool,alias:bool,out_wide:bool,os:&'static[usize],oo:usize,on:usize,leaf:bool,values:&'static[f64] }
    include!("../../tests/fixtures/dual_seed_goldens.rs");
    fn value(shape:&[usize],strides:&[usize],offset:usize,len:usize,wide:bool,live:bool)->Tensor {
        let data=if wide {Tensor::from_vec_f64((0..len).map(|i|i as f64*0.123456789123+0.7).collect(),&[len]).unwrap()}
            else {Tensor::from_vec((0..len).map(|i|i as f32*0.123456789123f32+0.7f32).collect(),&[len]).unwrap()};
        Tensor::from_parts(data.0.storage.clone(),shape.to_vec(),strides.to_vec(),offset,Device::Cpu,live,None)
    }
    #[test]
    fn actual_torch_four_hundred_seed_metadata_copy_controls() {
        for (i,c) in GOLDENS.iter().enumerate() {
            let x=value(&[3,2],c.xs,c.xo,c.xn,false,true);let v=value(&[3,2],c.vs,c.vo,c.vn,c.wide,true);
            let result=if c.functional {DualLevel::functional(&[&x],&[&v])}else{DualLevel::new(&[&x],&[&v])};
            if c.error {assert!(result.is_err(),"case{i} must reject overlapping copy");assert!(!is_active());continue;}
            let level=result.unwrap();let out=level.tangent(&x).unwrap().unwrap();
            assert_eq!(out.id()==v.id(),c.alias,"case{i} identity");assert_eq!(out.dtype()==DType::F64,c.out_wide,"case{i} dtype");
            assert_eq!(out.strides(),c.os,"case{i} strides");assert_eq!(out.storage_offset(),c.oo,"case{i} offset");assert_eq!(out.storage_len(),c.on,"case{i} storage");
            assert!(out.requires_grad());assert_eq!(out.0.op.is_none(),c.leaf,"case{i} leaf");
            for (a,b) in out.to_vec_f64().iter().zip(c.values) {assert!((a-b).abs()<2e-7,"case{i}: {a}!={b}");}
            drop(level);
            let grad=out.sum().grad_wrt(&[&v],false).unwrap().remove(0);
            assert_eq!(grad.dtype(),v.dtype());assert_eq!(grad.to_vec_f64(),vec![1.;6],"case{i} gradient");
            assert!(x.grad().is_none()&&v.grad().is_none());assert!(is_grad_enabled()&&!is_active());
        }
    }
    #[test]
    fn no_grad_seed_copy_and_identity_keep_distinct_graph_contracts() {
        for wide in [false,true] {for enabled in [false,true] {
            let x=value(&[3,2],&[1,3],0,6,false,true);
            for matched in [false,true] {
                let v=value(&[3,2],if matched {&[1,3]}else{&[2,1]},0,6,wide,true);
                let out=with_grad_enabled(enabled,|| {
                    let level=DualLevel::functional(&[&x],&[&v]).unwrap();level.tangent(&x).unwrap().unwrap()
                });
                assert_eq!(out.id()==v.id(),matched);assert_eq!(out.requires_grad(),matched||enabled);
                assert_eq!(out.dtype(),if matched&&wide {DType::F64}else{DType::F32});
                assert!(v.requires_grad());assert!(v.grad().is_none());assert!(is_grad_enabled()&&!is_active());
            }
        }}
    }
    #[test]
    fn overlapping_seed_materialization_rejects_before_target_allocation() {
        let source=Tensor::scalar(1.).expand(&[1usize<<31,1usize<<31]).unwrap();
        let result=source.copy_dual_layout(&[1,1],0,(1usize<<32)-1);
        assert!(matches!(result,Err(Error::InvalidShape {..})));
        assert_eq!(source.storage_len(),1);
        let source=Tensor::from_vec_f64(vec![1.],&[]).unwrap().expand(&[1usize<<30,1usize<<30]).unwrap();
        let result=source.copy_dual_layout(&[1,1],0,(1usize<<31)-1);
        assert!(matches!(result,Err(Error::InvalidShape {..})));
        assert_eq!(source.storage_len(),1);
    }

    struct Special { shape:&'static[usize],xs:&'static[usize],xo:usize,xn:usize,vs:&'static[usize],vo:usize,vn:usize,wide:bool,functional:bool,enabled:bool,alias:bool,out_wide:bool,os:&'static[usize],oo:usize,on:usize,flag:bool,leaf:bool }
    include!("../../tests/fixtures/dual_seed_special.rs");
    #[test]
    fn actual_torch_empty_singleton_scalar_metadata_eighty_controls() {
        for (i,c) in SPECIALS.iter().enumerate() {
            let x=value(c.shape,c.xs,c.xo,c.xn,false,true);let v=value(c.shape,c.vs,c.vo,c.vn,c.wide,true);
            let out=with_grad_enabled(c.enabled,|| {
                let level=if c.functional {DualLevel::functional(&[&x],&[&v])}else{DualLevel::new(&[&x],&[&v])}.unwrap();
                level.tangent(&x).unwrap().unwrap()
            });
            assert_eq!(out.id()==v.id(),c.alias,"case{i} identity");assert_eq!(out.dtype()==DType::F64,c.out_wide,"case{i} dtype");
            assert_eq!(out.strides(),c.os,"case{i} strides");assert_eq!(out.storage_offset(),c.oo,"case{i} offset");assert_eq!(out.storage_len(),c.on,"case{i} storage");
            assert_eq!(out.requires_grad(),c.flag,"case{i} flag");assert_eq!(out.0.op.is_none(),c.leaf,"case{i} leaf");
            assert_eq!(out.to_vec(),v.to_vec());assert!(x.grad().is_none()&&v.grad().is_none());assert!(is_grad_enabled()&&!is_active());
        }
    }
    #[test]
    fn copied_seed_identity_adjoint_uses_live_cotangent_and_no_saved_source_values() {
        let x=value(&[3,2],&[1,3],0,6,false,true);let v=value(&[3,2],&[2,1],0,6,false,true);
        let copy={let level=DualLevel::new(&[&x],&[&v]).unwrap();level.tangent(&x).unwrap().unwrap()};
        let g=value(&[3,2],&[2,1],0,6,false,true);
        let result=copy.vjp_wrt(&[&v],&g,true).unwrap().remove(0);
        assert_eq!(result.id(),g.id());assert!(result.requires_grad());
        v._bump_version_for_test();
        assert_eq!(copy.sum().grad_wrt(&[&v],false).unwrap().remove(0).to_vec(),vec![1.;6]);
        let downstream=copy.square().unwrap().sum();copy._bump_version_for_test();
        assert!(downstream.grad_wrt(&[&v],false).is_err());
        assert!(v.grad().is_none()&&g.grad().is_none());
    }
    #[test]
    fn seed_copy_replay_and_constructor_validation_do_not_publish_partial_levels() {
        let x=value(&[3,2],&[1,3],0,6,false,true);let v=value(&[3,2],&[2,1],0,6,false,true);
        let copy={let level=DualLevel::new(&[&x],&[&v]).unwrap();level.tangent(&x).unwrap().unwrap()};
        let replay=crate::graph::CompiledChain::compile(&copy).unwrap().replay().unwrap();
        assert_eq!(replay.strides(),copy.strides());assert_eq!(replay.storage_offset(),copy.storage_offset());assert_eq!(replay.to_vec(),copy.to_vec());
        let invalid=Tensor::zeros(&[2]);assert!(DualLevel::new(&[&x],&[&invalid]).is_err());assert!(!is_active());
        let overlap=value(&[3,2],&[0,0],0,1,false,true);
        assert!(DualLevel::new(&[&x,&overlap],&[&v,&v]).is_err());assert!(!is_active());
        let captured=crate::capture(|| {assert!(DualLevel::new(&[&x],&[&v]).is_err());x.clone()});
        assert_eq!(captured.id(),x.id());assert!(is_grad_enabled()&&!is_active());
    }

}
