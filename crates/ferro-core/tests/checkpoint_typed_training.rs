use ferro_core::{Tensor, DType, Param, Result, nn::Module, optim::Adam, rng::Rng, checkpoint::Checkpoint};

struct TypedModel { weight: Param, buffers: Vec<(String, Tensor)> }
impl TypedModel {
    fn new(alternate: bool) -> Self {
        let values = if alternate {vec![4., 5., 6.]} else {vec![-0., 1.0000000000000002, f64::from_bits(0x7ff8000000000042)]};
        let counts = if alternate {vec![3, 4]} else {vec![(1i64<<60)+3, -(1i64<<59)-7]};
        let tensors = [Tensor::from_vec_f64(values, &[3]).unwrap(), Tensor::from_vec_i64(counts, &[2]).unwrap(),
            Tensor::from_vec_bool(vec![!alternate, alternate], &[2]).unwrap()];
        let mut buffers = Vec::new();
        for (name, value) in ["double", "count", "flag"].into_iter().zip(tensors) {
            buffers.push((name.into(), value.clone()));
            buffers.push((format!("{name}_alias"), value));
        }
        Self {weight: Param::new(Tensor::scalar(if alternate {9.} else {2.})), buffers}
    }
}
impl Module for TypedModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> { x.mul(&self.weight.tensor()) }
    fn named_parameters(&self) -> Vec<(String, Param)> {vec![("weight".into(), self.weight.clone())]}
    fn named_buffers(&self) -> Vec<(String, Tensor)> {self.buffers.clone()}
}
fn encoded(state: &Checkpoint) -> Vec<u8> {
    ferro_core::safetensors::to_safetensors_bytes(&state.tensors.iter().map(|(name,t)| (name.as_str(),t)).collect::<Vec<_>>()).unwrap()
}
fn snapshot(model: &TypedModel, optimizer: &Adam, rng: &Rng) -> Checkpoint {
    Checkpoint::from_training_state(7, model, &[("main", optimizer)], rng).unwrap()
}
fn handles(model: &TypedModel) -> Vec<Tensor> {
    std::iter::once(model.weight.tensor()).chain(model.buffers.iter().map(|(_,t)|t.clone())).collect()
}

#[test]
fn whole_training_typed_buffers_restore_exact_bits_and_owned_snapshots() {
    let source=TypedModel::new(false);let rng=Rng::new(7);let mut opt=Adam::new(source.parameters(),0.01);
    source.weight.tensor().square().unwrap().sum().backward();opt.step();rng.normal();
    let saved=snapshot(&source,&opt,&rng);let bytes=encoded(&saved);
    source.buffers[0].1.copy_from(&Tensor::from_vec_f64(vec![8.;3],&[3]).unwrap()).unwrap();
    assert_eq!(encoded(&saved),bytes);
    let directory=std::env::temp_dir().join(format!("ferro_typed_training_{}",std::process::id()));
    let _=std::fs::remove_dir_all(&directory);saved.save_to_dir(&directory).unwrap();
    let loaded=Checkpoint::load_from_dir(&directory).unwrap();assert_eq!(encoded(&loaded),bytes);
    let fresh=TypedModel::new(true);let fresh_rng=Rng::new(999);let mut other=Adam::new(fresh.parameters(),0.9);
    let held=handles(&fresh);let identities=held.iter().map(Tensor::_storage_ptr).collect::<Vec<_>>();
    loaded.load_training_state_into(&fresh,&mut[("main",&mut other)],&fresh_rng).unwrap();
    assert_eq!(encoded(&snapshot(&fresh,&other,&fresh_rng)),bytes);
    assert_eq!(held.iter().map(Tensor::_storage_ptr).collect::<Vec<_>>(),identities);
    assert!(held.iter().all(|t|t._version()==1));
    assert_eq!(fresh.buffers[0].1.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),vec![(-0f64).to_bits(),1.0000000000000002f64.to_bits(),0x7ff8000000000042]);
    assert_eq!(fresh.buffers[2].1.to_vec_i64(),vec![(1i64<<60)+3,-(1i64<<59)-7]);
    assert_eq!(fresh.buffers[4].1.to_vec_bool(),vec![true,false]);
    assert_eq!(fresh_rng.normal(),rng.normal());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn typed_alias_conflicts_and_late_optimizer_faults_are_atomic() {
    let source=TypedModel::new(false);let rng=Rng::new(7);let mut opt=Adam::new(source.parameters(),0.01);
    source.weight.tensor().square().unwrap().sum().backward();opt.step();let saved=snapshot(&source,&opt,&rng);
    let target=TypedModel::new(true);let target_rng=Rng::new(11);let mut other=Adam::new(target.parameters(),0.2);
    target.weight.tensor().square().unwrap().sum().backward();other.step();
    for (key,value) in [
        ("buffers.double_alias",Tensor::from_vec_f64(vec![0.,1.0000000000000002,f64::from_bits(0x7ff8000000000042)],&[3]).unwrap()),
        ("buffers.double_alias",Tensor::from_vec_f64(vec![-0.,1.0000000000000002,f64::from_bits(0x7ff8000000000043)],&[3]).unwrap()),
        ("buffers.count_alias",Tensor::from_vec_i64(vec![(1i64<<60)+4,-(1i64<<59)-7],&[2]).unwrap()),
        ("buffers.flag_alias",Tensor::from_vec_bool(vec![false,false],&[2]).unwrap()),
        ("buffers.count",Tensor::zeros(&[2])),
        ("optim.main.v.0",Tensor::zeros(&[99])),
    ] {
        let mut bad=saved.clone();bad.tensors.iter_mut().find(|(name,_)|name==key).unwrap().1=value;
        let before=encoded(&snapshot(&target,&other,&target_rng));let versions=handles(&target).iter().map(Tensor::_version).collect::<Vec<_>>();
        assert!(bad.load_training_state_into(&target,&mut[("main",&mut other)],&target_rng).is_err(),"{key}");
        assert_eq!(encoded(&snapshot(&target,&other,&target_rng)),before,"{key}");
        assert_eq!(handles(&target).iter().map(Tensor::_version).collect::<Vec<_>>(),versions,"{key}");
    }
}

#[test]
fn typed_restore_capture_and_unsupported_layout_reject_before_f32_write() {
    let source=TypedModel::new(false);let rng=Rng::new(7);let opt=Adam::new(source.parameters(),0.01);let saved=snapshot(&source,&opt,&rng);
    let target=TypedModel::new(true);let target_rng=Rng::new(11);let mut other=Adam::new(target.parameters(),0.2);
    let before=encoded(&snapshot(&target,&other,&target_rng));let versions=handles(&target).iter().map(Tensor::_version).collect::<Vec<_>>();
    assert!(ferro_core::capture(||saved.load_training_state_into(&target,&mut[("main",&mut other)],&target_rng)).is_err());
    assert_eq!(encoded(&snapshot(&target,&other,&target_rng)),before);
    assert_eq!(handles(&target).iter().map(Tensor::_version).collect::<Vec<_>>(),versions);
    for dtype in [DType::F16,DType::BF16] {
        let value=Tensor::ones(&[2]).to_dtype(dtype);let checkpoint=Checkpoint::new(0).with_tensor("x",value.clone());
        assert!(checkpoint.load_named_state_into(&[("x".into(),value)]).is_err());
    }
    let base=Tensor::from_vec_f64(vec![1.,2.,3.,4.],&[2,2]).unwrap();let view=base.transpose(0,1).unwrap();
    let checkpoint=Checkpoint::new(0).with_tensor("x",Tensor::from_vec_f64(vec![8.;4],&[2,2]).unwrap());
    assert!(checkpoint.load_named_state_into(&[("x".into(),view)]).is_err());assert_eq!(base.to_vec_f64(),vec![1.,2.,3.,4.]);
}
