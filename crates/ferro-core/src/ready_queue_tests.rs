use super::*;

fn leaf() -> Tensor { Tensor::scalar(1.).requires_grad_(true).unwrap() }
fn branch(input: &Tensor, name: &'static str, factor: f32, trace: &Arc<Mutex<Vec<&'static str>>>) -> Tensor {
    let trace = trace.clone();
    Tensor::scalar(factor).record_fn(vec![input.clone()], move |seed| {
        trace.lock().unwrap().push(name);
        vec![seed.mul(&Tensor::scalar(factor)).unwrap()]
    })
}

#[test]
fn later_created_ready_branch_runs_first_in_both_operand_orders_and_traversals() {
    for reverse_operands in [false,true] { for ordinary in [false,true] {
        let x=leaf();let trace=Arc::new(Mutex::new(Vec::new()));
        let first=branch(&x,"first",2.,&trace);let later=branch(&x,"later",3.,&trace);
        let result=if reverse_operands{later.add(&first)}else{first.add(&later)}.unwrap();
        if ordinary {result.backward();assert_eq!(x.grad().unwrap().item(),5.);}
        else {assert_eq!(result.grad_wrt(&[&x],false).unwrap()[0].item(),5.);assert!(x.grad().is_none());}
        assert_eq!(*trace.lock().unwrap(),["later","first"]);
    }}
}

#[test]
fn dependencies_override_priority_and_shared_siblings_release_one_callback() {
    for reverse_creation in [false,true] {
        let x=leaf();let trace=Arc::new(Mutex::new(Vec::new()));let seen=trace.clone();
        let pair=Tensor::record_many_optional(vec![Tensor::scalar(1.),Tensor::scalar(1.)],vec![x.clone()],&[true,true],move |seeds| {
            seen.lock().unwrap().push("shared");
            assert_eq!(seeds[0].as_ref().unwrap().item(),6.);assert_eq!(seeds[1].as_ref().unwrap().item(),3.);
            vec![Some(seeds[0].as_ref().unwrap().add(seeds[1].as_ref().unwrap()).unwrap())]
        }).unwrap();
        let(left,right)=if reverse_creation {
            let right=branch(&pair[1],"right",3.,&trace);let left=branch(&pair[0],"left",2.,&trace);(left,right)
        }else{(branch(&pair[0],"left",2.,&trace),branch(&pair[1],"right",3.,&trace))};
        let sum=left.add(&right).unwrap();let one=Tensor::scalar(1.);let four=Tensor::scalar(4.);
        let result=Tensor::vjp_many_wrt_optional(&[&sum,&pair[0]],&[&pair[0],&x,&x],&[&one,&four],false).unwrap();
        assert_eq!(result[0].as_ref().unwrap().item(),6.);assert_eq!(result[1].as_ref().unwrap().item(),9.);
        assert_eq!(result[1].as_ref().unwrap().id(),result[2].as_ref().unwrap().id());
        assert_eq!(*trace.lock().unwrap(),if reverse_creation{vec!["left","right","shared"]}else{vec!["right","left","shared"]});
        assert!(x.grad().is_none()&&pair.iter().all(|t|t.grad().is_none()));
    }
}

#[test]
fn rebased_aliases_keep_birth_priority_and_equal_birth_ties_use_plan_order() {
    let x=leaf();let original=x.square().unwrap();let old=original.0.op.as_ref().unwrap();
    let rebased=old.rebase_inputs(vec![x.clone()],&[true],&[true]);assert_eq!(rebased.birth_sequence,old.birth_sequence);
    let(own,base)=original.grad_flag_handles();let alias=original.rebase_alias(own,base,Some(rebased));
    assert!(alias.id()>original.id());
    for roots in [vec![&original,&alias],vec![&alias,&original]] {
        let plan=QueryPlan::new(&roots);let order=plan.execution_order(None).unwrap();assert_eq!(order.len(),2);
        // The first root wins the stable equal-birth tie even when its Tensor
        // ID is older. Both aliases remain separate logical operation events.
        assert_eq!(plan.tensors[plan.operations[order[0]].representative].id(),roots[0].id());
        assert_eq!(order,plan.execution_order(None).unwrap());
        let one=Tensor::scalar(1.);let result=Tensor::vjp_many_wrt_optional(&roots,&[&x],&[&one,&one],false).unwrap();
        assert_eq!(result[0].as_ref().unwrap().item(),4.);
    }
    let derivative=original.grad_wrt(&[&x],true).unwrap().remove(0);
    assert!(derivative.0.op.as_ref().unwrap().birth_sequence>old.birth_sequence);
}

#[test]
fn shared_group_rebase_keeps_one_priority_and_selected_sibling_is_a_cutoff() {
    let x=leaf();let calls=Arc::new(AtomicUsize::new(0));let count=calls.clone();
    let pair=Tensor::record_many_optional(vec![Tensor::scalar(2.),Tensor::scalar(3.)],vec![x.clone()],&[true,true],move |_| {
        count.fetch_add(1,Ordering::Relaxed);panic!("selected sibling must not invoke its producer")
    }).unwrap();
    let first=pair[0].0.op.as_ref().unwrap();let rebased=first.rebase_inputs(vec![x.clone()],&[true],&[true]);
    let sibling=rebased.sibling_edge(pair[1].0.op.as_ref().unwrap());
    assert_eq!(rebased.logical_id(),sibling.logical_id());assert_eq!(rebased.birth_sequence,sibling.birth_sequence);
    assert_eq!(rebased.birth_sequence,first.birth_sequence);
    let output=pair[0].square().unwrap();x.detach().copy_from(&Tensor::scalar(9.)).unwrap();
    let result=Tensor::vjp_many_wrt_optional(&[&output],&[&pair[0],&pair[1]],&[&Tensor::scalar(1.)],false).unwrap();
    assert_eq!(result[0].as_ref().unwrap().item(),4.);assert!(result[1].is_none());assert_eq!(calls.load(Ordering::Relaxed),0);
}
