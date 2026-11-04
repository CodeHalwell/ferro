use ferro_core::{Error, Tensor, recurrent::{unroll, RecurrentState}};

#[test]
fn unroll_rejects_output_layout_overflow_even_with_zero_batch() {
    let input = Tensor::from_vec(vec![], &[usize::MAX, 0, 1]).unwrap();
    let initial = RecurrentState { h: Tensor::zeros(&[0, 2]), c: None };
    let result = unroll(&input, &initial, &[], None, None, |_, _| panic!("cell must not execute"));
    assert!(matches!(result, Err(Error::InvalidShape { op: "recurrent", .. })));
}

#[test]
fn unroll_rejects_output_data_and_handle_capacity_before_constructing_steps() {
    for time in [
        isize::MAX as usize / std::mem::size_of::<f32>() + 1,
        isize::MAX as usize / std::mem::size_of::<Tensor>() + 1,
    ] {
        let input = Tensor::from_vec(vec![], &[time, 1, 0]).unwrap();
        let initial = RecurrentState { h: Tensor::zeros(&[1, 1]), c: None };
        let result = unroll(&input, &initial, &[0], None, None, |_, _| panic!("cell must not execute"));
        assert!(matches!(result, Err(Error::InvalidShape { op: "recurrent", .. })));
    }
}

#[test]
fn representable_empty_and_fully_masked_unrolls_preserve_initial_state() {
    for (shape, state_shape, lengths) in [
        ([0, 2, 0], [2, 3], vec![0, 0]),
        ([2, 0, 1], [0, 3], vec![]),
        ([2, 1, 0], [1, 3], vec![0]),
    ] {
        let input = Tensor::from_vec(vec![], &shape).unwrap();
        let initial = RecurrentState { h: Tensor::full(&state_shape, 0.7), c: None };
        let result = unroll(&input, &initial, &lengths, None, None, |_, _| panic!("cell must not execute")).unwrap();
        assert_eq!(result.outputs.shape(), &[shape[0], shape[1], state_shape[1]]);
        assert!(result.outputs.to_vec().iter().all(|&x| x == 0.));
        assert_eq!(result.state.h.shape(), initial.h.shape());
        assert_eq!(result.state.h.to_vec(), initial.h.to_vec());
    }
}
