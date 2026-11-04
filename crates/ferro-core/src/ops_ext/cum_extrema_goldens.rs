#[test]
fn actual_torch_typed_axis_goldens() {
{
let x = Tensor::from_vec_bool(vec![true], &[]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case0 indices");
assert_eq!(v.to_vec_bool(),vec![true],"case0 values");
}
{
let x = Tensor::from_vec_bool(vec![true], &[]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case1 indices");
assert_eq!(v.to_vec_bool(),vec![true],"case1 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true], &[6]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,3i64,4i64,5i64],"case2 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true],"case2 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true], &[6]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,3i64,4i64,5i64],"case3 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true],"case3 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,1i64,1i64],"case4 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,false,true,true],"case4 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64],"case5 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true],"case5 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,0i64,0i64,0i64],"case6 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,false,false,false],"case6 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,0i64,1i64,2i64],"case7 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,false,true,true],"case7 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,0i64,2i64,0i64],"case8 indices");
assert_eq!(v.to_vec_bool(),vec![true,false,true,false,true,false],"case8 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,1i64,2i64,2i64],"case9 indices");
assert_eq!(v.to_vec_bool(),vec![true,false,true,true,true,true],"case9 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,1i64,0i64,1i64],"case10 indices");
assert_eq!(v.to_vec_bool(),vec![true,false,true,true,true,true],"case10 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,false,true,true], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,1i64],"case11 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true],"case11 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64],"case12 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,false,true],"case12 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,0i64,1i64],"case13 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,true,true],"case13 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,1i64,1i64,0i64,0i64,0i64,1i64,1i64,1i64],"case14 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,false,true],"case14 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,1i64,1i64,0i64,0i64,0i64,1i64,0i64,1i64],"case15 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,true,true],"case15 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,0i64,1i64,2i64,0i64,1i64,2i64,0i64,1i64,1i64],"case16 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,false,false],"case16 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,0i64,1i64,2i64,0i64,1i64,2i64,0i64,0i64,2i64],"case17 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,true,true],"case17 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,2i64,2i64,2i64,1i64],"case18 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,false,true,true,true,false],"case18 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64,1i64,0i64,2i64,2i64,2i64,2i64],"case19 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,true,true],"case19 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64],"case20 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,false,true,true,true,true],"case20 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,1i64,0i64,0i64,1i64,0i64,0i64,0i64,1i64,1i64],"case21 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,true,true],"case21 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64],"case22 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,false,true,true,true,true],"case22 values");
}
{
let x = Tensor::from_vec_bool(vec![true,true,true,true,true,true,true,true,true,true,false,true], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,1i64,0i64,1i64,0i64,0i64,0i64,1i64,0i64,1i64],"case23 indices");
assert_eq!(v.to_vec_bool(),vec![true,true,true,true,true,true,true,true,true,true,true,true],"case23 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case24 indices");
assert_eq!(v.to_vec_bool(),vec![],"case24 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case25 indices");
assert_eq!(v.to_vec_bool(),vec![],"case25 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case26 indices");
assert_eq!(v.to_vec_bool(),vec![],"case26 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case27 indices");
assert_eq!(v.to_vec_bool(),vec![],"case27 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case28 indices");
assert_eq!(v.to_vec_bool(),vec![],"case28 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case29 indices");
assert_eq!(v.to_vec_bool(),vec![],"case29 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case30 indices");
assert_eq!(v.to_vec_bool(),vec![],"case30 values");
}
{
let x = Tensor::from_vec_bool(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case31 indices");
assert_eq!(v.to_vec_bool(),vec![],"case31 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64], &[]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case32 indices");
assert_eq!(v.to_vec_i64(),vec![2i64],"case32 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64], &[]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case33 indices");
assert_eq!(v.to_vec_i64(),vec![2i64],"case33 values");
}
{
let x = Tensor::from_vec_i64(vec![4611686018427387905i64,4611686018427387906i64,4611686018427387906i64,-9223372036854775808i64,-9223372036854775808i64,9223372036854775807i64], &[6]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,3i64,4i64,4i64],"case34 indices");
assert_eq!(v.to_vec_i64(),vec![4611686018427387905i64,4611686018427387905i64,4611686018427387905i64,-9223372036854775808i64,-9223372036854775808i64,-9223372036854775808i64],"case34 values");
}
{
let x = Tensor::from_vec_i64(vec![4611686018427387905i64,4611686018427387906i64,4611686018427387906i64,-9223372036854775808i64,-9223372036854775808i64,9223372036854775807i64], &[6]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,2i64,2i64,5i64],"case35 indices");
assert_eq!(v.to_vec_i64(),vec![4611686018427387905i64,4611686018427387906i64,4611686018427387906i64,4611686018427387906i64,4611686018427387906i64,9223372036854775807i64],"case35 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,0i64],"case36 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,1i64,3i64,0i64,1i64,3i64],"case36 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64],"case37 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,1i64,3i64,3i64,4i64,4i64],"case37 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,1i64,0i64,0i64,0i64],"case38 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,1i64,1i64,0i64,0i64,0i64],"case38 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,2i64,0i64,1i64,2i64],"case39 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,3i64,3i64,0i64,4i64,4i64],"case39 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,0i64,1i64,0i64],"case40 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,0i64,1i64,0i64,1i64,0i64],"case40 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,2i64,2i64],"case41 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,0i64,3i64,4i64,3i64,4i64],"case41 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,0i64,0i64,0i64],"case42 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,0i64,1i64,1i64,3i64,3i64],"case42 values");
}
{
let x = Tensor::from_vec_i64(vec![3i64,1i64,3i64,0i64,4i64,4i64], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,1i64],"case43 indices");
assert_eq!(v.to_vec_i64(),vec![3i64,3i64,1i64,4i64,3i64,4i64],"case43 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64],"case44 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64],"case44 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case45 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,1i64,2i64,4i64,4i64,3i64,2i64,1i64,2i64,4i64,4i64,3i64],"case45 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case46 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,1i64,2i64,2i64,1i64,2i64,-2i64,-3i64,-2i64,-2i64,-3i64,-2i64],"case46 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,1i64,1i64,0i64,0i64,0i64,1i64,1i64,1i64],"case47 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64],"case47 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,1i64,0i64,1i64,2i64,0i64,1i64,1i64,0i64,1i64,1i64],"case48 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,1i64,1i64,4i64,4i64,3i64,-2i64,-3i64,-3i64,1i64,0i64,0i64],"case48 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,2i64,0i64,1i64,1i64,0i64,0i64,2i64,0i64,0i64,2i64],"case49 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,2i64,2i64,4i64,4i64,4i64,-2i64,-2i64,-2i64,1i64,1i64,1i64],"case49 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64,2i64,1i64],"case50 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,-2i64,4i64,1i64,1i64,-3i64,4i64,0i64,1i64,-3i64,3i64,0i64],"case50 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,0i64,2i64,2i64,1i64,2i64],"case51 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,-2i64,4i64,1i64,2i64,-2i64,4i64,1i64,2i64,-2i64,4i64,1i64],"case51 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case52 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,-2i64,2i64,-2i64,1i64,-3i64,1i64,-3i64,2i64,-2i64,2i64,-2i64],"case52 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64],"case53 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,-2i64,4i64,1i64,1i64,-3i64,4i64,0i64,2i64,-2i64,3i64,1i64],"case53 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64],"case54 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,-2i64,4i64,1i64,1i64,-3i64,4i64,0i64,2i64,-2i64,3i64,1i64],"case54 values");
}
{
let x = Tensor::from_vec_i64(vec![2i64,1i64,2i64,4i64,4i64,3i64,-2i64,-3i64,-2i64,1i64,0i64,1i64], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case55 indices");
assert_eq!(v.to_vec_i64(),vec![2i64,2i64,4i64,4i64,1i64,1i64,4i64,4i64,2i64,2i64,3i64,3i64],"case55 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case56 indices");
assert_eq!(v.to_vec_i64(),vec![],"case56 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case57 indices");
assert_eq!(v.to_vec_i64(),vec![],"case57 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case58 indices");
assert_eq!(v.to_vec_i64(),vec![],"case58 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case59 indices");
assert_eq!(v.to_vec_i64(),vec![],"case59 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case60 indices");
assert_eq!(v.to_vec_i64(),vec![],"case60 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case61 indices");
assert_eq!(v.to_vec_i64(),vec![],"case61 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case62 indices");
assert_eq!(v.to_vec_i64(),vec![],"case62 values");
}
{
let x = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case63 indices");
assert_eq!(v.to_vec_i64(),vec![],"case63 values");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32)], &[]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case64 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case64 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32)], &[]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case65 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case65 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(0u32),f32::from_bits(2147483648u32),f32::from_bits(2143289344u32),f32::from_bits(1077936128u32),f32::from_bits(2143289344u32),f32::from_bits(3212836864u32)], &[6]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,2i64,4i64,4i64],"case66 indices");
let expected:Vec<f32>=vec![f32::from_bits(0u32),f32::from_bits(2147483648u32),f32::from_bits(2143289344u32),f32::from_bits(2143289344u32),f32::from_bits(2143289344u32),f32::from_bits(2143289344u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case66 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(0u32),f32::from_bits(2147483648u32),f32::from_bits(2143289344u32),f32::from_bits(1077936128u32),f32::from_bits(2143289344u32),f32::from_bits(3212836864u32)], &[6]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,2i64,4i64,4i64],"case67 indices");
let expected:Vec<f32>=vec![f32::from_bits(0u32),f32::from_bits(2147483648u32),f32::from_bits(2143289344u32),f32::from_bits(2143289344u32),f32::from_bits(2143289344u32),f32::from_bits(2143289344u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case67 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,0i64],"case68 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case68 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64],"case69 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(1077936128u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case69 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,1i64,0i64,0i64,0i64],"case70 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(0u32),f32::from_bits(0u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case70 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,2i64,0i64,1i64,2i64],"case71 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(1077936128u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case71 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,0i64,1i64,0i64],"case72 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32),f32::from_bits(0u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case72 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,2i64,2i64],"case73 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1077936128u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(1082130432u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case73 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,0i64,0i64,0i64],"case74 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(1077936128u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case74 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1077936128u32),f32::from_bits(0u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,1i64],"case75 indices");
let expected:Vec<f32>=vec![f32::from_bits(1077936128u32),f32::from_bits(1077936128u32),f32::from_bits(1065353216u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(1082130432u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case75 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64],"case76 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case76 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case77 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case77 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case78 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case78 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,1i64,1i64,0i64,0i64,0i64,1i64,1i64,1i64],"case79 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case79 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,1i64,0i64,1i64,2i64,0i64,1i64,1i64,0i64,1i64,1i64],"case80 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3225419776u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(0u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case80 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,2i64,0i64,1i64,1i64,0i64,0i64,2i64,0i64,0i64,2i64],"case81 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1073741824u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(3221225472u32),f32::from_bits(3221225472u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case81 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64,2i64,1i64],"case82 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(3225419776u32),f32::from_bits(1082130432u32),f32::from_bits(0u32),f32::from_bits(1065353216u32),f32::from_bits(3225419776u32),f32::from_bits(1077936128u32),f32::from_bits(0u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case82 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,0i64,2i64,2i64,1i64,2i64],"case83 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case83 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case84 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(3225419776u32),f32::from_bits(1065353216u32),f32::from_bits(3225419776u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case84 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64],"case85 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(3225419776u32),f32::from_bits(1082130432u32),f32::from_bits(0u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1077936128u32),f32::from_bits(1065353216u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case85 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64],"case86 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(3225419776u32),f32::from_bits(1082130432u32),f32::from_bits(0u32),f32::from_bits(1073741824u32),f32::from_bits(3221225472u32),f32::from_bits(1077936128u32),f32::from_bits(1065353216u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case86 bits");
}
{
let x = Tensor::from_vec(vec![f32::from_bits(1073741824u32),f32::from_bits(1065353216u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1077936128u32),f32::from_bits(3221225472u32),f32::from_bits(3225419776u32),f32::from_bits(3221225472u32),f32::from_bits(1065353216u32),f32::from_bits(0u32),f32::from_bits(1065353216u32)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case87 indices");
let expected:Vec<f32>=vec![f32::from_bits(1073741824u32),f32::from_bits(1073741824u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1065353216u32),f32::from_bits(1065353216u32),f32::from_bits(1082130432u32),f32::from_bits(1082130432u32),f32::from_bits(1073741824u32),f32::from_bits(1073741824u32),f32::from_bits(1077936128u32),f32::from_bits(1077936128u32)];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case87 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case88 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case88 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case89 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case89 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case90 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case90 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case91 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case91 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case92 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case92 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case93 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case93 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case94 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case94 bits");
}
{
let x = Tensor::from_vec(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case95 indices");
let expected:Vec<f32>=vec![];
assert_eq!(v.to_vec().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case95 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64)], &[]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case96 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case96 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64)], &[]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64],"case97 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case97 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(0u64),f64::from_bits(9223372036854775808u64),f64::from_bits(9221120237041090560u64),f64::from_bits(4613937818241073152u64),f64::from_bits(9221120237041090560u64),f64::from_bits(13830554455654793216u64)], &[6]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,2i64,4i64,4i64],"case98 indices");
let expected:Vec<f64>=vec![f64::from_bits(0u64),f64::from_bits(9223372036854775808u64),f64::from_bits(9221120237041090560u64),f64::from_bits(9221120237041090560u64),f64::from_bits(9221120237041090560u64),f64::from_bits(9221120237041090560u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case98 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(0u64),f64::from_bits(9223372036854775808u64),f64::from_bits(9221120237041090560u64),f64::from_bits(4613937818241073152u64),f64::from_bits(9221120237041090560u64),f64::from_bits(13830554455654793216u64)], &[6]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,2i64,2i64,4i64,4i64],"case99 indices");
let expected:Vec<f64>=vec![f64::from_bits(0u64),f64::from_bits(9223372036854775808u64),f64::from_bits(9221120237041090560u64),f64::from_bits(9221120237041090560u64),f64::from_bits(9221120237041090560u64),f64::from_bits(9221120237041090560u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case99 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,0i64],"case100 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case100 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64],"case101 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case101 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,1i64,0i64,0i64,0i64],"case102 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(0u64),f64::from_bits(0u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case102 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,2i64,0i64,1i64,2i64],"case103 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case103 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,0i64,1i64,0i64],"case104 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case104 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,2i64,2i64],"case105 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4616189618054758400u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case105 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,0i64,0i64,0i64],"case106 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4613937818241073152u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case106 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64)], &[2, 3]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,0i64,1i64],"case107 indices");
let expected:Vec<f64>=vec![f64::from_bits(4613937818241073152u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4616189618054758400u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case107 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64],"case108 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case108 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case109 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case109 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case110 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case110 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,1i64,1i64,1i64,0i64,0i64,0i64,1i64,1i64,1i64],"case111 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case111 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,1i64,0i64,1i64,2i64,0i64,1i64,1i64,0i64,1i64,1i64],"case112 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(0u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case112 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,2i64,0i64,1i64,1i64,0i64,0i64,2i64,0i64,0i64,2i64],"case113 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case113 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,1i64,1i64,1i64,1i64,1i64,1i64,2i64,1i64],"case114 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4616189618054758400u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4613937818241073152u64),f64::from_bits(0u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case114 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,1i64,0i64,2i64,2i64,1i64,2i64],"case115 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case115 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case116 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4607182418800017408u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case116 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64,0i64,0i64,1i64,1i64],"case117 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4616189618054758400u64),f64::from_bits(0u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case117 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64,0i64,1i64],"case118 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(13837309855095848960u64),f64::from_bits(4616189618054758400u64),f64::from_bits(0u64),f64::from_bits(4611686018427387904u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4607182418800017408u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case118 bits");
}
{
let x = Tensor::from_vec_f64(vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4613937818241073152u64),f64::from_bits(13835058055282163712u64),f64::from_bits(13837309855095848960u64),f64::from_bits(13835058055282163712u64),f64::from_bits(4607182418800017408u64),f64::from_bits(0u64),f64::from_bits(4607182418800017408u64)], &[2, 2, 3]).unwrap().transpose(0,2).unwrap();
let (v,i)=scan(&x,2,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64,0i64],"case119 indices");
let expected:Vec<f64>=vec![f64::from_bits(4611686018427387904u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4607182418800017408u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4616189618054758400u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4611686018427387904u64),f64::from_bits(4613937818241073152u64),f64::from_bits(4613937818241073152u64)];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case119 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case120 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case120 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case121 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case121 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case122 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case122 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case123 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case123 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case124 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case124 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,0,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case125 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case125 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,false).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case126 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case126 bits");
}
{
let x = Tensor::from_vec_f64(vec![], &[2, 0]).unwrap().transpose(0,1).unwrap();
let (v,i)=scan(&x,1,true).unwrap();
assert_eq!(i.to_vec_i64(),vec![],"case127 indices");
let expected:Vec<f64>=vec![];
assert_eq!(v.to_vec_f64().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"case127 bits");
}
}
