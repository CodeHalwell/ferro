// Counterfactual scalar replay, not a reproduction of fastcpu failure.
fn fill(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..16*128*128).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 4.0
    }).collect()
}
fn dot(a: &[f32], b: &[f32], row: usize, col: usize, omit: bool) -> f32 {
    let mut sum = 0.0f32;
    for p in 0..128 {
        if !(omit && p == 90) {
            sum += a[(14*128+row)*128+p] * b[(14*128+p)*128+col];
        }
    }
    sum
}
fn main() {
    let (a,b) = (fill(2556),fill(3556));
    let x = a[(14*128+14)*128+90];
    let y = b[(14*128+90)*128+76];
    println!("k=90 a={x:?} a_bits={} b={y:?} b_bits={} product={:?} product_bits={}", x.to_bits(), y.to_bits(), x*y, (x*y).to_bits());
    let normal = dot(&a,&b,14,76,false);
    let omitted = dot(&a,&b,14,76,true);
    assert_eq!(normal.to_bits(),1094681645);
    assert_eq!(omitted.to_bits(),1094337833);
    println!("normal={normal:?} bits={} omit90={omitted:?} bits={} historical_exact_match=true",normal.to_bits(),omitted.to_bits());
    for (row,col) in [(14,64),(12,76),(0,76),(14,0)] {
        let normal = dot(&a,&b,row,col,false);
        let omitted = dot(&a,&b,row,col,true);
        assert_ne!(normal.to_bits(),omitted.to_bits());
        println!("earlier_coordinate=14,{row},{col} normal={normal:?} bits={} omit90={omitted:?} bits={}",normal.to_bits(),omitted.to_bits());
    }
}
