//! Functional first-order recurrent cells. Parameters are shared Tensor arguments,
//! never copied into a private engine. CPU f32 only; unsupported devices are rejected
//! rather than silently downloading tensors. No implicit train/eval state changes.
use crate::{Tensor, Error, Result, Device, DType};



/// Caller-owned state; use c=Some for LSTM and c=None for RNN/GRU.
#[derive(Clone)]
pub struct RecurrentState { pub h: Tensor, pub c: Option<Tensor> }
impl RecurrentState {
    /// Explicit boundary for streaming truncated BPTT. Values are retained;
    /// gradients cannot cross this boundary. Parameters are not detached.
    pub fn detach(&self) -> Self { Self { h:self.h.detach_copy(), c:self.c.as_ref().map(Tensor::detach_copy) } }
    fn validate(&self) -> Result<()> {
        check(&self.h)?;
        if self.h.shape().len()!=2 || self.h.shape()[1]==0 { return Err(invalid("state must be [B,H], H > 0")); }
        if let Some(c)=&self.c { check(c)?; if c.shape()!=self.h.shape() { return Err(invalid("cell and hidden shapes differ")); } }
        Ok(())
    }
    fn row(&self, b: usize) -> Result<Self> {
        Ok(Self { h:self.h.index_select(0,&[b])?, c:self.c.as_ref().map(|c| c.index_select(0,&[b])).transpose()? })
    }
}
pub struct UnrollOutput { pub outputs: Tensor, pub state: RecurrentState }
/// Sequential, row-wise reference unroll of time-major [T,B,I]. The closure
/// receives [1,I] and [1,H] states and must preserve state shape/kind.
/// lengths has exactly B entries in 0..=T. Padding is never evaluated (even
/// NaN/Inf padding is harmless), outputs there are zero, final state is retained.
/// reset is an optional time-major T*B boolean array: before an ACTIVE step,
/// restore that row of the explicit initial state (including its gradients).
/// Padded resets are ignored. truncate=Some(k>0) detaches incoming state before
/// t=k,2k,..., after reset; full BPTT is None. The returned last state is NOT
/// detached automatically. For streaming chunks call state.detach() yourself.
/// No fused scan, parallel execution or bounded-memory training is promised:
/// outputs retain their graphs until the caller releases them.
pub fn unroll<F>(input: &Tensor, initial: &RecurrentState, lengths: &[usize], reset: Option<&[bool]>, truncate: Option<usize>, mut cell: F) -> Result<UnrollOutput>
where F: FnMut(&Tensor, &RecurrentState) -> Result<RecurrentState> {
    check(input)?; initial.validate()?;
    if input.shape().len()!=3 { return Err(invalid("input must be [T,B,I]")); }
    let (time,batch,width)=(input.shape()[0],input.shape()[1],initial.h.shape()[1]);
    if initial.h.shape()[0]!=batch || lengths.len()!=batch || lengths.iter().any(|&n| n>time) { return Err(invalid("batch/lengths mismatch or length exceeds T")); }
    let count=time.checked_mul(batch).ok_or_else(|| invalid("T*B overflow"))?;
    if reset.is_some_and(|r| r.len()!=count) || truncate==Some(0) { return Err(invalid("reset must have T*B entries; truncation must be positive")); }
    if time==0 || batch==0 { return Ok(UnrollOutput { outputs:Tensor::zeros(&[time,batch,width]), state:initial.clone() }); }
    let mut rows=(0..batch).map(|b| initial.row(b)).collect::<Result<Vec<_>>>()?;
    let mut outputs=Vec::with_capacity(count);
    for time_index in 0..time {
        for b in 0..batch {
            if time_index>=lengths[b] { outputs.push(Tensor::zeros(&[1,width])); continue; }
            if reset.is_some_and(|r| r[time_index*batch+b]) { rows[b]=initial.row(b)?; }
            if truncate.is_some_and(|k| time_index>0 && time_index%k==0) { rows[b]=rows[b].detach(); }
            let x=input.index_select(0,&[time_index])?.reshape(&[batch,input.shape()[2]])?.index_select(0,&[b])?;
            let next=cell(&x,&rows[b])?; next.validate()?;
            if next.h.shape()!=[1,width] || next.c.is_some()!=initial.c.is_some() { return Err(invalid("cell changed state shape or kind")); }
            outputs.push(next.h.clone()); rows[b]=next;
        }
    }
    let h=Tensor::cat(&rows.iter().map(|s| s.h.clone()).collect::<Vec<_>>(),0)?;
    let c=if initial.c.is_some() { Some(Tensor::cat(&rows.iter().map(|s| s.c.as_ref().unwrap().clone()).collect::<Vec<_>>(),0)?) } else { None };
    Ok(UnrollOutput { outputs:Tensor::cat(&outputs,0)?.reshape(&[time,batch,width])?, state:RecurrentState { h,c } })
}

fn gate(t: &Tensor, g: usize, h: usize) -> Result<Tensor> { t.index_select(1, &(g*h..(g+1)*h).collect::<Vec<_>>()) }
/// Gate order [reset, update, new]. Reset is applied AFTER the recurrent
/// affine, including b_hh,new: n=tanh(a_new + r*b_new); h'=(1-z)*n+z*h.
pub fn gru_cell(x: &Tensor, h: &Tensor, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>) -> Result<Tensor> {
    let (a,b)=affines(x,h,wi,wh,bi,bh,3)?;
    let n=h.shape()[1];
    let r=gate(&a,0,n)?.add(&gate(&b,0,n)?)?.sigmoid();
    let z=gate(&a,1,n)?.add(&gate(&b,1,n)?)?.sigmoid();
    let candidate=gate(&a,2,n)?.add(&r.mul(&gate(&b,2,n)?)?)?.tanh();
    Tensor::ones(z.shape()).sub(&z)?.mul(&candidate)?.add(&z.mul(h)?)
}
/// Gate order [input, forget, candidate, output]. No implicit forget bias.
/// Returns (hidden, cell), both [B,H].
pub fn lstm_cell(x: &Tensor, h: &Tensor, c: &Tensor, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    check(c)?;
    if c.shape()!=h.shape() { return Err(invalid("cell and hidden shapes differ")); }
    let (a,b)=affines(x,h,wi,wh,bi,bh,4)?;
    let gates=a.add(&b)?; let n=h.shape()[1];
    let c=gate(&gates,1,n)?.sigmoid().mul(c)?.add(&gate(&gates,0,n)?.sigmoid().mul(&gate(&gates,2,n)?.tanh())?)?;
    let h=gate(&gates,3,n)?.sigmoid().mul(&c.tanh())?;
    Ok((h,c))
}

fn invalid(msg: &str) -> Error { Error::InvalidShape { op: "recurrent", msg: msg.into() } }
fn check(t: &Tensor) -> Result<()> {
    if t.device() != Device::Cpu { return Err(Error::Unsupported { op: "recurrent", msg: "CPU only".into() }); }
    if t.dtype() != DType::F32 { return Err(Error::DtypeMismatch { op: "recurrent", expected: DType::F32, got: t.dtype() }); }
    Ok(())
}
fn affines(x: &Tensor, h: &Tensor, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>, gates: usize) -> Result<(Tensor, Tensor)> {
    for t in [x,h,wi,wh].into_iter().chain(bi).chain(bh) { check(t)?; }
    if x.shape().len()!=2 || h.shape().len()!=2 || x.shape()[0]!=h.shape()[0] || h.shape()[1]==0 { return Err(invalid("expected x [B,I], h [B,H], H > 0")); }
    let n=h.shape()[1].checked_mul(gates).ok_or_else(|| invalid("gate width overflow"))?;
    if wi.shape()!=[n,x.shape()[1]] || wh.shape()!=[n,h.shape()[1]] { return Err(invalid("weights must be [G*H,I] and [G*H,H]")); }
    for b in [bi,bh].into_iter().flatten() { if b.shape()!=[n] { return Err(invalid("bias must be [G*H]")); } }
    let mut a=x.matmul(&wi.transpose(0,1)?)?;
    let mut b=h.matmul(&wh.transpose(0,1)?)?;
    if let Some(bias)=bi { a=a.add(bias)?; }
    if let Some(bias)=bh { b=b.add(bias)?; }
    Ok((a,b))
}
/// tanh(x W_ih^T + b_ih + h W_hh^T + b_hh). Biases are independently optional.
pub fn rnn_cell(x: &Tensor, h: &Tensor, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>) -> Result<Tensor> {
    let (a,b)=affines(x,h,wi,wh,bi,bh,1)?;
    Ok(a.add(&b)?.tanh())
}
