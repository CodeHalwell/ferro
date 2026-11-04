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
    fn rows(&self, rows: &[usize]) -> Result<Self> {
        Ok(Self { h:self.h.index_select(0,rows)?, c:self.c.as_ref().map(|c| c.index_select(0,rows)).transpose()? })
    }
    fn zip(&self, other: &Self, f: impl Fn(&Tensor,&Tensor) -> Result<Tensor>) -> Result<Self> {
        Ok(Self { h:f(&self.h,&other.h)?, c:match (&self.c,&other.c) { (Some(a),Some(b)) => Some(f(a,b)?), _ => None } })
    }
}
pub struct UnrollOutput { pub outputs: Tensor, pub state: RecurrentState }
/// Sequential unroll of time-major [T,B,I], batched over the rows active at
/// each step: the closure receives [A,I] inputs and [A,H] states for the A
/// active rows (all B when none has ended) and must preserve state shape/kind.
/// Runs with a reset, or with truncation and ragged lengths, step each row
/// alone ([1,I] and [1,H]) so a discarded history never joins a shared graph.
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
    let elements=crate::shape::checked_numel("recurrent", &[time,batch,width])?;
    if elements>isize::MAX as usize/std::mem::size_of::<f32>() { return Err(invalid("output buffer byte length overflow")); }
    if time==0 || batch==0 { return Ok(UnrollOutput { outputs:Tensor::zeros(&[time,batch,width]), state:initial.clone() }); }
    let mut outputs=Vec::new();
    outputs.try_reserve_exact(time).map_err(|e| invalid(&format!("cannot reserve timestep outputs: {e}")))?;
    // A reset or a per-row truncation discards one row's history while its
    // batch-mates keep theirs. In a batched graph that row still receives a
    // zero cotangent, and 0 * a nonfinite local derivative poisons the shared
    // weights, so those runs keep separate per-row graphs.
    if reset.is_some_and(|r| r.contains(&true)) || (truncate.is_some() && lengths.iter().any(|&n| n!=time)) {
        return unroll_rows(input,initial,lengths,reset,truncate,cell);
    }
    let mut state=initial.clone();
    for time_index in 0..time {
        let active=(0..batch).filter(|&b| time_index<lengths[b]).collect::<Vec<_>>();
        if active.is_empty() { outputs.push(Tensor::zeros(&[batch,width])); continue; }
        if truncate.is_some_and(|k| time_index>0 && time_index%k==0) { state=state.detach(); }
        let full=active.len()==batch;
        let x=input.narrow(0,time_index,1)?.reshape(&[batch,input.shape()[2]])?;
        let (x,prev)=if full { (x,state.clone()) } else { (x.index_select(0,&active)?,state.rows(&active)?) };
        let next=cell(&x,&prev)?; next.validate()?;
        if next.h.shape()!=[active.len(),width] || next.c.is_some()!=initial.c.is_some() { return Err(invalid("cell changed state shape or kind")); }
        if full { outputs.push(next.h.clone()); state=next; continue; }
        let mut slot=vec![active.len();batch];
        for (p,&b) in active.iter().enumerate() { slot[b]=p; }
        outputs.push(Tensor::cat(&[next.h.clone(),Tensor::zeros(&[1,width])],0)?.index_select(0,&slot)?);
        let pick=(0..batch).map(|b| if slot[b]<active.len() { batch+slot[b] } else { b }).collect::<Vec<_>>();
        state=state.zip(&next,|a,b| Tensor::cat(&[a.clone(),b.clone()],0)?.index_select(0,&pick))?;
    }
    Ok(UnrollOutput { outputs:Tensor::cat(&outputs,0)?.reshape(&[time,batch,width])?, state })
}

/// Row-wise unroll with an independent graph per row; inputs are validated by `unroll`.
fn unroll_rows<F>(input: &Tensor, initial: &RecurrentState, lengths: &[usize], reset: Option<&[bool]>, truncate: Option<usize>, mut cell: F) -> Result<UnrollOutput>
where F: FnMut(&Tensor, &RecurrentState) -> Result<RecurrentState> {
    let (time,batch,width)=(input.shape()[0],input.shape()[1],initial.h.shape()[1]);
    let mut outputs=Vec::new();
    outputs.try_reserve_exact(time*batch).map_err(|e| invalid(&format!("cannot reserve timestep outputs: {e}")))?;
    let mut rows=(0..batch).map(|b| initial.rows(&[b])).collect::<Result<Vec<_>>>()?;
    for time_index in 0..time {
        for b in 0..batch {
            if time_index>=lengths[b] { outputs.push(Tensor::zeros(&[1,width])); continue; }
            if reset.is_some_and(|r| r[time_index*batch+b]) { rows[b]=initial.rows(&[b])?; }
            if truncate.is_some_and(|k| time_index>0 && time_index%k==0) { rows[b]=rows[b].detach(); }
            let x=input.narrow(0,time_index,1)?.reshape(&[batch,input.shape()[2]])?.index_select(0,&[b])?;
            let next=cell(&x,&rows[b])?; next.validate()?;
            if next.h.shape()!=[1,width] || next.c.is_some()!=initial.c.is_some() { return Err(invalid("cell changed state shape or kind")); }
            outputs.push(next.h.clone()); rows[b]=next;
        }
    }
    let h=Tensor::cat(&rows.iter().map(|s| s.h.clone()).collect::<Vec<_>>(),0)?;
    let c=if initial.c.is_some() { Some(Tensor::cat(&rows.iter().map(|s| s.c.as_ref().unwrap().clone()).collect::<Vec<_>>(),0)?) } else { None };
    Ok(UnrollOutput { outputs:Tensor::cat(&outputs,0)?.reshape(&[time,batch,width])?, state:RecurrentState { h,c } })
}

fn gate(t: &Tensor, g: usize, h: usize) -> Result<Tensor> { t.narrow(1,g*h,h)?.contiguous() }
/// Gate order [reset, update, new]. Reset is applied AFTER the recurrent
/// affine, including b_hh,new: n=tanh(a_new + r*b_new); h'=(1-z)*n+z*h.
pub fn gru_cell(x: &Tensor, h: &Tensor, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>) -> Result<Tensor> {
    let (a,b)=affines(x,h,wi,wh,bi,bh,3)?;
    gru_gates(&a,&b,h)
}
fn gru_gates(a: &Tensor, b: &Tensor, h: &Tensor) -> Result<Tensor> {
    let n=h.shape()[1];
    let r=gate(a,0,n)?.add(&gate(b,0,n)?)?.sigmoid();
    let z=gate(a,1,n)?.add(&gate(b,1,n)?)?.sigmoid();
    let candidate=gate(a,2,n)?.add(&r.mul(&gate(b,2,n)?)?)?.tanh();
    Tensor::ones(z.shape()).sub(&z)?.mul(&candidate)?.add(&z.mul(h)?)
}
/// Gate order [input, forget, candidate, output]. No implicit forget bias.
/// Returns (hidden, cell), both [B,H].
pub fn lstm_cell(x: &Tensor, h: &Tensor, c: &Tensor, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    check(c)?;
    if c.shape()!=h.shape() { return Err(invalid("cell and hidden shapes differ")); }
    let (a,b)=affines(x,h,wi,wh,bi,bh,4)?;
    lstm_gates(&a,&b,c)
}
fn lstm_gates(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
    let gates=a.add(b)?; let n=c.shape()[1];
    let c=gate(&gates,1,n)?.sigmoid().mul(c)?.add(&gate(&gates,0,n)?.sigmoid().mul(&gate(&gates,2,n)?.tanh())?)?;
    let h=gate(&gates,3,n)?.sigmoid().mul(&c.tanh())?;
    Ok((h,c))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellKind { Rnn, Gru, Lstm }

/// `unroll` over a built-in tanh RNN, GRU or LSTM cell. W_ih is applied to every
/// timestep in one GEMM before the loop and W_hh is transposed once, so each
/// step costs one [B,H]x[H,G*H] product. Padded positions are zeroed before
/// the projection, which keeps NaN padding out of the weight gradient.
#[allow(clippy::too_many_arguments)]
pub fn unroll_cell(kind: CellKind, input: &Tensor, initial: &RecurrentState, wi: &Tensor, wh: &Tensor, bi: Option<&Tensor>, bh: Option<&Tensor>, lengths: &[usize], reset: Option<&[bool]>, truncate: Option<usize>) -> Result<UnrollOutput> {
    for t in [input,wi,wh].into_iter().chain(bi).chain(bh) { check(t)?; }
    initial.validate()?;
    if (kind==CellKind::Lstm)!=initial.c.is_some() { return Err(invalid("LSTM needs cell state; RNN/GRU must not carry one")); }
    if input.shape().len()!=3 { return Err(invalid("input must be [T,B,I]")); }
    let (time,batch,features)=(input.shape()[0],input.shape()[1],input.shape()[2]);
    let width=initial.h.shape()[1];
    let gates=width.checked_mul(match kind { CellKind::Rnn => 1, CellKind::Gru => 3, CellKind::Lstm => 4 }).ok_or_else(|| invalid("gate width overflow"))?;
    if wi.shape()!=[gates,features] || wh.shape()!=[gates,width] { return Err(invalid("weights must be [G*H,I] and [G*H,H]")); }
    if [bi,bh].into_iter().flatten().any(|b| b.shape()!=[gates]) { return Err(invalid("bias must be [G*H]")); }
    if lengths.len()!=batch || lengths.iter().any(|&n| n>time) { return Err(invalid("batch/lengths mismatch or length exceeds T")); }
    let x=if lengths.iter().all(|&n| n==time) { input.clone() } else {
        let keep=(0..time).flat_map(|t| lengths.iter().map(move |&n| t<n)).collect();
        Tensor::where_cond(&Tensor::from_vec_bool(keep,&[time,batch,1])?,input,&Tensor::zeros(&[1]))?
    };
    let mut projected=x.reshape(&[time*batch,features])?.matmul(&wi.transpose(0,1)?.contiguous()?)?;
    if let Some(b)=bi { projected=projected.add(b)?; }
    let projected=projected.reshape(&[time,batch,gates])?;
    let recurrent=wh.transpose(0,1)?.contiguous()?;
    // Bias rows are identical, so any A of them serve A active rows without a per-step broadcast.
    let bias=bh.map(|b| b.expand(&[batch,gates])?.contiguous()).transpose()?;
    unroll(&projected,initial,lengths,reset,truncate,|a,s| {
        let mut b=s.h.matmul(&recurrent)?;
        if let Some(bias)=&bias { b=b.add(&bias.narrow(0,0,b.shape()[0])?)?; }
        Ok(match kind {
            CellKind::Rnn => RecurrentState { h:a.add(&b)?.tanh(), c:None },
            CellKind::Gru => RecurrentState { h:gru_gates(a,&b,&s.h)?, c:None },
            CellKind::Lstm => { let (h,c)=lstm_gates(a,&b,s.c.as_ref().expect("validated LSTM state"))?; RecurrentState { h,c:Some(c) } }
        })
    })
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
