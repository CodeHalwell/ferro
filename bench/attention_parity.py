"""Composed rank-3 attention baseline, not Rust SDPA or Torch SDPA parity.

Score normalization includes scale, optional additive upper-left -1e9 mask,
and softmax. Full attention also includes K transpose/materialization, QK and PV.
Component wall times are non-additive; no device timeline or traffic claim.
"""
import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
import torch
import ferro

SHAPES=[('short',2,32,32,32,24),('long',2,512,512,64,48),
        ('rectangular',2,96,192,64,40),('rectangular_tall',2,192,96,32,56)]
MODES=['torch_eager','torch_inductor','ferro_eager','ferro_compiled','ferro_static_snapshot']
RTOL=2e-4
ATOL=2e-5

def measure(run,sync,warmup,iters):
    for _ in range(warmup):
        out=run()
        del out
    sync()
    samples=[]
    for _ in range(iters):
        sync()
        start=time.perf_counter_ns()
        out=run()
        sync()
        samples.append((time.perf_counter_ns()-start)/1000)
        del out
    return dict(samples_us=samples,median_us=statistics.median(samples),min_us=min(samples),max_us=max(samples))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json',type=Path,required=True)
    ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--iters',type=int,default=40)
    ap.add_argument('--warmup',type=int,default=10)
    args=ap.parse_args()
    assert args.iters>0 and args.warmup>=0 and not args.json.exists()
    assert torch.cuda.is_available() and ferro.cuda_init(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch._dynamo.config.suppress_errors=False
    modes=MODES[::-1] if args.reverse else MODES
    report=dict(shapes=SHAPES,modes=modes,rtol=RTOL,atol=ATOL,seed=180,
                dtype='float32',tf32=False,inference='Torch inference_mode; Ferro detached leaves',
                mask='upper-left additive -1e9; deliberately matching composed legacy finite-mask semantics, not torch SDPA',
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                binding_sha256=hashlib.sha256(Path(ferro.ferro.__file__).read_bytes()).hexdigest(),
                warmup=args.warmup,iters=args.iters,rows=[])
    failed=False
    with torch.inference_mode():
        for name,b,n,m,d,dv in SHAPES:
            gen=torch.Generator(device='cuda').manual_seed(180)
            q=torch.randn((b,n,d),device='cuda',generator=gen)
            k=torch.randn((b,m,d),device='cuda',generator=gen)
            v=torch.randn((b,m,dv),device='cuda',generator=gen)
            raw=q@k.transpose(1,2)
            scale=d**-0.5
            for causal in (False,True):
                mask=torch.triu(torch.full((b,n,m),-1e9,device='cuda'),diagonal=1) if causal else None
                def norm(s):
                    s=s*scale
                    return torch.softmax(s+mask if causal else s,dim=-1)
                for component in ('normalization','attention'):
                    tq=q.clone(); tk=k.clone(); tv=v.clone(); ts=raw.clone()
                    def reference():
                        return norm(ts) if component=='normalization' else norm(tq@tk.transpose(1,2))@tv
                    expected=reference().clone()
                    for mode in modes:
                        row=dict(shape=name,dimensions=[b,n,m,d,dv],causal=causal,component=component,mode=mode)
                        graph=None; compiled=None; root=None
                        try:
                            if mode.startswith('torch'):
                                run=reference
                                if mode=='torch_inductor':
                                    # Independent fixtures must not share Dynamo closure guards.
                                    torch._dynamo.reset()
                                    run=torch.compile(run,backend='inductor',fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
                                sync=torch.cuda.synchronize
                                export=lambda y:y
                                mutate=lambda: (tq.copy_(q*.73+.19),tv.copy_(v*.87+.013),ts.copy_(raw*.73+.19))
                                restore=lambda: (tq.copy_(q),tv.copy_(v),ts.copy_(raw))
                            else:
                                fq,fk,fv,fs=[ferro.from_dlpack(x) for x in (q,k,v,raw)]
                                scalar=ferro.Tensor([scale],[]).to('cuda')
                                fm=ferro.from_dlpack(mask) if causal else None
                                assert not any(x.requires_grad for x in (fq,fk,fv,fs,scalar))
                                def fn():
                                    s=fs if component=='normalization' else fq.bmm(fk.transpose(1,2).reshape([b,d,m]))
                                    s=s*scalar
                                    if causal: s=s+fm
                                    p=s.softmax(-1)
                                    return p if component=='normalization' else p.bmm(fv)
                                run=fn
                                if mode!='ferro_eager':
                                    root=ferro.capture(fn)
                                    compiled=root.compile_fused()
                                    row['structure']=dict(operations=compiled.num_steps,runs=compiled.num_runs,leaves=compiled.num_operands)
                                    run=compiled.replay
                                    if mode=='ferro_static_snapshot':
                                        graph=compiled.prepare_static()
                                        def run():
                                            graph.replay()
                                            return graph.snapshot()
                                sync=ferro.cuda_synchronize
                                export=torch.from_dlpack
                                def mutate():
                                    fq.copy_(ferro.from_dlpack(q*.73+.19))
                                    fv.copy_(ferro.from_dlpack(v*.87+.013))
                                    fs.copy_(ferro.from_dlpack(raw*.73+.19))
                                    tq.copy_(q*.73+.19);tv.copy_(v*.87+.013);ts.copy_(raw*.73+.19)
                                def restore():
                                    fq.copy_(ferro.from_dlpack(q));fv.copy_(ferro.from_dlpack(v));fs.copy_(ferro.from_dlpack(raw))
                                    tq.copy_(q);tv.copy_(v);ts.copy_(raw)
                            gates=[]
                            for phase in ('original','changed_q_v_or_scores','restored'):
                                if phase=='changed_q_v_or_scores': mutate()
                                if phase=='restored': restore()
                                oracle=reference()
                                if phase=='changed_q_v_or_scores': assert not torch.allclose(oracle,expected,rtol=RTOL,atol=ATOL)
                                out=run();sync();actual=export(out);torch.cuda.synchronize()
                                assert actual.is_cuda and actual.dtype==torch.float32 and not actual.requires_grad
                                torch.testing.assert_close(actual,oracle,rtol=RTOL,atol=ATOL)
                                diff=(actual.double()-oracle.double()).abs()
                                ratio=(diff/(ATOL+RTOL*oracle.double().abs())).max().item()
                                assert torch.isfinite(actual).all() and ratio<=1
                                gates.append(dict(phase=phase,max_abs_error=diff.max().item(),max_tolerance_ratio=ratio))
                                del actual,out
                            row['parity']=gates
                            row.update(measure(run,sync,args.warmup,args.iters),status='passed')
                        except Exception as exc:
                            missing=re.search(r"Cannot find a working triton installation\b|No module named ['\"]triton['\"]",str(exc))
                            status='blocked' if mode=='torch_inductor' and missing else 'failed'
                            row.update(status=status,error_type=type(exc).__name__,error=str(exc))
                            failed=failed or status=='failed'
                        finally:
                            tq.copy_(q);tv.copy_(v);ts.copy_(raw)
                            torch.cuda.synchronize();ferro.cuda_synchronize()
                        report['rows'].append(row)
                        args.json.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                        print(json.dumps({key:value for key,value in row.items() if key!='samples_us'}),flush=True)
    assert len(report['rows'])==len(SHAPES)*2*2*len(MODES)
    return int(failed)

if __name__=='__main__':
    raise SystemExit(main())
