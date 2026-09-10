"""Full-model and isolated component wall timings; not a GPU kernel attribution.

Run twice, second with --reverse. Components use actual full-transformer
intermediates. Their fence-inclusive latencies MUST NOT be summed or subtracted
from asynchronous full-model latency. No production instrumentation is added.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import ferro
import model_graphs as model


def measure(run, sync, warmup, iters):
    for _ in range(warmup):
        out = run()
    sync()
    total, call, fence = [], [], []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter_ns()
        out = run()
        t1 = time.perf_counter_ns()
        sync()
        t2 = time.perf_counter_ns()
        total.append((t2 - t0) / 1000)
        call.append((t1 - t0) / 1000)
        fence.append((t2 - t1) / 1000)
        del out
    return {key: dict(samples_us=xs, median_us=statistics.median(xs), min_us=min(xs),
                      max_us=max(xs), p25_us=sorted(xs)[len(xs)//4], p75_us=sorted(xs)[3*len(xs)//4])
            for key, xs in [('total', total), ('call_wall', call), ('final_fence', fence)]}


def components(case):
    """Torch intermediates are shared mathematical fixtures for both backends."""
    w, x = case['weights'], case['x']
    n, d, h = case['tokens'], case['width'], case['heads']
    z = F.layer_norm(x, (d,), w['ln1g'], w['ln1b'], eps=1e-5)
    projections = [z @ w['w'+k] + w['b'+k] for k in ('q', 'k', 'v')]
    q, k, v = [t.reshape(n, h, d//h).transpose(0, 1).contiguous() for t in projections]
    kt = k.transpose(1, 2).contiguous()
    scores = (q @ kt) * ((d//h)**-0.5)
    probs = scores.softmax(-1)
    attn = probs @ v
    flat = attn.transpose(0, 1).contiguous().reshape(n, d)
    y = x + flat @ w['wo'] + w['bo']
    ln2 = F.layer_norm(y, (d,), w['ln2g'], w['ln2b'], eps=1e-5)
    hidden = F.gelu(ln2 @ w['w1'] + w['b1'], approximate='tanh')
    specs = []
    def add(name, args, tf, ff):
        specs.append((name, args, tf, ff))
    for key, inp in [('ln1', x), ('ln2', y)]:
        add(key, [inp, w[key+'g'], w[key+'b']],
            lambda a,b,c: F.layer_norm(a, (d,), b,c,eps=1e-5), lambda a,b,c: a.layer_norm(b,c))
    for key in ('q','k','v'):
        add('projection_'+key, [z,w['w'+key],w['b'+key]], lambda a,b,c:a@b+c, lambda a,b,c:a.matmul(b)+c)
    add('layout_qkv_one', [projections[0]],
        lambda a:a.reshape(n,h,d//h).transpose(0,1).contiguous(),
        lambda a:a.reshape([n,h,d//h]).transpose(0,1).reshape([h,n,d//h]))
    add('layout_key_transpose', [k], lambda a:a.transpose(1,2).contiguous(),
        lambda a:a.transpose(1,2).reshape([h,d//h,n]))
    add('attention_qk_bmm', [q,kt], lambda a,b:a@b, lambda a,b:a.bmm(b))
    add('attention_softmax', [scores], lambda a:a.softmax(-1), lambda a:a.softmax(-1))
    add('attention_pv_bmm', [probs,v], lambda a,b:a@b, lambda a,b:a.bmm(b))
    add('layout_attention_merge', [attn], lambda a:a.transpose(0,1).contiguous().reshape(n,d),
        lambda a:a.transpose(0,1).reshape([n,d]))
    add('output_projection_residual', [x,flat,w['wo'],w['bo']], lambda a,b,c,e:a+b@c+e,
        lambda a,b,c,e:a+b.matmul(c)+e)
    add('mlp_up_bias_gelu', [ln2,w['w1'],w['b1']],lambda a,b,c:F.gelu(a@b+c,approximate='tanh'),
        lambda a,b,c:(a.matmul(b)+c).gelu())
    add('mlp_down_bias_residual', [y,hidden,w['w2'],w['b2']],lambda a,b,c,e:a+b@c+e,
        lambda a,b,c,e:a+b.matmul(c)+e)
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--warmup', type=int, default=15)
    args = ap.parse_args()
    if min(args.tokens,args.width,args.heads,args.iters)<=0 or args.width%args.heads or args.warmup<0:
        ap.error('invalid dimensions/iterations')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert ferro.cuda_init(0)
    report = dict(platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), dtype='float32', tf32=False,
                  args={**vars(args), 'json':str(args.json)}, rows=[],
                  profiler_activities=[str(x) for x in torch.profiler.supported_activities()],
                  caveat='call_wall includes host work, possible blocking CUDA calls and Python overhead; not pure scheduling. Components are isolated, not additive.')
    def save(row):
        report['rows'].append(row)
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in row.items() if k!='timing'} | ({'median_us': row['timing']['total']['median_us']} if 'timing' in row else {})),flush=True)
    order = ['torch_eager','ferro_eager','ferro_compiled']
    if args.reverse:
        order.reverse()
    with torch.inference_mode():
        case = model.make_case('transformer',args.tokens,args.width,args.heads,18)
        expected = model.torch_forward(case,case['x'])
        for backend in order + ['torch_compile']:
            start=time.perf_counter_ns()
            try:
                path = model.prepare_torch(case,backend=='torch_compile') if backend.startswith('torch') else (model.prepare_ferro_eager if backend=='ferro_eager' else model.prepare_ferro_compiled)(case,ferro)
                errors = {'original':model.parity(path,expected,2e-4,2e-5)}
                changed=case['x']*.73+.19
                path['bind'](changed)
                errors['changed_input']=model.parity(path,model.torch_forward(case,changed),2e-4,2e-5)
                path['bind'](case['x'])
                errors['restored_input']=model.parity(path,expected,2e-4,2e-5)
                original=case['weights']['wq'].clone()
                case['weights']['wq'].mul_(.81)
                if backend.startswith('ferro'):
                    path['keepalive'][0]['wq'].copy_(ferro.from_dlpack(case['weights']['wq']))
                errors['changed_weight']=model.parity(path,model.torch_forward(case,case['x']),2e-4,2e-5)
                case['weights']['wq'].copy_(original)
                if backend.startswith('ferro'):
                    path['keepalive'][0]['wq'].copy_(ferro.from_dlpack(original))
                errors['restored_weight']=model.parity(path,expected,2e-4,2e-5)
                setup=(time.perf_counter_ns()-start)/1e6
                save(dict(scope='full_model',backend=backend,status='passed',max_abs_errors=errors,
                          setup_validation_ms=setup,structure=path.get('structure'),
                          timing=measure(path['run'],path['sync'],args.warmup,args.iters)))
            except Exception as exc:
                save(dict(scope='full_model',backend=backend,status='blocked',error=str(exc)))
                if backend!='torch_compile':
                    raise
        for name, inputs, tf, ff in components(case):
            expected=tf(*inputs)
            finputs=[ferro.from_dlpack(x.contiguous()) for x in inputs]
            eager=lambda ff=ff,fin= finputs:ff(*fin)
            root=ferro.capture(eager)
            graph=root.compile_fused()
            paths={'torch_eager':(lambda tf=tf,inputs=inputs:tf(*inputs),torch.cuda.synchronize,lambda y:y),
                   'ferro_eager':(eager,ferro.cuda_synchronize,torch.from_dlpack),
                   'ferro_compiled':(graph.replay,ferro.cuda_synchronize,torch.from_dlpack)}
            for backend in order:
                run,sync,export=paths[backend]
                error=model.parity(dict(run=run,sync=sync,export=export),expected,2e-4,2e-5)
                save(dict(scope='isolated_component',component=name,backend=backend,status='passed',
                          max_abs_error=error,structure=dict(operations=graph.num_steps,runs=graph.num_runs) if backend=='ferro_compiled' else None,
                          timing=measure(run,sync,args.warmup,args.iters)))
        save(dict(scope='empty_fence_control',backend='ferro',timing=measure(lambda:None,ferro.cuda_synchronize,args.warmup,args.iters)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
