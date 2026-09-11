"""Full-model CUDA graph benchmarks; run only after serialized integration."""
import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
import ferro
import torch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('models',ROOT/'bench/model_graphs.py')
models=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=models
spec.loader.exec_module(models)
MODES=['torch_eager','torch_inductor_no_graphs','torch_reduce_overhead','ferro_eager','ferro_compiled','ferro_static','ferro_static_snapshot']

def prepare(case,mode):
    if mode.startswith('torch'):
        state={'x':case['x']}
        fn=lambda x:models.torch_forward(case,x)
        if mode=='torch_inductor_no_graphs':
            fn=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        if mode=='torch_reduce_overhead':
            fn=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False,mode='reduce-overhead')
        return dict(run=lambda:fn(state['x']),bind=lambda x:state['x'].copy_(x),weight=lambda w:case['weights']['w1'].copy_(w),sync=torch.cuda.synchronize,export=lambda x:x)
    path=models.prepare_ferro_eager(case,ferro)
    weights,state=path['keepalive']
    path['weight']=lambda w:weights['w1'].copy_(ferro.from_dlpack(w))
    if mode=='ferro_eager': return path
    root=ferro.capture(path['run'])
    compiled=root.compile_fused()
    assert compiled.num_steps=={'mlp':5,'residual_mlp':6,'transformer':34}[case['name']]
    path['structure']=dict(operations=compiled.num_steps,runs=compiled.num_runs,leaves=compiled.num_operands)
    if mode=='ferro_compiled':
        path['run']=compiled.replay
        return path
    graph=compiled.prepare_static()
    path['graph']=graph
    path['run']=graph.replay
    path['export']=lambda _:torch.from_dlpack(graph.snapshot())
    if mode=='ferro_static_snapshot':
        def run():
            graph.replay()
            return graph.snapshot()
        path['run']=run
        path['export']=lambda y:torch.from_dlpack(y)
    return path

def parity(path,expected):
    out=path['run']()
    path['sync']()
    actual=path['export'](out)
    torch.cuda.synchronize()
    assert actual.is_cuda and actual.dtype==torch.float32 and not actual.requires_grad
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
    return (actual-expected).abs().max().item()

def measure(path):
    for _ in range(10):
        out=path['run']()
        del out
    path['sync']()
    samples=[]
    for _ in range(40):
        path['sync']()
        start=time.perf_counter_ns()
        out=path['run']()
        path['sync']()
        samples.append((time.perf_counter_ns()-start)/1000)
        del out
    return dict(samples_us=samples,median_us=statistics.median(samples),min_us=min(samples),max_us=max(samples))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--json',required=True,type=Path)
    args=ap.parse_args()
    assert torch.cuda.is_available() and ferro.cuda_init(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch._dynamo.config.suppress_errors=False
    modes=list(reversed(MODES)) if args.reverse else MODES
    report=dict(platform=platform.platform(),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,dtype='float32',tf32=False,modes=modes,tokens=128,width=256,heads=8,warmup=10,samples=40,rows=[])
    with torch.inference_mode():
        for name in ['mlp','residual_mlp','transformer']:
            started=time.perf_counter_ns()
            case=models.make_case(name,128,256,8,18)
            original_x=case['x'].clone()
            original_w=case['weights']['w1'].clone()
            changed_x=original_x*0.73+0.19
            changed_w=original_w*0.87+0.013
            torch.cuda.synchronize()
            fixture_ms=(time.perf_counter_ns()-started)/1e6
            for mode in modes:
                path=None
                case['x'].copy_(original_x)
                case['weights']['w1'].copy_(original_w)
                torch.cuda.synchronize()
                row=dict(case=name,mode=mode,fixture_setup_ms=fixture_ms,output_semantics='reusable private output; copy-out excluded' if mode=='ferro_static' else 'independent result; allocation/copy included')
                start=time.perf_counter_ns()
                try:
                    path=prepare(case,mode)
                    prepared=time.perf_counter_ns()
                    out=path['run']()
                    path['sync']()
                    del out
                    row['prepare_ms']=(prepared-start)/1e6
                    row['first_call_ms']=(time.perf_counter_ns()-prepared)/1e6
                    errors=[]
                    baseline=models.torch_forward(case,case['x'])
                    errors.append(parity(path,baseline))
                    case['x'].copy_(changed_x)
                    path['bind'](changed_x)
                    expected=models.torch_forward(case,case['x'])
                    assert not torch.allclose(baseline,expected,rtol=2e-4,atol=2e-5)
                    errors.append(parity(path,expected))
                    case['weights']['w1'].copy_(changed_w)
                    path['weight'](changed_w)
                    changed_expected=models.torch_forward(case,case['x'])
                    assert not torch.allclose(expected,changed_expected,rtol=2e-4,atol=2e-5)
                    errors.append(parity(path,changed_expected))
                    case['x'].copy_(original_x)
                    case['weights']['w1'].copy_(original_w)
                    path['bind'](original_x)
                    path['weight'](original_w)
                    errors.append(parity(path,baseline))
                    row['freshness_max_abs_errors']=errors
                    if 'structure' in path: row['structure']=path['structure']
                    row.update(measure(path),status='passed')
                    if 'graph' in path: row['graph_replays']=path['graph'].replay_count
                except Exception as exc:
                    row.update(status='blocked' if mode.startswith('torch_') and mode!='torch_eager' else 'failed',error_type=type(exc).__name__,error=str(exc),attempt_ms=(time.perf_counter_ns()-start)/1e6)
                finally:
                    if path is not None:
                        path['sync']()
                        del path
                report['rows'].append(row)
                args.json.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
                print(json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
    assert len(report['rows'])==len(modes)*3
    assert not any(r['status']=='failed' for r in report['rows'])

if __name__=='__main__': main()
