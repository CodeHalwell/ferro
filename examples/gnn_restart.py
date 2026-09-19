"""One-hop message-passing CPU f32 learning and fresh-process restart proof.

Fixed gates BEFORE execution: seeds 7/11/43, 400 full-batch steps, held-out MSE
<=25% of training-mean baseline; Torch gradient atol=1e-5, rtol=1e-4.
32 disconnected four-node graphs; targets depend on each node and its neighbor.
Topology is immutable model config; plans are rebuilt, not checkpointed buffers.
No CUDA training/restart or data-loader cursor claim.
"""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import ferro as fr
from training_restart import SEEDS, STEPS, snapshot, state_digest

ROWS=(0,0,1,1,2,2,3,3)
COLS=(0,3,1,0,2,1,3,2)

class GraphModel(fr.nn.Module):
    seed: int
    graphs: int = 32
    rows: tuple = ROWS
    cols: tuple = COLS
    hidden: int = 16
    def build(self):
        graph=fr.graph.COO(4,4,list(self.rows),list(self.cols))
        batch, self.offsets=fr.graph.COO.batch([graph]*self.graphs)
        self.plan=batch.prepare('cpu')
        self.register_buffer('edge_weights',fr.Tensor([0.5]*batch.nnz,[batch.nnz]))
        self.w1=fr.nn.Parameter(fr.Tensor.randn([1,self.hidden],seed=self.seed))
        self.b1=fr.nn.Parameter(fr.Tensor.zeros([self.hidden]))
        self.w2=fr.nn.Parameter(fr.Tensor.randn([self.hidden,1],seed=self.seed+1)*self.hidden**-0.5)
        self.b2=fr.nn.Parameter(fr.Tensor.zeros([1]))
    def forward(self,x):
        neighbors=self.plan.spmm(self.edge_weights,x)
        return (neighbors @ self.w1.tensor()+self.b1.tensor()).tanh() @ self.w2.tensor()+self.b2.tensor()

def dataset(held_out=False,graphs=32):
    rng=random.Random(2027 if held_out else 2026)
    x=[rng.uniform(-1,1) for _ in range(graphs*4)]
    # Independent scalar target, no sparse engine or model implementation involved.
    means=[(x[i]+x[(i//4)*4+(i-1)%4])*0.5 for i in range(len(x))]
    y=[math.sin(1.5*v)+0.2*v*v for v in means]
    return fr.Tensor(x,[len(x),1]),fr.Tensor(y,[len(y),1]),y

def gradient_check(seed):
    import torch
    torch.set_num_threads(1)
    model=GraphModel(seed=seed,graphs=2)
    x,y,_=dataset(graphs=2); x=x.requires_grad_(True)
    tx=torch.tensor(x.tolist(),requires_grad=True)
    weights={name:torch.tensor(p.tensor().tolist(),requires_grad=True) for name,p in model.named_parameters()}
    adj=torch.zeros(8,8)
    for g in range(2):
        for r,c in zip(ROWS,COLS): adj[g*4+r,g*4+c]+=0.5
    expected=torch.tanh((adj @ tx) @ weights['w1']+weights['b1']) @ weights['w2']+weights['b2']
    reference=torch.nn.functional.mse_loss(expected,torch.tensor(y.tolist()))
    loss=fr.nn.functional.mse_loss(model(x),y)
    torch.testing.assert_close(torch.tensor(loss.item()),reference.detach(),atol=1e-5,rtol=1e-4)
    loss.backward(); reference.backward()
    pairs=[('input',x.grad,tx.grad)]+[(n,p.tensor().grad,weights[n].grad) for n,p in model.named_parameters()]
    errors={}
    for name,actual,expected in pairs:
        got=torch.tensor(actual.tolist())
        torch.testing.assert_close(got,expected,atol=1e-5,rtol=1e-4)
        errors[name]=float((got-expected).abs().max())
    return errors

def train(seed,resume=None,midpoint=None):
    model=GraphModel(seed=seed)
    rng=fr.Generator(seed+100 if resume is None else seed+999)
    opt=fr.optim.AdamW(model.parameters(),lr=0.025 if resume is None else 0.7,weight_decay=0.001)
    start=0 if resume is None else fr.checkpoint.load_training(resume,model,optimizers={'main':opt},generator=rng)
    x,y,_=dataset(); losses=[]
    for step in range(start,STEPS):
        opt.zero_grad()
        noisy=x+fr.Tensor.randn(x.shape,generator=rng)*0.005
        loss=fr.nn.functional.mse_loss(model(noisy),y)
        value=loss.item()
        assert math.isfinite(value)
        losses.append(value); loss.backward(); opt.step(); opt.zero_grad()
        if midpoint is not None and step+1==STEPS//2: snapshot(model,opt,rng,step+1).save(midpoint)
    state=snapshot(model,opt,rng,STEPS)
    with fr.no_grad():
        hx,hy,labels=dataset(True)
        mse=fr.nn.functional.mse_loss(model(hx),hy).item()
        train_labels=dataset()[2]; mean=sum(train_labels)/len(train_labels)
        baseline=sum((v-mean)**2 for v in labels)/len(labels)
        # Ablate neighbor aggregation using the same learned readout.
        own=(hx @ model.w1.tensor()+model.b1.tensor()).tanh() @ model.w2.tensor()+model.b2.tensor()
        ablated=fr.nn.functional.mse_loss(own,hy).item()
    return state,{'losses':losses,'metrics':{'held_out_mse':mse,'constant_baseline_mse':baseline,'ratio':mse/baseline,'no_message_passing_mse':ablated},'next_draws':fr.Tensor.randn([8],generator=rng).tolist()}

def reject_changed_topology(checkpoint,seed,directory):
    changed=list(COLS); changed[1]=2
    target=GraphModel(seed=seed,cols=tuple(changed)); rng=fr.Generator(999)
    opt=fr.optim.AdamW(target.parameters(),lr=0.7,weight_decay=0.001)
    before=state_digest(snapshot(target,opt,rng,0),directory/'before')
    try: fr.checkpoint.load_training(checkpoint,target,optimizers={'main':opt},generator=rng)
    except ValueError: pass
    else: raise AssertionError('Changed graph topology accepted')
    after=state_digest(snapshot(target,opt,rng,0),directory/'after')
    assert before==after,'Rejected topology restore mutated state'

def prove(seed):
    gradients=gradient_check(seed)
    with tempfile.TemporaryDirectory(prefix='ferro-gnn-') as tmp:
        directory=Path(tmp); midpoint=directory/'midpoint'
        state,expected=train(seed,midpoint=midpoint)
        expected_state=state_digest(state,directory/'expected')
        reject_changed_topology(midpoint,seed,directory)
        output=directory/'resumed.json'
        subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('train_restart_gnn.py')),'--seed',str(seed),'--resume',str(midpoint),'--worker-output',str(output)],check=True,timeout=120)
        resumed=json.loads(output.read_text())
        assert resumed['losses']==expected['losses'][STEPS//2:],'Restart loss trajectory differs'
        assert resumed['state']==expected_state,'Restart serialized state differs'
        assert resumed['metrics']==expected['metrics'],'Restart metrics differ'
        assert resumed['next_draws']==expected['next_draws'],'Restart RNG differs'
    assert expected['metrics']['ratio']<=0.25,expected['metrics']
    assert expected['metrics']['no_message_passing_mse']>expected['metrics']['held_out_mse'],'Fixture does not require graph messages'
    return {'task':'gnn','seed':seed,'steps':STEPS,'metrics':expected['metrics'],'gradient_max_abs_errors':gradients,'restart':'bit-exact','changed_topology_restore':'rejected without mutation','initial_loss':expected['losses'][0],'final_loss':expected['losses'][-1]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--worker-output',type=Path)
    args=parser.parse_args()
    if args.worker_output is not None:
        if args.resume is None or args.seed is None: parser.error('worker requires resume and seed')
        state,result=train(args.seed,resume=args.resume)
        result['state']=state_digest(state,args.worker_output.with_suffix('.state'))
        args.worker_output.write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    else:
        if args.resume is not None: parser.error('resume is reserved for worker')
        for seed in SEEDS if args.seed is None else (args.seed,): print(json.dumps(prove(seed),sort_keys=True,allow_nan=False),flush=True)
