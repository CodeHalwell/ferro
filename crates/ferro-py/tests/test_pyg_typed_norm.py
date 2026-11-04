"""Actual PyG grouping, live state, detached moments and typed checkpoint gates."""
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch_geometric.nn as pyg
import ferro as fr
from ferro.training import Trainer
from ferro.graph import InstanceNorm,HeteroBatchNorm,HeteroLayerNorm


def native(tensor,grad=False):
    return fr.tensor(tensor.detach().reshape(-1).tolist(),dtype=str(tensor.dtype).split('.')[-1],requires_grad=grad).reshape(list(tensor.shape))


def inputs(rows=6,channels=3):
    raw=torch.linspace(-.7,1.2,rows*channels).reshape(channels,rows)
    return native(raw,True).T,raw.T.detach().requires_grad_()


def models(name,*args,**kwargs):
    a,b=globals()[name](*args,**kwargs),getattr(pyg,name)(*args,**kwargs)
    with fr.no_grad(),torch.no_grad():
        for key,p in b.named_parameters():
            value=torch.linspace(.3,1.3,p.numel()).reshape(p.shape) if key=='weight' else torch.linspace(-.4,.2,p.numel()).reshape(p.shape)
            p.copy_(value);dict(a.named_parameters())[key].tensor().copy_(native(value))
    return a,b


class TypedNormTests(unittest.TestCase):
    def close(self,a,b):
        if a is None or b is None:self.assertIs(a,b);return
        actual=torch.tensor(a.tolist(),dtype=getattr(torch,a.dtype)).reshape(a.shape)
        torch.testing.assert_close(actual,b.detach(),rtol=2e-4,atol=3e-5,equal_nan=True)

    def state(self,a,b):
        aa,bb=dict(a.named_buffers()),dict(b.named_buffers());self.assertEqual(aa.keys(),bb.keys())
        for key in aa:
            self.close(aa[key],bb[key]);self.assertEqual(aa[key].requires_grad,bb[key].requires_grad,key)

    def gradients(self,out,wanted,a,b,x,tx):
        weight=torch.linspace(-.4,.9,wanted.numel()).reshape(wanted.shape)
        aa=[x]+[p.tensor()for p in a.parameters()];bb=[tx]+list(b.parameters())
        actual=fr.autograd.grad(out,aa,native(weight));expected=torch.autograd.grad(wanted,bb,weight)
        for left,right in zip(actual,expected):self.close(left,right)

    def test_instance_graph_and_no_batch_values_all_gradients_modes_and_running_state(self):
        for affine,tracking,training,batch in itertools.product((False,True),(False,True),(False,True),(None,[0,2,0,2,2,0])):
            a,b=models('InstanceNorm',3,affine=affine,track_running_stats=tracking,momentum=.2)
            a.train(training);b.train(training);x,tx=inputs()
            kwargs={} if batch is None else dict(batch_size=4)
            ids=None if batch is None else fr.tensor(batch,dtype='int64');tids=None if batch is None else torch.tensor(batch)
            out,wanted=a(x,ids,**kwargs),b(tx,tids,**kwargs);self.close(out,wanted);self.state(a,b);self.gradients(out,wanted,a,b,x,tx)

    def test_instance_live_running_graph_cross_call_gradients_and_reset_boundaries(self):
        a,b=models('InstanceNorm',3,affine=True,track_running_stats=True)
        x,tx=inputs();ids=fr.tensor([0,2,0,2,2,0],dtype='int64');tids=torch.tensor([0,2,0,2,2,0])
        a(x,ids,batch_size=4);b(tx,tids,batch_size=4);self.state(a,b)
        self.assertTrue(a.running_mean.requires_grad);self.assertTrue(a.running_var.requires_grad)
        a.eval();b.eval();z,tz=inputs(2)
        out,wanted=a(z,fr.tensor([0,1],dtype='int64'),2),b(tz,torch.tensor([0,1]),2)
        ga=fr.autograd.grad(out.sum(),[x,z]+[p.tensor()for p in a.parameters()]);gb=torch.autograd.grad(wanted.sum(),[tx,tz]+list(b.parameters()))
        for left,right in zip(ga,gb):self.close(left,right)
        self.assertGreater(sum(abs(v)for row in ga[0].tolist()for v in row),0.)
        before=a.running_mean.tolist()
        with self.assertRaisesRegex(NotImplementedError,'no_grad'):a.reset_running_stats()
        self.assertEqual(a.running_mean.tolist(),before)
        # Compare the supported no_grad reset path, which retains live history in PyG.
        old_a,old_b=a.running_mean,b.running_mean
        with fr.no_grad(),torch.no_grad():a.reset_running_stats();b.reset_running_stats()
        self.assertIs(a.running_mean,old_a);self.assertIs(b.running_mean,old_b);self.state(a,b)
        # Fresh history for the no-batch transition and post-reset adjoint probes.
        a,b=models('InstanceNorm',3,track_running_stats=True);x,tx=inputs()
        a(x,ids,4);b(tx,tids,4)
        with self.assertRaises(RuntimeError):a(z)
        with self.assertRaises(RuntimeError):b(tz)
        with fr.no_grad(),torch.no_grad():self.close(a(z),b(tz));self.state(a,b)
        with fr.no_grad(),torch.no_grad():a.reset_parameters();b.reset_parameters()
        self.state(a,b)
        self.close(fr.autograd.grad(a.running_mean.sum(),x)[0],torch.autograd.grad(b.running_mean.sum(),tx)[0])
        with fr.no_grad(),torch.no_grad():a(x,ids,4);b(tx,tids,4)
        self.assertFalse(a.running_mean.requires_grad);self.state(a,b)
        self.close(a(z),b(tz))

    def test_instance_empties_zero_channels_singletons_none_momentum_and_empty_groups(self):
        for channels,rows,batch_size,tracking in itertools.product((0,3),(0,1,3),(0,1,4),(False,True)):
            if rows and not batch_size:continue
            a,b=models('InstanceNorm',channels,affine=True,track_running_stats=tracking)
            x,tx=inputs(rows,channels);ids=fr.zeros(rows,dtype='int64');tids=torch.zeros(rows,dtype=torch.long)
            out,wanted=a(x,ids,batch_size),b(tx,tids,batch_size);self.close(out,wanted);self.state(a,b);self.gradients(out,wanted,a,b,x,tx)
        for channels,rows in itertools.product((0,3),(0,1,3)):
            a,b=models('InstanceNorm',channels,affine=True,track_running_stats=True);x,tx=inputs(rows,channels)
            if rows==1:
                with self.assertRaises(ValueError):a(x)
                with self.assertRaises(ValueError):b(tx)
            elif channels==0:
                with self.assertRaises(IndexError):a(x)
                with self.assertRaises(IndexError):b(tx)
            else:self.close(a(x),b(tx));self.state(a,b)
        for rows in (0,3):
            a,b=models('InstanceNorm',0,affine=False,track_running_stats=True);x,tx=inputs(rows,0)
            self.close(a(x),b(tx));self.state(a,b)
        for tracking,training in itertools.product((False,True),(False,True)):
            a,b=models('InstanceNorm',3,momentum=None,track_running_stats=tracking);a.train(training);b.train(training);x,tx=inputs()
            for model,value in ((a,x),(b,tx)):
                with self.assertRaises(TypeError):model(value)
            if tracking and training:
                with self.assertRaises(TypeError):a(x,fr.zeros(6,dtype='int64'))
                with self.assertRaises(TypeError):b(tx,torch.zeros(6,dtype=torch.long))
            else:self.close(a(x,fr.zeros(6,dtype='int64')),b(tx,torch.zeros(6,dtype=torch.long)))
        with self.assertRaises(ValueError):InstanceNorm(3)(fr.zeros(0,3),fr.zeros(0,dtype='int64'))

    def test_hetero_batch_values_detached_statistics_and_all_parameter_gradients(self):
        for affine,tracking,training,momentum in itertools.product((False,True),(False,True),(False,True),(.2,None)):
            a,b=models('HeteroBatchNorm',3,4,affine=affine,track_running_stats=tracking,momentum=momentum)
            a.train(training);b.train(training);x,tx=inputs();ids=fr.tensor([0,2,0,2,1,0],dtype='int64');tids=torch.tensor([0,2,0,2,1,0])
            out,wanted=a(x,ids),b(tx,tids);self.close(out,wanted);self.state(a,b);self.gradients(out,wanted,a,b,x,tx)
            if tracking:self.assertFalse(a.running_mean.requires_grad);self.assertFalse(a.running_var.requires_grad)
        a,b=models('HeteroBatchNorm',1,1);x,tx=inputs(1,1)
        out,wanted=a(x,fr.tensor([0],dtype='int64')),b(tx,torch.tensor([0]));self.close(out,wanted)
        ga=fr.autograd.grad(out.sum(),x)[0];gb=torch.autograd.grad(wanted.sum(),tx)[0];self.close(ga,gb);self.assertGreater(ga.item(),0.)
        for values in ([0.,1.],[10000.1,10000.2]):
            a,b=models('HeteroBatchNorm',1,1,affine=False,momentum=1.,eps=.5)
            tx=torch.tensor(values).reshape(2,1).requires_grad_();x=native(tx,True)
            out,wanted=a(x,fr.zeros(2,dtype='int64')),b(tx,torch.zeros(2,dtype=torch.long))
            self.close(out,wanted);self.state(a,b);self.gradients(out,wanted,a,b,x,tx)
            if values[0]==0.:
                wrong=(tx-tx.mean(0))/(tx.var(0,unbiased=False)+.5).sqrt()
                self.assertGreater((wanted-wrong).abs().max().item(),.1)
            else:self.assertLess(a.running_var.item(),0.)

    def test_hetero_batch_present_type_updates_empty_types_and_counter_wrap(self):
        for momentum in (.2,None):
            a,b=models('HeteroBatchNorm',3,4,momentum=momentum)
            for ids in ([0,2,0,2,1,0],[],[1,1,1],[3]):
                x,tx=inputs(len(ids));out,wanted=a(x,fr.tensor(ids,dtype='int64')),b(tx,torch.tensor(ids,dtype=torch.long));self.close(out,wanted);self.state(a,b)
            self.assertEqual(a.num_batches_tracked.item(),0 if momentum is not None else 4)
            a.eval();b.eval();x,tx=inputs();self.close(a(x,fr.tensor([0,1,2,3,0,1],dtype='int64')),b(tx,torch.tensor([0,1,2,3,0,1])));self.state(a,b)
        for count in (2**63-1,-1):
            a,b=models('HeteroBatchNorm',3,3,momentum=None)
            with fr.no_grad(),torch.no_grad():a.num_batches_tracked.copy_(fr.tensor(count,dtype='int64'));b.num_batches_tracked.fill_(count)
            x,tx=inputs();ids=fr.zeros(6,dtype='int64');tids=torch.zeros(6,dtype=torch.long)
            if count==-1:
                with self.assertRaises(ZeroDivisionError):a(x,ids)
                with self.assertRaises(ZeroDivisionError):b(tx,tids)
            else:self.close(a(x,ids),b(tx,tids))
            self.state(a,b)
        for channels,types,rows in ((0,3,2),(3,0,0),(0,0,0),(3,4,0)):
            a,b=models('HeteroBatchNorm',channels,types,momentum=None);x,tx=inputs(rows,channels)
            out,wanted=a(x,fr.zeros(rows,dtype='int64')),b(tx,torch.zeros(rows,dtype=torch.long));self.close(out,wanted);self.state(a,b);self.gradients(out,wanted,a,b,x,tx)

    def test_hetero_layer_vector_pointer_precedence_and_all_gradients(self):
        for affine,route in itertools.product((False,True),('vector','ptr_list','ptr_tensor','both')):
            a,b=models('HeteroLayerNorm',3,4,affine=affine);x,tx=inputs()
            kw,tw={},{}
            if route=='vector':kw['type_vec']=fr.tensor([0,2,0,2,1,0],dtype='int64');tw['type_vec']=torch.tensor([0,2,0,2,1,0])
            else:
                pointer=[0,2,2,5,6];kw['type_ptr']=fr.tensor(pointer,dtype='int64')if route=='ptr_tensor'else pointer;tw['type_ptr']=torch.tensor(pointer)if route=='ptr_tensor'else pointer
                if route=='both':kw['type_vec']=fr.tensor([-100],dtype='int64');tw['type_vec']=torch.tensor([-100])
            out,wanted=a(x,**kw),b(tx,**tw);self.close(out,wanted);self.gradients(out,wanted,a,b,x,tx)
        a,b=models('HeteroLayerNorm',3,4,affine=False);x,tx=inputs()
        self.close(a(x,type_vec='ignored'),b(tx,type_vec='ignored'))
        for channels,types,rows in ((0,3,2),(3,0,0),(0,0,0),(3,4,0)):
            a,b=models('HeteroLayerNorm',channels,types);x,tx=inputs(rows,channels)
            out,wanted=a(x,fr.zeros(rows,dtype='int64')),b(tx,torch.zeros(rows,dtype=torch.long));self.close(out,wanted);self.gradients(out,wanted,a,b,x,tx)

    def test_resets_optional_state_and_affine_parameter_flags(self):
        for name,args in [('InstanceNorm',(3,)),('HeteroBatchNorm',(3,4)),('HeteroLayerNorm',(3,4))]:
            kw={}if name=='HeteroLayerNorm'else dict(track_running_stats=True)
            a,b=models(name,*args,affine=True,**kw)
            with fr.no_grad(),torch.no_grad():a.reset_parameters();b.reset_parameters()
            for key,p in a.named_parameters():self.close(p.tensor(),dict(b.named_parameters())[key])
            self.state(a,b);a.requires_grad_(False);b.requires_grad_(False);x,tx=inputs()
            ids=fr.tensor([0,2,0,2,1,0],dtype='int64');tids=torch.tensor([0,2,0,2,1,0])
            out,wanted=a(x,ids),b(tx,tids);self.close(out,wanted);self.close(fr.autograd.grad(out.sum(),x)[0],torch.autograd.grad(wanted.sum(),tx)[0])
            self.assertTrue(all(not p.requires_grad for p in a.parameters()))
        for name,args in [('InstanceNorm',(3,)),('HeteroBatchNorm',(3,4))]:
            a=globals()[name](*args,affine=False,track_running_stats=False)
            self.assertEqual(list(a.parameters()),[]);self.assertIsNone(a.running_mean);self.assertIsNone(a.num_batches_tracked)
        # Manually live state follows the same explicit reset boundary.
        a,b=models('HeteroBatchNorm',2,2)
        a.running_mean.requires_grad_(True);b.running_mean.requires_grad_(True)
        old_a,old_b=a.running_mean,b.running_mean
        with self.assertRaisesRegex(NotImplementedError,'no_grad'):a.reset_running_stats()
        with self.assertRaises(RuntimeError):b.reset_running_stats()
        with fr.no_grad(),torch.no_grad():a.reset_running_stats();b.reset_running_stats()
        self.assertIs(a.running_mean,old_a);self.assertIs(b.running_mean,old_b);self.state(a,b)

    def test_metadata_preflight_config_and_explicit_trainer_live_rebinding_boundary(self):
        a=HeteroBatchNorm(3,4,momentum=None);x,_=inputs();before=a.running_mean.tolist()
        for bad in (fr.tensor([0.]*6),fr.tensor([-1]*6,dtype='int64'),fr.tensor([4]*6,dtype='int64'),fr.tensor([0],dtype='int64')):
            with self.assertRaises((ValueError,TypeError)):a(x,bad)
        self.assertEqual(a.num_batches_tracked.item(),0);self.assertEqual(before,a.running_mean.tolist())
        layer=HeteroLayerNorm(3,4)
        for pointer in ([0,1,2],[0,2,1,5,6],[1,2,3,5,6],[0,1,2,3,5]):
            with self.assertRaises(ValueError):layer(x,type_ptr=pointer)
        with self.assertRaises(ValueError):layer(x)
        with self.assertRaises(NotImplementedError):HeteroLayerNorm(3,4,mode='graph')
        with self.assertRaises(NotImplementedError):InstanceNorm(3,device='cuda')
        with self.assertRaises(AttributeError):a.momentum=.2
        with self.assertRaises(NotImplementedError):fr.capture(lambda:a(x,fr.zeros(6,dtype='int64')))
        import ferro._graph_typed_norm as helper
        with patch.object(helper.Tensor,'ones',side_effect=AssertionError('allocation before capacity')):
            with self.assertRaisesRegex(ValueError,'capacity'):HeteroLayerNorm(0,sys.maxsize//8+1)
        tracked=InstanceNorm(3,affine=True,track_running_stats=True);old=tracked.running_mean
        trainer=Trainer(tracked,train_step=lambda model,batch:{'loss':model(batch,fr.zeros(6,dtype='int64')).sum()})
        with self.assertRaisesRegex(RuntimeError,'structure lock'):trainer.fit([x])
        self.assertIs(tracked.running_mean,old);self.assertFalse(tracked._structure_locked)
        plain=InstanceNorm(3,affine=True,track_running_stats=False)
        self.assertEqual(len(Trainer(plain,train_step=lambda model,batch:{'loss':model(batch,fr.zeros(6,dtype='int64')).sum()}).fit([x])),1)

    def test_checkpoint_configuration_mismatch_is_atomic_for_typed_state(self):
        for name,args,kw,changed in [('InstanceNorm',(3,),dict(affine=True,track_running_stats=True),dict(momentum=.2)),('HeteroBatchNorm',(3,4),dict(momentum=None),dict(momentum=.2)),('HeteroLayerNorm',(3,4),dict(),dict(eps=.2))]:
            model=globals()[name](*args,**kw);saved=fr.checkpoint.snapshot_training(model,optimizers={},generator=fr.Generator(9))
            other=globals()[name](*args,**{**kw,**changed});before={k:v.tolist()for k,v in other.state_dict().items()}
            with self.assertRaises(ValueError):saved.restore(other,optimizers={},generator=fr.Generator(8))
            self.assertEqual(before,{k:v.tolist()for k,v in other.state_dict().items()})

    def test_huge_empty_zero_channel_groups_avoid_irrelevant_count_vectors(self):
        script='''
import ferro as f
from ferro.graph import InstanceNorm,HeteroBatchNorm,HeteroLayerNorm
x=f.zeros(0,0,requires_grad=True);ids=f.zeros(0,dtype='int64');size=10**12
for m in (InstanceNorm(0,affine=True,track_running_stats=True),HeteroBatchNorm(0,size,momentum=None),HeteroLayerNorm(0,size)):
    y=m(x,ids,size) if isinstance(m,InstanceNorm) else m(x,ids)
    gradients=f.autograd.grad(y.sum(),[x]+[p.tensor() for p in m.parameters()])
    if y.shape != [0,0] or gradients[0].shape != [0,0]:raise RuntimeError('empty shape changed')
    if isinstance(m,HeteroBatchNorm) and m.num_batches_tracked.item()!=1:raise RuntimeError('counter changed')
print('bounded-zero-channel-norms')
'''
        result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('bounded-zero-channel-norms',result.stdout)

    def test_fresh_process_complete_step_values_modes_aliases_optimizer_and_rng(self):
        script=r'''
import json,math,sys,ferro as f
from ferro.graph import InstanceNorm,HeteroBatchNorm,HeteroLayerNorm
class Model(f.nn.Module):
    def build(self):
        self.linear=f.nn.Linear(3,3)
        self.instance=InstanceNorm(3,affine=True,track_running_stats=True)
        self.typed=HeteroBatchNorm(3,4,momentum=None)
        self.tied=self.typed
        self.layer=HeteroLayerNorm(3,4)
    def forward(self,x):
        graph=f.tensor([0,2,0,2,2,0],dtype='int64');types=f.tensor([0,1,0,2,1,2],dtype='int64')
        z=self.instance(self.linear(x),graph,4)
        return self.layer(self.typed(z,types)+self.tied(z*.3,types),types)
def exact(v):
    if isinstance(v,float):
        if not math.isfinite(v):raise RuntimeError('nonfinite checkpoint fixture')
        return v.hex()
    return [exact(x)for x in v]if isinstance(v,list)else v
m=Model();m.layer.eval();opt=f.optim.AdamW(m.parameters(),lr=.002);rng=f.Generator(53)
def step():
    x=f.Tensor.rand([6,3],generator=rng)*2-1;target=f.Tensor.rand([6,3],generator=rng)
    out=m(x);loss=(out-target).square().mean();loss.backward();opt.step();opt.zero_grad();return exact(loss.item())
if sys.argv[1]=='save':
    step()
    # A checkpoint stores values, not an AD graph across earlier steps.
    m.instance.running_mean=m.instance.running_mean.detach();m.instance.running_var=m.instance.running_var.detach()
    f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=1).save(sys.argv[2])
else:f.checkpoint.load(sys.argv[2]).restore(m,optimizers={'main':opt},generator=rng)
losses=[step()for _ in range(3)]
s=f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=4).tensors()
result={'losses':losses,'state':{k:[v.dtype,v.shape,exact(v.tolist())]for k,v in s.items()},'alias':m.typed is m.tied,'modes':[m.training,m.layer.training],'flags':{k:v.requires_grad for k,v in m.named_buffers()},'draw':exact(f.Tensor.rand([5],generator=rng).tolist())}
print(json.dumps(result,sort_keys=True))
'''
        with tempfile.TemporaryDirectory() as directory:
            output=[subprocess.run([sys.executable,'-B','-c',script,mode,directory+'/state'],check=True,capture_output=True,text=True,timeout=60).stdout for mode in ('save','load')]
        self.assertEqual(json.loads(output[0]),json.loads(output[1]));result=json.loads(output[0]);self.assertTrue(result['alias']);self.assertEqual(result['modes'],[True,False])

if __name__=='__main__':unittest.main()
