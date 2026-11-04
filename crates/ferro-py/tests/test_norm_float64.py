"""CPU double normalization parameters, all gradients, buffers and typed restart."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ferro as fr
import torch
import test_sgd_f64 as states
import test_typed_training_buffers as buffers


def native(value, grad=None):
    result=fr.tensor(value.detach().tolist(),dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape))
    return result.requires_grad_(value.requires_grad if grad is None else grad)


def make(name, *, reference=False, dtype='float64', **options):
    namespace=torch.nn if reference else fr.nn
    dtype=getattr(torch,dtype) if reference else dtype
    if name in ('LayerNorm','RMSNorm'):
        return getattr(namespace,name)((3,4),dtype=dtype,**options)
    return getattr(namespace,name)(3,dtype=dtype,**options)


SHAPES={'LayerNorm':(2,3,4),'RMSNorm':(2,3,4),'BatchNorm1d':(2,3,4),'BatchNorm2d':(2,3,2,4),
        'BatchNorm3d':(2,3,2,2,3),'InstanceNorm1d':(2,3,4),'InstanceNorm2d':(2,3,2,4),'InstanceNorm3d':(2,3,2,2,3)}


class NormFloat64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def same(self,actual,expected):
        self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.dtype,str(expected.dtype).removeprefix('torch.'))
        values=torch.tensor(actual.tolist(),dtype=expected.dtype).reshape(expected.shape)
        tolerance=3e-11 if expected.dtype==torch.float64 else 5e-5
        torch.testing.assert_close(values,expected.detach(),rtol=tolerance,atol=tolerance,equal_nan=True)

    def state(self,actual,expected):
        for method in ('named_parameters','named_buffers'):
            left,right=dict(getattr(actual,method)()),dict(getattr(expected,method)())
            self.assertEqual(set(left),set(right))
            for key in left:self.same(left[key],right[key])

    def comparison(self,actual,expected,shape,*,strided=False):
        generator=torch.Generator().manual_seed(991)
        base=torch.randn(shape,dtype=torch.float64,generator=generator)*.4+.3
        if strided:
            source=base.transpose(0,-1).contiguous();x=source.transpose(0,-1).detach().requires_grad_();nx=native(source,False).transpose(0,len(shape)-1).detach().requires_grad_(True)
        else:x=base.requires_grad_();nx=native(x)
        y,z=actual(nx),expected(x);self.same(y,z)
        seed=torch.randn(z.shape,dtype=torch.float64,generator=generator)
        (y*native(seed)).sum().backward();(z*seed).sum().backward();self.same(nx.grad,x.grad)
        for key,p in actual.named_parameters():self.same(p.grad,dict(expected.named_parameters())[key].grad)
        self.state(actual,expected);actual.zero_grad();expected.zero_grad()

    def test_layer_and_rms_all_parameter_input_gradients_axes_and_bias(self):
        for name in ('LayerNorm','RMSNorm'):
            for affine in (False,True):
                for bias in ((False,True) if name=='LayerNorm' else (False,)):
                    options=dict(elementwise_affine=affine)
                    if name=='LayerNorm':options['bias']=bias
                    a,b=make(name,**options),make(name,reference=True,**options)
                    with fr.no_grad(),torch.no_grad():
                        for key,p in a.named_parameters():
                            values=torch.arange(p.numel(),dtype=torch.float64).reshape(p.shape)*.07000000000000002+.20000000000000004
                            p.copy_(native(values));dict(b.named_parameters())[key].copy_(values)
                    for training in (True,False):
                        a.train(training);b.train(training);self.comparison(a,b,(2,3,4),strided=True)

    def test_batchnorm_all_ranks_training_eval_running_state_and_every_gradient(self):
        for name in ('BatchNorm1d','BatchNorm2d','BatchNorm3d'):
            for tracking in (False,True):
                for bias in (False,True):
                    a=make(name,track_running_stats=tracking,momentum=None,bias=bias)
                    b=make(name,reference=True,track_running_stats=tracking,momentum=None,bias=bias)
                    self.comparison(a,b,SHAPES[name],strided=True)
                    self.comparison(a,b,SHAPES[name])
                    a.eval();b.eval();self.comparison(a,b,SHAPES[name],strided=True)
                    if tracking:self.assertEqual(a.num_batches_tracked.item(),2)

    def test_instancenorm_all_ranks_tracked_untracked_batched_and_unbatched(self):
        for name in ('InstanceNorm1d','InstanceNorm2d','InstanceNorm3d'):
            for tracking in (False,True):
                for bias in (False,True):
                    a=make(name,affine=True,track_running_stats=tracking,bias=bias,momentum=.30000000000000004)
                    b=make(name,reference=True,affine=True,track_running_stats=tracking,bias=bias,momentum=.30000000000000004)
                    self.comparison(a,b,SHAPES[name],strided=True)
                    self.comparison(a,b,SHAPES[name][1:])
                    a.eval();b.eval();self.comparison(a,b,SHAPES[name],strided=True)
                    if tracking:self.assertEqual(a.num_batches_tracked.item(),0)

    def test_typed_reset_preserves_parameter_buffer_ids_flags_and_live_grad_handles(self):
        for name in SHAPES:
            for dtype in ('float32','float64'):
                options=dict(affine=True,track_running_stats=True) if name.startswith('Instance') else {}
                a,b=make(name,dtype=dtype,**options),make(name,reference=True,dtype=dtype,**options)
                ids={key:p._id() for key,p in a.named_parameters()};bufids={key:t._id() for key,t in a.named_buffers()};grads={}
                with fr.no_grad(),torch.no_grad():
                    for key,p in a.named_parameters():
                        p.fill_(.7);p.requires_grad_(False);g=fr.ones(p.shape,dtype=dtype);p.grad=g;grads[key]=g
                        q=dict(b.named_parameters())[key];q.fill_(.7);q.requires_grad_(False);q.grad=torch.ones_like(q)
                    for key,t in a.named_buffers():
                        t.fill_(7 if t.dtype=='int64' else .3);dict(b.named_buffers())[key].fill_(7 if t.dtype=='int64' else .3)
                a.reset_parameters();b.reset_parameters();self.state(a,b)
                self.assertEqual(ids,{key:p._id() for key,p in a.named_parameters()});self.assertEqual(bufids,{key:t._id() for key,t in a.named_buffers()})
                for key,p in a.named_parameters():self.assertFalse(p.requires_grad);self.assertEqual(p.grad._id(),grads[key]._id())
                before={key:states.flat_bits(value) for key,value in (*a.named_parameters(),*a.named_buffers())}
                with self.assertRaises((ValueError,NotImplementedError)):
                    fr.func.jvp(lambda x:(a.reset_parameters(),x)[1],(fr.ones(1),),(fr.ones(1),))
                self.assertEqual(before,{key:states.flat_bits(value) for key,value in (*a.named_parameters(),*a.named_buffers())})
                a.weight.requires_grad_(True);b.weight.requires_grad_(True)
                stale=a.weight.square().sum();reference_stale=b.weight.square().sum()
                a.reset_parameters();b.reset_parameters()
                with self.assertRaises((ValueError,RuntimeError)):stale.backward()
                with self.assertRaises(RuntimeError):reference_stale.backward()
                self.assertEqual(a.weight.grad._id(),grads['weight']._id())

    def test_empty_dimensions_singleton_failure_and_exact_counter_order(self):
        for name in ('LayerNorm','RMSNorm'):
            for shape,norm in (((0,3,4),(3,4)),((2,0,3),(0,3)),((2,0),(0,))):
                a=getattr(fr.nn,name)(norm,dtype='float64');b=getattr(torch.nn,name)(norm,dtype=torch.float64)
                self.comparison(a,b,shape)
        for name in ('BatchNorm1d','BatchNorm2d','BatchNorm3d'):
            shape=SHAPES[name];a,b=make(name,momentum=None),make(name,reference=True,momentum=None)
            self.comparison(a,b,(0,*shape[1:]));self.comparison(a,b,(*shape[:-1],0))
            singleton=(1,3,*([1]*(len(shape)-2)))
            for model,value in ((a,fr.ones(singleton,dtype='float64')),(b,torch.ones(singleton,dtype=torch.float64))):
                with self.assertRaises(ValueError):model(value)
            self.state(a,b);self.assertEqual(a.num_batches_tracked.item(),3)
        for name in ('InstanceNorm1d','InstanceNorm2d','InstanceNorm3d'):
            a=make(name,affine=True,track_running_stats=True);b=make(name,reference=True,affine=True,track_running_stats=True);shape=SHAPES[name]
            self.comparison(a,b,(*shape[:-1],0))
            for model,value in ((a,fr.ones((0,*shape[1:]),dtype='float64')),(b,torch.ones((0,*shape[1:]),dtype=torch.float64))):
                with self.assertRaises(ValueError if model is a else IndexError):model(value)
            self.state(a,b)

    def test_typed_sgd_training_preserves_parameters_and_matches_torch(self):
        for name,shape in SHAPES.items():
            options=dict(affine=True,track_running_stats=True) if name.startswith('Instance') else {}
            a,b=make(name,**options),make(name,reference=True,**options)
            config=dict(lr=.010000000000000002,momentum=.9000000000000001,weight_decay=.030000000000000002)
            opt=fr.optim.SGD(a.parameters(),**config);ref=torch.optim.SGD(b.parameters(),**config)
            ids=[p._id() for p in a.parameters()];rng=torch.Generator().manual_seed(331)
            for _ in range(4):
                opt.zero_grad();ref.zero_grad();x=torch.randn(shape,dtype=torch.float64,generator=rng).requires_grad_();nx=native(x)
                target=torch.randn(shape,dtype=torch.float64,generator=rng)
                loss=(a(nx)-native(target)).square().mean();expected=(b(x)-target).square().mean();self.same(loss,expected)
                loss.backward();expected.backward();self.same(nx.grad,x.grad)
                for p,q in zip(a.parameters(),b.parameters()):self.same(p.grad,q.grad)
                opt.step();ref.step();self.state(a,b);self.assertEqual(ids,[p._id() for p in a.parameters()])
                for i,q in enumerate(b.parameters()):self.same(opt.state_dict()['state'][i]['momentum_buffer'],ref.state[q]['momentum_buffer'])

    def test_dtype_capacity_capture_and_checkpoint_corruption_reject_without_state_changes(self):
        for name in SHAPES:
            for dtype in ('int64','bool'):
                with self.assertRaises((TypeError,ValueError,NotImplementedError)):make(name,dtype=dtype)
            with self.assertRaises(NotImplementedError):make(name,device='cuda')
        with self.assertRaises(NotImplementedError):fr.nn.GroupNorm(1,3,dtype='bool')
        for name in ('LayerNorm','RMSNorm'):
            with self.assertRaisesRegex(ValueError,'capacity'):getattr(fr.nn,name)((sys.maxsize//8+1,),dtype='float64')
            model=make(name)
            with self.assertRaises((ValueError,NotImplementedError)):fr.capture(lambda:model(fr.ones(2,3,4,dtype='float64')))
        source=make('BatchNorm1d',momentum=None);opt=fr.optim.SGD(source.parameters(),lr=.1,momentum=.9);rng=fr.Generator(7)
        source(fr.randn([2,3,4],dtype='float64',generator=rng)).square().sum().backward();opt.step();opt.zero_grad()
        saved=fr.checkpoint.snapshot_training(source,optimizers={'sgd':opt},generator=rng)
        target=make('BatchNorm1d',momentum=None);other=fr.optim.SGD(target.parameters(),lr=.1,momentum=.9);r=fr.Generator(99)
        ids={key:t._id() for key,t in target.named_buffers()};ids.update({key:t._id() for key,t in target.named_parameters()})
        for key,value in (('buffers.running_mean',fr.zeros(3)),('buffers.num_batches_tracked',fr.tensor(3.)),('optim.sgd.velocity.1',fr.zeros(3))):
            with tempfile.TemporaryDirectory() as directory:
                bad=buffers.corrupt(saved,directory,key,value)
                before=states.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':other},generator=r))
                with self.assertRaises(ValueError):bad.restore(target,optimizers={'sgd':other},generator=r)
                self.assertEqual(states.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':other},generator=r)),before)
        self.assertEqual(ids,{**{key:t._id() for key,t in target.named_buffers()},**{key:t._id() for key,t in target.named_parameters()}})
        for bad in (make('BatchNorm1d',dtype='float32',momentum=None),make('BatchNorm1d',momentum=.1)):
            badopt=fr.optim.SGD(bad.parameters(),lr=.1,momentum=.9);before=states.digest(fr.checkpoint.snapshot_training(bad,optimizers={'sgd':badopt},generator=r))
            with self.assertRaises(ValueError):saved.restore(bad,optimizers={'sgd':badopt},generator=r)
            self.assertEqual(states.digest(fr.checkpoint.snapshot_training(bad,optimizers={'sgd':badopt},generator=r)),before)

    def test_fresh_process_full_typed_state_optimizer_rng_and_counter_continuation(self):
        script=r'''
import json,struct,sys
import ferro as fr
class Model(fr.nn.Module):
 def __init__(self):
  super().__init__();self.layer=fr.nn.LayerNorm((3,4),dtype='float64');self.rms=fr.nn.RMSNorm((3,4),dtype='float64')
  self.batch=fr.nn.BatchNorm1d(3,dtype='float64',momentum=None);self.instance=fr.nn.InstanceNorm1d(3,dtype='float64',affine=True,track_running_stats=True,momentum=.3)
 def forward(self,x):return self.instance(self.batch(self.rms(self.layer(x))))
m=Model();o=fr.optim.SGD(m.parameters(),lr=.010000000000000002,momentum=.9000000000000001,weight_decay=.03);r=fr.Generator(71)
def step(i):
 m.train(i%3!=2);x=fr.randn([2+i%2,3,4],dtype='float64',generator=r);target=fr.randn(x.shape,dtype='float64',generator=r)
 loss=(m(x)-target).square().mean();value=loss.item();loss.backward();o.step();o.zero_grad();return value
if sys.argv[1]=='save':
 for i in range(3):step(i)
 fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=3).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(m,optimizers={'sgd':o},generator=r)
losses=[step(i) for i in range(3,8)];state=fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=8).tensors()
def bits(t):
 values=t.reshape([t.numel()]).tolist();code={'float32':'f','float64':'d','int64':'q','bool':'?'}[t.dtype]
 return[t.dtype,t.shape,struct.pack('<'+code*len(values),*values).hex()]
print(json.dumps({'losses':losses,'state':{k:bits(v) for k,v in state.items()},'next':fr.randn([9],dtype='float64',generator=r).tolist(),'counts':[m.batch.num_batches_tracked.item(),m.instance.num_batches_tracked.item()]},sort_keys=True))
'''
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory() as directory:
            results=[subprocess.run([sys.executable,'-c',script,mode,str(Path(directory)/'state')],env=env,capture_output=True,text=True,timeout=60,check=True).stdout for mode in ('save','load')]
        first,second=map(json.loads,results);self.assertEqual(first,second);self.assertEqual(first['counts'],[6,0])


if __name__=='__main__':unittest.main(verbosity=2)
