"""Typed GroupNorm modules and existing double LocalResponseNorm training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ferro as fr
import torch
import test_norm_float64 as norm
import test_sgd_f64 as states
import test_typed_training_buffers as buffers


class GroupNormFloat64Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def same(self,actual,expected):
        self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.dtype,str(expected.dtype).removeprefix('torch.'))
        values=torch.tensor(actual.tolist(),dtype=expected.dtype).reshape(expected.shape)
        tolerance=3e-11 if expected.dtype==torch.float64 else 5e-5
        torch.testing.assert_close(values,expected.detach(),rtol=tolerance,atol=tolerance,equal_nan=True)

    def pair(self,channels=4,groups=2,**kw):
        a=fr.nn.GroupNorm(groups,channels,dtype='float64',**kw);b=torch.nn.GroupNorm(groups,channels,dtype=torch.float64,**kw)
        with fr.no_grad(),torch.no_grad():
            for key,p in a.named_parameters():
                q=dict(b.named_parameters())[key]
                values=torch.arange(p.numel(),dtype=torch.float64)*.07000000000000002+.20000000000000004
                p.copy_(norm.native(values));q.copy_(values)
        return a,b

    def compare(self,a,b,shape,strided=False):
        generator=torch.Generator().manual_seed(447)
        values=torch.randn(shape,dtype=torch.float64,generator=generator)*.4+.3
        if strided:
            source=values.transpose(0,-1).contiguous();x=source.transpose(0,-1).detach().requires_grad_()
            nx=norm.native(source,False).transpose(0,len(shape)-1).detach().requires_grad_(True)
        else:x=values.requires_grad_();nx=norm.native(x)
        actual,expected=a(nx),b(x);self.same(actual,expected)
        seed=torch.randn(shape,dtype=torch.float64,generator=generator)
        (actual*norm.native(seed)).sum().backward();(expected*seed).sum().backward();self.same(nx.grad,x.grad)
        for key,p in a.named_parameters():self.same(p.grad,dict(b.named_parameters())[key].grad)

    def test_factory_and_nonaffine_cross_dtype_contracts(self):
        for dtype in ('float32','float64'):
            for affine in (False,True):
                for bias in (False,True):
                    a=fr.nn.GroupNorm(2,4,dtype=dtype,affine=affine,bias=bias)
                    b=torch.nn.GroupNorm(2,4,dtype=getattr(torch,dtype),affine=affine,bias=bias)
                    self.assertEqual(set(a.state_dict()),set(b.state_dict()))
                    for key,p in a.named_parameters():self.same(p,dict(b.named_parameters())[key])
                    for input_dtype in ('float32','float64'):
                        x=torch.arange(24,dtype=getattr(torch,input_dtype)).reshape(2,4,3).requires_grad_();nx=norm.native(x)
                        if affine and input_dtype!=dtype:
                            with self.assertRaises(ValueError):a(nx)
                            with self.assertRaises(RuntimeError):b(x)
                        else:
                            actual,expected=a(nx),b(x);self.same(actual,expected)
                            actual.sum().backward();expected.sum().backward();self.same(nx.grad,x.grad)
                        a.zero_grad();b.zero_grad()
        for channels in (0,4):
            a=fr.nn.GroupNorm(2,channels,dtype='float64');b=torch.nn.GroupNorm(2,channels,dtype=torch.float64)
            self.same(a.weight,b.weight);self.same(a.bias,b.bias)

    def test_all_parameter_and_input_gradients_groups_ranks_strides_and_modes(self):
        for shape in ((2,4),(2,4,5),(2,4,2,3),(2,4,2,2,3)):
            for groups in (1,2,4):
                for bias in (False,True):
                    for training in (False,True):
                        with self.subTest(shape=shape,groups=groups,bias=bias,training=training):
                            a,b=self.pair(groups=groups,bias=bias);a.train(training);b.train(training)
                            self.compare(a,b,shape,True)

    def test_reset_preserves_leaf_flags_grad_handles_and_mutation_guards(self):
        for dtype in ('float32','float64'):
            a=fr.nn.GroupNorm(2,4,dtype=dtype);b=torch.nn.GroupNorm(2,4,dtype=getattr(torch,dtype))
            ids={key:p._id() for key,p in a.named_parameters()};grads={}
            with fr.no_grad(),torch.no_grad():
                for key,p in a.named_parameters():
                    p.fill_(.7);p.requires_grad_(False);g=fr.ones(4,dtype=dtype);p.grad=g;grads[key]=g
                    q=dict(b.named_parameters())[key];q.fill_(.7);q.requires_grad_(False);q.grad=torch.ones_like(q)
            a.reset_parameters();b.reset_parameters()
            for key,p in a.named_parameters():
                self.same(p,dict(b.named_parameters())[key]);self.assertEqual(p._id(),ids[key]);self.assertFalse(p.requires_grad)
                self.assertEqual(p.grad._id(),grads[key]._id())
            before={key:states.flat_bits(p) for key,p in a.named_parameters()}
            for call in (lambda:fr.func.jvp(lambda x:(a.reset_parameters(),x)[1],(fr.ones(1),),(fr.ones(1),)),lambda:fr.capture(lambda:(a.reset_parameters(),fr.ones(1))[1])):
                with self.assertRaises((ValueError,NotImplementedError)):call()
                self.assertEqual(before,{key:states.flat_bits(p) for key,p in a.named_parameters()})
            a.weight.requires_grad_(True);b.weight.requires_grad_(True)
            stale=a.weight.square().sum();ref=b.weight.square().sum();a.reset_parameters();b.reset_parameters()
            with self.assertRaises((ValueError,RuntimeError)):stale.backward()
            with self.assertRaises(RuntimeError):ref.backward()
            self.assertEqual(a.weight.grad._id(),grads['weight']._id())

    def test_empty_statistics_and_singleton_boundaries(self):
        for shape in ((0,4,3),(2,4,0),(2,4,0,2),(2,0,3),(0,4,0)):
            for affine in (False,True):
                with self.subTest(shape=shape,affine=affine):
                    a,b=self.pair(channels=shape[1],affine=affine);self.compare(a,b,shape)
                    if affine and shape[0]>0 and shape[1]>0 and 0 in shape[2:]:
                        self.assertTrue(all(v!=v for v in a.weight.grad.tolist()))
        for shape in ((1,4),(1,4,1,1)):
            a,b=self.pair(groups=4)
            with self.assertRaises(ValueError):a(fr.ones(shape,dtype='float64'))
            with self.assertRaises(ValueError):b(torch.ones(shape,dtype=torch.float64))

    def test_local_response_norm_typed_module_modes_empty_identity_and_gradients(self):
        for size in (1,2,5):
            for shape in ((2,4,3),(2,4,2,3),(2,4,2,2,3),(0,4,3),(2,0,3)):
                for training in (False,True):
                    with self.subTest(size=size,shape=shape,training=training):
                        a=fr.nn.LocalResponseNorm(size,alpha=.03,beta=.7,k=1.2);b=torch.nn.LocalResponseNorm(size,alpha=.03,beta=.7,k=1.2)
                        a.train(training);b.train(training);self.assertEqual(list(a.parameters()),[]);self.assertEqual(a.state_dict(),{})
                        self.compare(a,b,shape)
                        if 0 in shape:
                            x=fr.empty(shape,dtype='float64');tx=torch.empty(shape,dtype=torch.float64)
                            self.assertIs(a(x),x);self.assertIs(b(tx),tx)
        for values in ((float('inf'),1.,0.),(float('nan'),2.,-1.)):
            a=fr.nn.LocalResponseNorm(3);b=torch.nn.LocalResponseNorm(3)
            x=torch.tensor(values,dtype=torch.float64).reshape(1,3,1).requires_grad_();nx=norm.native(x)
            actual,expected=a(nx),b(x);self.same(actual,expected);actual.sum().backward();expected.sum().backward();self.same(nx.grad,x.grad)

    def test_typed_sgd_groupnorm_lrn_trajectory_and_momentum(self):
        a,b=self.pair();al=fr.nn.LocalResponseNorm(4,alpha=.03);bl=torch.nn.LocalResponseNorm(4,alpha=.03)
        options=dict(lr=.010000000000000002,momentum=.9000000000000001,weight_decay=.030000000000000002)
        opt=fr.optim.SGD(a.parameters(),**options);ref=torch.optim.SGD(b.parameters(),**options)
        ids=[p._id() for p in a.parameters()];rng=torch.Generator().manual_seed(337)
        for i in range(6):
            opt.zero_grad();ref.zero_grad();a.train(i%2==0);b.train(i%2==0)
            x=torch.randn(2+i%2,4,3,dtype=torch.float64,generator=rng).requires_grad_();nx=norm.native(x)
            target=torch.randn(x.shape,dtype=torch.float64,generator=rng)
            loss=(al(a(nx))-norm.native(target)).square().mean();expected=(bl(b(x))-target).square().mean();self.same(loss,expected)
            loss.backward();expected.backward();self.same(nx.grad,x.grad)
            for p,q in zip(a.parameters(),b.parameters()):self.same(p.grad,q.grad)
            opt.step();ref.step();self.assertEqual(ids,[p._id() for p in a.parameters()])
            for j,(p,q) in enumerate(zip(a.parameters(),b.parameters())):
                self.same(p,q);self.same(opt.state_dict()['state'][j]['momentum_buffer'],ref.state[q]['momentum_buffer'])

    def test_validation_and_full_checkpoint_rejection_are_atomic(self):
        for dtype in ('int64','bool'):
            with self.assertRaises(NotImplementedError):fr.nn.GroupNorm(1,4,dtype=dtype)
        with self.assertRaises(NotImplementedError):fr.nn.GroupNorm(1,4,dtype='float64',device='cuda')
        with self.assertRaisesRegex(ValueError,'capacity'):fr.nn.GroupNorm(1,sys.maxsize//8+1,dtype='float64')
        for a in (fr.nn.GroupNorm(2,4,dtype='float64'),fr.nn.LocalResponseNorm(3)):
            with self.assertRaises(NotImplementedError):fr.capture(lambda:a(fr.ones(2,4,3,dtype='float64')))
        source,_=self.pair();opt=fr.optim.SGD(source.parameters(),lr=.1,momentum=.9);rng=fr.Generator(7)
        source(fr.randn([2,4,3],dtype='float64',generator=rng)).square().sum().backward();opt.step();opt.zero_grad()
        saved=fr.checkpoint.snapshot_training(source,optimizers={'sgd':opt},generator=rng)
        target,_=self.pair();other=fr.optim.SGD(target.parameters(),lr=.1,momentum=.9);r=fr.Generator(99)
        ids=[p._id() for p in target.parameters()]
        for key in ('params.weight','optim.sgd.velocity.1'):
            with tempfile.TemporaryDirectory() as directory:
                bad=buffers.corrupt(saved,directory,key,fr.zeros(4))
                before=states.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':other},generator=r))
                with self.assertRaises(ValueError):bad.restore(target,optimizers={'sgd':other},generator=r)
                self.assertEqual(before,states.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':other},generator=r)))
        for bad in (fr.nn.GroupNorm(1,4,dtype='float64'),fr.nn.GroupNorm(2,4)):
            badopt=fr.optim.SGD(bad.parameters(),lr=.1,momentum=.9);before=states.digest(fr.checkpoint.snapshot_training(bad,optimizers={'sgd':badopt},generator=r))
            with self.assertRaises(ValueError):saved.restore(bad,optimizers={'sgd':badopt},generator=r)
            self.assertEqual(before,states.digest(fr.checkpoint.snapshot_training(bad,optimizers={'sgd':badopt},generator=r)))
        self.assertEqual(ids,[p._id() for p in target.parameters()]);saved.restore(target,optimizers={'sgd':other},generator=r)
        self.assertEqual(states.digest(saved),states.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':other},generator=r)))

    def test_fresh_process_typed_groupnorm_lrn_sgd_state_rng_continuation(self):
        script=r'''
import json,struct,sys
import ferro as fr
class Model(fr.nn.Module):
 def __init__(self):
  super().__init__();self.group=fr.nn.GroupNorm(2,4,dtype='float64');self.local=fr.nn.LocalResponseNorm(4,alpha=.03,beta=.7,k=1.2)
 def forward(self,x):return self.local(self.group(x))
m=Model();o=fr.optim.SGD(m.parameters(),lr=.010000000000000002,momentum=.9000000000000001,weight_decay=.03);r=fr.Generator(71)
def step(i):
 m.train(i%2==0);x=fr.randn([2+i%2,4,3],dtype='float64',generator=r);target=fr.randn(x.shape,dtype='float64',generator=r)
 loss=(m(x)-target).square().mean();value=loss.item();loss.backward();o.step();o.zero_grad();return value
if sys.argv[1]=='save':
 for i in range(3):step(i)
 fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=3).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(m,optimizers={'sgd':o},generator=r)
losses=[step(i) for i in range(3,8)];state=fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=8).tensors()
def bits(t):
 values=t.reshape([t.numel()]).tolist();code={'float32':'f','float64':'d','int64':'q','bool':'?'}[t.dtype]
 return[t.dtype,t.shape,struct.pack('<'+code*len(values),*values).hex()]
print(json.dumps({'losses':losses,'state':{k:bits(v) for k,v in state.items()},'next':fr.randn([9],dtype='float64',generator=r).tolist()},sort_keys=True))
'''
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory() as directory:
            results=[subprocess.run([sys.executable,'-c',script,mode,str(Path(directory)/'state')],env=env,capture_output=True,text=True,timeout=60,check=True).stdout for mode in ('save','load')]
        self.assertEqual(*map(json.loads,results))


if __name__=='__main__':unittest.main(verbosity=2)
