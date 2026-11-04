"""Typed embedding modules reuse native dense lookup/bags and exact SGD state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import ferro as fr
import torch
import test_embedding_options as helpers
import test_sgd_f64 as state_helpers


def tensor(data, grad=False, dtype='float64'):
    return fr.tensor(data, dtype=dtype, requires_grad=grad)


class EmbeddingFloat64Tests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        helpers.EmbeddingOptionsTests.same(self, actual, expected)

    def test_factories_and_supplied_weight_dtype_precedence_and_alias_flags(self):
        for cls, reference in ((fr.nn.Embedding, torch.nn.Embedding), (fr.nn.EmbeddingBag, torch.nn.EmbeddingBag)):
            self.assertEqual(cls(3, 2).weight.dtype, 'float32')
            self.assertEqual(cls(3, 2, dtype='float64').weight.dtype, 'float64')
            for dtype, refdtype in ((None, None), ('float32', torch.float32), ('float64', torch.float64), ('int64', torch.int64)):
                for device in (None, 'cpu', 'cuda'):
                    source = tensor([[.10000000000000002, 1e-200], [-0., 1e200]], True)
                    original = torch.tensor(source.tolist(), dtype=torch.float64, requires_grad=True)
                    actual = cls(2, 2, _weight=source, dtype=dtype, device=device)
                    expected = reference(2, 2, _weight=original, dtype=refdtype, device=device)
                    self.same(actual.weight, expected.weight)
                    self.assertTrue(actual.weight._shares_storage(source))
                    self.assertNotEqual(actual.weight._id(), source._id())
                    self.assertTrue(source.requires_grad)
                    self.assertTrue(actual.weight.trainable)
                    self.assertIsNone(source.grad)
            for freeze in (False, True):
                source = tensor([[.2, .7], [.5, -.3]], True)
                actual = cls.from_pretrained(source, freeze=freeze, padding_idx=0)
                expected = reference.from_pretrained(torch.tensor(source.tolist(), dtype=torch.float64), freeze=freeze, padding_idx=0)
                self.same(actual.weight, expected.weight)
                self.assertEqual(actual.weight.requires_grad, not freeze)
                self.assertTrue(actual.weight._shares_storage(source))
                self.assertTrue(source.requires_grad)
                with fr.no_grad(): source.fill_(.25)
                self.assertEqual(actual.weight.tolist(), [[.25, .25], [.25, .25]])

    def test_lookup_padding_frequency_renorm_and_parameter_gradients(self):
        for padding in (None, 0, -1):
            for scale in (False, True):
                for shape in ((), (4,), (2, 2)):
                    with self.subTest(padding=padding, scale=scale, shape=shape):
                        initial = torch.tensor([[3.,4.,1e-100],[-2.,.5,.3],[.2,.7,-.9],[1.1,-.4,.6]], dtype=torch.float64)
                        actual = fr.nn.Embedding.from_pretrained(helpers.native(initial), freeze=False, padding_idx=padding, max_norm=1., scale_grad_by_freq=scale)
                        expected = torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False, padding_idx=padding, max_norm=1., scale_grad_by_freq=scale)
                        ids = torch.tensor(1 if not shape else [0,1,1,3]).reshape(shape)
                        a, b = actual(helpers.native(ids)), expected(ids)
                        c, d = actual(helpers.native(ids)), expected(ids)
                        self.same(a,b);self.same(c,d);self.same(actual.weight,expected.weight)
                        seed=torch.arange(b.numel(),dtype=torch.float64).reshape(b.shape)*.13+.2
                        ((a+c)*helpers.native(seed)).sum().backward();((b+d)*seed).sum().backward()
                        self.same(actual.weight.grad,expected.weight.grad)

    def test_bag_modes_live_samples_and_dense_weight_gradients(self):
        for mode in ('sum','mean','max'):
            for scale in ((False,True) if mode!='max' else (False,)):
                for weighted in ((False,True) if mode=='sum' else (False,)):
                    w=torch.tensor([[.2,-.7],[.8,.1],[-.5,1.3],[1.7,-.4]],dtype=torch.float64)
                    actual=fr.nn.EmbeddingBag.from_pretrained(helpers.native(w),freeze=False,mode=mode,padding_idx=0,scale_grad_by_freq=scale,include_last_offset=True)
                    expected=torch.nn.EmbeddingBag.from_pretrained(w.clone(),freeze=False,mode=mode,padding_idx=0,scale_grad_by_freq=scale,include_last_offset=True)
                    ids=torch.tensor([0,1,1,2,3]);offsets=torch.tensor([0,2,2,5])
                    ps=torch.tensor([.2,.4,-.1,.7,1.3],dtype=torch.float64,requires_grad=True) if weighted else None
                    ns=None if ps is None else helpers.native(ps,True)
                    a=actual(helpers.native(ids),helpers.native(offsets),ns);b=expected(ids,offsets,ps)
                    self.same(a,b);a.square().sum().backward();b.square().sum().backward()
                    self.same(actual.weight.grad,expected.weight.grad)
                    if weighted:self.same(ns.grad,ps.grad)

    def test_reset_preserves_shared_leaf_flags_grad_handles_and_double_precision(self):
        for cls,reference in ((fr.nn.Embedding,torch.nn.Embedding),(fr.nn.EmbeddingBag,torch.nn.EmbeddingBag)):
            for dtype,tdtype in (('float32',torch.float32),('float64',torch.float64)):
                source=fr.ones(5,7,dtype=dtype);model=cls.from_pretrained(source,freeze=True,padding_idx=-1)
                expected=reference.from_pretrained(torch.ones(5,7,dtype=tdtype),freeze=True,padding_idx=-1)
                param=model.weight;identity=param._id();gradient=fr.ones(5,7,dtype=dtype);param.grad=gradient
                torch_gradient=torch.ones(5,7,dtype=tdtype);expected.weight.grad=torch_gradient
                model.reset_parameters();expected.reset_parameters()
                self.assertIs(model.weight,param);self.assertEqual(param._id(),identity);self.assertEqual(param.grad._id(),gradient._id())
                self.assertIs(expected.weight.grad,torch_gradient);self.assertFalse(param.requires_grad)
                self.assertTrue(param._shares_storage(source));self.assertEqual(param.tolist()[-1],[0.]*7)
                self.assertTrue(any(v != 0 for row in param.tolist()[:-1] for v in row))
                if dtype=='float64':self.assertTrue(any(float(torch.tensor(v,dtype=torch.float32))!=v for row in param.tolist()[:-1] for v in row))
                before=state_helpers.flat_bits(param)
                with self.assertRaises(NotImplementedError):fr.capture(model.reset_parameters)
                self.assertEqual(state_helpers.flat_bits(param),before);self.assertEqual(param.grad._id(),gradient._id())
                with self.assertRaises(NotImplementedError):
                    fr.func.jvp(lambda x:(model.reset_parameters(),x)[1],(fr.ones(1),),(fr.ones(1),))
                self.assertEqual(state_helpers.flat_bits(param),before);self.assertEqual(param.grad._id(),gradient._id())
                param.requires_grad_(True);expected.weight.requires_grad_(True)
                stale=param.square().sum();reference_stale=expected.weight.square().sum()
                model.reset_parameters();expected.reset_parameters()
                with self.assertRaises((ValueError,RuntimeError)):stale.backward()
                with self.assertRaises(RuntimeError):reference_stale.backward()
                self.assertEqual(param.grad._id(),gradient._id());self.assertIs(expected.weight.grad,torch_gradient)

    def test_empty_shapes_and_validation_keep_unsupported_boundaries(self):
        for cls,reference in ((fr.nn.Embedding,torch.nn.Embedding),(fr.nn.EmbeddingBag,torch.nn.EmbeddingBag)):
            for rows,width in ((0,3),(3,0),(0,0)):
                actual=cls(rows,width,dtype='float64');expected=reference(rows,width,dtype=torch.float64)
                ids=fr.tensor([],dtype='int64');ti=torch.tensor([],dtype=torch.int64)
                a=actual(ids) if cls is fr.nn.Embedding else actual(ids,fr.tensor([0]))
                b=expected(ti) if cls is fr.nn.Embedding else expected(ti,torch.tensor([0]))
                self.same(a,b);a.sum().backward();b.sum().backward();self.same(actual.weight.grad,expected.weight.grad)
            for kw in ({'dtype':'int64'},{'dtype':'bool'},{'dtype':'float64','device':'cuda'},{'sparse':True}):
                with self.assertRaises((ValueError,TypeError,NotImplementedError)):cls(3,2,**kw)
            with self.assertRaisesRegex(ValueError,'capacity'):cls(sys.maxsize//8+1,1,dtype='float64')
            with self.assertRaises(NotImplementedError):fr.capture(lambda:cls(3,2,dtype='float64'))
            source=tensor([[.2,.7],[.5,-.3]],True);before=state_helpers.flat_bits(source)
            with self.assertRaises(ValueError):cls(3,2,_weight=source)
            with self.assertRaises((ValueError,TypeError)):cls.from_pretrained(source.transpose(0,1))
            self.assertEqual(state_helpers.flat_bits(source),before);self.assertIsNone(source.grad)

    def test_typed_sgd_multistep_matches_torch_and_retains_parameter_slots(self):
        for bag in (False,True):
            cls=fr.nn.EmbeddingBag if bag else fr.nn.Embedding;reference=torch.nn.EmbeddingBag if bag else torch.nn.Embedding
            options={'mode':'sum'} if bag else {}
            w=torch.tensor([[.2,.7],[.5,-.3],[1.2,.4]],dtype=torch.float64)
            actual=cls.from_pretrained(helpers.native(w),freeze=False,padding_idx=0,**options)
            expected=reference.from_pretrained(w.clone(),freeze=False,padding_idx=0,**options)
            config=dict(lr=.10000000000000002,momentum=.9000000000000001,weight_decay=.030000000000000002,dampening=.10000000000000002)
            optimizer=fr.optim.SGD(actual.parameters(),**config);oracle=torch.optim.SGD(expected.parameters(),**config)
            identity=actual.weight._id();ids=torch.tensor([0,1,1,2]);offsets=torch.tensor([0,2])
            for step in range(9):
                optimizer.zero_grad();oracle.zero_grad()
                a=actual(helpers.native(ids),helpers.native(offsets)) if bag else actual(helpers.native(ids))
                b=expected(ids,offsets) if bag else expected(ids)
                loss=(a-.17*step).square().sum();ref=(b-.17*step).square().sum()
                self.same(loss,ref);loss.backward();ref.backward();self.same(actual.weight.grad,expected.weight.grad)
                held=actual.weight.grad;optimizer.step();oracle.step()
                self.assertEqual(actual.weight.grad._id(),held._id());self.assertEqual(actual.weight._id(),identity)
                self.same(actual.weight,expected.weight)
                self.same(optimizer.state_dict()['state'][0]['momentum_buffer'],oracle.state[expected.weight]['momentum_buffer'])

    def test_typed_checkpoint_dtype_and_configuration_fail_atomically(self):
        source=fr.nn.EmbeddingBag(3,2,dtype='float64',mode='sum')
        optimizer=fr.optim.SGD(source.parameters(),lr=.10000000000000002,momentum=.9)
        source(fr.tensor([0,1]),fr.tensor([0])).square().sum().backward();optimizer.step();optimizer.zero_grad()
        rng=fr.Generator(7);saved=fr.checkpoint.snapshot_training(source,optimizers={'sgd':optimizer},generator=rng,step=1)
        for dtype,mode in (('float32','sum'),('float64','mean')):
            target=fr.nn.EmbeddingBag(3,2,dtype=dtype,mode=mode);opt=fr.optim.SGD(target.parameters(),lr=.1,momentum=.9);other=fr.Generator(11)
            before=state_helpers.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':opt},generator=other))
            identity=target.weight._id()
            with self.assertRaises(ValueError):saved.restore(target,optimizers={'sgd':opt},generator=other)
            self.assertEqual(state_helpers.digest(fr.checkpoint.snapshot_training(target,optimizers={'sgd':opt},generator=other)),before)
            self.assertEqual(target.weight._id(),identity)
        bad=optimizer.state_dict();bad['state'][0]['momentum_buffer']=fr.ones(3,2)
        before=state_helpers.state_bits(optimizer.state_dict())
        with self.assertRaises((ValueError,TypeError)):optimizer.load_state_dict(bad)
        self.assertEqual(state_helpers.state_bits(optimizer.state_dict()),before)

    def test_fresh_process_typed_sgd_aliases_full_checkpoint_and_rng_suffix(self):
        script=r'''
import json,struct,sys
import ferro as fr
class Model(fr.nn.Module):
 def __init__(self):
  super().__init__()
  source=fr.tensor([[.20000000000000004,.7],[.5,-.3],[1.2,.4]],dtype='float64')
  self.lookup=fr.nn.Embedding.from_pretrained(source,freeze=False,padding_idx=0,max_norm=1.)
  self.bag=fr.nn.EmbeddingBag.from_pretrained(source,freeze=False,padding_idx=0,mode='sum',scale_grad_by_freq=True,include_last_offset=True)
  self.register_buffer('source',source)
m=Model();o=fr.optim.SGD([{'params':m.lookup.parameters(),'lr':.10000000000000002,'momentum':.9000000000000001},{'params':m.bag.parameters(),'lr':.030000000000000002,'momentum':.7,'nesterov':True}],weight_decay=.03)
r=fr.Generator(71)
def step():
 ids=fr.tensor([0,1,1,2]);offsets=fr.tensor([0,2,4]);a=m.lookup(ids);b=m.bag(ids,offsets)
 loss=(a-fr.randn([4,2],dtype='float64',generator=r)).square().sum()+(b-fr.randn([2,2],dtype='float64',generator=r)).square().sum()
 value=loss.item();loss.backward();o.step();o.zero_grad();return value
if sys.argv[1]=='save':
 for _ in range(3):step()
 fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=3).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(m,optimizers={'sgd':o},generator=r)
losses=[step() for _ in range(5)]
state=fr.checkpoint.snapshot_training(m,optimizers={'sgd':o},generator=r,step=8).tensors()
def bits(t):
 flat=t.reshape([t.numel()]).tolist();code={'float32':'f','float64':'d','int64':'q','bool':'?'}[t.dtype]
 return[t.dtype,t.shape,struct.pack('<'+code*len(flat),*flat).hex()]
print(json.dumps({'losses':losses,'state':{k:bits(v) for k,v in state.items()},'next':fr.randn([9],dtype='float64',generator=r).tolist(),'aliases':[m.lookup.weight._shares_storage(m.source),m.bag.weight._shares_storage(m.source)]},sort_keys=True))
'''
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory() as directory:
            results=[subprocess.run([sys.executable,'-c',script,mode,str(Path(directory)/'state')],env=env,capture_output=True,text=True,timeout=60,check=True).stdout for mode in ('save','load')]
        first,second=map(json.loads,results);self.assertEqual(first,second);self.assertEqual(first['aliases'],[True,True])


if __name__=='__main__':unittest.main(verbosity=2)
