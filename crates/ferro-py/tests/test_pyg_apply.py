"""Typed graph field ownership and selection against actual PyG."""
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData, HeteroData as TorchHeteroData, Batch as TorchBatch
import ferro as fr
from ferro.graph import Data, Batch, HeteroData, HeteroBatch, EdgeIndex
from test_pyg_clone import native
from test_pyg_batch_hooks import pairs, PairData
from test_pyg_batch_options import fixtures
from test_pyg_hetero_batch import fixtures as hetero_fixtures


class GraphApplyTests(unittest.TestCase):
    def close(self,a,b):
        value=torch.tensor(a.tolist(),dtype=getattr(torch,a.dtype)).reshape(a.shape)
        torch.testing.assert_close(value,b.detach(),rtol=0,atol=0,equal_nan=True)
        mask=b.detach()==0
        self.assertEqual(torch.signbit(value[mask]).tolist(),torch.signbit(b.detach()[mask]).tolist())

    def fields(self,a,b):
        for key,value in a.items():
            if isinstance(value,int):self.assertEqual(value,b[key])
            else:self.close(value,b[key].reshape(value.shape))

    def test_apply_selected_order_duplicates_current_values_and_sequential_failures(self):
        for selected in ((),('y','x'),('x','x'),('absent',),('y','absent','x','y')):
            x=native(torch.tensor([1.]),True);y=native(torch.tensor([2.]),True)
            a=Data(x=x,y=y);b=TorchData(x=torch.tensor([1.],requires_grad=True),y=torch.tensor([2.],requires_grad=True))
            old_b=(b.x,b.y);seen_a=[];seen_b=[]
            def fa(value):seen_a.append(value.tolist());return value*2
            def fb(value):seen_b.append(value.tolist());return value*2
            self.assertIs(a.apply(fa,*selected),a);self.assertIs(b.apply(fb,*selected),b)
            self.assertEqual(seen_a,seen_b);self.fields(a,b)
            (a.x.square().sum()+a.y.sum()).backward();(b.x.square().sum()+b.y.sum()).backward()
            for actual,wanted in zip((x,y),old_b):self.close(actual.grad,wanted.grad)
        a,b=Data(x=fr.tensor([1.]),y=fr.tensor([2.])),TorchData(x=torch.tensor([1.]),y=torch.tensor([2.]))
        def fail(value):
            if value.tolist()==[2.]:raise RuntimeError('second callback')
            return value+10
        for data in (a,b):
            with self.assertRaisesRegex(RuntimeError,'second callback'):data.apply(fail)
        self.fields(a,b);self.assertEqual(a.x.tolist(),[11.])
        original=a.y
        with self.assertRaises(TypeError):a.apply(lambda value:None,'y')
        self.assertIs(a.y,original)

    def test_detach_replaces_only_slots_preserving_old_graph_and_storage_version(self):
        for dtype in (torch.float32,torch.float64):
            x=native(torch.tensor([.25,-0.,2.],dtype=dtype),True);expected=torch.tensor([.25,-0.,2.],dtype=dtype,requires_grad=True)
            a,b=Data(x=x,alias=x),TorchData(x=expected,alias=expected)
            old,wanted=x.square().sum(),expected.square().sum()
            self.assertIs(a.detach('x'),a);self.assertIs(b.detach('x'),b)
            self.assertIsNot(a.x,x);self.assertFalse(a.x.requires_grad);self.assertTrue(x.requires_grad)
            self.assertIs(a.alias,x);self.assertIs(b.alias,expected)
            old.backward();wanted.backward();self.close(x.grad,expected.grad)
            stale=x.square().sum();torch_stale=expected.square().sum()
            with fr.no_grad(),torch.no_grad():a.x.fill_(3.);b.x.fill_(3.)
            self.close(x,expected)
            with self.assertRaisesRegex(ValueError,'modified by an inplace operation'):stale.backward()
            with self.assertRaisesRegex(RuntimeError,'modified by an inplace operation'):torch_stale.backward()
        a=Data(x=fr.nn.Parameter(fr.ones(2)))
        old=a.x;a.detach();self.assertNotIsInstance(a.x,fr.nn.Parameter);self.assertTrue(old.requires_grad)
        a=Data(x=fr.ones(1,requires_grad=True))
        old=a.x
        with self.assertRaises(NotImplementedError):a.detach_()
        self.assertIs(a.x,old);self.assertTrue(a.x.requires_grad)

    def test_contiguous_typed_layout_aliases_cpu_identity_and_input_gradients(self):
        for dtype in (torch.float32,torch.float64,torch.int64,torch.bool):
            raw=torch.tensor([[0,1,0],[1,0,1]],dtype=dtype)
            if dtype.is_floating_point:raw[0,0]=-0.
            x=native(raw,dtype.is_floating_point).T;expected=raw.T.detach().requires_grad_(dtype.is_floating_point)
            a,b=Data(x=x,alias=x),TorchData(x=expected,alias=expected)
            self.assertIs(a.contiguous('x'),a);self.assertIs(b.contiguous('x'),b)
            self.assertIsNot(a.x,x);self.assertIs(a.alias,x);self.assertEqual(a.x.stride(),b.x.stride());self.close(a.x,b.x)
            new=a.x
            a.contiguous('x');self.assertIs(a.x,new)
            for data in (a,b):
                saved=data.x
                self.assertIs(data.cpu('x','absent'),data);self.assertIs(data.x,saved)
                self.assertIs(data.to('cpu','x',non_blocking=True),data);self.assertIs(data.x,saved)
            if dtype.is_floating_point:
                a.x.square().sum().backward();b.x.square().sum().backward();self.close(x.grad,expected.grad)
        tiny=torch.tensor([torch.nextafter(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64)).item(),-0.],dtype=torch.float64)
        a=Data(x=native(tiny));a.contiguous().detach().cpu();self.close(a.x,tiny)

    def test_standard_and_custom_batches_keep_current_inverse_slices_and_repeated_gradients(self):
        for custom in (False,True):
            actual,expected=pairs('float64') if custom else fixtures(True)
            field='x_s' if custom else 'x'
            a,b=Batch.from_data_list(actual,follow_batch=[field]),TorchBatch.from_data_list(expected,follow_batch=[field])
            cls=type(a);metadata=a._hook_plans if custom else a._slices
            a._marker=object();marker=a._marker
            a.apply(lambda value:value*2+.25,field);b.apply(lambda value:value*2+.25,field)
            self.assertIs(type(a),cls);self.assertIs(a._marker,marker)
            self.assertIs(metadata,a._hook_plans if custom else a._slices)
            for left,right in zip(a[[2,0,2]],b[[2,0,2]]):self.fields(left,right)
            losses_a=[getattr(item,field).square().sum() for item in a[[2,0,2]]]
            losses_b=[getattr(item,field).square().sum() for item in b[[2,0,2]]]
            sum(losses_a[1:],losses_a[0]).backward();sum(losses_b[1:],losses_b[0]).backward()
            for left,right in zip(actual,expected):
                if getattr(left,field).grad is None:self.assertIsNone(getattr(right,field).grad)
                else:self.close(getattr(left,field).grad,getattr(right,field).grad)
            a.detach().contiguous().cpu();b.detach().contiguous().cpu()
            self.fields(a,b)
            for left,right in zip(a.to_data_list(),b.to_data_list()):
                self.fields(left,right);self.assertFalse(getattr(left,field).requires_grad)
            self.assertIs(metadata,a._hook_plans if custom else a._slices)

    def test_heterogeneous_store_order_selected_names_and_batched_parent_identity(self):
        a,b=HeteroData(),TorchHeteroData()
        a.context,b.context=fr.tensor([1.]),torch.tensor([1.])
        for key,number in (('a',2.),('b',3.),(('a','to','b'),4.)):
            a[key].value,b[key].value=fr.tensor([number]),torch.tensor([number])
        seen_a=[];seen_b=[]
        def fa(v):seen_a.append(v.tolist());return v*2
        def fb(v):seen_b.append(v.tolist());return v*2
        a.apply(fa,'value','context','value');b.apply(fb,'value','context','value')
        self.assertEqual(seen_a,seen_b)
        for left,right in zip(a.stores,b.stores):self.fields(left,right)
        actual,expected=hetero_fixtures(True)
        a,b=HeteroBatch.from_data_list(actual,follow_batch=['x']),TorchBatch.from_data_list(expected,follow_batch=['x'])
        stores=tuple(a.stores);metadata=a._slices;parents=[s._parent for s in stores]
        a.apply(lambda v:v*1.5+.125,'x','edge_attr');b.apply(lambda v:v*1.5+.125,'x','edge_attr')
        for left,right in zip(a[[2,0,2]],b[[2,0,2]]):
            for sa,sb in zip(left.stores,right.stores):self.fields(sa,sb)
        (a['a'].x.square().sum()+a['a','to','b'].edge_attr.sum()).backward()
        (b['a'].x.square().sum()+b['a','to','b'].edge_attr.sum()).backward()
        for left,right in zip(actual,expected):
            self.close(left['a'].x.grad,right['a'].x.grad)
            self.close(left['a','to','b'].edge_attr.grad,right['a','to','b'].edge_attr.grad)
        a.detach('x').contiguous().cpu();b.detach('x').contiguous().cpu()
        self.assertEqual(tuple(a.stores),stores);self.assertIs(a._slices,metadata)
        self.assertTrue(all(s._parent is parent is a for s,parent in zip(stores,parents)))
        self.assertIs(a._global_view._mapping,a._global)
        for left,right in zip(a.to_data_list(),b.to_data_list()):
            for sa,sb in zip(left.stores,right.stores):self.fields(sa,sb)
            self.assertFalse(left['a'].x.requires_grad)

    def test_typed_metadata_native_topology_and_invalid_requests_are_explicit(self):
        topology=EdgeIndex(7,[0],[1]);a=Data(x=fr.ones(2),edge_index=topology,num_nodes=7)
        seen=[]
        a.apply(lambda value:(seen.append(value),value)[1])
        self.assertEqual(seen,[a.x]);self.assertIs(a.edge_index,topology);self.assertEqual(a.num_nodes,7)
        a.detach().contiguous().cpu();self.assertIs(a.edge_index,topology)
        h=HeteroData();h['a'].num_nodes=3;h['a'].x=fr.ones(3)
        seen=[];h.apply(lambda value:(seen.append(value),value)[1])
        self.assertEqual(seen,[h['a'].x]);self.assertEqual(h['a'].num_nodes,3)
        # PyG invokes callbacks on integer metadata too; this typed API does not.
        target=TorchData(x=torch.ones(2),num_nodes=7);target.apply(lambda value:value*2)
        self.assertEqual(target.num_nodes,14)
        for data in (a,h,Batch.from_data_list([Data(x=fr.ones(1))]),HeteroBatch.from_data_list([h])):
            for method in ('detach','contiguous','cpu'):
                with self.assertRaises(TypeError):getattr(data,method)(1)
            with self.assertRaises(TypeError):data.apply(None)
            with self.assertRaises(TypeError):data.apply(lambda value:value,False)
            with self.assertRaises(TypeError):data.to('cpu',non_blocking=1)
            for device in ('cuda','cuda:0','cpu:0',None):
                with self.assertRaises(NotImplementedError):data.to(device)
            with self.assertRaises(NotImplementedError):data.detach_('x')
        edge=fr.tensor([[0],[1]],dtype='int64');a=Data(edge_index=edge)
        with self.assertRaises((TypeError,ValueError)):a.apply(lambda value:value.float(),'edge_index')
        self.assertIs(a.edge_index,edge)
        import ferro._graph_apply as implementation
        oversized=fr.zeros(2,1,dtype='int64').expand([2,sys.maxsize//16+1])
        with patch.object(implementation,'_edge_parts',side_effect=AssertionError('read before capacity check')):
            with self.assertRaisesRegex(ValueError,'capacity'):a.apply(lambda value:oversized,'edge_index')
        self.assertIs(a.edge_index,edge)

    def test_huge_empty_detach_contiguous_and_cpu_preserve_metadata_without_host_reads(self):
        script='''
import ferro as fr
from ferro.graph import Data
for dtype in ('float32','float64','int64','bool'):
 source=fr.zeros(1,0,dtype=dtype,requires_grad=dtype.startswith('float'))
 value=source.expand([10**12,0]);data=Data(value=value)
 old=data.value
 if data.cpu().value is not old:raise RuntimeError('CPU handle changed')
 data.contiguous()
 if data.value.shape != [10**12,0]:raise RuntimeError('wrong contiguous shape')
 if dtype.startswith('float'):
  data.value.sum().backward()
  if source.grad.shape != [1,0]:raise RuntimeError('wrong input gradient')
 data.detach()
 if data.value.requires_grad or data.value.shape != [10**12,0]:raise RuntimeError('wrong detached field')
print('huge-empty-apply-ok')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('huge-empty-apply-ok',result.stdout)


if __name__=='__main__':unittest.main()
