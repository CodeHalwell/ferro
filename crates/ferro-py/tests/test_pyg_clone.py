"""Native graph clone ownership against actual PyG, including batch state."""
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData, HeteroData as TorchHeteroData, Batch as TorchBatch
import ferro as fr
from ferro.graph import Data, Batch, HeteroData, HeteroBatch, EdgeIndex
from test_pyg_batch_hooks import pairs, PairData
from test_pyg_batch_options import fixtures
from test_pyg_hetero_batch import fixtures as hetero_fixtures


def native(value, grad=False):
    return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).removeprefix('torch.')).reshape(list(value.shape)).requires_grad_(grad)


class GraphCloneTests(unittest.TestCase):
    def close(self,a,b):
        actual=torch.tensor(a.tolist(),dtype=getattr(torch,a.dtype)).reshape(a.shape)
        torch.testing.assert_close(actual,b.detach(),atol=0,rtol=0,equal_nan=True)
        mask=b.detach()==0
        self.assertEqual(torch.signbit(actual[mask]).tolist(),torch.signbit(b.detach()[mask]).tolist())

    def fields(self,a,b):
        for key,value in a.items():
            if isinstance(value,int):
                self.assertEqual(value,b[key]);continue
            target=b[key]
            if not value.shape:target=target.reshape([])
            self.close(value,target)

    def test_homogeneous_selected_clones_preserve_class_layout_typed_bytes_and_gradients(self):
        class Custom(Data):
            __slots__=('_slot',)
            calls=0
            def __init__(self,marker,**kwargs):
                type(self).calls+=1;super().__init__(**kwargs);self._marker=marker;self._slot='slot'
        class TorchCustom(TorchData):
            __slots__=('_slot',)
            calls=0
            def __init__(self,marker,**kwargs):
                type(self).calls+=1;super().__init__(**kwargs);self._marker=marker;self._slot='slot'
        for dtype in (torch.float32,torch.float64):
            raw=torch.tensor([[.2,-0.,.7],[1.2,-.8,.5]],dtype=dtype)
            x=native(raw,True).T;expected=raw.T.detach().requires_grad_()
            ids=torch.tensor([2**60+3,-2**60+1],dtype=torch.int64)
            actual=Custom(object(),x=x,alias=x,ids=native(ids),mask=fr.tensor([True,False],dtype='bool'),num_nodes=3)
            reference=TorchCustom(actual._marker,x=expected,alias=expected,ids=ids,mask=torch.tensor([True,False]),num_nodes=3)
            for selected in ((),('x',),('absent',)):
                cloned,wanted=actual.clone(*selected),reference.clone(*selected)
                self.assertIs(type(cloned),Custom);self.assertIs(cloned._marker,actual._marker);self.assertEqual(cloned._slot,'slot')
                self.assertEqual(cloned.x.stride(),wanted.x.stride());self.fields(cloned,wanted)
                self.assertEqual(cloned.x is actual.x,wanted.x is reference.x)
                self.assertEqual(cloned.alias is actual.alias,wanted.alias is reference.alias)
                self.assertEqual(cloned.x is cloned.alias,wanted.x is wanted.alias)
            cloned,wanted=actual.clone(),reference.clone()
            (cloned.x.square().sum()+cloned.alias.sum()).backward()
            (wanted.x.square().sum()+wanted.alias.sum()).backward()
            self.close(x.grad,expected.grad)
            with fr.no_grad(),torch.no_grad():
                a,b=actual.clone('x'),reference.clone('x')
                self.assertEqual(a.x.requires_grad,b.x.requires_grad)
                self.assertIs(a.alias,actual.alias)
            self.assertEqual(Custom.calls,TorchCustom.calls)
        tiny=torch.tensor([torch.nextafter(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64)).item(),-0.],dtype=torch.float64)
        self.close(Data(value=native(tiny)).clone().value,tiny.clone())
        p=fr.nn.Parameter(fr.ones(2));q=torch.nn.Parameter(torch.ones(2))
        a,b=Data(x=p,num_nodes=2).clone(),TorchData(x=q,num_nodes=2).clone()
        self.assertNotIsInstance(a.x,fr.nn.Parameter)
        a.x.sum().backward();b.x.sum().backward();self.close(p.grad,q.grad)

    def test_standard_batch_current_extra_fields_selected_aliases_and_inverse_metadata(self):
        actual,reference=fixtures(True)
        batch,oracle=Batch.from_data_list(actual,follow_batch=['x']),TorchBatch.from_data_list(reference,follow_batch=['x'])
        batch.extra,oracle.extra=fr.tensor([7.,8.]),torch.tensor([7.,8.])
        batch.x=batch.x*1.25;oracle.x=oracle.x*1.25
        cloned,wanted=batch.clone('x'),oracle.clone('x')
        self.assertIs(type(cloned),type(batch));self.fields(cloned,wanted)
        self.assertIs(cloned.extra,batch.extra);self.assertIs(cloned.ptr,batch.ptr)
        self.assertIsNot(cloned.x,batch.x)
        for a,b in zip(cloned.to_data_list(),wanted.to_data_list()):self.fields(a,b)
        self.assertIsNot(cloned._slices,batch._slices)
        self.assertIs(wanted._slice_dict,oracle._slice_dict)  # Our metadata isolation is intentionally stronger.
        cloned._slices['x'][1]+=1
        self.assertNotEqual(cloned._slices['x'][1],batch._slices['x'][1])
        cloned=batch.clone()
        with fr.no_grad(),torch.no_grad():
            cloned.x.detach().fill_(3.);wanted.x.detach().fill_(3.)
        self.assertNotEqual(cloned.x.tolist(),batch.x.tolist())
        wanted=oracle.clone()
        cloned=batch.clone()
        cloned.x.square().sum().backward();wanted.x.square().sum().backward()
        for a,b in zip(actual,reference):self.close(a.x.grad,b.x.grad)

    def test_heterogeneous_clone_keeps_subclass_store_parents_extras_and_selected_layout(self):
        class Custom(HeteroData):
            calls=0
            def __init__(self,marker):type(self).calls+=1;super().__init__();self._marker=marker
        class TorchCustom(TorchHeteroData):
            calls=0
            def __init__(self,marker):type(self).calls+=1;super().__init__();self._marker=marker
        original,reference=Custom('m'),TorchCustom('m')
        raw=torch.arange(6.).reshape(2,3)
        original['a'].x=native(raw,True).T;reference['a'].x=raw.T.detach().requires_grad_()
        original['a'].num_nodes=reference['a'].num_nodes=3
        original['a','to','a'].edge_index=fr.tensor([[0],[1]],dtype='int64');reference['a','to','a'].edge_index=torch.tensor([[0],[1]])
        cloned,wanted=original.clone('x'),reference.clone('x')
        self.assertIs(type(cloned),Custom);self.assertEqual(cloned._marker,'m');self.assertEqual(Custom.calls,1)
        self.assertIs(cloned['a']._parent,cloned);self.assertIsNot(cloned['a'],original['a'])
        self.assertIs(cloned['a','to','a'].edge_index,original['a','to','a'].edge_index)
        self.assertEqual(cloned['a'].x.stride(),wanted['a'].x.stride())
        cloned['a'].num_nodes=5
        self.assertEqual(cloned['a','to','a'].size(),(5,5));self.assertEqual(original['a','to','a'].size(),(3,3))
        actual,reference=hetero_fixtures(True)
        batch,oracle=HeteroBatch.from_data_list(actual),TorchBatch.from_data_list(reference)
        batch.extra,oracle.extra=fr.tensor([7.]),torch.tensor([7.])
        batch['a'].extra,oracle['a'].extra=fr.ones(batch['a'].num_nodes),torch.ones(oracle['a'].num_nodes)
        cloned,wanted=batch.clone('x'),oracle.clone('x')
        self.assertEqual(set(cloned._global),set(wanted._global_store))
        self.assertEqual(set(cloned['a'].keys()),set(wanted['a'].keys()))
        self.assertIs(cloned.extra,batch.extra);self.assertIs(cloned['a'].extra,batch['a'].extra)
        self.assertIs(type(cloned['a']),type(batch['a']));self.assertIs(cloned['a']._parent,cloned)
        self.assertIs(cloned._global_view._mapping,cloned._global);self.assertIs(cloned._global_view._parent,cloned)
        for a,b in zip(cloned.to_data_list(),wanted.to_data_list()):self.fields(a['a'],b['a'])
        cloned['a'].x.square().sum().backward();wanted['a'].x.square().sum().backward()
        for a,b in zip(actual,reference):self.close(a['a'].x.grad,b['a'].x.grad)
        cloned._slices['a']['x']=(0,1,1,1,1)
        self.assertNotEqual(cloned._slices['a']['x'],batch._slices['a']['x'])

    def test_custom_batch_clone_preserves_live_offsets_class_and_private_templates(self):
        actual,reference=pairs('float64')
        batch,oracle=Batch.from_data_list(actual,follow_batch=['x_s']),TorchBatch.from_data_list(reference,follow_batch=['x_s'])
        batch._marker=oracle._marker=object()
        batch.x_s=batch.x_s*2;oracle.x_s=oracle.x_s*2
        cloned,wanted=batch.clone('x_s'),oracle.clone('x_s')
        self.assertIsInstance(cloned,PairData);self.assertIs(type(cloned),type(batch));self.assertIs(cloned._marker,batch._marker)
        self.assertIs(cloned.x_t,batch.x_t);self.assertIsNot(cloned._hook_plans,batch._hook_plans)
        self.assertIsNot(cloned._hook_templates,batch._hook_templates)
        cloned._hook_templates[0]._field_roles['new']='graph'
        self.assertNotIn('new',batch._hook_templates[0]._field_roles)
        for a,b in zip(cloned[[2,0,2]],wanted[[2,0,2]]):
            self.assertIsInstance(a,PairData);self.fields(a,b)
        cloned.x_s.square().sum().backward();wanted.x_s.square().sum().backward()
        for a,b in zip(actual,reference):self.close(a.x_s.grad,b.x_s.grad)

    def test_native_topology_counts_empty_clone_and_selector_validation(self):
        original=Data(edge_index=EdgeIndex(5,[0],[1]))
        cloned=original.clone()
        self.assertIsNot(cloned.edge_index,original.edge_index)
        self.assertEqual(cloned.num_nodes,5);self.assertEqual(cloned.edge_index.src,[0]);self.assertEqual(cloned.edge_index.dst,[1])
        self.assertIs(original.clone('absent').edge_index,original.edge_index)
        for name in (3,False,['x']):
            with self.assertRaises(TypeError):original.clone(name)
        import ferro._graph_clone as implementation
        with patch.object(implementation,'_clone',side_effect=AssertionError('cloned before metadata validation')):
            bad=Data(num_nodes=0,first=fr.tensor([1.]),last=fr.ones(1,dtype='float64').expand([sys.maxsize//8+1]))
            with self.assertRaisesRegex(ValueError,'capacity'):bad.clone()
        empty=Data(num_nodes=0,x=fr.zeros(0,8,dtype='float64',requires_grad=True)[:,4:])
        value=empty.clone().x;value.sum().backward()
        self.assertEqual(value.shape,[0,4]);self.assertEqual(empty.x.grad.shape,[0,4])

    def test_huge_empty_typed_fields_clone_without_host_materialization(self):
        script='''
import ferro as fr
from ferro.graph import Data
from ferro._graph_clone import clone
for dtype in ('float32','float64','int64','bool'):
    source=fr.zeros(1,0,dtype=dtype,requires_grad=dtype.startswith('float'))
    data=Data(value=source.expand([10**12,0]))
    copied=clone(data)
    if copied.value.shape != [10**12,0] or copied.value.dtype != dtype:raise RuntimeError('wrong empty clone')
    if dtype.startswith('float'):
        copied.value.sum().backward()
        if source.grad.shape != [1,0]:raise RuntimeError('wrong empty input gradient')
print('huge-empty-clone-ok')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('huge-empty-clone-ok',result.stdout)


if __name__=='__main__':unittest.main()
