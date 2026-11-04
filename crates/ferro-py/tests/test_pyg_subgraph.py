"""Aligned container subgraphs against actual PyG with explicit schema limits."""
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data as TorchData, HeteroData as TorchHeteroData, Batch as TorchBatch
import ferro as fr
from ferro.graph import Data,Batch,HeteroData,HeteroBatch,EdgeIndex,NeighborLoader
from test_pyg_batch_hooks import PairData
from test_pyg_clone import native


def ids(values):
    return fr.tensor(values,dtype='bool' if values and isinstance(values[0],bool) else 'int64')


def torch_ids(values):
    return torch.tensor(values,dtype=torch.bool if values and isinstance(values[0],bool) else torch.int64)


def homogeneous(dtype):
    raw=torch.arange(12,dtype=dtype).reshape(3,4)/13-.4
    x=raw.T.detach().requires_grad_();edge=torch.tensor([[0,1,2,2,3,0],[1,2,0,2,0,3]])
    attr=(torch.arange(12,dtype=dtype).reshape(6,2)/17).requires_grad_()
    other=torch.arange(4,dtype=dtype).requires_grad_();context=torch.tensor(.7,dtype=dtype,requires_grad=True)
    a=Data(x=native(raw,True).T,edge_index=native(edge),edge_attr=native(attr,True),num_nodes=4,
           extra=native(other,True),labels=fr.tensor([2**60+i for i in range(4)],dtype='int64'),mask=ids([True,False,True,False]),context=native(context,True))
    b=TorchData(x=x,edge_index=edge,edge_attr=attr,num_nodes=4,extra=other,labels=torch.tensor([2**60+i for i in range(4)]),mask=torch.tensor([True,False,True,False]),context=context)
    return a,b


def heterogeneous(dtype):
    a,b=HeteroData(),TorchHeteroData()
    for key,count in (('a',4),('b',3),('z',0)):
        raw=torch.arange(2*count,dtype=dtype).reshape(2,count)/11
        a[key].x=native(raw,True).T;b[key].x=raw.T.detach().requires_grad_()
        a[key].num_nodes=b[key].num_nodes=count
        a[key].y=fr.tensor([2**60+i for i in range(count)],dtype='int64');b[key].y=torch.tensor([2**60+i for i in range(count)],dtype=torch.int64)
        a[key].scale=fr.tensor(.3,dtype=str(dtype).split('.')[-1],requires_grad=True);b[key].scale=torch.tensor(.3,dtype=dtype,requires_grad=True)
    for relation,edge in ((('a','to','b'),[[0,1,2,2,3],[1,2,0,2,0]]),(('b','back','a'),[[0,1,2],[2,0,3]]),(('z','self','z'),[[],[]])):
        edge=torch.tensor(edge,dtype=torch.int64).reshape(2,-1)
        a[relation].edge_index=native(edge);b[relation].edge_index=edge
        attr=torch.arange(edge.shape[1]*2,dtype=dtype).reshape(-1,2)/19
        a[relation].edge_attr=native(attr,True);b[relation].edge_attr=attr.detach().requires_grad_()
        a[relation].scale=fr.tensor(.8,dtype=str(dtype).split('.')[-1]);b[relation].scale=torch.tensor(.8,dtype=dtype)
    a.context=fr.tensor(.7,dtype=str(dtype).split('.')[-1],requires_grad=True);b.context=torch.tensor(.7,dtype=dtype,requires_grad=True)
    return a,b


class ContainerSubgraphTests(unittest.TestCase):
    def close(self,a,b):
        value=torch.tensor(a.tolist(),dtype=getattr(torch,a.dtype)).reshape(a.shape)
        torch.testing.assert_close(value,b.detach(),rtol=0,atol=0,equal_nan=True)

    def fields(self,a,b):
        for key,value in a.items():
            if isinstance(value,int):self.assertEqual(value,b[key])
            else:self.close(value,b[key])

    def hetero_fields(self,a,b):
        self.assertEqual(a.metadata(),b.metadata())
        self.fields(a._global_view,b._global_store)
        for key in (*a.node_types,*a.edge_types):
            self.fields(a[key],b[key]);self.assertIs(a[key]._parent,a)

    def test_homogeneous_node_order_repeats_masks_empties_and_all_field_gradients(self):
        for dtype in (torch.float32,torch.float64):
            for selection in ([2,0],[2,0,2],[True,False,True,False],[],[3,2,1,0]):
                a,b=homogeneous(dtype);out,wanted=a.subgraph(subset=ids(selection)),b.subgraph(subset=torch_ids(selection))
                self.fields(out,wanted);self.assertTrue(out.validate());self.assertEqual(out.num_nodes,wanted.num_nodes)
                self.assertIs(out.context,a.context);self.assertIsNot(out.x,a.x);self.assertIsNot(out.edge_attr,a.edge_attr)
                loss=out.x.square().sum()+out.extra.square().sum()+out.edge_attr.square().sum()+out.context
                reference=wanted.x.square().sum()+wanted.extra.square().sum()+wanted.edge_attr.square().sum()+wanted.context
                loss.backward();reference.backward()
                for key in ('x','extra','edge_attr','context'):self.close(a[key].grad,b[key].grad)
                self.assertEqual(a.num_nodes,4);self.assertIsNot(out._store,a._store);self.assertIsNot(out._field_roles,a._field_roles)

    def test_homogeneous_edge_order_repeats_aliases_and_native_isolates(self):
        for dtype in (torch.float32,torch.float64):
            for selection in ([3,1,3],[],[True,False,True,False,False,True]):
                a,b=homogeneous(dtype);out,wanted=a.edge_subgraph(subset=ids(selection)),b.edge_subgraph(subset=torch_ids(selection))
                self.fields(out,wanted);self.assertEqual(out.num_nodes,4);self.assertIs(out.x,a.x);self.assertIs(out.context,a.context)
                out.edge_attr.square().sum().backward();wanted.edge_attr.square().sum().backward();self.close(a.edge_attr.grad,b.edge_attr.grad)
        a=Data(edge_index=EdgeIndex(7,[0,2],[1,0]))
        out=a.edge_subgraph(ids([]));self.assertEqual(out.num_nodes,7);self.assertEqual(out.edge_index.shape,[2,0]);self.assertEqual(out.edge_index.dtype,'int64')
        out=a.subgraph(ids([6,2,0]));self.assertEqual(out.num_nodes,3);self.assertEqual(out.edge_index.tolist(),[[1],[2]])
        # Actual PyG's edge-only inference loses isolates; we retain the original count explicitly.
        b=TorchData(edge_index=torch.tensor([[0,2],[1,0]]));self.assertEqual(b.num_nodes,3)
        self.assertEqual(b.edge_subgraph(torch_ids([])).num_nodes,0)
        self.assertEqual(Data(edge_index=fr.tensor([[0,2],[1,0]],dtype='int64')).edge_subgraph(ids([])).num_nodes,3)

    def test_heterogeneous_partial_node_subsets_rectangular_topology_and_gradients(self):
        for dtype in (torch.float32,torch.float64):
            for selection in ({'a':[2,0,2]},{'a':[],'b':[2,0]},{'a':[True,False,True,False],'b':[2,0]},{}):
                a,b=heterogeneous(dtype)
                out=a.subgraph(subset_dict={key:ids(value) for key,value in selection.items()})
                wanted=b.subgraph(subset_dict={key:torch_ids(value) for key,value in selection.items()})
                self.hetero_fields(out,wanted);self.assertTrue(out.validate());self.assertIs(out.context,a.context)
                for key in a.node_types:
                    self.assertEqual(out[key].x is a[key].x,key not in selection)
                    self.assertIs(out[key].scale,a[key].scale);self.assertIsNot(out[key],a[key])
                loss=out.context
                reference=wanted.context
                for key in a.node_types:
                    loss=loss+out[key].x.square().sum()+out[key].scale
                    reference=reference+wanted[key].x.square().sum()+wanted[key].scale
                for key in a.edge_types:
                    loss=loss+out[key].edge_attr.square().sum();reference=reference+wanted[key].edge_attr.square().sum()
                loss.backward();reference.backward()
                self.close(a.context.grad,b.context.grad)
                for key in a.node_types:
                    self.close(a[key].x.grad,b[key].x.grad);self.close(a[key].scale.grad,b[key].scale.grad)
                for key in a.edge_types:self.close(a[key].edge_attr.grad,b[key].edge_attr.grad)

    def test_heterogeneous_edge_subsets_keep_untouched_tensor_aliases(self):
        a,b=heterogeneous(torch.float64);relation=('a','to','b')
        for selection in ([4,0,4],[],[True,False,True,False,True]):
            out=a.edge_subgraph(subset_dict={relation:ids(selection)});wanted=b.edge_subgraph(subset_dict={relation:torch_ids(selection)})
            self.hetero_fields(out,wanted);self.assertTrue(out.validate())
            for key in a.node_types:self.assertIs(out[key].x,a[key].x)
            self.assertIs(out['b','back','a'].edge_index,a['b','back','a'].edge_index)
            self.assertIs(out['b','back','a'].edge_attr,a['b','back','a'].edge_attr)
        out=a.edge_subgraph({relation:ids([4,0,4])});wanted=b.edge_subgraph({relation:torch_ids([4,0,4])})
        out[relation].edge_attr.square().sum().backward();wanted[relation].edge_attr.square().sum().backward()
        self.close(a[relation].edge_attr.grad,b[relation].edge_attr.grad)

    def test_subclasses_private_slots_graph_roles_and_store_mappings_survive(self):
        class Custom(Data):
            __slots__=('_slot',)
            def __init__(self,marker):super().__init__(num_nodes=4);self._marker=marker;self._slot='private';self.x=fr.ones(4,2)
        original=Custom(object());out=original.subgraph(ids([3,1,3]))
        self.assertIs(type(out),Custom);self.assertIs(out._marker,original._marker);self.assertEqual(out._slot,'private')
        class CustomHetero(HeteroData):
            def __init__(self,marker):super().__init__();self._marker=marker;self['a'].x=fr.ones(2,1)
        original=CustomHetero(object());out=original.subgraph({'a':ids([1])})
        self.assertIs(type(out),CustomHetero);self.assertIs(out._marker,original._marker)
        self.assertIs(out._global_view._parent,out);self.assertIs(out._global_view._mapping,out._global)
        out['a'].num_nodes=7;self.assertEqual(original['a'].num_nodes,2)
        a=Data(x=fr.ones(4,1),edge_index=fr.tensor([[0,1,2,2],[1,0,0,2]],dtype='int64'),
               custom=fr.tensor([1.,2.,3.,4.]),field_roles={'custom':'node'})
        out=a.subgraph(ids([0,1]));self.assertEqual(out.num_edges,2);self.assertEqual(out.num_nodes,2)
        self.assertEqual(out._role('custom',2,2),'node');self.assertTrue(out.validate())
        graph=Data(num_nodes=3,context=fr.tensor(.7));out=graph.subgraph(ids([2,0]));self.assertEqual(out.num_nodes,2);self.assertIs(out.context,graph.context)

    def test_schema_and_selector_validation_precede_all_field_gathers(self):
        a,b=homogeneous(torch.float32)
        for bad in (fr.tensor([.1]),fr.tensor(1,dtype='int64'),fr.tensor([[0]],dtype='int64'),ids([-1]),ids([4]),ids([True])):
            with self.assertRaises((TypeError,ValueError)):a.subgraph(bad)
        with self.assertRaises(TypeError):a.subgraph([0])
        for operation in ('subgraph','edge_subgraph'):
            with self.assertRaises(NotImplementedError):getattr(Batch.from_data_list([a]),operation)(ids([]))
            with self.assertRaises(NotImplementedError):getattr(PairData(x_s=fr.ones(1,1)),operation)(ids([]))
            indexed=a.clone();indexed.edge_label_index=fr.tensor([[0],[1]],dtype='int64')
            with self.assertRaises(NotImplementedError):getattr(indexed,operation)(ids([]))
        h,_=heterogeneous(torch.float32);before=h.metadata()
        with self.assertRaises(KeyError):h.subgraph({'unknown':ids([])})
        self.assertEqual(h.metadata(),before)
        with self.assertRaises(KeyError):h.edge_subgraph({('unknown','to','a'):ids([])})
        with self.assertRaises(TypeError):h.subgraph(ids([]))
        with self.assertRaises(NotImplementedError):HeteroBatch.from_data_list([h]).subgraph({})
        sampled=next(iter(NeighborLoader(a,[1],input_nodes=ids([0]),batch_size=1)))
        with self.assertRaises(NotImplementedError):sampled.subgraph(ids([]))
        h=HeteroData();h['a'].x=fr.ones(2,1);h['a','to','a'].edge_attr=fr.ones(1,1)
        with self.assertRaises(NotImplementedError):h.subgraph({})
        import ferro._graph_subgraph as implementation
        with patch.object(implementation,'_gather',side_effect=AssertionError('gather during capture')):
            for graph, selection in ((a,ids([0])),(heterogeneous(torch.float32)[0],{})):
                for operation in ('subgraph','edge_subgraph'):
                    with self.assertRaisesRegex(NotImplementedError,'eager'):
                        fr.capture(lambda: getattr(graph,operation)(selection))
        oversized=Data(x=fr.ones(1,1).expand([1,sys.maxsize//8+1]))
        with patch.object(implementation,'_gather',side_effect=AssertionError('gather before preflight')):
            with self.assertRaisesRegex(ValueError,'capacity'):oversized.subgraph(ids([0,0]))
        oversized_ids=fr.zeros(1,dtype='int64').expand([sys.maxsize//16+1])
        with self.assertRaisesRegex(ValueError,'capacity'):a.edge_subgraph(oversized_ids)

    def test_huge_empty_features_and_omitted_typed_nodes_do_not_build_dense_metadata(self):
        script='''
import ferro as fr
from ferro.graph import Data,HeteroData
size=10**12
source=fr.zeros(1,0,dtype='float64',requires_grad=True)
data=Data(x=source.expand([size,0]),edge_index=fr.tensor([[],[]],dtype='int64'))
selected=data.subgraph(fr.tensor([size-1,0,size-1],dtype='int64'))
if selected.x.shape != [3,0] or selected.num_nodes != 3:raise RuntimeError('wrong empty feature selection')
selected.x.sum().backward()
if source.grad.shape != [1,0]:raise RuntimeError('wrong empty source gradient')
h=HeteroData();h['a'].x=fr.ones(1,1);h['b'].x=fr.zeros(1,0).expand([size,0]);h['a','to','b'].edge_index=fr.tensor([[],[]],dtype='int64')
out=h.subgraph({'a':fr.tensor([],dtype='int64')})
if out['b'].x is not h['b'].x or out['b'].num_nodes != size:raise RuntimeError('omitted type lost ownership')
print('huge-empty-subgraph-ok')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('huge-empty-subgraph-ok',result.stdout)


if __name__=='__main__':unittest.main()
