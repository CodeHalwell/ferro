"""Actual PyG incidence cleanup, typed field ownership, gradients and state."""
import json,math,subprocess,sys,tempfile,unittest,warnings
from unittest.mock import patch
import torch
from torch_geometric.data import Data as TData,HeteroData as THetero
from torch_geometric.transforms import RemoveIsolatedNodes as Reference
import ferro as fr
from ferro.graph import Data,HeteroData,Batch,HeteroBatch,EdgeIndex,InMemoryDataset
from ferro.transforms import RemoveIsolatedNodes


def native(value,grad=False):return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).split('.')[-1],requires_grad=grad).reshape(list(value.shape))
def values(value):return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)


class RemoveIsolatedTests(unittest.TestCase):
    def close(self,a,b):torch.testing.assert_close(values(a),b.detach(),rtol=1e-6,atol=1e-7,equal_nan=True)
    def homogeneous(self,dtype=torch.float32,explicit=True,edge=None):
        edge=[[5,1,5,3,1],[1,5,1,3,1]]if edge is None else edge
        x=torch.arange(21,dtype=dtype).reshape(3,7).T.detach().requires_grad_();y=torch.linspace(-.3,.6,7,dtype=dtype,requires_grad=True);a=torch.linspace(.1,.7,len(edge[0]),dtype=dtype,requires_grad=True);index=torch.tensor(edge,dtype=torch.long).reshape(2,-1)
        extra=dict(num_nodes=7)if explicit else{}
        left=Data(x=native(x,True),y=native(y,True),edge_index=native(index),edge_attr=native(a,True),mask=fr.tensor([True,False,True,False,True,False,True]),node_type=fr.tensor([0,1,2,3,4,5,6],dtype='int64'),**extra)
        right=TData(x=x,y=y,edge_index=index,edge_attr=a,mask=torch.tensor([True,False,True,False,True,False,True]),node_type=torch.arange(7),**extra)
        return left,right
    def compare_fields(self,left,right,keys):
        loss=target=None
        for key in keys:
            self.close(left[key],right[key])
            if right[key].requires_grad:
                seed=torch.linspace(-.2,.9,right[key].numel(),dtype=right[key].dtype).reshape(right[key].shape)
                term=(left[key]*native(seed)).sum();wanted=(right[key]*seed).sum();loss=term if loss is None else loss+term;target=wanted if target is None else target+wanted
        return loss,target

    def test_actual_homogeneous_incidence_self_loops_parallel_fields_and_gradients(self):
        for dtype in (torch.float32,torch.float64):
            for explicit in (False,True):
                with self.subTest(dtype=dtype,explicit=explicit):
                    data,ref=self.homogeneous(dtype,explicit);out,want=RemoveIsolatedNodes()(data),Reference()(ref)
                    self.assertEqual(out.num_nodes,want.num_nodes);self.assertEqual(out.edge_index.tolist(),want.edge_index.tolist());self.assertEqual(out._field_roles['y'],'node');self.assertTrue(out.validate())
                    self.assertIs(out.edge_attr,data.edge_attr);self.assertIsNot(out.x,data.x);self.assertFalse(out.x._shares_storage(data.x));self.assertFalse(torch._C._is_alias_of(want.x,ref.x))
                    loss,target=self.compare_fields(out,want,('x','y','edge_attr','mask','node_type'));loss.backward();target.backward()
                    for key in ('x','y','edge_attr'):self.close(data[key].grad,ref[key].grad)
                    self.assertEqual(data.x.shape,[7,3]);self.assertEqual(data.num_nodes,7)

    def test_empty_missing_topology_unconnected_fields_and_native_topology(self):
        for edge in ([[],[]],[[3],[3]]):
            a,b=self.homogeneous(torch.float64,edge=edge);out,wanted=RemoveIsolatedNodes()(a),Reference()(b);loss,target=self.compare_fields(out,wanted,('x','y','edge_attr'));loss.backward();target.backward();self.close(a.x.grad,b.x.grad);self.assertEqual(out.num_nodes,wanted.num_nodes)
        a,b=self.homogeneous();a.edge_index=None;b.edge_index=None;a.edge_attr=None;b.edge_attr=None
        out,wanted=RemoveIsolatedNodes()(a),Reference()(b);self.close(out.x,wanted.x);self.close(out.y,wanted.y);self.assertEqual(out.num_nodes,0);self.assertIsNone(out.edge_index)
        a=Data(edge_index=EdgeIndex(8,[5,1],[1,5]),edge_attr=fr.tensor([.2,.7],requires_grad=True));out=RemoveIsolatedNodes()(a)
        self.assertEqual(out.num_nodes,2);self.assertEqual(out.edge_index.tolist(),[[1,0],[0,1]]);self.assertIs(out.edge_attr,a.edge_attr)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');wanted=Reference()(TData(edge_index=torch.tensor([[5,1],[1,5]])))
        self.assertEqual(out.edge_index.tolist(),wanted.edge_index.tolist())

    def test_actual_ambiguous_row_rule_and_explicit_role_override(self):
        edge=torch.tensor([[1,3,1,3],[3,1,1,3]])
        a=Data(x=fr.arange(4).float().reshape([4,1]),edge_index=native(edge),y=fr.arange(4),edge_signal=fr.arange(4),score=fr.arange(4),num_nodes=4)
        b=TData(x=torch.arange(4.).reshape(4,1),edge_index=edge,y=torch.arange(4),edge_signal=torch.arange(4),score=torch.arange(4),num_nodes=4)
        out,ref=RemoveIsolatedNodes()(a),Reference()(b)
        for key in ('x','y','edge_signal','score'):self.close(out[key],ref[key])
        self.assertIs(out.edge_signal,a.edge_signal);self.assertEqual(out._field_roles['score'],'node');self.assertEqual(out._field_roles['edge_signal'],'edge');self.assertTrue(out.validate())
        a=Data(x=fr.ones(4,1),edge_index=native(edge),y=fr.arange(4),num_nodes=4,field_roles={'y':'graph'});out=RemoveIsolatedNodes()(a)
        self.assertIs(out.y,a.y);self.assertEqual(out._field_roles['y'],'graph');self.assertEqual(out.num_nodes,2)
        a=Data(node_type=fr.arange(4),edge_index=native(edge),field_roles={'node_type':'graph'});out=RemoveIsolatedNodes()(a);self.assertEqual(out.num_nodes,2);self.assertIs(out.node_type,a.node_type)

    def heterogeneous(self):
        a,b=HeteroData(),THetero()
        for name,count in (('paper',6),('author',4),('isolated',3)):
            x=torch.linspace(-.4,.8,2*count,dtype=torch.float64).reshape(2,count).T.detach().requires_grad_();label=torch.arange(count)
            a[name].x=native(x,True);b[name].x=x;a[name].y=native(label);b[name].y=label;a[name].num_nodes=count;b[name].num_nodes=count
            a[name].scalar=fr.tensor(.2,requires_grad=True);b[name].scalar=torch.tensor(.2,requires_grad=True)
        for key,edge in ((('author','writes','paper'),[[3,1,3,1],[5,2,2,5]]),(('paper','cites','paper'),[[5,2],[2,2]]),(('paper','empty','isolated'),[[],[]])):
            index=torch.tensor(edge,dtype=torch.long).reshape(2,-1);weight=torch.linspace(.2,.8,index.shape[1],dtype=torch.float64,requires_grad=True)
            a[key].edge_index=native(index);b[key].edge_index=index;a[key].edge_attr=native(weight,True);b[key].edge_attr=weight
        a.global_value=fr.tensor(4.,dtype='float64',requires_grad=True);b.global_value=torch.tensor(4.,dtype=torch.float64,requires_grad=True)
        return a,b

    def test_actual_heterogeneous_all_relations_unconnected_types_and_all_gradients(self):
        a,b=self.heterogeneous();out,ref=RemoveIsolatedNodes()(a),Reference()(b);self.assertEqual(out.metadata(),ref.metadata());self.assertTrue(out.validate());loss=target=None
        for key in out.node_types:
            self.assertIs(out[key]._parent,out);self.assertEqual(out[key].num_nodes,ref[key].num_nodes);self.assertIs(out[key].scalar,a[key].scalar)
            left,right=self.compare_fields(out[key],ref[key],('x','y','scalar'));loss=left if loss is None else loss+left;target=right if target is None else target+right
        for key in out.edge_types:
            self.assertIs(out[key]._parent,out);self.assertIs(out[key].edge_attr,a[key].edge_attr);self.assertEqual(out[key].edge_index.tolist(),ref[key].edge_index.tolist())
            left,right=self.compare_fields(out[key],ref[key],('edge_attr',));loss=loss+left;target=target+right
        self.assertIs(out.global_value,a.global_value);loss=loss+out.global_value;target=target+ref.global_value;loss.backward();target.backward()
        for key in a.node_types:
            self.close(a[key].x.grad,b[key].x.grad);self.close(a[key].scalar.grad,b[key].scalar.grad)
        for key in a.edge_types:self.close(a[key].edge_attr.grad,b[key].edge_attr.grad)
        self.close(a.global_value.grad,b.global_value.grad);self.assertEqual(out['isolated'].num_nodes,0)

    def test_class_and_private_state_copy_direct_forward_and_typed_exact_bytes(self):
        class Custom(Data):
            __slots__=('_token',)
            def __init__(self,token):super().__init__(x=fr.tensor([[0.],[-0.],[1e-300],[2e-300]],dtype='float64'),edge_index=fr.tensor([[1,2],[2,1]],dtype='int64'));self._token=token
        data=Custom(object());data._history=object();out=RemoveIsolatedNodes()(data)
        self.assertIs(type(out),Custom);self.assertIs(out._token,data._token);self.assertIs(out._history,data._history);self.assertIsNot(out._store,data._store);self.assertEqual(out.x.tolist()[0][0].hex(),(-0.).hex());self.assertEqual(out.x.tolist()[1][0].hex(),float(1e-300).hex())
        self.assertIs(RemoveIsolatedNodes().forward(data),data);self.assertEqual(data.num_nodes,2)
        class HCustom(HeteroData):pass
        h=HCustom();h['a'].x=fr.ones(3,2);h['a','r','a'].edge_index=fr.tensor([[2],[2]],dtype='int64');h._tag=object();out=RemoveIsolatedNodes()(h);self.assertIs(type(out),HCustom);self.assertIs(out._tag,h._tag);self.assertIs(out['a']._parent,out)

    def test_metadata_rejects_before_gathers_or_partial_changes(self):
        a,b=self.homogeneous();a.edge_label_index=fr.tensor([[0],[1]],dtype='int64')
        with self.assertRaises(NotImplementedError):RemoveIsolatedNodes()(a)
        with self.assertRaises(NotImplementedError):RemoveIsolatedNodes()(Batch.from_data_list([Data(x=fr.ones(2,1),edge_index=fr.tensor([[0],[1]],dtype='int64'))]))
        h,_=self.heterogeneous()
        with self.assertRaises(NotImplementedError):RemoveIsolatedNodes()(HeteroBatch.from_data_list([h]))
        class Hooks(Data):
            def __inc__(self,key,value,*args):return 0
        with self.assertRaises(NotImplementedError):RemoveIsolatedNodes()(Hooks(num_nodes=2))
        with self.assertRaises(ValueError):RemoveIsolatedNodes()(Data())
        with self.assertRaises(NotImplementedError):RemoveIsolatedNodes()(Data(edge_index=fr.tensor([[4],[2]],dtype='int64'),score=fr.ones(5)))
        a=Data(x=fr.ones(3,1),edge_index=fr.tensor([[0],[3]],dtype='int64'));old=a.x
        with patch.object(fr.Tensor,'index_select',side_effect=AssertionError('gather before metadata rejection')):
            with self.assertRaises(ValueError):RemoveIsolatedNodes().forward(a)
        self.assertIs(a.x,old)
        h,_=self.heterogeneous();old=h['paper'].x;h['author'].x=fr.ones(5,2)
        with patch.object(fr.Tensor,'index_select',side_effect=AssertionError('gather before later-store validation')):
            with self.assertRaises(ValueError):RemoveIsolatedNodes().forward(h)
        self.assertIs(h['paper'].x,old)
        a=Data(edge_index=fr.tensor([[],[]],dtype='int64'));a._store['edge_index']=fr.zeros(2,1,dtype='int64').expand([2,sys.maxsize//8])
        with patch.object(fr.Tensor,'tolist',side_effect=AssertionError('metadata capacity before topology read')):
            with self.assertRaisesRegex(ValueError,'capacity'):RemoveIsolatedNodes()(a)
        a,_=self.homogeneous()
        with self.assertRaises(NotImplementedError):fr.capture(lambda:RemoveIsolatedNodes()(a).x)

    def test_sparse_remapping_of_huge_zero_width_nodes_is_bounded_and_live(self):
        script=r'''
import ferro as f
from ferro.transforms import RemoveIsolatedNodes
x=f.zeros(1,0,dtype='float64',requires_grad=True);n=10**12
data=f.graph.Data(x=x.expand([n,0]),edge_index=f.tensor([[n-1,2],[2,n-1]],dtype='int64'),num_nodes=n)
out=RemoveIsolatedNodes()(data)
if out.num_nodes!=2 or out.x.shape!=[2,0]or out.edge_index.tolist()!=[[1,0],[0,1]]:raise RuntimeError('bad sparse incidence remap')
out.x.sum().backward()
if x.grad.shape!=[1,0]:raise RuntimeError('lost actual zero gradient')
print('bounded sparse incidence')
'''
        result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,timeout=15);self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('bounded sparse incidence',result.stdout)

    def test_saved_cleaned_data_and_model_exact_continuation(self):
        class Model(fr.nn.Module):
            def build(self):self.linear=fr.nn.Linear(3,2)
            def forward(self,data):return self.linear(RemoveIsolatedNodes()(data).x)
        def exact(v):
            if isinstance(v,float):self.assertTrue(math.isfinite(v));return v.hex()
            return [exact(t)for t in v]if isinstance(v,list)else v
        def step(model,opt,data):
            out=model(data);loss=out.square().mean();loss.backward();opt.step();opt.zero_grad();return exact(out.tolist()),exact(loss.item())
        data,_=self.homogeneous();data=data.detach();model=Model();opt=fr.optim.AdamW(model.parameters(),lr=.003);rng=fr.Generator(7);step(model,opt,data)
        with tempfile.TemporaryDirectory()as path:
            InMemoryDataset.save([data],path+'/data');fr.checkpoint.snapshot_training(model,optimizers={'main':opt},generator=rng,step=1).save(path+'/state')
            dataset=InMemoryDataset();dataset.load(path+'/data');other=Model();optimizer=fr.optim.AdamW(other.parameters(),lr=.2);generator=fr.Generator(8);fr.checkpoint.load(path+'/state').restore(other,optimizers={'main':optimizer},generator=generator)
            for _ in range(3):self.assertEqual(step(model,opt,data),step(other,optimizer,dataset[0]))
            left=fr.checkpoint.snapshot_training(model,optimizers={'main':opt},generator=rng,step=4).tensors();right=fr.checkpoint.snapshot_training(other,optimizers={'main':optimizer},generator=generator,step=4).tensors()
            self.assertEqual(left.keys(),right.keys())
            for key in left:self.assertEqual((left[key].dtype,left[key].shape,exact(left[key].tolist())),(right[key].dtype,right[key].shape,exact(right[key].tolist())))

if __name__=='__main__':unittest.main()
