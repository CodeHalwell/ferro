"""Actual PyG dense random-walk contracts and bounded native ownership/state."""
import itertools,json,math,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import torch
from torch_geometric.data import Data as TData,Batch as TBatch
from torch_geometric.transforms import AddRandomWalkPE as Reference
import ferro as fr
from ferro.graph import Data,Batch,EdgeIndex,InMemoryDataset,HeteroData
from ferro.transforms import AddRandomWalkPE


def native(value,grad=False):return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).split('.')[-1],requires_grad=grad).reshape(list(value.shape))
def values(value):return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)


class RandomWalkPETests(unittest.TestCase):
    def close(self,a,b):torch.testing.assert_close(values(a),b.detach(),rtol=3e-5,atol=3e-6,equal_nan=True)
    def pair(self,n,edge,weight=None,x=None):
        index=torch.tensor(edge,dtype=torch.long).reshape(2,-1);kwargs={};tk={}
        if weight is not None:kwargs['edge_weight']=native(weight,weight.requires_grad);tk['edge_weight']=weight
        if x is not None:kwargs['x']=native(x,x.requires_grad);tk['x']=x
        return Data(edge_index=native(index),num_nodes=n,**kwargs),TData(edge_index=index,num_nodes=n,**tk)
    def compare(self,a,b,length=3,attr='random_walk_pe'):
        left,right=AddRandomWalkPE(length,attr)(a),Reference(length,attr)(b);key='x'if attr is None else attr
        self.close(left[key],right[key]);self.assertEqual(left[key].requires_grad,right[key].requires_grad)
        self.assertIsNot(left,a);self.assertIs(left.edge_index,a.edge_index);self.assertEqual(left._field_roles[key],'node');self.assertTrue(left.validate())
        if right[key].requires_grad:
            seed=torch.linspace(-.4,.9,right[key].numel(),dtype=right[key].dtype).reshape(right[key].shape)
            (left[key]*native(seed)).sum().backward();(right[key]*seed).sum().backward()
            for field in ('x','edge_weight'):
                if field in a and a[field].requires_grad:
                    if b[field].grad is None:self.assertIsNone(a[field].grad)
                    else:self.close(a[field].grad,b[field].grad)
        return left,right

    def test_actual_directed_weighted_unweighted_parallel_and_zero_length_values(self):
        graphs=[(5,[[0,1,1,2,3],[1,0,2,1,3]],None),(3,[[0,0,1,2],[1,1,0,2]],None),(4,[[0,0,1,2],[1,1,0,2]],torch.tensor([2.,3.,4.,.5])),(3,[[],[]],None),(0,[[],[]],None)]
        for (n,edge,w),length in itertools.product(graphs,(0,1,2,5)):
            with self.subTest(nodes=n,edges=len(edge[0]),weighted=w is not None,length=length):self.compare(*self.pair(n,edge,w),length=length)
        self.assertEqual(AddRandomWalkPE(0)(Data(edge_index=fr.tensor([[0],[0]],dtype='int64'),num_nodes=1)).random_walk_pe.shape,[1,1])

    def test_actual_live_weight_gradients_clamp_and_nonfinite_values(self):
        edge=[[0,0,1,2],[1,2,0,2]]
        for raw,length in itertools.product(([2.,3.,4.,.5],[.3,.7,1.,2.],[0.,0.,-2.,.5],[float('inf'),2.,3.,4.],[float('nan'),2.,3.,4.]),(1,2,4)):
            with self.subTest(weight=raw,length=length):self.compare(*self.pair(4,edge,torch.tensor(raw,requires_grad=True)),length=length)
        weight=torch.empty(0,requires_grad=True);self.compare(*self.pair(0,[[],[]],weight),length=4)
        weight=torch.linspace(.3,1.8,8)[::2].detach().requires_grad_();self.compare(*self.pair(4,edge,weight),length=3)

    def test_append_cast_rank_one_strided_fields_and_all_gradients(self):
        for dtype,rank in itertools.product((torch.float32,torch.float64,torch.int64,torch.bool),(1,2)):
            with self.subTest(dtype=dtype,rank=rank):
                x=torch.tensor([0,1,2,3]if rank==1 else[[0,1,2,3],[3,2,1,0]],dtype=dtype)
                if rank==2:x=x.T
                if dtype in (torch.float32,torch.float64):x=x.detach().requires_grad_()
                weights=torch.tensor([2.,3.,4.,.5],requires_grad=True)
                self.compare(*self.pair(4,[[0,0,1,2],[1,2,0,2]],weights,x),attr=None)
        a,b=self.pair(3,[[0,1],[1,0]]);self.compare(a,b,attr=None)
        a,b=self.pair(3,[[0,1],[1,0]]);a.extra=fr.zeros(3,2);b.extra=torch.zeros(3,2);before=a.extra;out,_=self.compare(a,b,attr='extra');self.assertIs(a.extra,before);self.assertIsNot(out.extra,before)

    def test_parallel_live_weight_derivative_counterexample_and_explicit_rejection(self):
        weights=torch.tensor([2.,3.,4.],requires_grad=True);a,b=self.pair(2,[[0,0,1],[1,1,0]],weights)
        Reference(2)(b).random_walk_pe.sum().backward();torch.testing.assert_close(weights.grad,torch.tensor([-.04,-.04,-.025]))
        denominator=torch.tensor([5.,5.,4.],requires_grad=True);values_=1./denominator;adj=torch.zeros(4);adj[torch.tensor([1,1,2])]=values_;adj=adj.reshape(2,2);(adj@adj).diagonal().sum().backward()
        native_weight=fr.tensor([2.,3.,4.],requires_grad=True);degree=fr.scatter(native_weight.reshape([3,1]),[0,0,1],2,'sum').reshape([2]);v=1./degree.index_select(0,[0,0,1]);native_adj=fr.zeros(4).scatter(0,fr.tensor([1,1,2],dtype='int64'),v).reshape([2,2]);fr.linalg.diagonal(native_adj@native_adj).sum().backward()
        self.assertGreater(abs(native_weight.grad.tolist()[0]-weights.grad.tolist()[0]),.01)
        import ferro._graph_positional as impl
        before=dict(a.items())
        with patch.object(impl,'scatter',side_effect=AssertionError('degree allocation before duplicate rejection')):
            with self.assertRaisesRegex(NotImplementedError,'overwrite gradients'):AddRandomWalkPE(2).forward(a)
        for key,value in before.items():self.assertIs(a[key],value)
        detached=weights.detach();self.compare(*self.pair(2,[[0,0,1],[1,1,0]],detached),length=2)

    def test_shallow_class_copy_native_topology_and_batch_inverse_metadata(self):
        class Custom(Data):
            __slots__=('_tag',)
            def __init__(self,tag):super().__init__(x=fr.ones(3,1),edge_index=fr.tensor([[0,1],[1,0]],dtype='int64'));self._tag=tag
        source=Custom(object());source._marker=object();out=AddRandomWalkPE(2)(source)
        self.assertIs(type(out),Custom);self.assertIs(out._tag,source._tag);self.assertIs(out._marker,source._marker);self.assertIs(out.x,source.x);self.assertNotIn('random_walk_pe',source)
        self.assertIs(AddRandomWalkPE(2).forward(source),source)
        native_data=Data(edge_index=EdgeIndex(4,[0,1],[1,0]),num_nodes=4)
        coo=Data(edge_index=fr.tensor([[0,1],[1,0]],dtype='int64'),num_nodes=4)
        self.close(AddRandomWalkPE(4)(native_data).random_walk_pe,values(AddRandomWalkPE(4)(coo).random_walk_pe));self.assertIs(AddRandomWalkPE(2)(native_data).edge_index,native_data.edge_index)
        first,tf=self.pair(3,[[0,1],[1,0]],x=torch.arange(3,dtype=torch.float32).reshape(3,1));second,ts=self.pair(2,[[0],[0]],x=torch.ones(2,1))
        batch,tbatch=Batch.from_data_list([first,second]),TBatch.from_data_list([tf,ts]);out,reference=self.compare(batch,tbatch,attr=None)
        self.assertIsInstance(out,Batch);self.assertIs(out.ptr,batch.ptr);self.assertIs(out.batch,batch.batch)
        for result,wanted in zip(out.to_data_list(),reference.to_data_list()):self.close(result.x,wanted.x);self.assertEqual(result.num_nodes,wanted.num_nodes)
        transformed=[AddRandomWalkPE(2)(first),AddRandomWalkPE(2)(second)];stored=Batch.from_data_list(transformed);self.assertEqual(stored.random_walk_pe.shape,[5,2])
        for result,wanted in zip(stored.to_data_list(),transformed):self.assertEqual(result.random_walk_pe.tolist(),wanted.random_walk_pe.tolist())

    def test_dtype_shape_reserved_fields_work_and_capacity_preflight(self):
        import ferro._graph_positional as impl
        for length in (-1,True,1.5,2**63):
            with self.assertRaises(ValueError):AddRandomWalkPE(length)
        with self.assertRaises(TypeError):AddRandomWalkPE(2,3)
        a,b=self.pair(2,[[0,1],[1,0]],torch.ones(2,dtype=torch.float64))
        with self.assertRaises(RuntimeError):Reference(2)(b)
        with self.assertRaisesRegex(ValueError,'float32'):AddRandomWalkPE(2)(a)
        bad=[Data(edge_index=fr.tensor([[],[]],dtype='int64'),num_nodes=2001),Data(edge_index=fr.tensor([[],[]],dtype='int64'),num_nodes=500),Data(edge_index=fr.tensor([[0],[2]],dtype='int64'),num_nodes=2),Data(edge_index=fr.tensor([[0],[1]],dtype='int64'),num_nodes=2,edge_weight=fr.ones(1,1))]
        with patch.object(impl,'scatter',side_effect=AssertionError('allocation before invalid metadata')):
            for data in bad:
                with self.assertRaises((ValueError,NotImplementedError)):AddRandomWalkPE(3).forward(data)
            for attr in ('edge_index','num_nodes','batch','_private','clone'):
                with self.assertRaises(ValueError):AddRandomWalkPE(2,attr).forward(a)
            huge=Data(edge_index=fr.tensor([[],[]],dtype='int64'),num_nodes=1,x=fr.ones(1,1,dtype='float64').expand([1,sys.maxsize//8]))
            with self.assertRaisesRegex(ValueError,'capacity'):AddRandomWalkPE(2,None).forward(huge)
        with self.assertRaises(TypeError):AddRandomWalkPE(2)(HeteroData())
        with self.assertRaises(ValueError):AddRandomWalkPE(2)(Data(num_nodes=3))
        d=Data(edge_index=fr.tensor([[0],[0]],dtype='int64'),num_nodes=1)
        with self.assertRaises(NotImplementedError):fr.capture(lambda:AddRandomWalkPE(2)(d).random_walk_pe)
        transform=AddRandomWalkPE(2);transform.walk_length=-2
        with self.assertRaises(ValueError):transform.forward(d)

    def test_dense_node_limit_and_topology_metadata_before_host_reads(self):
        data=Data(edge_index=fr.tensor([[],[]],dtype='int64'),num_nodes=2000)
        out=AddRandomWalkPE(0)(data).random_walk_pe
        self.assertEqual(out.shape,[2000,1]);self.assertEqual(out.sum().item(),0.)
        data=Data(edge_index=fr.tensor([[],[]],dtype='int64'),num_nodes=1)
        data._store['edge_index']=fr.zeros(2,1,dtype='int64').expand([2,10**12])
        with patch.object(fr.Tensor,'tolist',side_effect=AssertionError('host read before topology work bound')):
            with self.assertRaisesRegex(ValueError,'work capacity'):AddRandomWalkPE(1).forward(data)

    def test_deterministic_reference_rng_and_saved_typed_dataset(self):
        a,b=self.pair(3,[[0,1],[1,0]],x=torch.tensor([[.1,.2],[.3,.4],[.5,.6]],dtype=torch.float64))
        torch.manual_seed(19);state=torch.get_rng_state().clone();expected=Reference(3)(b);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        transform=AddRandomWalkPE(3);first=transform(a);second=transform(a);self.assertEqual(first.random_walk_pe.tolist(),second.random_walk_pe.tolist())
        generator=fr.Generator(29);control=fr.Generator(29);transform(a);self.assertEqual(fr.Tensor.rand([4],generator=generator).tolist(),fr.Tensor.rand([4],generator=control).tolist())
        with tempfile.TemporaryDirectory()as directory:
            InMemoryDataset.save([first],directory+'/data');dataset=InMemoryDataset();dataset.load(directory+'/data');restored=dataset[0]
            self.assertEqual(restored._field_roles['random_walk_pe'],'node');self.assertEqual(restored.x.dtype,'float64');self.close(restored.random_walk_pe,expected.random_walk_pe);self.close(restored.x,b.x)
            self.assertEqual(transform(restored).random_walk_pe.tolist(),first.random_walk_pe.tolist())

    def test_huge_empty_output_does_not_loop_over_walk_length_and_keeps_gradients(self):
        script=r'''
import ferro as f
from ferro.transforms import AddRandomWalkPE
weight=f.zeros(0,requires_grad=True);x=f.zeros(0,3,requires_grad=True)
d=f.graph.Data(x=x,edge_index=f.tensor([[],[]],dtype='int64'),edge_weight=weight,num_nodes=0)
y=AddRandomWalkPE(10**12,None)(d).x
if y.shape!=[0,10**12+3]:raise RuntimeError('wrong output shape')
y.sum().backward()
if x.grad.shape!=[0,3]or weight.grad.shape!=[0]:raise RuntimeError('lost actual empty gradients')
print('bounded empty walk')
'''
        result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,timeout=15);self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('bounded empty walk',result.stdout)

    def test_fresh_process_complete_training_state_and_saved_actual_data(self):
        script=r'''
import json,math,sys,ferro as f
from ferro.transforms import AddRandomWalkPE
class Model(f.nn.Module):
    steps:int=3
    def build(self):
        self.weights=f.nn.Parameter(f.tensor([1.3,1.4,1.5,1.6,1.7,1.8]))
        self.head=f.nn.Linear(2+self.steps,2)
    def forward(self,data,x):
        graph=f.graph.Data(x=x,edge_index=data.edge_index,num_nodes=data.num_nodes,edge_weight=self.weights.tensor())
        return self.head(AddRandomWalkPE(self.steps,None)(graph).x)
def exact(value):
    if isinstance(value,float):
        if not math.isfinite(value):raise RuntimeError('nonfinite fixture')
        return value.hex()
    return [exact(v)for v in value]if isinstance(value,list)else value
m=Model();rng=f.Generator(91);opt=f.optim.AdamW(m.parameters(),lr=.003);root=sys.argv[2]
if sys.argv[1]=='save':
    data=f.graph.Data(x=f.tensor([[.2,.4],[-.3,.7],[.8,-.1],[.4,.6]]),edge_index=f.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]],dtype='int64'),y=f.ones(4,2)*.2,num_nodes=4)
    f.graph.InMemoryDataset.save([data],root+'/data')
else:
    dataset=f.graph.InMemoryDataset();dataset.load(root+'/data');data=dataset[0]
def step():
    x=data.x+f.Tensor.rand([4,2],generator=rng)*.1;out=m(data,x);loss=(out-data.y).square().mean();loss.backward()
    if m.weights.grad is None or m.weights.grad.abs().sum().item()==0:raise RuntimeError('missing live weight gradient')
    opt.step();opt.zero_grad();return exact(out.tolist()),exact(loss.item())
if sys.argv[1]=='save':
    for _ in range(2):step()
    m.head.bias.trainable=False
    f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=2).save(root+'/state')
else:f.checkpoint.load(root+'/state').restore(m,optimizers={'main':opt},generator=rng)
losses=[step()for _ in range(3)];snapshot=f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=5)
wrong=Model(steps=2);before={k:v.tolist()for k,v in wrong.state_dict().items()}
try:snapshot.restore(wrong,optimizers={'main':f.optim.AdamW(wrong.parameters(),lr=.003)},generator=f.Generator(2))
except ValueError:pass
else:raise RuntimeError('configuration mismatch accepted')
if before!={k:v.tolist()for k,v in wrong.state_dict().items()}:raise RuntimeError('failed restore mutated model')
print(json.dumps({'losses':losses,'tensors':{k:[v.dtype,v.shape,exact(v.tolist())]for k,v in snapshot.tensors().items()},'frozen':not m.head.bias.trainable,'data':{k:[v.dtype,v.shape,exact(v.tolist())]for k,v in data.items()},'nodes':data.num_nodes,'draw':exact(f.Tensor.rand([5],generator=rng).tolist())},sort_keys=True))
'''
        with tempfile.TemporaryDirectory()as directory:
            results=[subprocess.run([sys.executable,'-B','-c',script,mode,directory],check=True,capture_output=True,text=True,timeout=60).stdout for mode in ('save','load')]
        first,second=map(json.loads,results);self.assertEqual(first,second);self.assertTrue(first['frozen']);self.assertEqual(first['nodes'],4)

if __name__=='__main__':unittest.main()
