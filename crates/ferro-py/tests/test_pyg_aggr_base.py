"""Actual PyG custom aggregation calls, native reductions and sorted padding."""
import inspect
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn.aggr import Aggregation as TorchAggregation
import ferro as fr
from ferro.graph import Aggregation
import test_pyg_modulated as helpers

native = helpers.native


class Reduce(Aggregation):
    def __init__(self, kind='sum'):
        super().__init__(); self.kind = kind

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, gain=1.):
        return self.reduce(x * gain, index, ptr, dim_size, dim, self.kind)


class TorchReduce(TorchAggregation):
    def __init__(self, kind='sum'):
        super().__init__(); self.kind = kind

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, gain=1.):
        return self.reduce(x * gain, index, ptr, dim_size, dim, self.kind)


class Learned(Aggregation):
    def __init__(self, channels, out_channels, kind='sum', p=0.):
        super().__init__()
        self.channels, self.out_channels, self.kind, self.p = channels, out_channels, kind, p
        self.lin, self.drop = fr.nn.Linear(channels, out_channels), fr.nn.Dropout(p, seed=181)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, gain=1.):
        return self.reduce(self.drop(self.lin(x)) * gain, index, ptr, dim_size, dim, self.kind)

    def get_out_channels(self, in_channels):
        if in_channels != self.channels: raise ValueError('wrong input width')
        return self.out_channels


class TorchLearned(TorchAggregation):
    def __init__(self, channels, out_channels, kind='sum', p=0.):
        super().__init__()
        self.channels, self.out_channels, self.kind, self.p = channels, out_channels, kind, p
        self.lin, self.drop = torch.nn.Linear(channels, out_channels), torch.nn.Dropout(p)

    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, gain=1.):
        return self.reduce(self.drop(self.lin(x)) * gain, index, ptr, dim_size, dim, self.kind)


class AggregationBaseTests(unittest.TestCase):
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def close(self, actual, expected):
        torch.testing.assert_close(helpers.torch_value(actual), expected.detach(), atol=3e-5, rtol=4e-4, equal_nan=True)

    def setUp(self):
        torch.manual_seed(2557)

    def test_custom_reductions_arbitrary_axes_index_csr_empty_and_all_input_gradients(self):
        for kind in ('sum', 'mean', 'min', 'max'):
            for shape, dim in (((5,), 0), ((5, 3), -2), ((2, 5, 3), 1), ((2, 3, 5), -1), ((0, 3), 0), ((5, 0), 0)):
                rows = shape[dim]
                for csr in ((False, True) if dim % len(shape) == 0 else (False,)):
                    model, oracle = Reduce(kind), TorchReduce(kind)
                    x = (torch.rand(shape) + .5).requires_grad_(); nx = native(x, True)
                    ids = torch.tensor([0, 0, 2, 2, 2], dtype=torch.long) if rows else torch.empty(0,dtype=torch.long)
                    ptr = torch.tensor([0, 2, 2, 5, 5],dtype=torch.long) if rows else torch.tensor([0,0,0,0,0],dtype=torch.long)
                    kwargs = dict(ptr=ptr, dim_size=4, dim=dim) if csr else dict(index=ids, dim_size=4, dim=dim)
                    options = {key:native(value) if isinstance(value,torch.Tensor) else value for key,value in kwargs.items()}
                    self.gradients(model,oracle,model(nx,**options),oracle(x,**kwargs),[('x',nx,x)])

    def test_nonleading_csr_with_actual_pyg_leading_axis_reference(self):
        for shape,dim in (((2,5,3),1),((2,3,5),2)):
            for kind in ('sum','mean','min','max'):
                model,oracle=Reduce(kind),TorchReduce(kind)
                x=(torch.rand(shape)+.5).requires_grad_();nx=native(x,True)
                ptr=torch.tensor([0,2,2,5,5])
                try:
                    expected=oracle(x,ptr=ptr,dim_size=4,dim=dim)
                except ImportError as error:
                    self.assertIn('torch-scatter',str(error))
                    order=[dim]+[i for i in range(len(shape)) if i!=dim]
                    leading=x.permute(order).contiguous()
                    expected=oracle(leading,ptr=ptr,dim_size=4,dim=0).permute([order.index(i) for i in range(len(order))])
                self.gradients(model,oracle,model(nx,ptr=native(ptr),dim_size=4,dim=dim),expected,[('x',nx,x)])

    def test_learned_handwritten_constructor_and_node_axis_parameter_gradients(self):
        self.assertEqual(str(inspect.signature(Learned)), str(inspect.signature(TorchLearned)))
        for kind in ('sum','mean','max'):
            model,oracle=Learned(3,2,kind),TorchLearned(3,2,kind)
            self.copy(model,oracle)
            base=torch.randn(2,3,5,requires_grad=True); nb=native(base,True)
            ids=torch.tensor([0,2,0,1,2])
            self.gradients(model,oracle,model(nb.transpose(1,2),index=native(ids),dim=1,gain=.7),oracle(base.transpose(1,2),index=ids,dim=1,gain=.7),[('x',nb,base)])
            self.assertEqual(model.get_out_channels(3),2)

    def test_public_custom_aggregation_resolvers_multi_simple_gen_and_sage(self):
        from torch_geometric.nn import GENConv as TorchGEN, SimpleConv as TorchSimple, SAGEConv as TorchSAGE
        from torch_geometric.nn.aggr import MultiAggregation as TorchMulti
        models=[(fr.graph.MultiAggregation([Learned(3,3),'sum'],mode='proj',mode_kwargs=dict(in_channels=3,out_channels=4)),
                 TorchMulti([TorchLearned(3,3),'sum'],mode='proj',mode_kwargs=dict(in_channels=3,out_channels=4)),True),
                (fr.graph.SimpleConv(Learned(3,2),combine_root='cat'),TorchSimple(TorchLearned(3,2),combine_root='cat'),False),
                (fr.graph.GENConv(3,3,aggr=Learned(3,3),norm=None),TorchGEN(3,3,aggr=TorchLearned(3,3),norm=None),False),
                (fr.graph.SAGEConv(3,3,aggr=Learned(3,3)),TorchSAGE(3,3,aggr=TorchLearned(3,3)),False)]
        index=torch.tensor([0,2,0,1,2]);edge=torch.tensor([[0,1,2,3,4,0],[1,2,0,2,3,4]])
        for model,oracle,multi in models:
            self.copy(model,oracle)
            x=torch.randn(5,3,requires_grad=True);nx=native(x,True)
            topology=index if multi else edge
            if isinstance(model,fr.graph.SimpleConv):
                weight=torch.rand(6,requires_grad=True);nw=native(weight,True)
                self.gradients(model,oracle,model(nx,native(topology),nw),oracle(x,topology,weight),[('x',nx,x),('edge_weight',nw,weight)])
            else:
                self.gradients(model,oracle,model(nx,native(topology)),oracle(x,topology),[('x',nx,x)])
        attention=fr.graph.AttentionalAggregation(fr.nn.Linear(3,1),fr.nn.Dropout(.2,seed=31))
        model=fr.graph.MultiAggregation([attention,Learned(4,3)])
        before=attention.nn.rng_state.tolist()
        with self.assertRaises(ValueError):model(fr.Tensor.ones([5,3]),[0,2,0,1,2])
        self.assertEqual(before,attention.nn.rng_state.tolist())

    def test_standard_module_hooks_receive_normalized_arguments_before_forward(self):
        for mode in ('default','index','ptr'):
            model,oracle=Learned(3,2),TorchLearned(3,2);self.copy(model,oracle)
            records=[[],[]]
            handles=[]
            for position, layer in enumerate((model,oracle)):
                def pre(module,args,kwargs,position=position):
                    index=kwargs['index']
                    records[position].append((len(args),sorted(kwargs),None if index is None else index.tolist(),kwargs['dim_size'],kwargs['dim']))
                    kwargs['gain']=.7;return args,kwargs
                handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
                handles.append(layer.register_forward_hook(lambda module,args,kwargs,out:out*2.,with_kwargs=True))
            x=torch.randn(5,3,requires_grad=True);nx=native(x,True)
            kwargs={} if mode=='default' else dict(index=torch.tensor([0,2,0,1,2])) if mode=='index' else dict(ptr=torch.tensor([0,2,2,5]))
            options={key:native(value) for key,value in kwargs.items()}
            self.gradients(model,oracle,model(nx,**options),oracle(x,**kwargs),[('x',nx,x)])
            self.assertEqual(records[0],records[1])
            for handle in handles:handle.remove()

    def test_dense_batch_sorted_truncation_fill_and_original_input_gradients(self):
        model,oracle=Aggregation(),TorchAggregation()
        ids=torch.tensor([0,0,0,2,2,3],dtype=torch.long)
        for time in (None,0,2,4):
            for fill in (0.,-.25,float('nan')):
                x=torch.randn(3,6,requires_grad=True);nx=native(x,True)
                actual,mask=model.to_dense_batch(nx.transpose(0,1),native(ids),dim_size=5,fill_value=fill,max_num_elements=time)
                expected,reference_mask=oracle.to_dense_batch(x.transpose(0,1),ids,dim_size=5,fill_value=fill,max_num_elements=time)
                self.assertEqual(mask.tolist(),reference_mask.tolist())
                self.gradients(model,oracle,actual,expected,[('x',nx,x)])
                if time==2:self.assertEqual(nx.grad.transpose(0,1).narrow(0,2,1).tolist(),[[0.]*3])

    def test_dense_empty_features_rows_and_zero_time_virtual_group_capacity(self):
        model,oracle=Aggregation(),TorchAggregation()
        for rows,width,count,time in ((0,3,0,0),(0,3,3,2),(3,0,2,2),(0,0,2,0)):
            x=torch.randn(rows,width,requires_grad=True);nx=native(x,True)
            ids=torch.tensor([0,0,1],dtype=torch.long) if rows else torch.empty(0,dtype=torch.long)
            actual,mask=model.to_dense_batch(nx,native(ids),dim_size=count,max_num_elements=time)
            expected,reference_mask=oracle.to_dense_batch(x,ids,dim_size=count,max_num_elements=time)
            self.assertEqual(mask.tolist(),reference_mask.tolist())
            self.gradients(model,oracle,actual,expected,[('x',nx,x)])
        x=fr.Tensor.ones([1,2]).requires_grad_(True)
        with patch('ferro._graph_aggr_base.cat',side_effect=AssertionError('zero-time batching allocated padding')):
            out,mask=model.to_dense_batch(x,[0],dim_size=10**12,max_num_elements=0)
        self.assertEqual(out.shape,[10**12,0,2]);self.assertEqual(mask.shape,[10**12,0])
        out.sum().backward();self.assertEqual(x.grad.tolist(),[[0.,0.]])

    def test_serialized_custom_alias_dropout_optimizer_generator_continuation(self):
        def factory(kind='sum'):
            model=Learned(3,2,kind,p=.2);model.alias=model.lin.weight;return model
        def step(model,optimizer,rng):
            output=model(fr.Tensor.randn([5,3],generator=rng),index=[0,2,0,1,2],dim_size=4)
            loss=output.square().mean();values=output.tolist(),loss.item()
            loss.backward();optimizer.step();optimizer.zero_grad();return values
        model,rng=factory(),fr.Generator(521);optimizer=fr.optim.AdamW(model.parameters(),lr=.003)
        step(model,optimizer,rng)
        saved=fr.checkpoint.snapshot_training(model,optimizers={'aggregation':optimizer},generator=rng,step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory+'/state');saved=fr.checkpoint.load(directory+'/state')
        expected=[step(model,optimizer,rng) for _ in range(3)]
        other,other_rng=factory(),fr.Generator(1);other_opt=fr.optim.AdamW(other.parameters(),lr=.8)
        saved.restore(other,optimizers={'aggregation':other_opt},generator=other_rng)
        self.assertIs(other.alias,other.lin.weight)
        self.assertEqual(expected,[step(other,other_opt,other_rng) for _ in range(3)])
        left=fr.checkpoint.snapshot_training(model,optimizers={'aggregation':optimizer},generator=rng,step=4).tensors()
        right=fr.checkpoint.snapshot_training(other,optimizers={'aggregation':other_opt},generator=other_rng,step=4).tensors()
        self.assertEqual(set(left),set(right))
        for key in left:self.assertEqual((left[key].dtype,left[key].shape,left[key].tolist()),(right[key].dtype,right[key].shape,right[key].tolist()),key)
        changed,changed_rng=factory('mean'),fr.Generator(1);changed_opt=fr.optim.AdamW(changed.parameters())
        def state():return {key:value.tolist() for key,value in fr.checkpoint.snapshot_training(changed,optimizers={'aggregation':changed_opt},generator=changed_rng).tensors().items()}
        before=state()
        with self.assertRaises(ValueError):saved.restore(changed,optimizers={'aggregation':changed_opt},generator=changed_rng)
        self.assertEqual(before,state())

    def test_metadata_and_helper_boundaries_before_custom_callbacks(self):
        model,x=Reduce(),fr.Tensor.ones([3,2]);calls=[]
        handle=model.register_forward_pre_hook(lambda *args:calls.append(1))
        for kwargs in (dict(index=[0,1]),dict(index=[0,-1,1]),dict(index=[0,1,2],dim_size=2),dict(ptr=fr.Tensor.from_i64([0,2],[2])),dict(dim=2),dict(index=[0,0,0],ptr=fr.Tensor.from_i64([0,3],[2]))):
            with self.assertRaises((ValueError,TypeError,NotImplementedError)):model(x,**kwargs)
        with self.assertRaises(NotImplementedError):fr.capture(lambda:model(x,index=[0,0,0]))
        self.assertEqual(calls,[]);handle.remove()
        with self.assertRaises(RuntimeError):model.reduce(x)
        with self.assertRaises(NotImplementedError):model.reduce(x,index=[0,0,0],reduce='mul')
        with self.assertRaises(NotImplementedError):model.to_dense_batch(x)
        empty,index=fr.Tensor.zeros([0,2]),fr.Tensor.from_i64([],[0])
        with self.assertRaises(ValueError):model.to_dense_batch(empty,index,max_num_elements=0)
        with self.assertRaises(ValueError):model.to_dense_batch(empty,index,dim_size=0)
        with self.assertRaises(RuntimeError):TorchAggregation().to_dense_batch(torch.empty(0,2),torch.empty(0,dtype=torch.long),max_num_elements=0)
        with self.assertRaises(RuntimeError):TorchAggregation().to_dense_batch(torch.empty(0,2),torch.empty(0,dtype=torch.long),dim_size=0)
        with self.assertRaises(ValueError):model.to_dense_batch(x,[1,0,1])
        with self.assertRaises(ValueError):model.to_dense_batch(x,[0,0,0],fill_value=1e100)
        with self.assertRaises(NotImplementedError):model.assert_index_present(None)
        model.assert_sorted_index(None)
        with self.assertRaises(ValueError):model.assert_sorted_index([1,0])
        with self.assertRaises(ValueError):model.assert_two_dimensional_input(x,1)
        with patch('ferro._graph_aggr_base.scatter',side_effect=AssertionError('scatter before capacity rejection')):
            with self.assertRaisesRegex(ValueError,'capacity'):model(x,index=[0,0,0],dim_size=sys.maxsize//8+1)
        index=fr.Tensor.from_i64([0],[1]).expand([sys.maxsize//8+1])
        with patch.object(fr.Tensor,'tolist',side_effect=AssertionError('index read before capacity rejection')):
            with self.assertRaisesRegex(ValueError,'capacity'):model(x,index=index)


if __name__=='__main__':unittest.main(verbosity=2)
