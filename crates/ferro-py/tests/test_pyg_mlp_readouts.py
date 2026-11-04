"""MLP-specific PyG readout routing, gradients, preflight and exact state."""
import itertools,json,math,subprocess,sys,tempfile,unittest
from unittest.mock import patch
import torch
from torch_geometric.nn import MLP as TorchMLP
from torch_geometric.nn.aggr import DeepSetsAggregation as TorchDeep,AttentionalAggregation as TorchAttention,MultiAggregation as TorchMulti
import ferro as fr
from ferro.graph import MLP
from ferro.graph import DeepSetsAggregation,AttentionalAggregation,MultiAggregation


def native(value,grad=False):
    return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).split('.')[-1],requires_grad=grad).reshape(list(value.shape))
def values(value):return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)
def streams(module):return {name:v.tolist()for name,v in module.named_buffers()if name.endswith('rng_state')}


class MLPReadoutTests(unittest.TestCase):
    def close(self,a,b):torch.testing.assert_close(values(a),b.detach(),atol=6e-5,rtol=6e-4,equal_nan=True)
    def copy(self,a,b):
        aa,bb=dict(a.named_parameters()),dict(b.named_parameters());self.assertEqual(aa.keys(),bb.keys());gen=torch.Generator().manual_seed(681)
        with fr.no_grad(),torch.no_grad():
            for name,p in bb.items():
                val=torch.randn(p.shape,generator=gen)*.23
                if 'norms' in name and name.endswith('weight'):val=val+1
                p.copy_(val);aa[name].tensor().copy_(native(val))
    def compare(self,a,b,*,csr=False,rows=6,groups=4):
        self.copy(a,b);source=torch.randn(4,rows,generator=torch.Generator().manual_seed(682),requires_grad=True);nx=native(source,True)
        ids=torch.tensor([0,0,0,2,2,2]if csr else[0,2,0,2,2,0],dtype=torch.long)if rows else torch.empty(0,dtype=torch.long)
        ptr=torch.tensor([0,3,3,6,6]if rows else[0,0,0,0,0],dtype=torch.long)
        kw={'ptr':native(ptr)}if csr else{'index':native(ids)};tk={'ptr':ptr}if csr else{'index':ids}
        out,expected=a(nx.T,dim_size=groups,**kw),b(source.T,dim_size=groups,**tk);self.close(out,expected)
        seed=torch.linspace(-.5,.8,expected.numel()).reshape(expected.shape)
        (out*native(seed)).sum().backward();(expected*seed).sum().backward();self.close(nx.grad,source.grad)
        bp=dict(b.named_parameters())
        for name,p in a.named_parameters():
            with self.subTest(parameter=name):
                if bp[name].grad is None:self.assertIsNone(p.tensor().grad)
                else:self.assertIsNotNone(p.tensor().grad);self.close(p.tensor().grad,bp[name].grad)
        ab={k:v for k,v in a.named_buffers()if not k.endswith('rng_state')};bb=dict(b.named_buffers());self.assertEqual(ab.keys(),bb.keys())
        for name,v in ab.items():self.close(v,bb[name])
        return out

    def test_deep_sets_actual_mlp_paths_norms_csr_and_all_gradients(self):
        for local_norm,global_norm,csr in itertools.product((None,'graph_norm','layer_norm','batch_norm'),(None,'batch_norm'),(False,True)):
            with self.subTest(local=local_norm,global_=global_norm,csr=csr):
                a=DeepSetsAggregation(MLP([4,5,3],norm=local_norm,act='tanh'),MLP([3,5,2],norm=global_norm,act='tanh'))
                b=TorchDeep(TorchMLP([4,5,3],norm=local_norm,act='tanh'),TorchMLP([3,5,2],norm=global_norm,act='tanh'))
                self.assertIsNone(a.local_nn);self.assertIsNone(a.global_nn);self.assertEqual(a.get_out_channels(4),2);self.compare(a,b,csr=csr)
        for csr in (False,True):
            a=DeepSetsAggregation(MLP([4,5,3],norm='graph_norm'),MLP([3,5,2],norm=None));b=TorchDeep(TorchMLP([4,5,3],norm='graph_norm'),TorchMLP([3,5,2],norm=None));self.compare(a,b,csr=csr,rows=0)

    def test_attention_mlp_gate_value_mixed_modules_and_all_gradients(self):
        for norm,gate_width,csr in itertools.product((None,'graph_norm','layer_norm','batch_norm'),(1,3),(False,True)):
            with self.subTest(norm=norm,gate_width=gate_width,csr=csr):
                a=AttentionalAggregation(MLP([4,5,gate_width],norm=norm,act='tanh'),MLP([4,5,3],norm=norm,act='tanh'))
                b=TorchAttention(TorchMLP([4,5,gate_width],norm=norm,act='tanh'),TorchMLP([4,5,3],norm=norm,act='tanh'))
                self.assertIsNone(a.gate_nn);self.assertIsNone(a.nn);self.assertEqual(a.get_out_channels(4),3);self.compare(a,b,csr=csr)
        for gate_mlp in (False,True):
            a=AttentionalAggregation(MLP([4,5,1],norm='graph_norm')if gate_mlp else fr.nn.Linear(4,1),fr.nn.Linear(4,3)if gate_mlp else MLP([4,5,3],norm='graph_norm'))
            b=TorchAttention(TorchMLP([4,5,1],norm='graph_norm')if gate_mlp else torch.nn.Linear(4,1),torch.nn.Linear(4,3)if gate_mlp else TorchMLP([4,5,3],norm='graph_norm'));self.compare(a,b)
        a=AttentionalAggregation(MLP([4,5,1],norm='graph_norm'));b=TorchAttention(TorchMLP([4,5,1],norm='graph_norm'));self.compare(a,b,rows=0)

    def test_original_index_and_csr_normalization_routes_match_reference(self):
        x=torch.tensor([[1.,2.,3.,4.],[2.,4.,1.,3.],[8.,9.,10.,11.],[9.,12.,7.,10.]])
        ids=torch.tensor([0,0,1,1]);ptr=torch.tensor([0,2,4])
        a=DeepSetsAggregation(MLP([4,4],norm='graph_norm',plain_last=False,act='tanh'));b=TorchDeep(TorchMLP([4,4],norm='graph_norm',plain_last=False,act='tanh'));self.copy(a,b)
        coo,csr=a(native(x),native(ids),dim_size=2),a(native(x),ptr=native(ptr),dim_size=2)
        self.close(coo,b(x,ids,dim_size=2));self.close(csr,b(x,ptr=ptr,dim_size=2));self.assertGreater((values(coo)-values(csr)).abs().max().item(),.01)
        a=DeepSetsAggregation(MLP([4,4],norm=None,dropout=[.4]),MLP([4,4],norm='graph_norm',plain_last=False))
        b=TorchDeep(TorchMLP([4,4],norm=None,dropout=[.4]),TorchMLP([4,4],norm='graph_norm',plain_last=False));before=streams(a)
        with self.assertRaises((RuntimeError,ValueError)):b(x,ids,dim_size=2)
        with self.assertRaises(ValueError):a(native(x),native(ids),dim_size=2)
        self.assertEqual(before,streams(a))
        self.copy(a,b);a.eval();b.eval();self.close(a(native(x),ptr=native(ptr),dim_size=2),b(x,ptr=ptr,dim_size=2))

    def test_independent_dropout_masks_for_both_readout_routes(self):
        for attention in (False,True):
            if attention:
                a=AttentionalAggregation(MLP([4,5,1],norm='graph_norm',dropout=[.2,.3]),MLP([4,5,3],norm='graph_norm',dropout=[.3,.4]))
                b=TorchAttention(TorchMLP([4,5,1],norm='graph_norm',dropout=[.2,.3]),TorchMLP([4,5,3],norm='graph_norm',dropout=[.3,.4]))
            else:
                a=DeepSetsAggregation(MLP([4,5,3],norm='graph_norm',dropout=[.2,.3]),MLP([3,5,2],norm=None,dropout=[.3,.4]))
                b=TorchDeep(TorchMLP([4,5,3],norm='graph_norm',dropout=[.2,.3]),TorchMLP([3,5,2],norm=None,dropout=[.3,.4]))
            layers=[layer for network in a._networks()for layer in network.dropouts];samplers=[fr.nn.Dropout(layer.p,seed=2)for layer in layers]
            with fr.no_grad():
                for sampler,layer in zip(samplers,layers):sampler.rng_state.copy_(layer.rng_state)
            calls=[]
            def dropout(value,p=.5,training=True,inplace=False):
                self.assertTrue(training);self.assertFalse(inplace);sampler=samplers[len(calls)];self.assertEqual(sampler.p,p);calls.append(tuple(value.shape));return value*values(sampler(fr.ones(list(value.shape))))
            with patch('torch.nn.functional.dropout',side_effect=dropout):self.compare(a,b)
            self.assertEqual(len(calls),4)
            for sampler,layer in zip(samplers,layers):self.assertEqual(sampler.rng_state.tolist(),layer.rng_state.tolist())
            a.eval();b.eval();a.zero_grad();b.zero_grad();before=streams(a);self.compare(a,b);self.assertEqual(before,streams(a))

    def test_multi_width_routing_and_shared_mlp_gradients(self):
        shared=MLP([4,5,3],norm='graph_norm',act='tanh');tshared=TorchMLP([4,5,3],norm='graph_norm',act='tanh')
        a=AttentionalAggregation(shared,shared);b=TorchAttention(tshared,tshared);self.assertIs(a.gate_mlp,a.mlp);self.compare(a,b)
        for mode in ('cat','sum'):
            a=MultiAggregation([DeepSetsAggregation(MLP([4,5,3],norm='graph_norm')),AttentionalAggregation(MLP([4,5,1],norm='graph_norm'),MLP([4,5,3],norm=None))],mode=mode)
            b=TorchMulti([TorchDeep(TorchMLP([4,5,3],norm='graph_norm')),TorchAttention(TorchMLP([4,5,1],norm='graph_norm'),TorchMLP([4,5,3],norm=None))],mode=mode)
            self.assertEqual(a.get_out_channels(4),6 if mode=='cat' else 3);self.compare(a,b)

    def test_simple_conv_mlp_aggregation_and_live_edge_weight_gradients(self):
        from torch_geometric.nn import SimpleConv as TorchSimple
        for attention in (False,True):
            if attention:
                aggregation=AttentionalAggregation(MLP([4,5,1],norm='graph_norm',act='tanh'),MLP([4,5,3],norm='graph_norm',act='tanh'))
                reference=TorchAttention(TorchMLP([4,5,1],norm='graph_norm',act='tanh'),TorchMLP([4,5,3],norm='graph_norm',act='tanh'))
            else:
                aggregation=DeepSetsAggregation(MLP([4,5,3],norm='graph_norm',act='tanh'),MLP([3,5,2],norm=None,act='tanh'))
                reference=TorchDeep(TorchMLP([4,5,3],norm='graph_norm',act='tanh'),TorchMLP([3,5,2],norm=None,act='tanh'))
            a,b=fr.graph.SimpleConv(aggregation,'cat'),TorchSimple(reference,'cat');self.copy(a,b)
            generator=torch.Generator().manual_seed(683)
            x=torch.randn(6,4,generator=generator,requires_grad=True);target=torch.randn(5,2,generator=generator,requires_grad=True);weight=torch.rand(7,generator=generator,requires_grad=True)
            nx,nt,nw=native(x,True),native(target,True),native(weight,True);edge=torch.tensor([[0,1,2,3,4,5,1],[2,0,2,1,2,0,4]])
            actual,expected=a((nx,nt),native(edge),nw),b((x,target),edge,weight);self.close(actual,expected)
            seed=torch.linspace(-.6,.7,expected.numel()).reshape(expected.shape)
            (actual*native(seed)).sum().backward();(expected*seed).sum().backward()
            for native_input,reference_input in ((nx,x),(nt,target),(nw,weight)):self.close(native_input.grad,reference_input.grad)
            parameters=dict(b.named_parameters())
            for name,parameter in a.named_parameters():self.close(parameter.tensor().grad,parameters[name].grad)

    def test_known_later_metadata_and_singleton_fail_before_any_stream(self):
        local=MLP([4,5,3],norm=None,dropout=[.3,.4]);global_=MLP([3,5,2],dropout=.2);a=DeepSetsAggregation(local,global_);before=streams(a)
        with self.assertRaisesRegex(ValueError,'more than one row'):a(fr.ones(6,4),fr.zeros(6,dtype='int64'),dim_size=1)
        self.assertEqual(before,streams(a));self.assertEqual(global_.norms[0].module.num_batches_tracked.item(),0)
        bad=MLP([4,4,3],norm=fr.graph.GraphNorm(4),plain_last=False);a=AttentionalAggregation(MLP([4,5,1],norm=None,dropout=.3),bad);before=streams(a)
        with self.assertRaisesRegex(ValueError,'normalization width'):a(fr.ones(6,4),fr.zeros(6,dtype='int64'))
        self.assertEqual(before,streams(a))
        valid=AttentionalAggregation(MLP([4,5,1],norm='graph_norm',dropout=.3),MLP([4,5,3],norm=None));before=streams(valid)
        with self.assertRaises(ValueError):valid(fr.ones(6,4),fr.zeros(5,dtype='int64'))
        self.assertEqual(before,streams(valid))

    def test_reset_aliases_and_atomic_checkpoint_configuration(self):
        shared=MLP([4,4,4],norm=None,dropout=.3);a=AttentionalAggregation(shared,shared);before=streams(a);a.reset_parameters();self.assertIs(a.gate_mlp,a.mlp);self.assertEqual(before,streams(a))
        snapshot=fr.checkpoint.snapshot_training(a,optimizers={},generator=fr.Generator(3))
        bshared=MLP([4,4,4],norm=None,dropout=.4);b=AttentionalAggregation(bshared,bshared);before={k:v.tolist()for k,v in b.state_dict().items()}
        with self.assertRaises(ValueError):snapshot.restore(b,optimizers={},generator=fr.Generator(4))
        self.assertEqual(before,{k:v.tolist()for k,v in b.state_dict().items()})

    def test_fresh_process_aliased_readout_full_state_and_rng(self):
        script=r'''
import json,math,sys,ferro as f
from ferro.graph import MLP
class Model(f.nn.Module):
    def build(self):
        self.norm=f.graph.BatchNorm(4,momentum=None)
        shared=MLP([4,4,4],norm=self.norm,plain_last=False,dropout=[.2,.3],act='tanh')
        shared.lins[1].weight=shared.lins[0].weight
        self.deep=f.graph.DeepSetsAggregation(shared,MLP([4,4],norm=None,dropout=[.2]))
        self.attention=f.graph.AttentionalAggregation(shared,shared)
    def forward(self,x):
        index=f.tensor([0,2,0,2,2,0],dtype='int64')
        return self.deep(x,index,dim_size=4)+self.attention(x,index,dim_size=4)
def exact(value):
    if isinstance(value,float):
        if not math.isfinite(value):raise RuntimeError('nonfinite fixture')
        return value.hex()
    return [exact(v)for v in value]if isinstance(value,list)else value
m=Model();rng=f.Generator(87);opt=f.optim.AdamW(m.parameters(),lr=.003)
def step():
    x=f.Tensor.rand([6,4],generator=rng)*2-1;target=f.Tensor.rand([4,4],generator=rng)
    loss=(m(x)-target).square().mean();loss.backward();opt.step();opt.zero_grad();return exact(loss.item())
if sys.argv[1]=='save':
    for _ in range(2):step()
    m.norm.eval();m.deep.global_mlp.lins[0].bias.trainable=False
    f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=2).save(sys.argv[2])
else:f.checkpoint.load(sys.argv[2]).restore(m,optimizers={'main':opt},generator=rng)
losses=[step()for _ in range(3)];s=f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=5)
shared=m.deep.local_mlp
print(json.dumps({'losses':losses,'tensors':{k:[v.dtype,v.shape,exact(v.tolist())]for k,v in s.tensors().items()},'aliases':shared is m.attention.gate_mlp and shared is m.attention.mlp and shared.norms[0] is m.norm and shared.norms[1] is m.norm and shared.lins[0].weight is shared.lins[1].weight,'norm_mode':m.norm.training,'frozen':not m.deep.global_mlp.lins[0].bias.trainable,'draw':exact(f.Tensor.rand([5],generator=rng).tolist())},sort_keys=True))
'''
        with tempfile.TemporaryDirectory()as directory:
            results=[subprocess.run([sys.executable,'-B','-c',script,mode,directory+'/state'],check=True,capture_output=True,text=True,timeout=60).stdout for mode in ('save','load')]
        first,second=map(json.loads,results);self.assertEqual(first,second);self.assertTrue(first['aliases']);self.assertTrue(first['frozen']);self.assertFalse(first['norm_mode'])

if __name__=='__main__':unittest.main()
