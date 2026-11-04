"""Actual PyG relational attention, all active/unused gradients and native state."""
import itertools,json,math,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch_geometric.nn import RGATConv as TorchRGAT
import ferro as fr
from ferro.graph import RGATConv


def native(value,grad=False):
    return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).split('.')[-1],requires_grad=grad).reshape(list(value.shape))


def values(value):
    return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)


class RGATTests(unittest.TestCase):
    def setUp(self):
        self.edges=torch.tensor([[0,1,2,2,3,0,1,2,1],[1,0,1,3,1,1,3,1,1]])
        self.types=torch.tensor([0,1,2,0,2,1,0,0,2])

    def close(self,a,b):
        torch.testing.assert_close(values(a),b.detach(),atol=4e-5,rtol=4e-4,equal_nan=True)

    def pair(self,**kwargs):
        a,b=RGATConv(4,2,4,**kwargs),TorchRGAT(4,2,4,**kwargs)
        aa,bb=dict(a.named_parameters()),dict(b.named_parameters());self.assertEqual(aa.keys(),bb.keys())
        with torch.no_grad(),fr.no_grad():
            for i,(name,p)in enumerate(bb.items()):
                val=torch.linspace(-.31,.37,p.numel()).reshape(p.shape)+.013*i
                p.copy_(val);aa[name].tensor().copy_(native(val))
        return a,b

    def compare(self,a,b,*,edges=None,types=None,rows=5,strided=False,edge=True,returned=True):
        te=self.edges if edges is None else edges;tt=self.types if types is None else types
        source=torch.linspace(-.8,.9,4*rows).reshape(4,rows).requires_grad_();x=native(source,True)
        tx=source.T;nx=x.T
        if not strided:tx=tx.contiguous();nx=nx.contiguous()
        ta=None
        if a.edge_dim is not None and edge:
            ta=torch.linspace(-.4,.7,te.shape[1]*a.edge_dim).reshape(te.shape[1],a.edge_dim).requires_grad_()
            if a.edge_dim==1:ta=ta.reshape(-1);ta.retain_grad()
        na=None if ta is None else native(ta,True)
        ne=native(te);nt=native(tt)
        actual=a(nx,ne,nt,na,return_attention_weights=returned)
        wanted=b(tx,te,tt,ta,return_attention_weights=returned)
        if returned is not None:
            out,(top,alpha)=actual;target,(_,refalpha)=wanted;self.assertIs(top,ne);self.close(alpha,refalpha)
        else:out,target=actual,wanted;alpha=refalpha=None
        self.close(out,target)
        seed=torch.linspace(-.4,.8,target.numel()).reshape(target.shape)
        loss=(out*native(seed)).sum();expected=(target*seed).sum()
        if alpha is not None:
            coeff=torch.linspace(-.3,.6,refalpha.numel()).reshape(refalpha.shape)
            loss=loss+(alpha*native(coeff)).sum();expected=expected+(refalpha*coeff).sum()
        loss.backward();expected.backward()
        pairs=[('x',x,source)]+([]if ta is None else [('edge_attr',na,ta)])
        bp=dict(b.named_parameters());pairs +=[(name,p.tensor(),bp[name])for name,p in a.named_parameters()]
        for name,left,right in pairs:
            with self.subTest(gradient=name):
                if right.grad is None:self.assertIsNone(left.grad)
                else:self.assertIsNotNone(left.grad);self.close(left.grad,right.grad)
        return out,alpha

    def test_attention_modes_relations_cardinality_head_reductions_and_all_gradients(self):
        for mechanism,mode,mod,concat in itertools.product(('within-relation','across-relation'),('additive-self-attention','multiplicative-self-attention'),(None,'additive','scaled','f-additive','f-scaled'),(False,True)):
            with self.subTest(mechanism=mechanism,mode=mode,mod=mod,concat=concat):
                a,b=self.pair(attention_mechanism=mechanism,attention_mode=mode,mod=mod,concat=concat,heads=2,dim=1 if mode.startswith('additive')else 2,edge_dim=2)
                self.compare(a,b,strided=True,returned=False)

    def test_basis_block_relation_weights_scalar_edges_missing_edges_and_native_topology(self):
        for regularization,edge_dim,edge in itertools.product(({},dict(num_bases=2),dict(num_blocks=2)),(None,1,2),(False,True)):
            a,b=self.pair(heads=2,edge_dim=edge_dim,**regularization);self.compare(a,b,edge=edge,returned=None)
        a,b=self.pair(heads=2,num_blocks=2,attention_mode='multiplicative-self-attention',dim=2)
        x=torch.linspace(-.3,.7,20).reshape(5,4);top=fr.graph.EdgeIndex(5,self.edges[0].tolist(),self.edges[1].tolist())
        actual,(edges,alpha)=a(native(x),top,native(self.types),return_attention_weights=True)
        expected,(_,wanted)=b(x,self.edges,self.types,return_attention_weights=True)
        self.close(actual,expected);self.close(alpha,wanted);self.assertEqual(edges.tolist(),self.edges.tolist())
        a,b=self.pair(heads=1,bias=False,concat=False,negative_slope=-.1,aggr='sum',attention_mechanism='within-relation')
        self.compare(a,b)
        order=torch.tensor([8,2,5,0,7,3,6,1,4]);edge=self.edges[:,order];types=self.types[order]
        output,(_,attention)=a(native(x),native(self.edges),native(self.types),return_attention_weights=True)
        permuted,(_,reordered)=a(native(x),native(edge),native(types),return_attention_weights=True)
        self.close(permuted,values(output));self.close(reordered,values(attention)[order])

    def test_empty_nodes_edges_and_unused_relations_retain_actual_zero_gradients(self):
        for rows,mode,mechanism,regularization in itertools.product((0,5),('additive-self-attention','multiplicative-self-attention'),('within-relation','across-relation'),({},dict(num_bases=2),dict(num_blocks=2))):
            with self.subTest(rows=rows,mode=mode,mechanism=mechanism,reg=regularization):
                a,b=self.pair(heads=2,edge_dim=2,attention_mode=mode,attention_mechanism=mechanism,dim=1 if mode.startswith('additive')else 2,**regularization)
                self.compare(a,b,rows=rows,edges=torch.empty(2,0,dtype=torch.long),types=torch.empty(0,dtype=torch.long))

    def test_independent_fixed_dropout_masks_pre_dropout_attention_and_eval(self):
        for mode,p in itertools.product(('additive-self-attention','multiplicative-self-attention'),(.35,1.)):
            a,b=self.pair(heads=2,edge_dim=2,attention_mode=mode,dim=1 if mode.startswith('additive')else 2,dropout=p)
            sampler=fr.nn.Dropout(p,seed=29)
            with fr.no_grad():sampler.rng_state.copy_(a.attn_dropout.rng_state)
            calls=[]
            def dropout(value,p=.5,training=True,inplace=False):
                self.assertTrue(training);self.assertFalse(inplace);calls.append(tuple(value.shape))
                return value*values(sampler(fr.ones(list(value.shape))))
            with patch('torch.nn.functional.dropout',side_effect=dropout):self.compare(a,b)
            self.assertEqual(calls,[(9,2*a.dim)]);self.assertEqual(a.attn_dropout.rng_state.tolist(),sampler.rng_state.tolist())
            before=a.attn_dropout.rng_state.tolist();a.eval();b.eval()
            a.zero_grad();b.zero_grad();self.compare(a,b)
            self.assertEqual(before,a.attn_dropout.rng_state.tolist())

    def test_reset_bounds_safe_l2_initialization_and_preserved_reset_slots(self):
        a=RGATConv(4,2,4,heads=2,num_bases=3,edge_dim=2,dropout=.3)
        self.assertEqual(a.l2.tolist(),[[.5,.5],[.5,.5]])
        b=TorchRGAT(4,2,4,heads=2,num_bases=3,edge_dim=2,dropout=.3)
        with fr.no_grad(),torch.no_grad():
            a.l2.tensor().fill_(.7);a.w.tensor().fill_(.4);b.l2.fill_(.7);b.w.fill_(.4)
        before=a.attn_dropout.rng_state.tolist();a.reset_parameters();b.reset_parameters()
        self.close(a.l2.tensor(),b.l2);self.close(a.w.tensor(),b.w);self.assertEqual(before,a.attn_dropout.rng_state.tolist())
        for name in ('q','k','att','basis','e'):
            value=getattr(a,name);bound=math.sqrt(6./(value.shape[-2]+value.shape[-1]))
            self.assertLessEqual(values(value).abs().max().item(),bound);self.assertGreater(values(value).abs().sum().item(),0)
        self.close(a.l1.tensor(),b.l1);self.close(a.b1.tensor(),b.b1);self.close(a.b2.tensor(),b.b2);self.close(a.bias.tensor(),b.bias)

    def test_metadata_capacity_and_invalid_options_precede_allocation_or_streams(self):
        for options in (dict(num_bases=2,num_blocks=2),dict(num_blocks=3),dict(attention_mode='bad'),dict(attention_mechanism='bad'),dict(dim=2),dict(mod='scaled',dropout=.2),dict(dropout=math.nan),dict(edge_dim=0),dict(aggr='mean'),dict(flow='target_to_source'),dict(mod='unknown')):
            with self.assertRaises((ValueError,TypeError,NotImplementedError)):RGATConv(4,2,4,**options)
        import ferro._graph_rgat as helper
        with patch.object(helper.Tensor,'zeros',side_effect=AssertionError('allocated before capacity check')):
            with self.assertRaisesRegex(ValueError,'capacity'):RGATConv(1,1,sys.maxsize)
        a=RGATConv(4,2,4,heads=2,edge_dim=2,dropout=.3);x=fr.ones(5,4);edges=native(self.edges);types=native(self.types);attr=fr.ones(9,2);before=a.attn_dropout.rng_state.tolist()
        bad=[dict(edge_type=None),dict(edge_type=fr.zeros(9)),dict(edge_type=fr.tensor([-1]*9,dtype='int64')),dict(edge_type=fr.tensor([4]*9,dtype='int64')),dict(edge_attr=fr.tensor(1.)),dict(edge_attr=fr.ones(9,3)),dict(edge_attr=fr.ones(9,2,dtype='float64')),dict(size=(5,4)),dict(return_attention_weights=1),dict(edge_index=fr.tensor([[0],[5]],dtype='int64'))]
        for changes in bad:
            kwargs=dict(edge_index=edges,edge_type=types,edge_attr=attr);kwargs.update(changes)
            with self.assertRaises((ValueError,TypeError,NotImplementedError)):a(x,**kwargs)
            self.assertEqual(before,a.attn_dropout.rng_state.tolist())
        with self.assertRaises(NotImplementedError):fr.capture(lambda:a(x,edges,types,attr))
        with self.assertRaises(AttributeError):a.heads=4
        with patch.object(helper.Tensor,'index_select',side_effect=AssertionError('gather before preflight')):
            huge=fr.zeros(1,4).expand([sys.maxsize//8+1,4])
            with self.assertRaisesRegex(ValueError,'capacity'):a(huge,fr.zeros(2,0,dtype='int64'),fr.zeros(0,dtype='int64'))

    def test_checkpoint_config_mismatch_rejects_before_any_state_write(self):
        a,_=self.pair(heads=2,edge_dim=2,dropout=.3);saved=fr.checkpoint.snapshot_training(a,optimizers={},generator=fr.Generator(5))
        b=RGATConv(4,2,4,heads=2,edge_dim=2,dropout=.3,attention_mechanism='within-relation')
        before={k:v.tolist()for k,v in b.state_dict().items()}
        with self.assertRaises(ValueError):saved.restore(b,optimizers={},generator=fr.Generator(9))
        self.assertEqual(before,{k:v.tolist()for k,v in b.state_dict().items()})

    def test_fresh_process_complete_state_aliases_dropout_optimizer_and_rng(self):
        script=r'''
import json,math,sys,ferro as f
from ferro.graph import RGATConv
class Model(f.nn.Module):
    def build(self):
        self.layer=RGATConv(4,2,4,heads=2,num_bases=2,edge_dim=2,dropout=.3)
        self.layer.q=self.layer.k
        self.alias=self.layer
    def forward(self,x,edge):
        top=f.tensor([[0,1,2,2,3,0],[1,0,1,3,1,1]],dtype='int64');types=f.tensor([0,1,2,0,2,1],dtype='int64')
        return self.layer(x,top,types,edge,return_attention_weights=True)
def exact(v):
    if isinstance(v,float):
        if not math.isfinite(v):raise RuntimeError('nonfinite fixture')
        return v.hex()
    return [exact(x)for x in v]if isinstance(v,list)else v
m=Model();rng=f.Generator(77);opt=f.optim.AdamW(m.parameters(),lr=.003)
def step():
    x=f.Tensor.rand([5,4],generator=rng)-.5;edge=f.Tensor.rand([6,2],generator=rng)-.5
    out,(_,alpha)=m(x,edge);target=f.Tensor.rand([5,4],generator=rng)
    loss=(out-target).square().mean()+alpha.square().mean()*.2
    loss.backward();opt.step();opt.zero_grad();return exact(loss.item())
if sys.argv[1]=='save':
    m.layer.w.trainable=False;m.layer.activation.eval()
    for _ in range(2):step()
    f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=2).save(sys.argv[2])
else:f.checkpoint.load(sys.argv[2]).restore(m,optimizers={'main':opt},generator=rng)
losses=[step()for _ in range(3)]
s=f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=5)
print(json.dumps({'losses':losses,'tensors':{k:[v.dtype,v.shape,exact(v.tolist())]for k,v in s.tensors().items()},'alias':m.layer is m.alias and m.layer.q is m.layer.k,'mode':m.layer.activation.training,'frozen':not m.layer.w.trainable,'draw':exact(f.Tensor.rand([7],generator=rng).tolist())},sort_keys=True))
'''
        with tempfile.TemporaryDirectory()as directory:
            outputs=[subprocess.run([sys.executable,'-B','-c',script,mode,directory+'/state'],check=True,capture_output=True,text=True,timeout=60).stdout for mode in ('save','load')]
        first,second=map(json.loads,outputs);self.assertEqual(first,second);self.assertTrue(first['alias']);self.assertTrue(first['frozen']);self.assertFalse(first['mode'])

if __name__=='__main__':unittest.main()

