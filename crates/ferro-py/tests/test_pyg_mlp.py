"""Actual PyG MLP construction, state, gradients and registered random streams."""
import itertools,json,math,subprocess,sys,tempfile,unittest,warnings
from pathlib import Path
from unittest.mock import patch
import torch
from torch_geometric.nn import MLP as TorchMLP,BatchNorm as TorchBatchNorm
import ferro as fr
from ferro.graph import MLP


def native(value,grad=False):
    return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).split('.')[-1],requires_grad=grad).reshape(list(value.shape))
def values(value):return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)


class MLPTests(unittest.TestCase):
    def close(self,a,b):torch.testing.assert_close(values(a),b.detach(),atol=4e-5,rtol=4e-4,equal_nan=True)
    def copy(self,a,b):
        aa,bb=dict(a.named_parameters()),dict(b.named_parameters());self.assertEqual(aa.keys(),bb.keys());gen=torch.Generator().manual_seed(991)
        with fr.no_grad(),torch.no_grad():
            for name,p in bb.items():
                val=torch.randn(p.shape,generator=gen)*.23
                if 'norms' in name and name.endswith('weight'):val=val+1
                p.copy_(val);aa[name].tensor().copy_(native(val))
    def pair(self,*args,**kwargs):
        a,b=MLP(*args,**kwargs),TorchMLP(*args,**kwargs);self.copy(a,b);return a,b
    def compare(self,a,b,rows=6,return_emb=True,batch=True):
        source=torch.randn(a.in_channels,rows,generator=torch.Generator().manual_seed(992),requires_grad=True);nx=native(source,True)
        ids=torch.tensor([0,2,0,2,2,0])if rows==6 else torch.zeros(rows,dtype=torch.long)
        kw=dict(batch=native(ids),batch_size=4)if batch else {};tw=dict(batch=ids,batch_size=4)if batch else {}
        out=a(nx.T,return_emb=return_emb,**kw);target=b(source.T,return_emb=return_emb,**tw)
        if isinstance(return_emb,bool):
            output,emb=out;expected,ref_emb=target
            self.assertEqual(emb is None,ref_emb is None)
            if emb is not None:
                self.close(emb,ref_emb)
                if not a.plain_last:self.assertIs(output,emb);self.assertIs(expected,ref_emb)
        else:output,expected=out,target;emb=ref_emb=None
        self.close(output,expected);seed=torch.linspace(-.5,.7,expected.numel()).reshape(expected.shape)
        loss=(output*native(seed)).sum();reference=(expected*seed).sum()
        if emb is not None:
            seed=torch.linspace(.1,.6,ref_emb.numel()).reshape(ref_emb.shape)
            loss=loss+(emb*native(seed)).sum();reference=reference+(ref_emb*seed).sum()
        loss.backward();reference.backward();self.close(nx.grad,source.grad)
        bp=dict(b.named_parameters())
        for name,p in a.named_parameters():
            with self.subTest(parameter=name):
                if bp[name].grad is None:self.assertIsNone(p.tensor().grad)
                else:self.assertIsNotNone(p.tensor().grad);self.close(p.tensor().grad,bp[name].grad)
        ab={k:v for k,v in a.named_buffers()if not k.startswith('dropouts.')};bb=dict(b.named_buffers());self.assertEqual(ab.keys(),bb.keys())
        for name,v in ab.items():self.close(v,bb[name])
        return output,emb

    def test_actual_pyg_layer_norm_order_modes_embeddings_and_all_gradients(self):
        for norm,plain,first,training in itertools.product((None,'batch_norm','layer_norm','graph_norm'),(False,True),(False,True),(False,True)):
            with self.subTest(norm=norm,plain=plain,first=first,training=training):
                a,b=self.pair([4,6,3],norm=norm,plain_last=plain,act_first=first,bias=[True,False]);a.train(training);b.train(training);self.compare(a,b)

    def test_constructor_forms_return_embedding_and_dropout_lists(self):
        configs=[(([4,6,3],),{}),((4,),dict(hidden_channels=5,out_channels=3,num_layers=3)),((),dict(in_channels=4,hidden_channels=5,out_channels=3,num_layers=3)),(([99,99],),dict(in_channels=4,out_channels=3,num_layers=1)),((4,),dict(in_channels=99,out_channels=3,num_layers=1))]
        for args,kwargs in configs:
            for returned in (None,False,True):
                a,b=self.pair(*args,norm=None,**kwargs);self.assertEqual(list(a.channel_list),b.channel_list);self.assertEqual((a.in_channels,a.out_channels,a.num_layers),(b.in_channels,b.out_channels,b.num_layers));self.compare(a,b,return_emb=returned)
        for dropout,plain in ((.3,True),([.2,.4],True),(.3,False)):
            a,b=self.pair([4,4,3],dropout=dropout,plain_last=plain,norm=None)
            self.assertEqual(list(a.dropout),b.dropout)
        with warnings.catch_warnings(record=True)as seen:
            warnings.simplefilter('always');a,b=self.pair([4,5,3],relu_first=True,batch_norm=False)
        self.assertEqual(len(seen),2);self.assertTrue(a.act_first);self.compare(a,b)

    def test_activation_options_and_supplied_shared_registered_modules(self):
        for name,options in ((None,{}),('tanh',{}),('gelu',dict(approximate='tanh')),('leaky_relu',dict(negative_slope=.1)),('sigmoid',{})):
            a,b=self.pair([4,5,3],norm=None,act=name,act_kwargs=options,plain_last=False);self.compare(a,b)
        act,tact=fr.nn.PReLU(),torch.nn.PReLU();norm,tnorm=fr.graph.BatchNorm(4),TorchBatchNorm(4)
        a,b=MLP([4,4,4],act=act,norm=norm,plain_last=False),TorchMLP([4,4,4],act=tact,norm=tnorm,plain_last=False)
        self.assertIs(a.norms[0],a.norms[1]);self.assertIs(b.norms[0],b.norms[1]);self.assertIs(a.act,act);self.copy(a,b);self.compare(a,b)
        self.assertEqual(norm.module.num_batches_tracked.item(),2)
        saved=act.weight.tolist();a.reset_parameters();self.assertEqual(saved,act.weight.tolist())

    def test_empty_inputs_singletons_and_running_state_transitions(self):
        for norm,plain in itertools.product((None,'batch_norm','layer_norm','graph_norm'),(False,True)):
            a,b=self.pair([4,5,3],norm=norm,plain_last=plain);self.compare(a,b,rows=0)
        a,b=self.pair([4,5,3],norm_kwargs=dict(momentum=None),dropout=.2)
        before=[m.rng_state.tolist()for m in a.dropouts]
        for m,x in ((a,fr.ones(1,4)),(b,torch.ones(1,4))):
            with self.assertRaises(ValueError):m(x)
        self.assertEqual(a.norms[0].module.num_batches_tracked.item(),1);self.assertEqual(len(a.norms),1)
        self.assertEqual(before,[m.rng_state.tolist()for m in a.dropouts]);a.eval();b.eval();self.compare(a,b,rows=1)
        a,b=self.pair([4,5,3],norm_kwargs=dict(momentum=None,track_running_stats=False));a.eval();b.eval();self.compare(a,b)

    def test_independent_per_layer_dropout_masks_final_list_probability_and_eval(self):
        for plain,probabilities in ((True,[.2,.4]),(False,[.2,.4]),(True,.3)):
            a,b=self.pair([4,5,3],norm='graph_norm',plain_last=plain,dropout=probabilities)
            samplers=[fr.nn.Dropout(layer.p,seed=19)for layer in a.dropouts]
            with fr.no_grad():
                for sampler,layer in zip(samplers,a.dropouts):sampler.rng_state.copy_(layer.rng_state)
            calls=[]
            def dropout(value,p=.5,training=True,inplace=False):
                self.assertTrue(training);self.assertFalse(inplace);sampler=samplers[len(calls)];self.assertEqual(p,sampler.p);calls.append(tuple(value.shape))
                return value*values(sampler(fr.ones(list(value.shape))))
            with patch('torch.nn.functional.dropout',side_effect=dropout):self.compare(a,b)
            self.assertEqual(calls,[(6,5),(6,3)])
            for sampler,layer in zip(samplers,a.dropouts):self.assertEqual(sampler.rng_state.tolist(),layer.rng_state.tolist())
            before=[layer.rng_state.tolist()for layer in a.dropouts];a.zero_grad();b.zero_grad();a.eval();b.eval();self.compare(a,b)
            self.assertEqual(before,[layer.rng_state.tolist()for layer in a.dropouts])

    def test_reset_initialization_and_untouched_activation_and_stream_state(self):
        act,tact=fr.nn.PReLU(),torch.nn.PReLU();a,b=MLP([4,5,3],act=act,dropout=.3),TorchMLP([4,5,3],act=tact,dropout=.3)
        with fr.no_grad(),torch.no_grad():act.weight.tensor().fill_(.73);tact.weight.fill_(.73)
        a.norms[0].module.num_batches_tracked.copy_(fr.tensor(17,dtype='int64'))
        before=[v.rng_state.tolist()for v in a.dropouts];a.reset_parameters();b.reset_parameters()
        self.close(act.weight.tensor(),tact.weight);self.assertEqual(before,[v.rng_state.tolist()for v in a.dropouts]);self.assertEqual(a.norms[0].module.num_batches_tracked.item(),0)
        for source,linear in zip(a.channel_list,a.lins):
            self.assertLessEqual(values(linear.weight).abs().max().item(),source**-.5)
            self.assertLessEqual(values(linear.bias).abs().max().item(),source**-.5);self.assertGreater(values(linear.bias).abs().sum().item(),0)
        self.close(a.norms[0].module.weight.tensor(),b.norms[0].module.weight);self.close(a.norms[0].module.bias.tensor(),b.norms[0].module.bias)

    def test_metadata_validation_capacity_and_known_batch_preflight(self):
        bad=[dict(channel_list=[4]),dict(channel_list=[4,0,3]),dict(in_channels=4,out_channels=3),dict(channel_list=[4,5,3],dropout=[.2]),dict(channel_list=[4,5,3],dropout=[.2,math.nan]),dict(channel_list=[4,5,3],bias=[True]),dict(channel_list=[4,5,3],norm='unsupported'),dict(channel_list=[4,5,3],act=lambda x:x),dict(channel_list=[4,5,3],unknown=True)]
        for kwargs in bad:
            with self.assertRaises((ValueError,TypeError,NotImplementedError)):MLP(**kwargs)
        import ferro._graph_mlp as helper
        with patch.object(helper.nn,'Linear',side_effect=AssertionError('linear allocation before capacity')):
            with self.assertRaisesRegex(ValueError,'capacity'):MLP([1,sys.maxsize],norm=None)
        a=MLP([4,5,3],norm='graph_norm',dropout=[.3,.4]);x=fr.ones(6,4);before=[m.rng_state.tolist()for m in a.dropouts]
        for kwargs in (dict(batch=fr.tensor([-1]*6,dtype='int64')),dict(batch=fr.zeros(5,dtype='int64')),dict(batch=fr.ones(6,dtype='int64'),batch_size=1),dict(batch_size=-1),dict(return_emb=1)):
            with self.assertRaises((ValueError,TypeError)):a(x,**kwargs)
            self.assertEqual(before,[m.rng_state.tolist()for m in a.dropouts])
        with self.assertRaises(ValueError):a(fr.ones(6,4,dtype='float64'))
        with self.assertRaises(NotImplementedError):fr.capture(lambda:a(x))
        with self.assertRaises(AttributeError):a.dropout=(0.,0.)
        wrong=MLP([4,4,3],norm=fr.graph.GraphNorm(4),plain_last=False,dropout=.3);before=[m.rng_state.tolist()for m in wrong.dropouts]
        with self.assertRaisesRegex(ValueError,'normalization width'):wrong(x)
        self.assertEqual(before,[m.rng_state.tolist()for m in wrong.dropouts])
        a,b=self.pair([4,5,3],norm=None)
        self.close(a(x,batch='ignored',batch_size='ignored'),b(torch.ones(6,4),batch='ignored',batch_size='ignored'))

    def test_checkpoint_configuration_mismatch_is_atomic(self):
        a=MLP([4,5,3],dropout=.3);saved=fr.checkpoint.snapshot_training(a,optimizers={},generator=fr.Generator(2))
        for kwargs in (dict(dropout=[.3,.2]),dict(act_first=True),dict(norm_kwargs=dict(momentum=.2))):
            b=MLP([4,5,3],**kwargs);before={k:v.tolist()for k,v in b.state_dict().items()}
            with self.assertRaises(ValueError):saved.restore(b,optimizers={},generator=fr.Generator(3))
            self.assertEqual(before,{k:v.tolist()for k,v in b.state_dict().items()})

    def test_gin_registered_mlp_composition_and_all_gradients(self):
        from torch_geometric.nn import GINConv as TorchGIN
        a=fr.graph.GINConv(MLP([4,5,3],norm=None,act='tanh'),train_eps=True)
        b=TorchGIN(TorchMLP([4,5,3],norm=None,act='tanh'),train_eps=True)
        self.copy(a,b)
        source=torch.randn(6,4,generator=torch.Generator().manual_seed(993),requires_grad=True);x=native(source,True)
        topology=torch.tensor([[0,1,2,3,0,3],[1,0,1,2,1,3]])
        out,wanted=a(x,native(topology)),b(source,topology);self.close(out,wanted)
        seed=torch.linspace(-.4,.8,wanted.numel()).reshape(wanted.shape)
        (out*native(seed)).sum().backward();(wanted*seed).sum().backward();self.close(x.grad,source.grad)
        expected=dict(b.named_parameters())
        for name,p in a.named_parameters():self.close(p.tensor().grad,expected[name].grad)

    def test_fresh_process_complete_state_shared_norm_parameters_and_rng(self):
        script=r'''
import json,math,sys,ferro as f
from ferro.graph import MLP
class Model(f.nn.Module):
    def build(self):
        self.norm=f.graph.BatchNorm(4,momentum=None)
        self.mlp=MLP([4,4,4],norm=self.norm,act=f.nn.PReLU(),plain_last=False,dropout=[.2,.3])
        self.mlp.lins[1].weight=self.mlp.lins[0].weight
        self.alias=self.mlp
    def forward(self,x):return self.mlp(x,return_emb=True)[0]
def exact(v):
    if isinstance(v,float):
        if not math.isfinite(v):raise RuntimeError('nonfinite fixture')
        return v.hex()
    return [exact(x)for x in v]if isinstance(v,list)else v
m=Model();rng=f.Generator(73);opt=f.optim.AdamW(m.parameters(),lr=.003)
def step():
    x=f.Tensor.rand([6,4],generator=rng)*2-1;target=f.Tensor.rand([6,4],generator=rng)
    loss=(m(x)-target).square().mean();loss.backward();opt.step();opt.zero_grad();return exact(loss.item())
if sys.argv[1]=='save':
    for _ in range(2):step()
    m.norm.eval();m.mlp.act.weight.trainable=False
    f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=2).save(sys.argv[2])
else:f.checkpoint.load(sys.argv[2]).restore(m,optimizers={'main':opt},generator=rng)
losses=[step()for _ in range(3)]
s=f.checkpoint.snapshot_training(m,optimizers={'main':opt},generator=rng,step=5)
print(json.dumps({'losses':losses,'tensors':{k:[v.dtype,v.shape,exact(v.tolist())]for k,v in s.tensors().items()},'aliases':m.alias is m.mlp and m.mlp.norms[0] is m.norm and m.mlp.norms[1] is m.norm and m.mlp.lins[0].weight is m.mlp.lins[1].weight,'norm_mode':m.norm.training,'act_frozen':not m.mlp.act.weight.trainable,'draw':exact(f.Tensor.rand([5],generator=rng).tolist())},sort_keys=True))
'''
        with tempfile.TemporaryDirectory()as directory:
            results=[subprocess.run([sys.executable,'-B','-c',script,mode,directory+'/state'],check=True,capture_output=True,text=True,timeout=60).stdout for mode in ('save','load')]
        first,second=map(json.loads,results);self.assertEqual(first,second);self.assertTrue(first['aliases']);self.assertTrue(first['act_frozen']);self.assertFalse(first['norm_mode'])

if __name__=='__main__':unittest.main()

