"""Channel mask contracts against actual Torch kernels and native continuation."""
import math
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import torch
import ferro as fr
from ferro.nn import functional as F
import test_embedding_options as helpers

native = helpers.native


class ChannelModel(fr.nn.Module):
    probability: float = .3

    def build(self):
        self.dropout = fr.nn.Dropout1d(self.probability, seed=91)
        self.alias = self.dropout
        self.projection = fr.nn.Linear(3, 2)

    def forward(self, x):
        return self.projection(self.alias(self.dropout(x)).transpose(1, 2))


class ChannelDropoutTests(unittest.TestCase):
    same = helpers.EmbeddingOptionsTests.same
    shapes = ((1, (2, 4)), (1, (2, 3, 4)), (2, (2, 3, 4)), (2, (2, 3, 4, 5)), (3, (2, 3, 4, 5)), (3, (2, 3, 3, 4, 5)))

    def matched_seed(self, function, expected_mask, p):
        ones = torch.ones_like(expected_mask)
        with warnings.catch_warnings(), torch.random.fork_rng():
            warnings.simplefilter('ignore')
            for seed in range(4096):
                torch.manual_seed(seed)
                if torch.equal(function(ones, p) != 0, expected_mask != 0): return seed
        self.fail('no matching small channel mask found in bounded Torch seed search')

    def test_actual_torch_matched_channel_masks_values_strides_and_input_gradients(self):
        for dimension, shape in self.shapes:
            for dtype in ('float32', 'float64'):
                with self.subTest(dimension=dimension, shape=shape, dtype=dtype):
                    function, oracle = getattr(F, f'dropout{dimension}d'), getattr(torch.nn.functional, f'dropout{dimension}d')
                    base = torch.linspace(-1.1, 1.3, math.prod(shape), dtype=getattr(torch, dtype)).reshape(shape)
                    x = base.transpose(-1, -2).detach().requires_grad_(True)
                    nx = native(base).transpose(-1, -2).detach().requires_grad_(True)
                    multiplier = function(fr.ones(x.shape, dtype=dtype), .5, seed=81, offset=9)
                    tm = torch.tensor(multiplier.tolist(), dtype=x.dtype)
                    seed = self.matched_seed(oracle, tm, .5)
                    with warnings.catch_warnings(), torch.random.fork_rng():
                        warnings.simplefilter('ignore');torch.manual_seed(seed)
                        expected = oracle(x, .5)
                    actual = function(nx, .5, seed=81, offset=9)
                    self.same(actual, expected)
                    weights = torch.linspace(.3, 1.3, x.numel(), dtype=x.dtype).reshape(x.shape)
                    (actual.square() * native(weights)).sum().backward(); (expected.square() * weights).sum().backward()
                    self.same(nx.grad, x.grad)

    def test_dtype_probability_precision_nonfinite_values_and_full_drop(self):
        for dtype in ('float32', 'float64'):
            for p in (.123456789, 1.-1e-10, 1.):
                x = torch.tensor([[[1., -2., float('inf')], [float('nan'), -float('inf'), -0.]]], dtype=getattr(torch,dtype), requires_grad=True)
                nx = native(x, True)
                multiplier = F.dropout1d(fr.ones(x.shape,dtype=dtype),p,seed=31)
                seed = self.matched_seed(torch.nn.functional.dropout1d,torch.tensor(multiplier.tolist(),dtype=x.dtype),p)
                with torch.random.fork_rng():
                    torch.manual_seed(seed);expected=torch.nn.functional.dropout1d(x,p)
                actual=F.dropout1d(nx,p,seed=31);self.same(actual,expected)
                actual.sum().backward();expected.sum().backward();self.same(nx.grad,x.grad)

    def test_float32_surviving_scale_matches_actual_torch_exactly(self):
        for p in (.27981547357298786, .123456789, .7, .31):
            actual = F.dropout1d(fr.ones(2,8,3),p,seed=31)
            with torch.random.fork_rng():
                torch.manual_seed(31);expected = torch.nn.functional.dropout1d(torch.ones(2,8,3),p)
            av=torch.tensor(actual.tolist());reference=expected[expected!=0]
            self.assertGreater(reference.numel(),0);self.assertGreater((av!=0).sum().item(),0)
            self.assertTrue(torch.equal(av[av!=0],torch.full_like(av[av!=0],reference[0].item())))

    def test_identity_alias_empty_and_mode_behavior_matches_torch(self):
        for dimension,shape in self.shapes:
            function,oracle=getattr(F,f'dropout{dimension}d'),getattr(torch.nn.functional,f'dropout{dimension}d')
            for empty in (False,True):
                size=list(shape)
                if empty:size[-1]=0
                for p,training in ((0.,True),(.5,False),(.5,True),(1.,True)):
                    if not empty and training and p not in (0.,):continue
                    for enabled in (False,True):
                        x=torch.ones(size,requires_grad=True);nx=fr.ones(size,requires_grad=True)
                        with warnings.catch_warnings(),torch.set_grad_enabled(enabled),(fr.enable_grad() if enabled else fr.no_grad()):
                            warnings.simplefilter('ignore');expected=oracle(x,p,training);actual=function(nx,p,training)
                        self.same(actual,expected);self.assertEqual(actual is nx,expected is x)
                        self.assertEqual(actual.requires_grad,expected.requires_grad)
                        self.assertTrue(actual._shares_storage(nx))
        empty=fr.zeros(sys.maxsize//3,0,3)
        result=F.dropout1d(empty,.5,seed=7)
        self.assertIs(result,empty)

    def test_module_channel_counter_evaluation_failures_and_serialized_stream(self):
        for dimension,shape in self.shapes:
            module=getattr(fr.nn,f'Dropout{dimension}d')(.3,seed=19)
            x=fr.ones(shape,dtype='float64')
            function=getattr(F,f'dropout{dimension}d')
            count=shape[0] if dimension!=2 and len(shape)==dimension+1 else shape[0]*shape[1]
            self.same(module(x),torch.tensor(function(x,.3,seed=19,offset=0).tolist(),dtype=torch.float64))
            self.assertEqual(module._offset,count)
            self.same(module(x),torch.tensor(function(x,.3,seed=19,offset=count).tolist(),dtype=torch.float64))
            self.assertEqual(module._offset,2*count)
            before=module.rng_state.tolist();module.eval();module(x);self.assertEqual(module.rng_state.tolist(),before)
            module.train();module(fr.zeros(list(shape[:-1])+[0],dtype='float64'));self.assertEqual(module.rng_state.tolist(),before)
            with fr.no_grad():module.rng_state.copy_(module._encode(19,2**64-2))
            before=module.rng_state.tolist()
            with self.assertRaises(ValueError):module(x)
            self.assertEqual(module.rng_state.tolist(),before)
            with fr.no_grad():module.rng_state.copy_(module._encode(19,0))
            with self.assertRaises((ValueError,NotImplementedError)):fr.capture(lambda:module(x))
            self.assertEqual(module._offset,0)
            with self.assertRaises(NotImplementedError):fr.func.jvp(module,(x.float(),),(fr.ones(shape),))
            self.assertEqual(module._offset,0)
            for p in (0.,1.):
                fixed=getattr(fr.nn,f'Dropout{dimension}d')(p,seed=19);before=fixed.rng_state.tolist();fixed(x)
                self.assertEqual(fixed.rng_state.tolist(),before)

    def test_shape_dtype_option_and_capacity_guards_precede_mask_and_stream(self):
        for dimension in (1,2,3):
            module=getattr(fr.nn,f'Dropout{dimension}d')(.5,seed=19);before=module.rng_state.tolist()
            for shape in ((),(3,),tuple([2]*(dimension+3))):
                with self.assertRaises(ValueError):module(fr.ones(shape))
                self.assertEqual(module.rng_state.tolist(),before)
            shape=[2]*(dimension+2)
            with self.assertRaises(NotImplementedError):module(fr.ones(shape,dtype='int64'))
            self.assertEqual(module.rng_state.tolist(),before)
        for option in (dict(p=-1.),dict(p=2.),dict(p=float('nan')),dict(inplace=True),dict(seed=-1)):
            with self.assertRaises((ValueError,TypeError,NotImplementedError)):fr.nn.Dropout1d(**option)
        function=F.dropout1d;value=fr.ones(2,3,4)
        for options in (dict(seed=-1),dict(offset=-1),dict(offset=2**64-2),dict(training=1),dict(inplace=True)):
            with self.assertRaises((ValueError,TypeError,NotImplementedError)):function(value,**options)
        module=fr.nn.Dropout1d(.3,seed=17);before=module.rng_state.tolist()
        with patch('ferro.nn._channel_dropout.Tensor.ones',side_effect=AssertionError('mask allocated before validation')):
            with self.assertRaisesRegex(ValueError,'capacity'):module(fr.ones(1,1,1).expand(sys.maxsize//4+1,1,1))
        self.assertEqual(module.rng_state.tolist(),before)

    def test_native_higher_derivative_and_seeded_functional_jvp(self):
        x=torch.tensor([[[.3,-.8],[.7,1.1]]],requires_grad=True);nx=native(x,True)
        mask=F.dropout1d(fr.ones(x.shape),.5,seed=8)
        tm=torch.tensor(mask.tolist())
        output=F.dropout1d(nx,.5,seed=8);actual=(output*output).sum();expected=(x*tm).square().sum()
        ga=fr.autograd.grad(actual,(nx,),create_graph=True)[0];ge=torch.autograd.grad(expected,x,create_graph=True)[0]
        self.same(ga,ge)
        self.same(fr.autograd.grad(ga.sum(),(nx,))[0],torch.autograd.grad(ge.sum(),x)[0])
        value,tangent=fr.func.jvp(lambda value:F.dropout1d(value,.5,seed=8),(nx,),(fr.ones(nx.shape),))
        self.same(value,x.detach()*tm);self.same(tangent,tm)

    def test_complete_checkpoint_restores_aliases_optimizer_rng_and_channel_masks(self):
        def step(model,optimizer,rng):
            output=model(fr.Tensor.randn([2,3,4],generator=rng));loss=output.square().mean()
            result=output.tolist(),loss.item();loss.backward();optimizer.step();optimizer.zero_grad();return result
        model,rng=ChannelModel(),fr.Generator(19);optimizer=fr.optim.AdamW(model.parameters(),lr=.01)
        step(model,optimizer,rng)
        saved=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=rng,step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory+'/state');saved=fr.checkpoint.load(directory+'/state')
        expected=[step(model,optimizer,rng) for _ in range(3)]
        target,other_rng=ChannelModel(),fr.Generator(1);other=fr.optim.AdamW(target.parameters(),lr=.7)
        saved.restore(target,optimizers={'main':other},generator=other_rng)
        self.assertIs(target.alias,target.dropout);self.assertEqual(expected,[step(target,other,other_rng) for _ in range(3)])
        left=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=rng,step=4).tensors()
        right=fr.checkpoint.snapshot_training(target,optimizers={'main':other},generator=other_rng,step=4).tensors()
        self.assertEqual(set(left),set(right))
        for key in left:self.assertEqual((left[key].dtype,left[key].shape,left[key].tolist()),(right[key].dtype,right[key].shape,right[key].tolist()),key)
        wrong=ChannelModel(probability=.4);before=[p.tensor().tolist() for p in wrong.parameters()]
        with self.assertRaises(ValueError):saved.restore(wrong,optimizers={'main':fr.optim.AdamW(wrong.parameters())},generator=fr.Generator(1))
        self.assertEqual(before,[p.tensor().tolist() for p in wrong.parameters()])


if __name__=='__main__':unittest.main(verbosity=2)
