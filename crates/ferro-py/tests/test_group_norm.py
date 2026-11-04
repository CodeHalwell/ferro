"""Actual Torch grouped normalization, empty statistics and state contracts."""
import inspect
import sys
import unittest
from unittest.mock import patch
import torch
import ferro as fr
import ferro.nn as norm
from ferro.nn import functional as F
from test_layer_norm import native,close


class GroupNormTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(2063)

    def compare(self, shape, groups, dtype=torch.float32, affine=True, bias=True, strided=False, eps=1e-5):
        value=torch.randn(shape,dtype=dtype)
        x=value.transpose(0,-1).contiguous().transpose(0,-1).detach().requires_grad_() if strided else value.requires_grad_()
        nx=native(value.transpose(0,-1).contiguous()).transpose(0,len(shape)-1).requires_grad_(True) if strided else native(x)
        w=torch.randn(shape[1],dtype=dtype,requires_grad=True) if affine else None
        b=torch.randn(shape[1],dtype=dtype,requires_grad=True) if bias else None
        nw,nb=None if w is None else native(w),None if b is None else native(b)
        actual=F.group_norm(nx,groups,nw,nb,eps);expected=torch.nn.functional.group_norm(x,groups,w,b,eps)
        if not affine and bias:
            direct=expected
            expected=torch.nn.functional.group_norm(x,groups,None,None,eps)+b.reshape([1,shape[1]]+[1]*(len(shape)-2))
            tolerance=3e-11 if dtype==torch.float64 else 5e-5
            torch.testing.assert_close(direct,expected,atol=tolerance,rtol=tolerance)
        close(actual,expected)
        seed=torch.randn(shape,dtype=dtype)
        (actual*native(seed)).sum().backward();(expected*seed).sum().backward()
        for a,t in ((nx,x),(nw,w),(nb,b)):
            if a is not None:close(a.grad,t.grad)

    def test_all_ranks_dtypes_groups_strides_and_affine_adjoints(self):
        for dtype in (torch.float32,torch.float64):
            for shape in ((2,6),(2,6,5),(2,6,3,4),(2,6,2,3,4),(2,6,2,2,2,3)):
                for groups in (1,2,6):
                    for affine,bias in ((False,False),(True,False),(True,True),(False,True)):
                        with self.subTest(shape=shape,dtype=dtype,groups=groups,affine=affine,bias=bias):
                            self.compare(shape,groups,dtype,affine,bias,True)

    def test_empty_outer_spatial_and_channel_dimensions_and_singleton_rule(self):
        for dtype in (torch.float32,torch.float64):
            for shape in ((0,4,3),(2,4,0),(2,4,0,2),(2,0,3),(0,4,0)):
                for groups in (1,2,4):
                    for affine in (False,True):
                        with self.subTest(shape=shape,dtype=dtype,groups=groups,affine=affine):self.compare(shape,groups,dtype,affine,affine)
            self.compare((2,4),4,dtype)
        for shape in ((1,4),(1,4,1,1)):
            with self.assertRaises(ValueError):F.group_norm(fr.ones(shape),4)
            with self.assertRaises(ValueError):torch.nn.functional.group_norm(torch.ones(shape),4)

    def test_module_factory_signature_reset_flags_and_float64_nonaffine(self):
        self.assertEqual(list(inspect.signature(norm.GroupNorm).parameters),list(inspect.signature(torch.nn.GroupNorm).parameters))
        for affine,bias,dtype in ((True,True,torch.float32),(True,False,torch.float32),(False,True,torch.float32),(False,True,torch.float64)):
            model=norm.GroupNorm(2,4,affine=affine,bias=bias,dtype=str(dtype).removeprefix('torch.'))
            reference=torch.nn.GroupNorm(2,4,affine=affine,bias=bias,dtype=dtype)
            self.assertEqual(model.affine,reference.affine)
            for training in (False,True):
                model.train(training);reference.train(training)
                value=torch.randn(2,4,2,3,4,dtype=dtype,requires_grad=True);x=native(value)
                actual,expected=model(x),reference(value);close(actual,expected)
                probe=torch.randn_like(value);(actual*native(probe)).sum().backward();(expected*probe).sum().backward();close(x.grad,value.grad)
                for name,p in model.named_parameters():close(p.grad,dict(reference.named_parameters())[name].grad);p.tensor().zero_grad()
                reference.zero_grad()
            held={name:p.tensor()._id() for name,p in model.named_parameters()}
            with fr.no_grad(),torch.no_grad():
                for name,p in model.named_parameters():p.tensor().copy_(fr.full(p.tensor().shape,.4));dict(reference.named_parameters())[name].fill_(.4)
            model.reset_parameters();reference.reset_parameters()
            self.assertEqual(held,{name:p.tensor()._id() for name,p in model.named_parameters()})
            for name,p in model.named_parameters():close(p.tensor(),dict(reference.named_parameters())[name])
        self.assertEqual(norm.GroupNorm(3,0).weight.tensor().shape,[0])

    def test_epsilon_nonfinite_and_zero_variance_native_boundaries(self):
        for dtype in (torch.float32,torch.float64):
            for eps in (0.,.001,-.01,float('inf'),float('nan')):
                self.compare((2,4,3),2,dtype,True,True,eps=eps)
            for shape in ((2,4),(1,4,3)):
                x=torch.ones(shape,dtype=dtype,requires_grad=True);nx=native(x)
                actual=F.group_norm(nx,2);expected=torch.nn.functional.group_norm(x,2)
                close(actual,expected);actual.sum().backward();expected.sum().backward();close(nx.grad,x.grad)

    def test_early_capacity_shape_dtype_capture_and_higher_order_validation(self):
        for call in (lambda:norm.GroupNorm(0,4),lambda:norm.GroupNorm(True,4),lambda:norm.GroupNorm(3,4),lambda:norm.GroupNorm(1,-1),
                     lambda:norm.GroupNorm(1,sys.maxsize),lambda:norm.GroupNorm(1,4,affine=1)):
            with self.assertRaises((ValueError,TypeError)):call()
        for call in (lambda:norm.GroupNorm(1,4,dtype='int64'),lambda:norm.GroupNorm(1,4,device='cuda')):
            with self.assertRaises(NotImplementedError):call()
        for x,groups in ((fr.ones([4]),1),(fr.ones([2,4]),3)):
            with self.assertRaises(ValueError):F.group_norm(x,groups)
        with self.assertRaises(NotImplementedError):F.group_norm(fr.ones([2,4],dtype='int64'),2)
        with self.assertRaises(ValueError):F.group_norm(fr.ones([2,4]),2,fr.ones([4],dtype='float64'))
        virtual=fr.ones([1,1,1]).expand(2,0,sys.maxsize//4)
        with self.assertRaises(ValueError):F.group_norm(virtual,sys.maxsize//4)
        with self.assertRaises(NotImplementedError):fr.capture(lambda:F.group_norm(fr.ones([2,4]),2))
        for dtype in ('float32','float64'):
            x=fr.ones([2,4],dtype=dtype).requires_grad_(True);y=F.group_norm(x,2)
            with self.assertRaises((ValueError,RuntimeError)):fr.autograd.grad(y.sum(),(x,),create_graph=True)

    def test_bias_only_gradient_extension_matches_explicit_torch_affine_algebra(self):
        x=torch.randn(2,4,3,requires_grad=True);bias=torch.randn(4,requires_grad=True)
        direct=torch.nn.functional.group_norm(x,2,None,bias)
        with self.assertRaisesRegex(RuntimeError,'does not have a device'):direct.sum().backward()
        for dtype in (torch.float32,torch.float64):self.compare((2,4,3),2,dtype,False,True)

    def test_checkpoint_same_shape_config_and_exact_suffix(self):
        model=norm.GroupNorm(2,4);optimizer=fr.optim.AdamW(model.parameters(),lr=.003);generator=fr.Generator(6)
        value=fr.Tensor.randn([3,4,2],generator=generator);model(value).square().sum().backward();optimizer.step();optimizer.zero_grad()
        saved=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)
        restored=norm.GroupNorm(2,4);other=fr.optim.AdamW(restored.parameters(),lr=.003);rng=fr.Generator(77)
        saved.restore(restored,optimizers={'main':other},generator=rng)
        for _ in range(3):
            a=model(fr.Tensor.randn([2,4,3],generator=generator));b=restored(fr.Tensor.randn([2,4,3],generator=rng));self.assertEqual(a.tolist(),b.tolist())
            a.square().sum().backward();b.square().sum().backward();optimizer.step();other.step();optimizer.zero_grad();other.zero_grad()
        def state(m,o,g):return {k:[v.dtype,v.shape,v.tolist()] for k,v in fr.checkpoint.snapshot_training(m,optimizers={'main':o},generator=g).tensors().items()}
        self.assertEqual(state(model,optimizer,generator),state(restored,other,rng))
        wrong=norm.GroupNorm(1,4);before=[p.tensor().tolist() for p in wrong.parameters()]
        with self.assertRaises(ValueError):saved.restore(wrong,optimizers={'main':fr.optim.AdamW(wrong.parameters(),lr=.003)},generator=fr.Generator(7))
        self.assertEqual(before,[p.tensor().tolist() for p in wrong.parameters()])


if __name__=='__main__':unittest.main(verbosity=2)
