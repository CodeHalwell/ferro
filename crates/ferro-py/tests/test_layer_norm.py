"""Actual Torch multi-axis normalization, factory/state and derivative contracts."""
import inspect
import sys
import unittest
from unittest.mock import patch
import torch
import ferro as fr
import ferro.nn as norm
from ferro.nn import functional as F


def native(value, requires_grad=None):
    make = fr.Tensor.from_f64 if value.dtype == torch.float64 else fr.Tensor
    return make(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(value.requires_grad if requires_grad is None else requires_grad)


def close(actual, expected):
    if actual.shape != list(expected.shape) or actual.dtype != str(expected.dtype).removeprefix('torch.'):
        raise AssertionError('shape/dtype mismatch')
    tolerance = 3e-11 if expected.dtype == torch.float64 else 5e-5
    torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape), expected.detach(), atol=tolerance, rtol=tolerance, equal_nan=True)


class LayerNormTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)

    def functional_case(self, name, shape, normalized, dtype, affine, bias=False, strided=False, eps=None):
        base = torch.randn(shape, dtype=dtype)
        x = base.transpose(0, -1).contiguous().transpose(0, -1).detach().requires_grad_() if strided else base.requires_grad_()
        nx = native(base.transpose(0,-1).contiguous()).transpose(0,len(shape)-1).requires_grad_(True) if strided else native(x)
        weight = torch.randn(normalized, dtype=dtype, requires_grad=True) if affine else None
        offset = torch.randn(normalized, dtype=dtype, requires_grad=True) if bias else None
        nw, nb = None if weight is None else native(weight), None if offset is None else native(offset)
        options = {'eps': eps} if eps is not None else {}
        if name == 'layer_norm': options['bias'] = nb
        actual = getattr(F,name)(nx, normalized, nw, **options)
        if name == 'layer_norm': options['bias'] = offset
        expected = getattr(torch.nn.functional,name)(x,normalized,weight,**options)
        close(actual,expected)
        seed = torch.randn(shape,dtype=dtype)
        (actual*native(seed)).sum().backward(); (expected*seed).sum().backward()
        for left,right in ((nx,x),(nw,weight),(nb,offset)):
            if left is not None: close(left.grad,right.grad)

    def test_functional_all_axes_dtypes_strides_and_affine_gradients(self):
        for name in ('layer_norm','rms_norm'):
            for dtype in (torch.float32,torch.float64):
                for shape,normalized in (((3,), (3,)),((2,3,4),(3,4)),((2,2,3,4),(2,3,4))):
                    for affine in (False,True):
                        with self.subTest(name=name,dtype=dtype,shape=shape,affine=affine):
                            self.functional_case(name,shape,normalized,dtype,affine,name=='layer_norm' and affine,True)

    def test_modules_shapes_factory_reset_and_public_signature(self):
        for name in ('LayerNorm','RMSNorm'):
            cls,reference=getattr(norm,name),getattr(torch.nn,name)
            self.assertEqual(list(inspect.signature(cls).parameters),list(inspect.signature(reference).parameters))
            for affine in (False,True):
                for dtype in (torch.float32,torch.float64):
                    if dtype==torch.float64 and affine: continue
                    options={'elementwise_affine':affine,'dtype':str(dtype).removeprefix('torch.')}
                    model=cls((3,4),**options);options['dtype']=dtype;oracle=reference((3,4),**options)
                    self.assertEqual(model.normalized_shape,oracle.normalized_shape)
                    self.assertEqual(model.elementwise_affine,oracle.elementwise_affine)
                    for training in (False,True):
                        model.train(training);oracle.train(training)
                        x=torch.randn(2,3,4,dtype=dtype,requires_grad=True);nx=native(x)
                        actual,expected=model(nx),oracle(x);close(actual,expected)
                        probe=torch.randn_like(x);(actual*native(probe)).sum().backward();(expected*probe).sum().backward();close(nx.grad,x.grad)
                        for key,value in model.named_parameters():close(value.grad,dict(oracle.named_parameters())[key].grad)
                        for parameter in model.parameters(): parameter.tensor().zero_grad()
                        oracle.zero_grad()
                    held={key:value.tensor()._id() for key,value in model.named_parameters()}
                    with fr.no_grad(),torch.no_grad():
                        for key,value in model.named_parameters():
                            value.tensor().copy_(fr.full(value.tensor().shape,.7));dict(oracle.named_parameters())[key].fill_(.7)
                    model.reset_parameters();oracle.reset_parameters()
                    self.assertEqual(held,{key:value.tensor()._id() for key,value in model.named_parameters()})
                    for key,value in model.named_parameters():close(value.tensor(),dict(oracle.named_parameters())[key])
            if name=='LayerNorm':
                model=cls((3,4),bias=False);self.assertIsNone(model.bias)
                close(model(fr.ones([2,3,4])),reference((3,4),bias=False)(torch.ones(2,3,4)))
        class Derived(norm.LayerNorm):
            def reset_parameters(self):
                super().reset_parameters()
                with fr.no_grad():self.weight.tensor().copy_(fr.full(self.normalized_shape,2.))
        self.assertEqual(Derived((3,4)).weight.tensor().tolist(),[[2.]*4]*3)

    def test_empty_outer_and_normalized_dimensions_preserve_gradients(self):
        for name in ('layer_norm','rms_norm'):
            for dtype in (torch.float32,torch.float64):
                for shape,normalized in (((0,3,4),(3,4)),((2,0,3),(0,3)),((2,0),(0,))):
                    for affine in (False,True):
                        self.functional_case(name,shape,normalized,dtype,affine,name=='layer_norm' and affine)
        huge=fr.ones([1,1]).expand(sys.maxsize,0).requires_grad_(True)
        for apply in (F.layer_norm,F.rms_norm):
            output=apply(huge,(0,));self.assertEqual(output.shape,[sys.maxsize,0]);output.sum().backward();self.assertEqual(huge.grad.shape,[sys.maxsize,0]);huge.zero_grad()

    def test_dtype_epsilon_and_nonfinite_values_follow_torch(self):
        for name in ('layer_norm','rms_norm'):
            for dtype in (torch.float32,torch.float64):
                for eps in (0.,1e-2,-.01,float('nan'),float('inf')):
                    self.functional_case(name,(2,3,4),(3,4),dtype,True,name=='layer_norm',eps=eps)
        for dtype in (torch.float32,torch.float64):
            value=torch.tensor([[0.,0.,0.],[1e-8,-1e-8,2e-8]],dtype=dtype,requires_grad=True)
            nx=native(value);actual=F.rms_norm(nx,(3,));expected=torch.nn.functional.rms_norm(value,(3,));close(actual,expected)
            actual.sum().backward();expected.sum().backward();close(nx.grad,value.grad)

    def test_validation_precedes_materialization_and_explicit_dtype_limits(self):
        for cls in (norm.LayerNorm,norm.RMSNorm):
            for shape in ((),(-1,),True,(2,False),(sys.maxsize,2)):
                with self.assertRaises((ValueError,TypeError)):cls(shape)
            with self.assertRaises(NotImplementedError):cls(3,dtype='int64')
            with self.assertRaises(NotImplementedError):cls(3,device='cuda')
            with self.assertRaises(TypeError):cls(3,elementwise_affine=1)
        for apply in (F.layer_norm,F.rms_norm):
            for value,shape in ((fr.ones([2,3]),(2,)),(fr.ones([]),(1,)),(fr.ones([2,3]),())):
                with self.assertRaises(ValueError):apply(value,shape)
            with self.assertRaises(NotImplementedError):apply(fr.ones([2,3],dtype='int64'),(3,))
            with self.assertRaises(ValueError):apply(fr.ones([2,3]),(3,),fr.ones(3,dtype='float64'))
            with self.assertRaises(ValueError):apply(fr.ones([2,3]),(3,),fr.ones([1,3]))
            with self.assertRaises(NotImplementedError):fr.capture(lambda:apply(fr.ones([2,3],dtype='float64'),(3,)))
            virtual=fr.ones([1]).expand(sys.maxsize//4+1)
            with self.assertRaises(ValueError):apply(virtual,virtual.shape)
            leaf=fr.ones([2,3],dtype='float64').requires_grad_(True)
            with self.assertRaises((ValueError,RuntimeError)):fr.autograd.grad(apply(leaf,(3,)).sum(),(leaf,),create_graph=True)

    def test_checkpoint_same_shape_configuration_atomicity_and_exact_state(self):
        for cls in (norm.LayerNorm,norm.RMSNorm):
            model=cls((2,3));optimizer=fr.optim.AdamW(model.parameters(),lr=.003);generator=fr.Generator(2027)
            x=fr.Tensor.randn([4,2,3],generator=generator);model(x).square().sum().backward();optimizer.step();optimizer.zero_grad()
            saved=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=1)
            other=cls((2,3));other_optimizer=fr.optim.AdamW(other.parameters(),lr=.003);other_generator=fr.Generator(2)
            saved.restore(other,optimizers={'main':other_optimizer},generator=other_generator)
            for left,right in zip(model.parameters(),other.parameters()):self.assertEqual(left.tensor().tolist(),right.tensor().tolist())
            x=fr.Tensor.randn([3,2,3],generator=generator);xx=fr.Tensor.randn([3,2,3],generator=other_generator)
            actual,expected=model(x),other(xx);self.assertEqual(actual.tolist(),expected.tolist());actual.square().sum().backward();expected.square().sum().backward()
            optimizer.step();other_optimizer.step();optimizer.zero_grad();other_optimizer.zero_grad()
            def state(m,o,g):return {k:[v.dtype,v.shape,v.tolist()] for k,v in fr.checkpoint.snapshot_training(m,optimizers={'main':o},generator=g,step=2).tensors().items()}
            self.assertEqual(state(model,optimizer,generator),state(other,other_optimizer,other_generator))
            bad=cls((2,3),eps=.01);before=[p.tensor().tolist() for p in bad.parameters()]
            with self.assertRaises(ValueError):saved.restore(bad,optimizers={'main':fr.optim.AdamW(bad.parameters(),lr=.003)},generator=fr.Generator(1))
            self.assertEqual(before,[p.tensor().tolist() for p in bad.parameters()])

    def test_multi_axis_layer_norm_preserves_native_higher_order_boundary(self):
        x=torch.randn(2,2,3,requires_grad=True);w=torch.randn(2,3,requires_grad=True);b=torch.randn(2,3,requires_grad=True)
        nx,nw,nb=native(x),native(w),native(b)
        actual=F.layer_norm(nx,(2,3),nw,nb).tanh().sum();expected=torch.nn.functional.layer_norm(x,(2,3),w,b).tanh().sum()
        ga=fr.autograd.grad(actual,(nx,nw,nb));ge=torch.autograd.grad(expected,(x,w,b))
        for left,right in zip(ga,ge):close(left,right)
        with self.assertRaisesRegex(ValueError,'no differentiable VJP'):
            fr.autograd.grad(actual,(nx,nw,nb),create_graph=True)
        self.assertTrue(all(value.grad is None for value in (nx,nw,nb)))

    def test_layer_norm_native_capture_replays_live_multiaxis_inputs_and_affine(self):
        model=norm.LayerNorm((2,3));value=fr.Tensor.randn([3,2,3])
        graph=fr.capture(lambda:model(value)).compile_fused()
        for index in range(2):
            with fr.no_grad():
                value.copy_(fr.Tensor.randn(value.shape))
                model.weight.tensor().copy_(fr.Tensor.randn([2,3]))
                model.bias.tensor().copy_(fr.Tensor.randn([2,3]))
            actual=graph.replay();expected=model(value)
            torch.testing.assert_close(torch.tensor(actual.tolist()),torch.tensor(expected.tolist()),rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main(verbosity=2)
