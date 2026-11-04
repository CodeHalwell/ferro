"""Physical seed batches preserve real Jacobian layouts and storage aliases."""
import itertools
import unittest
import torch
import ferro as fr
from ferro import _native
from test_func_forward_ad import tensor, same
from test_func_forward_composition import gradients_match


class JacfwdLayoutTests(unittest.TestCase):
    def compare(self, actual, expected, gradients=()):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.stride(), tuple(expected.stride()))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.dtype, str(expected.dtype).split('.')[-1])
        same(actual, expected)
        if gradients:
            gradients_match(self, actual, expected, gradients, second=True)

    def test_views_materialized_outputs_and_empty_shapes(self):
        for shape, enabled, live in itertools.product(((2,3),(2,1),(0,3)),(False,True),(False,True)):
            tx=torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32).reshape(shape).mul(.1).requires_grad_(live)
            x=tensor(tx)
            functions=(lambda a:a,lambda a:a*a,lambda a:a.transpose(0,1),lambda a:a[:,::2],
                       lambda a:a[:,::2].square(),lambda a:a[:,:1].expand(a.shape[0],3),
                       lambda a:a[:,:1].expand(a.shape[0],3).square(),lambda a:a.sum(0),lambda a:a.sum(1))
            for index,function in enumerate(functions):
                with self.subTest(shape=shape,enabled=enabled,live=live,function=index):
                    with fr.enable_grad() if enabled else fr.no_grad(): actual=fr.func.jacfwd(function)(x)
                    with torch.enable_grad() if enabled else torch.no_grad(): expected=torch.func.jacfwd(function)(tx)
                    self.compare(actual,expected,[(x,tx)])
            self.assertIsNone(x.grad)

    def test_shared_basis_tree_blocks_and_duplicate_storage(self):
        for enabled in (False,True):
            tx=torch.tensor([[.2,-.4],[.3,.5]],requires_grad=True)
            tw=torch.tensor([[.6,-.3],[.4,.1]],requires_grad=True)
            x,w=map(tensor,(tx,tw));calls=[]
            def objective(a,b):
                calls.append(1)
                h=(a@b).tanh()
                return h,h.transpose(0,1),h,h.sum()
            with fr.enable_grad() if enabled else fr.no_grad(): actual=fr.func.jacfwd(objective,argnums=(0,1))(x,w)
            with torch.enable_grad() if enabled else torch.no_grad(): expected=torch.func.jacfwd(objective,argnums=(0,1))(tx,tw)
            self.assertEqual(len(calls),2)
            av=[value for row in actual for value in row];tv=[value for row in expected for value in row]
            for a,b in zip(av,tv):self.compare(a,b,[(x,tx),(w,tw)])
            for i,j in itertools.product(range(len(av)),repeat=2):
                self.assertEqual(av[i]._shares_storage(av[j]),torch._C._is_alias_of(tv[i],tv[j]))

    def test_array_index_axis_and_index_rank_layouts(self):
        for axis, rank, enabled in itertools.product((0,1),(1,2),(False,True)):
            tx=torch.tensor([[.2,.3,.4],[.5,.6,.7]],requires_grad=True);x=tensor(tx)
            data=[tx.shape[axis]-1,0,tx.shape[axis]-1]
            ti=torch.tensor(data).reshape((3,) if rank==1 else (3,1));i=fr.tensor(data,dtype='int64').reshape(list(ti.shape))
            def objective(a,index):
                key=[slice(None),slice(None)];key[axis]=index
                return a[tuple(key)].square()
            with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(lambda a:objective(a,i))(x)
            with torch.enable_grad() if enabled else torch.no_grad():expected=torch.func.jacfwd(lambda a:objective(a,ti))(tx)
            self.compare(actual,expected,[(x,tx)])

    def test_scalar_empty_mixed_output_and_multiple_argument_blocks(self):
        for shape in ((),(0,),(2,0)):
            tx=torch.empty(shape,requires_grad=True) if 0 in shape else torch.tensor(.3,requires_grad=True)
            x=tensor(tx);y=fr.tensor([.2,-.4],requires_grad=True);ty=torch.tensor([.2,-.4],requires_grad=True)
            def objective(library,a,b):return a.square(),b.square(),library.ones(2,dtype=library.float64),a.to(dtype=library.float64)
            actual=fr.func.jacfwd(lambda a,b:objective(fr,a,b),argnums=(0,1))(x,y)
            expected=torch.func.jacfwd(lambda a,b:objective(torch,a,b),argnums=(0,1))(tx,ty)
            for av,tv in zip(actual,expected):
                for a,b in zip(av,tv):self.compare(a,b,[(x,tx),(y,ty)])

    def test_private_batch_validation_and_live_wide_directions(self):
        x=fr.tensor([.2,-.4]);seed=fr.tensor([[1.,2.],[3.,4.]],dtype='float64',requires_grad=True)
        with self.assertRaises((ValueError,RuntimeError)):_native._ForwardBatched([x],[seed],3)
        with _native._ForwardBatched([x],[seed],2) as level:
            with self.assertRaises((ValueError,RuntimeError)):_native._forward_ad_tangent(x)
            result=x*x
            direction=level.tangents([result])[0]
        self.assertEqual(direction.dtype,'float64')
        tx=torch.tensor([.2,-.4]);ts=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64,requires_grad=True)
        expected=ts*(tx*2)
        same(direction,expected)
        with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):
            fr.autograd.grad(direction.sum(),seed,create_graph=True)
        actual_gradient,=fr.autograd.grad(direction.sum(),seed,create_graph=False)
        expected_gradient,=torch.autograd.grad(expected.sum(),ts,create_graph=False)
        same(actual_gradient,expected_gradient)
        self.assertIsNone(seed.grad)
        self.assertTrue(fr.is_grad_enabled())
        x=fr.tensor([-2.,0.,2.])
        seed=fr.tensor([[float('nan')]*3],requires_grad=True)
        with _native._ForwardBatched([x],[seed],1) as level:
            direction=level.tangents([x.clamp(-1.,1.)])[0]
        tx=torch.tensor([-2.,0.,2.]);ts=torch.tensor([[float('nan')]*3],requires_grad=True)
        expected=torch.func.jvp(lambda a:a.clamp(-1.,1.),(tx,),(ts[0],))[1]
        same(direction,expected.unsqueeze(0))
        gradients_match(self,direction,expected.unsqueeze(0),[(seed,ts)])
        a=fr.tensor(.3,requires_grad=True);ta=torch.tensor(.3,requires_grad=True)
        self.compare(fr.func.jacfwd(lambda a:a.softmax(0))(a),torch.func.jacfwd(lambda a:a.softmax(0))(ta),[(a,ta)])

    def test_same_op_opaque_fallback_retains_first_order_numerical_coverage(self):
        for shape in ((2,),(0,2)):
            tx=torch.full(shape,.2);x=tensor(tx)
            calls=[]
            def function(a):
                calls.append(1)
                return fr.nn.functional.gelu(a)
            actual=fr.func.jacfwd(function)(x)
            expected=torch.func.jacfwd(torch.nn.functional.gelu)(tx)
            same(actual,expected)
            self.assertEqual(actual.requires_grad,expected.requires_grad)
            self.assertEqual(calls,[1])
        x=fr.tensor([.2],requires_grad=True)
        with self.assertRaises((ValueError,RuntimeError,NotImplementedError)):
            fr.func.jacfwd(fr.nn.functional.gelu)(x)
        self.assertTrue(fr.is_grad_enabled())


if __name__=='__main__':unittest.main()
