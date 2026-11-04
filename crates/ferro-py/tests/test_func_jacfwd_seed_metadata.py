"""Physical basis buffers and logical feature metadata at Jacfwd entry."""
import itertools
import math
import unittest
import torch
import ferro as fr
from test_func_seed_layout import make


def leaves(value):
    if isinstance(value,(tuple,list)):return sum((leaves(x) for x in value),[])
    return [value]


class JacfwdSeedMetadataTests(unittest.TestCase):
    def same(self,actual,expected):
        aa,bb=leaves(actual),leaves(expected)
        self.assertEqual(len(aa),len(bb))
        for a,b in zip(aa,bb):
            self.assertEqual(a.shape,list(b.shape));self.assertEqual(a.stride(),tuple(b.stride()))
            self.assertEqual(a.storage_offset(),b.storage_offset())
            self.assertEqual(a.dtype,str(b.dtype).split('.')[-1]);self.assertEqual(a.requires_grad,b.requires_grad)
            torch.testing.assert_close(torch.tensor(a.reshape(-1).tolist(),dtype=b.dtype).reshape(b.shape),b.detach().clone(),rtol=2e-6,atol=1e-7,equal_nan=True)
        self.assertEqual([[a._shares_storage(b) for b in aa] for a in aa],[[torch._C._is_alias_of(a,b) for b in bb] for a in bb])

    def test_noncontiguous_offset_extra_storage_and_overlap_entry_rules(self):
        layouts=('contiguous','transpose','gap','expanded','row_expand','offset_contiguous','offset_transpose','extra_contiguous','extra_transpose')
        functions=(lambda a:a,lambda a:a.transpose(0,1),lambda a:a.square(),lambda a:a.sum())
        for kind,index,enabled,live in itertools.product(layouts,range(len(functions)),(False,True),(False,True)):
            with self.subTest(kind=kind,index=index,enabled=enabled,live=live):
                x=make(fr,kind,live=live);tx=make(torch,kind,live=live);calls=[]
                def objective(a):calls.append(1);return functions[index](a)
                with torch.set_grad_enabled(enabled):
                    try:expected=torch.func.jacfwd(functions[index])(tx)
                    except RuntimeError:
                        with (fr.enable_grad() if enabled else fr.no_grad()),self.assertRaises((ValueError,RuntimeError,NotImplementedError)):
                            fr.func.jacfwd(objective)(x)
                        self.assertEqual(calls,[]);continue
                with fr.enable_grad() if enabled else fr.no_grad():actual=fr.func.jacfwd(objective)(x)
                self.assertEqual(calls,[1]);self.same(actual,expected)
                if expected.requires_grad:
                    actual_grad=fr.autograd.grad(actual.sum(),x,create_graph=True)[0]
                    expected_grad=torch.autograd.grad(expected.sum(),tx,create_graph=True)[0]
                    self.assertEqual(actual_grad.requires_grad,expected_grad.requires_grad)
                    torch.testing.assert_close(torch.tensor(actual_grad.tolist()),expected_grad.detach().clone(),rtol=2e-6,atol=1e-7)
                self.assertIsNone(x.grad)

    def test_per_input_basis_pitch_and_full_cross_output_alias_matrix(self):
        configurations=(((2,),(3,)),((),(3,)),((0,),(3,)),((0,2),(0,1)))
        for shapes,enabled,live in itertools.product(configurations,(False,True),(False,True)):
            with self.subTest(shapes=shapes,enabled=enabled,live=live):
                expected_inputs=[torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape).requires_grad_(live) for shape in shapes]
                inputs=[fr.tensor(t.detach().tolist()).reshape(t.shape).requires_grad_(live) for t in expected_inputs]
                calls=[]
                def function(a,b):calls.append(1);return a,b,a.reshape(1,*a.shape)
                with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                    actual=fr.func.jacfwd(function,argnums=(0,1))(*inputs)
                    expected=torch.func.jacfwd(lambda a,b:(a,b,a.reshape(1,*a.shape)),argnums=(0,1))(*expected_inputs)
                self.same(actual,expected);self.assertEqual(calls,[1])
        x=make(fr,'offset_transpose');tx=make(torch,'offset_transpose')
        f=lambda a,b:(a,b,a+b)
        self.same(fr.func.jacfwd(f,argnums=(0,1))(x,x),torch.func.jacfwd(f,argnums=(0,1))(tx,tx))

    def test_empty_offsets_and_singleton_strides_keep_compatible_empty_basis(self):
        constructors=(lambda lib:lib.ones(8)[7:7].reshape(0,1),lambda lib:lib.zeros(0,8).narrow(1,4,4).transpose(0,1),lambda lib:lib.tensor([[.2],[.7]]).transpose(0,1),lambda lib:lib.tensor([.2,.7,.3])[1])
        for constructor,enabled in itertools.product(constructors,(False,True)):
            x=constructor(fr).requires_grad_(True);tx=constructor(torch).requires_grad_()
            with (fr.enable_grad() if enabled else fr.no_grad()),torch.set_grad_enabled(enabled):
                actual=fr.func.jacfwd(lambda a:(a,a.square()))(x)
                expected=torch.func.jacfwd(lambda a:(a,a.square()))(tx)
            self.same(actual,expected)

    def test_rejections_precede_function_rng_and_leave_a_fresh_level_usable(self):
        generator=fr.Generator(19);calls=[]
        def operation(a):calls.append(1);return a+fr.rand(1,generator=generator)
        for value in (fr.ones(1).expand(1<<31),fr.ones(1).expand(3,2)):
            with self.assertRaises((ValueError,OverflowError,RuntimeError,NotImplementedError)):
                fr.func.jacfwd(operation,randomness='same')(value)
            self.assertEqual(calls,[]);self.assertTrue(fr.is_grad_enabled());self.assertIsNone(value.grad)
        self.assertEqual(fr.rand(3,generator=generator).tolist(),fr.rand(3,generator=fr.Generator(19)).tolist())
        self.same(fr.func.jacfwd(lambda a:a.square())(fr.tensor([.2,.7])),torch.func.jacfwd(lambda a:a.square())(torch.tensor([.2,.7])))


if __name__=='__main__':unittest.main()
