"""CPU pointwise/copy layouts and real derivative ownership against Torch."""
import unittest
import torch
import ferro as fr


def native(value,live=False):
    return fr.tensor(value.detach().tolist(),dtype=str(value.dtype).split('.')[-1]).reshape(value.shape).clone().requires_grad_(live)


def observed(value):
    return torch.tensor(value.tolist(),dtype=getattr(torch,value.dtype)).reshape(value.shape)


class PointwiseLayoutTests(unittest.TestCase):
    def same(self,actual,expected):
        value=observed(actual)
        self.assertEqual(actual.shape,list(expected.shape));self.assertEqual(actual.stride(),tuple(expected.stride()))
        torch.testing.assert_close(value,expected.detach(),rtol=0,atol=0,equal_nan=True)
        mask=expected.detach()==0
        self.assertEqual(torch.signbit(value[mask]).tolist(),torch.signbit(expected.detach()[mask]).tolist())

    def test_binary_physical_order_dense_gapped_broadcast_and_mixed_promotion(self):
        for da,db in ((torch.float32,torch.float32),(torch.float64,torch.float64),(torch.float32,torch.float64)):
            a=torch.arange(1,25,dtype=da).reshape(2,3,4)/8
            b=torch.arange(1,25,dtype=db).reshape(2,3,4)/4
            aa,bb=native(a),native(b)
            layouts=(lambda x:x,lambda x:x.permute(2,0,1),lambda x:x[:,:,::2],lambda x:x[:1,:,:].expand(5,3,4))
            for layout in layouts:
                left,right=layout(aa),layout(bb);tl,tr=layout(a),layout(b)
                for name,fn in (('add',lambda x,y:x+y),('sub',lambda x,y:x-y),('mul',lambda x,y:x*y),('div',lambda x,y:x/y)):
                    with self.subTest(dtype=(da,db),layout=tl.stride(),op=name):self.same(fn(left,right),fn(tl,tr))
            for op in (lambda x,y:x*y,lambda x,y:x/y):self.same(op(aa.T,bb.T[:,0:1,:]),op(a.T,b.T[:,0:1,:]))

    def test_canonical_typed_copies_preserve_dense_layout_and_materialize_gaps(self):
        for before in (torch.float32,torch.float64,torch.int64,torch.bool):
            values=torch.tensor([[0,1,1],[1,0,1]],dtype=before)
            source=native(values,before.is_floating_point)
            for transform in (lambda x:x.T,lambda x:x[:,::2],lambda x:x[:1,:].expand(4,3)):
                a,t=transform(source),transform(values)
                for after in (torch.float32,torch.float64,torch.int64,torch.bool):
                    with self.subTest(before=before,after=after,stride=t.stride()):
                        result=a.to(dtype=str(after).split('.')[-1],copy=True)
                        expected=t.to(dtype=after,copy=True)
                        self.same(result,expected)
                        self.assertFalse(result._shares_storage(a))
                        self.assertEqual(result.requires_grad,before.is_floating_point and after.is_floating_point)
                self.assertIs(a.to(dtype=str(before).split('.')[-1]),a)
            tiny=torch.tensor([1e-300,-0.,1.0000000000000002,1e300],dtype=torch.float64).reshape(2,2).T
            self.same(native(tiny.T).T.to(dtype='float64',copy=True),tiny.to(copy=True))

    def test_pointwise_adjoint_layout_grad_slots_and_nonaliasing_are_native(self):
        for dtype in (torch.float32,torch.float64):
            values=torch.tensor([[.25,-.5,.75],[1.,-.25,.5]],dtype=dtype)
            base=native(values,True);tbase=values.clone().requires_grad_()
            x,tx=base.T,tbase.T
            v=fr.ones(2,3,dtype=str(dtype).split('.')[-1]).T
            tv=torch.ones(2,3,dtype=dtype).T
            y=x*x+x;ty=tx*tx+tx
            self.same(y,ty);self.assertFalse(y._shares_storage(x))
            gradient=fr.autograd.grad(y,base,grad_outputs=v,create_graph=dtype==torch.float32,retain_graph=True)[0]
            expected=torch.autograd.grad(ty,tbase,tv,create_graph=dtype==torch.float32,retain_graph=True)[0]
            self.same(gradient,expected);self.assertIsNone(base.grad)
            y.sum().backward();ty.sum().backward();self.same(base.grad,tbase.grad)
            old=observed(y).clone()
            with fr.no_grad(),torch.no_grad():base.fill_(9.);tbase.fill_(9.)
            torch.testing.assert_close(observed(y),old,rtol=0,atol=0)
            with self.assertRaises(ValueError):fr.autograd.grad(y.sum(),base)

    def test_jvp_and_structural_zero_products_keep_markers_and_layouts(self):
        values=torch.tensor([[.25,-.5,.75],[1.,-.25,.5]])
        x=native(values,True).T;tx=values.T.detach().requires_grad_()
        v=fr.ones(2,3).T;tv=torch.ones(2,3).T
        c=fr.tensor(2.,requires_grad=True);tc=torch.tensor(2.,requires_grad=True)
        y,dy=fr.func.jvp(lambda a:a*c,(x,),(v,));ty,tdy=torch.func.jvp(lambda a:a*tc,(tx,),(tv,))
        self.same(y,ty);self.same(dy,tdy)
        actual=fr.autograd.grad(dy.sum(),(x,c),create_graph=True)
        wanted=torch.autograd.grad(tdy.sum(),(tx,tc),create_graph=True)
        self.assertEqual(actual[0]._is_structural_zero(),torch._is_zerotensor(wanted[0]))
        self.same(actual[0].clone(),wanted[0].clone());self.same(actual[1],wanted[1])
        self.assertTrue(actual[0]._is_structural_zero());self.assertFalse(actual[0].clone()._is_structural_zero())
        for op in (lambda a,z:a+z,lambda a,z:z+a,lambda a,z:a-z,lambda a,z:z-a):
            self.same(op(x,actual[0]),op(tx,wanted[0]))
        with self.assertRaises(ValueError):actual[0].detach().zero_()

    def test_empty_copies_and_pointwise_outputs_keep_offsets_safe(self):
        for dtype in ('float32','float64'):
            a=fr.zeros(0,8,dtype=dtype,requires_grad=True);t=torch.zeros(0,8,dtype=getattr(torch,dtype),requires_grad=True)
            left=a.narrow(1,4,4).T;right=t.narrow(1,4,4).T
            self.assertEqual(left.storage_offset(),right.storage_offset())
            self.same(left+left,right+right)
            self.same(left.to(dtype=dtype,copy=True),right.to(copy=True))
            (left+left).sum().backward();(right+right).sum().backward();self.same(a.grad,t.grad)
            scalar=fr.ones(1,1,dtype=dtype);tscalar=torch.ones(1,1,dtype=getattr(torch,dtype))
            expanded=scalar.expand(1<<45,1);texpanded=tscalar.expand(1<<45,1)
            empty=fr.zeros(1,0,dtype=dtype);tempty=torch.zeros(1,0,dtype=getattr(torch,dtype))
            actual,expected=expanded*empty,texpanded*tempty
            # The tensor is empty; nested tolist() would still build 2^45 lists.
            self.assertEqual(actual.shape,list(expected.shape))
            self.assertEqual(actual.stride(),tuple(expected.stride()))
            self.assertEqual(actual.numel(),expected.numel())
            self.assertEqual(actual.numel(),0)


if __name__=='__main__':unittest.main()
