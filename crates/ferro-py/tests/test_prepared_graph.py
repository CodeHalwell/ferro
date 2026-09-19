"""Independent Torch values/VJPs for prepared graph algebra on CPU and CUDA."""
import math
import os
import unittest
import torch
import ferro as fr

class PreparedGraphCPU(unittest.TestCase):
    device = 'cpu'
    def tensor(self, values, shape):
        return fr.Tensor(values, shape).to(self.device).requires_grad_(True)
    def close(self, x, y):
        torch.testing.assert_close(torch.tensor(x.tolist()).reshape(y.shape), y.detach(), atol=1e-5, rtol=1e-4)
    def test_sparse_torch_values_and_both_gradients(self):
        for n, m, rows, cols, width in [(4,3,[2,0,2,1,0],[1,2,1,0,0],3), (3,2,[],[],2), (0,0,[],[],2), (3,2,[1,0,1],[0,1,0],0)]:
            a=fr.graph.COO(n,m,rows,cols).prepare(self.device)
            for sampled in (False, True):
                shapes=([n,width],[m,width]) if sampled else ([len(rows)],[m,width])
                vals=[[((i*13)%17-8)/10 for i in range(math.prod(shape))] for shape in shapes]
                x,y=[self.tensor(v,s) for v,s in zip(vals,shapes)]
                tx,ty=[torch.tensor(v).reshape(s).requires_grad_(True) for v,s in zip(vals,shapes)]
                if sampled:
                    actual=a.sddmm(x,y)
                    expected=(tx[rows]*ty[cols]).sum(1)
                else:
                    actual=a.spmm(x,y)
                    dense=torch.zeros(n,m).index_put((torch.tensor(rows,dtype=torch.long),torch.tensor(cols,dtype=torch.long)),tx,accumulate=True)
                    expected=dense @ ty
                g=torch.arange(expected.numel(),dtype=torch.float32).reshape(expected.shape)/7+0.1
                actual.backward_with(fr.Tensor(g.flatten().tolist(),list(g.shape)).to(self.device))
                expected.backward(g)
                self.close(actual,expected); self.close(x.grad,tx.grad); self.close(y.grad,ty.grad)
                self.assertEqual(x.grad.device,self.device); self.assertEqual(y.grad.device,self.device)
    def test_arbitrary_axis_select_and_scatter_torch(self):
        ids=[2,0,2]; p=fr.graph.PreparedSegments(ids,4,self.device)
        ids[:]=[1,1,1]
        for dim in range(3):
            shape=[2,3,2]; shape[dim]=4
            vals=[i/10 for i in range(math.prod(shape))]
            for scatter in (False,True):
                x=self.tensor(vals,shape); tx=torch.tensor(vals).reshape(shape).requires_grad_(True)
                outshape=shape.copy(); outshape[dim]=3
                srcvals=[0.2+i/30 for i in range(math.prod(outshape))]
                src=self.tensor(srcvals,outshape); ts=torch.tensor(srcvals).reshape(outshape).requires_grad_(True)
                if scatter:
                    actual=p.scatter_add(x,src,dim); expected=tx.index_add(dim,torch.tensor([2,0,2]),ts)
                else:
                    actual=p.select(x,dim); expected=tx.index_select(dim,torch.tensor([2,0,2]))
                g=torch.arange(expected.numel(),dtype=torch.float32).reshape(expected.shape)/20+0.1
                actual.backward_with(fr.Tensor(g.flatten().tolist(),list(g.shape)).to(self.device)); expected.backward(g)
                self.close(actual,expected); self.close(x.grad,tx.grad)
                if scatter: self.close(src.grad,ts.grad)
    def test_batch_and_permuted_edge_order(self):
        a=fr.graph.COO(3,2,[1,0,1],[0,1,0]); b=fr.graph.COO(2,3,[0,0],[1,2])
        batch, offsets=fr.graph.COO.batch([a,b])
        self.assertEqual([list(x) for x in offsets],[[0,0],[3,2],[5,5]])
        x=self.tensor([1.,2.,3.,4.,5.],[5,1]); w=self.tensor([1.,2.,3.,4.,5.],[5])
        y=batch.prepare(self.device).spmm(w,x)
        self.close(y,torch.tensor([[4.],[4.],[0.],[41.],[0.]]))
        order=[4,2,0,3,1]
        perm=fr.graph.COO(5,5,[batch.rows[i] for i in order],[batch.cols[i] for i in order])
        self.close(perm.prepare(self.device).spmm(w.index_select(0,order).to(self.device),x),torch.tensor(y.tolist()))
    def test_invalid_inputs_and_capture(self):
        a=fr.graph.COO(2,2,[0],[1]).prepare(self.device)
        x=self.tensor([1.,2.],[2,1]); w=self.tensor([1.],[1])
        with self.assertRaises(ValueError): a.spmm(x,w)
        with self.assertRaises(ValueError): a.sddmm(x,w)
        with self.assertRaises(ValueError): a.spmm(fr.Tensor.from_i64([1],[1]).to(self.device),x)
        with self.assertRaisesRegex(ValueError,'capture/replay'): fr.capture(lambda:a.spmm(w,x))
        p=fr.graph.PreparedSegments([1],2,self.device)
        with self.assertRaises(ValueError): p.select(x,2)
        with self.assertRaises(ValueError): p.scatter_add(x,x)

class PreparedGraphCUDA(PreparedGraphCPU):
    device='cuda:0'
    @classmethod
    def setUpClass(cls):
        try: fr.cuda_init()
        except Exception as exc:
            if os.environ.get('FERRO_REQUIRE_CUDA'): raise
            raise unittest.SkipTest(str(exc))

if __name__ == '__main__': unittest.main()
