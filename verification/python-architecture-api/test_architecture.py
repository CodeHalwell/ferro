import importlib
import importlib.util
import unittest
import ferro as fr


def t(v, shape):
    return fr.Tensor(v, shape).requires_grad_(True)


class Architecture(unittest.TestCase):
    def test_validation_and_state_boundaries(self):
        g,r=fr.graph,fr.recurrent
        bad=fr.Tensor.from_i64([1],[1])
        for op in [g.segment_sum,g.segment_mean,g.segment_max,g.segment_softmax]:
            with self.assertRaises(ValueError): op(bad,[0],1)
            with self.assertRaises((ValueError,OverflowError)): op(fr.Tensor.ones([1]),[-1],1)
        with self.assertRaises(ValueError): g.segment_max(fr.Tensor([float('nan')],[1]),[0],1)
        with self.assertRaises(ValueError): g.segment_softmax(fr.Tensor([float('inf')],[1]),[0],1)
        with self.assertRaises(ValueError): g.COO(1,1,[0],[0]).spmm(bad,fr.Tensor.ones([1,1]))
        with self.assertRaises(ValueError): fr.basis.BSplineBasis([0,0,1,1],1).evaluate(bad)
        with self.assertRaises(ValueError): fr.nn.functional.unfold2d(bad,1)
        with self.assertRaises(ValueError): r.rnn_cell(bad,fr.Tensor.zeros([1,1]),fr.Tensor.ones([1,1]),fr.Tensor.ones([1,1]))
        state=r.RecurrentState(t([.1,.8],[2,1]),t([.2,.9],[2,1]))
        w=fr.Tensor([.2,-.3,.4,.1],[4,1])
        x=fr.Tensor([.2,float('nan'),.3,float('inf')],[2,2,1])
        out=r.unroll('lstm',x,state,w,w,lengths=[2,0],reset=[False,False,True,True])
        expected=r.lstm_cell(fr.Tensor([.3],[1,1]),state.h.index_select(0,[0]),state.c.index_select(0,[0]),w,w)
        self.assertEqual(out.state.h.tolist()[0],expected[0].tolist()[0])
        self.assertAlmostEqual(out.state.h.tolist()[1][0],.8,places=6)
        self.assertAlmostEqual(out.state.c.tolist()[1][0],.9,places=6)
        self.assertFalse(out.state.detach().c.requires_grad)
        out.state.c.sum().backward(); self.assertIsNotNone(state.c.grad)
        empty=r.unroll('lstm',fr.Tensor.zeros([0,2,1]),state,w,w,lengths=[0,0])
        self.assertEqual(list(empty.outputs.shape),[0,2,1]); self.assertEqual(empty.state.c.tolist(),state.c.tolist())
        with self.assertRaises(ValueError): r.unroll('gru',x,state,w,w,lengths=[2,0])
        with self.assertRaises(ValueError): r.unroll('lstm',x,state,w,w,lengths=[2,0],reset=[])
        with self.assertRaises(ValueError): r.unroll('lstm',x,state,w,w,lengths=[3,0])

    def test_cpu_training_examples(self):
        from pathlib import Path
        path=Path(__file__).with_name('training_examples.py')
        self.assertTrue(path.exists(), 'training examples missing')
        spec=importlib.util.spec_from_file_location('architecture_examples',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        results=module.run_examples()
        self.assertEqual(set(results), {'graph','sequence','kan','convolution'})
        for name,(first,last) in results.items(): self.assertLess(last,first*.1,name)

    def test_modular_exports(self):
        self.assertTrue(hasattr(fr, 'graph'), 'graph must be available on import ferro')
        self.assertTrue(hasattr(fr, 'recurrent'), 'recurrent must be available on import ferro')
        self.assertTrue(hasattr(fr, 'basis'), 'basis must be available on import ferro')
        self.assertIs(fr.nn.KanLayer,fr.basis.KanLayer)
        self.assertIs(fr.nn.functional.rnn_cell,fr.recurrent.rnn_cell)
        self.assertIs(fr.nn.functional.gru_cell,fr.recurrent.gru_cell)
        self.assertIs(fr.nn.functional.lstm_cell,fr.recurrent.lstm_cell)
        self.assertIs(fr.nn.functional.kan,fr.basis.kan)

    def test_convolution_windows_torch_adjoints(self):
        from ferro.nn import functional as f
        self.assertTrue(hasattr(f,'conv2d'), 'conv2d options facade missing')
        import torch
        xv=[(i%13-6)/7 for i in range(2*4*5*6)]
        wv=[(i%11-5)/6 for i in range(6*2*2*3)]
        x=t(xv,[2,4,5,6]); w=t(wv,[6,2,2,3]); b=t([.1]*6,[6])
        tx=torch.tensor(x.tolist(),requires_grad=True); tw=torch.tensor(w.tolist(),requires_grad=True); tb=torch.tensor(b.tolist(),requires_grad=True)
        y=f.conv2d(x,w,b,stride=(2,1),padding=(1,2),dilation=(2,1),groups=2)
        ty=torch.nn.functional.conv2d(tx,tw,tb,stride=(2,1),padding=(1,2),dilation=(2,1),groups=2)
        self.assertTrue(torch.allclose(torch.tensor(y.tolist()),ty,atol=2e-6))
        (y*y).sum().backward(); (ty*ty).sum().backward()
        for a,c in [(x,tx),(w,tw),(b,tb)]: self.assertTrue(torch.allclose(torch.tensor(a.grad.tolist()),c.grad,atol=2e-5,rtol=2e-5))
        x=t(xv,[2,4,5,6]); tx=torch.tensor(x.tolist(),requires_grad=True)
        u=f.unfold2d(x,(2,3),stride=(2,1),padding=(1,2),dilation=(2,1))
        tu=torch.nn.functional.unfold(tx,(2,3),stride=(2,1),padding=(1,2),dilation=(2,1))
        self.assertEqual(u.tolist(),tu.tolist())
        folded=f.fold2d(u,(5,6),(2,3),stride=(2,1),padding=(1,2),dilation=(2,1))
        tf=torch.nn.functional.fold(tu,(5,6),(2,3),stride=(2,1),padding=(1,2),dilation=(2,1))
        self.assertEqual(folded.tolist(),tf.tolist())
        folded.sum().backward(); tf.sum().backward(); self.assertEqual(x.grad.tolist(),tx.grad.tolist())
        self.assertEqual(f.conv2d(fr.Tensor.ones([1,1,2,2]),fr.Tensor.ones([1,1,1,1])).tolist(),[[[[1,1],[1,1]]]])
        with self.assertRaises(ValueError): f.conv2d(x,w,stride=0)
        with self.assertRaises(ValueError): f.unfold2d(x,(2,))

    def test_basis_kan_and_parameter_identity(self):
        self.assertIsNotNone(importlib.util.find_spec('ferro.basis'), 'basis facade missing')
        from ferro.basis import BSplineBasis, kan, KanLayer
        basis=BSplineBasis([0,0,0,0,1,1,1,1],3)
        self.assertEqual(basis.num_basis,4); self.assertEqual(basis.degree,3)
        self.assertEqual(basis.domain,(0,1)); self.assertEqual(basis.knots,[0,0,0,0,1,1,1,1])
        x=t([.25],[1,1]); b=basis.evaluate(x)
        self.assertEqual(b.tolist(), [[[.421875,.421875,.140625,.015625]]])
        (b * fr.Tensor([1,2,3,4],[4])).sum().backward()
        self.assertAlmostEqual(x.grad.item(),3,places=6)
        x=t([.25],[1,1]); c=t([0,1,2,3],[1,1,4])
        y=kan(x,c,basis); self.assertAlmostEqual(y.item(),.75,places=6)
        y.sum().backward(); self.assertAlmostEqual(x.grad.item(),3,places=6)
        self.assertEqual(c.grad.tolist(),b.tolist())
        layer=KanLayer(basis,c.detach())
        param=layer.coefficients; self.assertEqual(list(layer.parameters()),[param])
        opt=fr.optim.SGD(layer.parameters(),lr=.1)
        before=param.tensor().tolist(); layer(x).sum().backward(); opt.step()
        self.assertIs(layer.coefficients,param); self.assertNotEqual(param.tensor().tolist(),before)
        self.assertEqual(c.tolist(),[[[0,1,2,3]]])
        self.assertEqual(basis.evaluate(fr.Tensor([-1,2],[2])).tolist(),[[0]*4,[0]*4])
        with self.assertRaises(ValueError): BSplineBasis([0,0,1,1],1,outside='bad')
        with self.assertRaises(ValueError): BSplineBasis([0,0,1,1],1,outside='error').evaluate(fr.Tensor([2],[1]))
        with self.assertRaises(ValueError): KanLayer(basis,fr.Tensor.zeros([1,1,3]))
        z=t([.4],[1,1])
        with self.assertRaises(ValueError): basis.evaluate(z).sum().grad_wrt([z],create_graph=True)

    def test_recurrent_cells_and_unroll(self):
        self.assertIsNotNone(importlib.util.find_spec('ferro.recurrent'), 'recurrent facade missing')
        from ferro import recurrent as r
        import torch
        for kind, gates in [('rnn',1),('gru',3),('lstm',4)]:
            xv=[.2,-.3]; hv=[.1,.4]; cv=[.3,-.2]
            wv=[.2,-.1,.3,.4]*gates; bv=[.1,-.2]*gates
            x=t(xv,[1,2]); h=t(hv,[1,2]); c=t(cv,[1,2])
            wi=t(wv,[gates*2,2]); wh=t(wv,[gates*2,2]); bi=t(bv,[gates*2]); bh=t(bv,[gates*2])
            cell=getattr(torch.nn, {'rnn':'RNNCell','gru':'GRUCell','lstm':'LSTMCell'}[kind])(2,2)
            with torch.no_grad():
                cell.weight_ih.copy_(torch.tensor(wv).reshape(gates*2,2)); cell.weight_hh.copy_(torch.tensor(wv).reshape(gates*2,2))
                cell.bias_ih.copy_(torch.tensor(bv)); cell.bias_hh.copy_(torch.tensor(bv))
            tx=torch.tensor([xv],requires_grad=True); th=torch.tensor([hv],requires_grad=True); tc=torch.tensor([cv],requires_grad=True)
            if kind=='lstm':
                y,cy=r.lstm_cell(x,h,c,wi,wh,bi,bh); ty,tcy=cell(tx,(th,tc)); loss=y.sum()+cy.sum(); tloss=ty.sum()+tcy.sum()
            else:
                y=getattr(r,kind+'_cell')(x,h,wi,wh,bi,bh); ty=cell(tx,th); loss=y.sum(); tloss=ty.sum()
            self.assertTrue(torch.allclose(torch.tensor(y.tolist()),ty,atol=1e-6))
            loss.backward(); tloss.backward()
            for a,b in [(x,tx),(h,th),(wi,cell.weight_ih),(wh,cell.weight_hh),(bi,cell.bias_ih),(bh,cell.bias_hh)]:
                self.assertTrue(torch.allclose(torch.tensor(a.grad.tolist()),b.grad,atol=1e-6),kind)
            initial=r.RecurrentState(fr.Tensor.zeros([1,2]), fr.Tensor.zeros([1,2]) if kind=='lstm' else None)
            out=r.unroll(kind,fr.Tensor([.2,-.3,.4,.1],[2,1,2]),initial,wi,wh,lengths=[1])
            self.assertEqual(out.outputs.tolist()[1], [[0,0]])
            self.assertEqual(out.state.h.tolist(),out.outputs.tolist()[0])
            self.assertFalse(out.state.detach().h.requires_grad)
        x=t([.3,.1,-.2,.4],[4,1,1]); wi=t([.7],[1,1]); wh=t([.6],[1,1]); h=t([.2],[1,1])
        out=r.unroll('rnn',x,r.RecurrentState(h),wi,wh,lengths=[4],truncate=2)
        out.state.h.sum().backward()
        self.assertEqual(x.grad.tolist()[:2], [[[0]],[[0]]]); self.assertIsNone(h.grad)
        with self.assertRaises(ValueError): r.unroll('bad',x,r.RecurrentState(h),wi,wh,lengths=[4])
        with self.assertRaises(ValueError): r.unroll('rnn',x,r.RecurrentState(h),wi,wh,lengths=[4],truncate=0)

    def test_sparse_topology_and_adjoints(self):
        from ferro import graph as g
        self.assertTrue(hasattr(g, 'COO'), 'native COO missing')
        a = g.COO(3, 2, [1,0,1], [0,1,0])
        w = t([2,3,-1], [3]); x = t([1,2,4,5], [2,2])
        y = a.spmm(w, x)
        self.assertEqual(y.tolist(), [[12,15],[1,2],[0,0]])
        (y * fr.Tensor([1,2,3,4,5,6], [3,2])).sum().backward()
        self.assertEqual(w.grad.tolist(), [11,14,11])
        self.assertEqual(x.grad.tolist(), [[3,4],[3,6]])
        csr, order = a.to_csr()
        self.assertEqual(order, [1,0,2])
        self.assertEqual(csr.row_offsets, [0,1,3,3])
        self.assertEqual(csr.spmm(w.index_select(0,order), x).tolist(), y.tolist())
        coalesced, values = a.coalesce(w)
        self.assertEqual(values.tolist(), [3,1])
        self.assertEqual(coalesced.spmm(values,x).tolist(), y.tolist())
        w.zero_grad(); (values * fr.Tensor([7,11],[2])).sum().backward()
        self.assertEqual(w.grad.tolist(),[11,7,11])
        left=t([1,2,3,4,5,6], [3,2]); right=t([2,3,4,5], [2,2])
        scores=a.sddmm(left,right)
        self.assertEqual(scores.tolist(), [18,14,18])
        scores.sum().backward()
        self.assertEqual(left.grad.tolist(), [[4,5],[4,6],[0,0]])
        self.assertEqual(right.grad.tolist(), [[6,8],[1,2]])
        self.assertEqual(csr.sddmm(left,right).tolist(), [14,18,18])
        self.assertEqual(csr.to_coo().rows, [0,1,1])
        self.assertEqual(a.shape, [3,2]); self.assertEqual(a.nnz,3)
        with self.assertRaises(ValueError): g.COO(1,1,[1],[0])
        with self.assertRaises(ValueError): g.CSR(1,1,[1,1],[0])
        with self.assertRaises(TypeError): g.COO(1,1,[0.5],[0])

    def test_segment_native_adjoints(self):
        self.assertIsNotNone(importlib.util.find_spec('ferro.graph'), 'graph facade missing')
        g = importlib.import_module('ferro.graph')
        for name, expected, grad in [
            ('sum', [3,4,6,2,0,0], [1]*6),
            ('mean', [3,4,3,1,0,0], [.5,.5,1,1,.5,.5]),
            ('max', [3,4,5,2,float('-inf'),float('-inf')], [0,1,1,1,1,0])]:
            x = t([1,2,3,4,5,0], [3,2])
            y = getattr(g, 'segment_'+name)(x, [1,0,1], 3)
            self.assertEqual(y.reshape([6]).tolist(), expected)
            y.sum().backward()
            self.assertEqual(x.grad.reshape([6]).tolist(), grad)
        x = t([1000,1001,-1000], [3])
        y = g.segment_softmax(x, [0,0,1], 3)
        self.assertAlmostEqual(y.tolist()[0], .26894142, places=6)
        (y * fr.Tensor([2,-1,7], [3])).sum().backward()
        self.assertAlmostEqual(x.grad.tolist()[0], .5898358, places=6)
        with self.assertRaises(ValueError):
            g.segment_sum(x, [0], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
