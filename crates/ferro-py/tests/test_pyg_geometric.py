"""Coordinate transforms against actual PyG, including live field gradients."""
import itertools
import math
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch_geometric.transforms as reference
from torch_geometric.data import Data as TData, HeteroData as THetero, Batch as TBatch
import ferro as fr
from ferro.graph import Data,HeteroData,Batch,HeteroBatch,EdgeIndex,InMemoryDataset
from ferro.transforms import Center,NormalizeScale,LinearTransformation,Cartesian,Distance,Polar,Spherical


def native(value, grad=False):
    return fr.tensor(value.detach().reshape(-1).tolist(),dtype=str(value.dtype).split('.')[-1],requires_grad=grad).reshape(list(value.shape))


def pair(width=3,dtype=torch.float64,attr=True):
    raw=torch.tensor([[.3,-.8,1.3,.6],[-.4,1.1,.2,-.7],[1.2,.4,-.5,.9]],dtype=dtype)[:width]
    p=raw.T.detach().requires_grad_();pos=native(raw,True).T
    edges=torch.tensor([[0,1,2,3,0],[1,2,3,0,2]],dtype=torch.int64)
    a,b=Data(pos=pos,edge_index=native(edges),num_nodes=4),TData(pos=p,edge_index=edges,num_nodes=4)
    if attr:
        q=torch.linspace(-.3,.6,5,dtype=dtype).requires_grad_();a.edge_attr=native(q,True);b.edge_attr=q
    a.x=fr.ones(4,1,dtype=str(dtype).split('.')[-1]);b.x=torch.ones(4,1,dtype=dtype)
    return a,b


class GeometricTests(unittest.TestCase):
    def close(self,a,b,atol=None):
        if a is None or b is None:
            self.assertIs(a,b);return
        actual=torch.tensor(a.tolist(),dtype=getattr(torch,a.dtype)).reshape(a.shape)
        self.assertEqual(actual.dtype,b.dtype)
        tolerance=(8e-6 if a.dtype=='float32' else 3e-12) if atol is None else atol
        torch.testing.assert_close(actual,b.detach(),rtol=tolerance,atol=tolerance,equal_nan=True)

    def compare_edge(self,kind,width,options,dtype=torch.float64,attr=True):
        a,b=pair(width,dtype,attr)
        left,right=globals()[kind](**options)(a),getattr(reference,kind)(**options)(b)
        self.close(left.edge_attr,right.edge_attr);self.assertIs(left.pos,a.pos);self.assertIs(left.x,a.x)
        self.assertIsNot(left._store,a._store)
        weight=torch.linspace(.3,1.1,right.edge_attr.numel(),dtype=right.edge_attr.dtype).reshape(right.edge_attr.shape)
        (left.edge_attr*native(weight)).sum().backward();(right.edge_attr*weight).sum().backward()
        self.close(a.pos.grad,b.pos.grad)
        if attr:self.close(a.edge_attr.grad,b.edge_attr.grad)

    def test_center_homogeneous_typed_stores_and_forward_copy_contract(self):
        for dtype in (torch.float32,torch.float64):
            a,b=pair(dtype=dtype);left,right=Center()(a),reference.Center()(b)
            self.close(left.pos,right.pos);self.assertIs(left.x,a.x);self.assertIsNot(left.pos,a.pos)
            left.pos.square().sum().backward();right.pos.square().sum().backward();self.close(a.pos.grad,b.pos.grad)
            a,b=HeteroData(),THetero()
            for key,n in [('a',3),('b',2),('empty',0)]:
                pos=torch.arange(n*2,dtype=dtype).reshape(n,2).requires_grad_()
                a[key].pos=native(pos,True);b[key].pos=pos
            a['none'].x=fr.ones(1,1);b['none'].x=torch.ones(1,1)
            left,right=Center()(a),reference.Center()(b)
            for key in ('a','b','empty'):
                self.close(left[key].pos,right[key].pos);self.assertIs(left[key]._parent,left)
            self.assertIs(left['none'].x,a['none'].x)
            loss=sum(left[key].pos.square().sum() for key in ('a','b','empty'));wanted=sum(right[key].pos.square().sum() for key in ('a','b','empty'))
            loss.backward();wanted.backward()
            for key in ('a','b','empty'):self.close(a[key].pos.grad,b[key].pos.grad)
        a,b=pair();old=a.pos;self.assertIs(Center().forward(a),a);self.assertIsNot(a.pos,old)
        empty=Data(x=fr.ones(2,1));out=Center()(empty);self.assertIsNot(out,empty);self.assertIs(out.x,empty.x)

    def test_normalize_scale_live_maximum_and_singular_values(self):
        for dtype in (torch.float32,torch.float64):
            for raw in (torch.tensor([[1.,2.],[3.,-2.],[-4.,.5]],dtype=dtype),torch.zeros(2,2,dtype=dtype)):
                p=raw.detach().requires_grad_();x=native(raw,True);a,b=Data(pos=x),TData(pos=p)
                out,wanted=NormalizeScale()(a),reference.NormalizeScale()(b);self.close(out.pos,wanted.pos)
                weight=torch.arange(1,7,dtype=dtype)[:raw.numel()].reshape(raw.shape)
                (out.pos*native(weight)).sum().backward();(wanted.pos*weight).sum().backward();self.close(x.grad,p.grad)
            for shape in ((0,2),(2,0)):
                with self.assertRaises(ValueError):NormalizeScale()(Data(pos=fr.zeros(*shape)))
                with self.assertRaises(RuntimeError):reference.NormalizeScale()(TData(pos=torch.zeros(shape)))
        a,b=pair();self.assertIsNot(NormalizeScale().forward(a),a)

    def test_linear_matrix_transpose_dtype_conversion_and_all_gradients(self):
        for pos_dtype,matrix_dtype in itertools.product((torch.float32,torch.float64),repeat=2):
            for width in (1,3):
                a,b=pair(width,pos_dtype);raw=torch.linspace(-.7,.9,width*width,dtype=matrix_dtype).reshape(width,width)
                matrix=raw.detach().requires_grad_();fm=native(raw,True)
                if width==1:a.pos=a.pos.reshape([4]);b.pos=b.pos.detach().reshape(4).requires_grad_()
                source=a.pos;out,wanted=LinearTransformation(fm)(a),reference.LinearTransformation(matrix)(b)
                self.close(out.pos,wanted.pos);out.pos.square().sum().backward();wanted.pos.square().sum().backward()
                self.close(source.grad,b.pos.grad);self.close(fm.grad,matrix.grad,atol=2e-5 if pos_dtype==torch.float32 else None)
        a,b=HeteroData(),THetero();matrix=torch.eye(2,dtype=torch.float64,requires_grad=True);fm=native(matrix,True)
        for key,n in [('a',3),('b',1),('empty',0)]:a[key].pos=fr.ones(n,2,dtype='float64',requires_grad=True);b[key].pos=torch.ones(n,2,dtype=torch.float64,requires_grad=True)
        out,wanted=LinearTransformation(fm)(a),reference.LinearTransformation(matrix)(b)
        for key in a.node_types:self.close(out[key].pos,wanted[key].pos)
        sum(out[key].pos.sum() for key in a.node_types).backward();sum(wanted[key].pos.sum() for key in a.node_types).backward();self.close(fm.grad,matrix.grad)

    def test_cartesian_distance_options_detached_normalization_and_gradients(self):
        for kind,width,dtype,norm,maximum,cat in itertools.product(('Cartesian','Distance'),(2,3),(torch.float32,torch.float64),(False,True),(None,2.3),(False,True)):
            self.compare_edge(kind,width,dict(norm=norm,max_value=maximum,cat=cat,interval=(-.4,1.7)),dtype)
        self.compare_edge('Cartesian',1,dict(norm=True),torch.float64)

    def test_polar_spherical_live_normalization_and_gradients(self):
        for kind,width in [('Polar',2),('Spherical',3)]:
            for dtype,norm,maximum,cat in itertools.product((torch.float32,torch.float64),(False,True),(None,2.3),(False,True)):
                self.compare_edge(kind,width,dict(norm=norm,max_value=maximum,cat=cat),dtype)

    def test_typed_edge_attributes_empties_zero_distance_and_native_topology(self):
        for kind,width in [('Cartesian',3),('Distance',3),('Polar',2),('Spherical',3)]:
            for dtype in (torch.int64,torch.bool,torch.float32,torch.float64):
                a,b=pair(width,torch.float64);q=torch.arange(5).to(dtype);a.edge_attr=native(q);b.edge_attr=q
                out,wanted=globals()[kind](max_value=3.)(a),getattr(reference,kind)(max_value=3.)(b);self.close(out.edge_attr,wanted.edge_attr)
            for norm,maximum in [(False,None),(True,2.)]:
                a,b=pair(width);a.edge_index=fr.tensor([[],[]],dtype='int64');b.edge_index=torch.empty(2,0,dtype=torch.int64);a.edge_attr=fr.zeros(0);b.edge_attr=torch.empty(0)
                out,wanted=globals()[kind](norm=norm,max_value=maximum)(a),getattr(reference,kind)(norm=norm,max_value=maximum)(b);self.close(out.edge_attr,wanted.edge_attr)
            for norm in (False,True):
                a,b=pair(width,attr=False);a.edge_index=EdgeIndex(4,[0,1],[0,1]);b.edge_index=torch.tensor([[0,1],[0,1]])
                out,wanted=globals()[kind](norm=norm)(a),getattr(reference,kind)(norm=norm)(b);self.close(out.edge_attr,wanted.edge_attr)
                out.edge_attr.sum().backward();wanted.edge_attr.sum().backward();self.close(a.pos.grad,b.pos.grad)
        for kind,width in [('Polar',2),('Spherical',3)]:
            a,b=pair(width,attr=False);a.edge_index=fr.tensor([[],[]],dtype='int64');b.edge_index=torch.empty(2,0,dtype=torch.int64)
            with self.assertRaises(ValueError):globals()[kind]()(a)
            with self.assertRaises(RuntimeError):getattr(reference,kind)()(b)

    def test_batch_global_centering_current_inverse_and_subclass_preservation(self):
        aa,bb=zip(*(pair(3) for _ in range(2)));a,b=Batch.from_data_list(aa),TBatch.from_data_list(bb)
        for left,right in [(Center(),reference.Center()),(Cartesian(),reference.Cartesian())]:
            out,wanted=left(a),right(b);self.assertIsInstance(out,Batch);self.assertIs(out._slices,a._slices)
            for x,y in zip(out.to_data_list(),wanted.to_data_list()):self.close(x.pos,y.pos);self.close(x.edge_attr,y.edge_attr)
            self.assertIs(out.batch,a.batch);self.assertIs(out.ptr,a.ptr)
        class Custom(Data):
            __slots__=('_slot',)
            def __init__(self,marker):super().__init__(pos=fr.ones(2,2));self._marker=marker;self._slot='kept'
        a=Custom(object());out=Center()(a);self.assertIs(type(out),Custom);self.assertIs(out._marker,a._marker);self.assertEqual(out._slot,'kept')
        h=HeteroData();h['a'].pos=fr.ones(2,2);h['b'].pos=fr.zeros(1,2)
        batch=HeteroBatch.from_data_list([h,h]);out=Center()(batch)
        self.assertIs(out['a']._parent,out);self.assertIsInstance(out,HeteroBatch)
        self.assertEqual([x['a'].pos.shape for x in out.to_data_list()],[[2,2],[2,2]])

    def test_preflight_shapes_capacity_capture_and_direct_forward_atomicity(self):
        with self.assertRaises(ValueError):LinearTransformation(fr.ones(2,3))
        with self.assertRaises(TypeError):LinearTransformation([[1.]])
        h=HeteroData();h['a'].pos=fr.ones(2,2);h['b'].pos=fr.ones(2,3);old=h['a'].pos
        with self.assertRaises(ValueError):LinearTransformation(fr.ones(2,2)).forward(h)
        self.assertIs(h['a'].pos,old)
        width=1<<30;huge_matrix=fr.ones(1,1).expand([width,width])
        typed=HeteroData();typed['a'].pos=fr.zeros(0,1).expand([0,width]);typed['b'].pos=fr.zeros(0,1,dtype='float64').expand([0,width]);original=typed['a'].pos
        with patch.object(fr.Tensor,'__matmul__',side_effect=AssertionError('write before cast capacity preflight')):
            with self.assertRaisesRegex(ValueError,'capacity'):LinearTransformation(huge_matrix).forward(typed)
        self.assertIs(typed['a'].pos,original)
        for kind,width in [('Cartesian',3),('Distance',3),('Polar',2),('Spherical',3)]:
            a,b=pair(width)
            with self.assertRaises(NotImplementedError):fr.capture(lambda:globals()[kind]()(a).edge_attr)
            a.edge_attr=fr.ones(2,1)
            with self.assertRaises(ValueError):globals()[kind]()(a)
        import ferro._graph_geometric as impl
        oversized=Data(pos=fr.ones(1,1).expand([2,sys.maxsize//8+1]))
        with patch.object(impl,'_center',side_effect=AssertionError('center before capacity')):
            with self.assertRaisesRegex(ValueError,'capacity'):Center()(oversized)
        with self.assertRaises(ValueError):Center()(Data(pos=fr.ones(2,dtype='int64')))
        wide=Data(pos=fr.ones(1,1).expand([1,sys.maxsize//8+1]),edge_index=fr.tensor([[0,0,0],[0,0,0]],dtype='int64'))
        with patch.object(fr.Tensor,'index_select',side_effect=AssertionError('gather before derived capacity')):
            with self.assertRaisesRegex(ValueError,'capacity'):Distance()(wide)

    def test_saved_dataset_and_registered_matrix_exact_training_continuation(self):
        class Project(fr.nn.Module):
            def build(self):self.matrix=fr.nn.Parameter(fr.tensor([[.7,.2],[-.3,.8]]))
            def forward(self,data):return LinearTransformation(self.matrix.tensor())(Center()(data)).pos
        def exact(value):
            if isinstance(value,list):return [exact(item) for item in value]
            if isinstance(value,float):
                self.assertTrue(math.isfinite(value));return value.hex()
            return value
        def step(model,opt,data):
            opt.zero_grad();out=model(data);loss=out.square().mean();loss.backward();opt.step();opt.zero_grad();return exact(out.tolist()),exact(loss.item())
        data,_=pair(2,torch.float32);model=Project();opt=fr.optim.AdamW(model.parameters(),lr=.01);rng=fr.Generator(17)
        step(model,opt,data)
        state=fr.checkpoint.snapshot_training(model,optimizers={'projection':opt},generator=rng,step=1)
        with tempfile.TemporaryDirectory() as directory:
            state.save(directory+'/state');InMemoryDataset.save([data],directory+'/data')
            loaded=InMemoryDataset();loaded.load(directory+'/data');resumed_data=loaded[0]
            saved=fr.checkpoint.load(directory+'/state')
            other=Project();other_opt=fr.optim.AdamW(other.parameters(),lr=.4);other_rng=fr.Generator(9)
            saved.restore(other,optimizers={'projection':other_opt},generator=other_rng)
            for _ in range(3):self.assertEqual(step(model,opt,data),step(other,other_opt,resumed_data))
        left=fr.checkpoint.snapshot_training(model,optimizers={'projection':opt},generator=rng,step=4).tensors()
        right=fr.checkpoint.snapshot_training(other,optimizers={'projection':other_opt},generator=other_rng,step=4).tensors()
        self.assertEqual(left.keys(),right.keys())
        for key in left:
            self.assertEqual(left[key].dtype,right[key].dtype,key);self.assertEqual(left[key].shape,right[key].shape,key)
            self.assertEqual(exact(left[key].tolist()),exact(right[key].tolist()),key)

    def test_huge_empty_center_is_bounded_and_keeps_zero_input_gradient(self):
        script='''
import ferro as f
from ferro.graph import Data
from ferro.transforms import Center
x=f.zeros(1,0,dtype='float64',requires_grad=True)
y=Center()(Data(pos=x.expand([10**12,0]))).pos
if y.shape != [10**12,0]:raise RuntimeError('shape changed')
y.sum().backward()
if x.grad.shape != [1,0]:raise RuntimeError('gradient shape changed')
print('bounded-empty-center')
'''
        result=subprocess.run([sys.executable,'-O','-B','-c',script],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('bounded-empty-center',result.stdout)


if __name__=='__main__':unittest.main()
