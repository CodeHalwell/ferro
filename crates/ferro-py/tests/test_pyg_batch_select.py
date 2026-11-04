"""Actual PyG positive graph selectors plus explicit negative/mask repairs."""
import sys
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch as TorchBatch, Data as TorchData
import ferro as fr
from ferro.graph import Data, Batch, HeteroData, HeteroBatch
from test_pyg_batch_options import fixtures
from test_pyg_batch_hooks import pairs, PairData
from test_pyg_hetero_batch import fixtures as hetero_fixtures


class BatchSelectionTests(unittest.TestCase):
    def close(self, value, expected):
        actual = torch.tensor(value.tolist(), dtype=getattr(torch, value.dtype)).reshape(value.shape)
        torch.testing.assert_close(actual, expected.detach(), atol=3e-6, rtol=3e-5)

    def fields(self, actual, expected):
        for key in actual.keys():
            wanted = expected[key]
            if not actual[key].shape: wanted = wanted.reshape([])
            self.close(actual[key], wanted)

    def hetero(self, actual, expected):
        self.assertEqual(actual.metadata(), expected.metadata())
        for key in actual._global: self.close(actual._global[key], expected[key].reshape(actual._global[key].shape))
        for key in (*actual.node_types, *actual.edge_types):
            for field, value in actual[key].items():
                if field == 'num_nodes': self.assertEqual(value, expected[key].num_nodes)
                else: self.close(value, expected[key][field].reshape(value.shape))

    def test_positive_scalar_slice_int_tensor_and_repeated_selection(self):
        actual, reference = fixtures(True)
        batch, oracle = Batch.from_data_list(actual, follow_batch=['x']), TorchBatch.from_data_list(reference, follow_batch=['x'])
        self.assertIs(batch['x'], batch.x)
        for index in range(4):
            self.fields(batch[index], oracle[index])
            self.fields(batch.get_example(index), oracle.get_example(index))
            self.fields(batch[fr.tensor(index, dtype='int64')], oracle[torch.tensor(index)])
        for left, right in ((slice(None), slice(None)), (slice(None, None, -1), slice(None,None,-1)),
                            (slice(1, 4, 2), slice(1,4,2)), ([3,0,3], [3,0,3]), ((2,0), (2,0)), ([], []),
                            (fr.tensor([[3,0],[2,3]],dtype='int64'),torch.tensor([[3,0],[2,3]])),
                            (fr.tensor([[True,False],[False,True]],dtype='bool'),torch.tensor([[True,False],[False,True]]))):
            selected, wanted = batch[left], oracle[right]
            self.assertEqual(len(selected),len(wanted))
            for a,b in zip(selected,wanted):self.fields(a,b)
        selected, wanted = batch[[3,0,3]], oracle[[3,0,3]]
        loss=sum((item.x.square().sum()+item.edge_attr.sum()+item.y.square().sum() for item in selected),fr.tensor(0.))
        expected=sum(item.x.square().sum()+item.edge_attr.sum()+item.y.square().sum() for item in wanted)
        loss.backward();expected.backward()
        for a,b in zip(actual,reference):
            for field in ('x','edge_attr','y'):
                if b[field].grad is None:self.assertIsNone(a[field].grad)
                else:self.close(a[field].grad,b[field].grad)

    def test_negative_python_normalization_and_strict_bool_mask_extensions(self):
        actual, reference = fixtures()
        batch, oracle = Batch.from_data_list(actual), TorchBatch.from_data_list(reference)
        with self.assertRaises(RuntimeError):oracle[-1]
        self.assertNotEqual(oracle[-2].x.shape[0],reference[-2].x.shape[0])
        for i in range(-4,0):self.fields(batch[i],oracle[i+4])
        for a,b in zip(batch.index_select([-1,-4,-1]),oracle.index_select([3,0,3])):self.fields(a,b)
        for a,b in zip(batch[fr.tensor([-2,-1],dtype='int64')],oracle[torch.tensor([2,3])]):self.fields(a,b)
        for a,b in zip(batch[[True,False,False,True]],oracle[torch.tensor([True,False,False,True])]):self.fields(a,b)
        for mask in ([True],fr.tensor([True],dtype='bool'),fr.tensor([False]*5,dtype='bool')):
            with self.assertRaisesRegex(IndexError,'num_graphs'):batch[mask]
        self.assertEqual(len(oracle[torch.tensor([True])]),1)  # Strict length is our boundary.

    def test_custom_hooks_current_value_offsets_subclasses_and_duplicate_gradients(self):
        for dtype in ('float32','float64'):
            actual,reference=pairs(dtype)
            batch,oracle=Batch.from_data_list(actual),TorchBatch.from_data_list(reference)
            batch.x_s=batch.x_s*1.7;oracle.x_s=oracle.x_s*1.7
            batch.edge_index_s=batch.edge_index_s+fr.tensor(1,dtype='int64')
            oracle.edge_index_s=oracle.edge_index_s+1
            selected,wanted=batch[[2,0,2]],oracle[[2,0,2]]
            for a,b in zip(selected,wanted):
                self.assertIsInstance(a,PairData);self.fields(a,b)
            selected[0]._field_roles['new']='graph'
            self.assertNotIn('new',batch[2]._field_roles)
            loss=sum((item.x_s.square().sum()+item.x_t.sum() for item in selected),fr.tensor(0.,dtype=dtype))
            expected=sum(item.x_s.square().sum()+item.x_t.sum() for item in wanted)
            loss.backward();expected.backward()
            for a,b in zip(actual,reference):
                for key in ('x_s','x_t'):
                    if b[key].grad is None:self.assertIsNone(a[key].grad)
                    else:self.close(a[key].grad,b[key].grad)
        calls, wanted_calls = [], []
        class Getter(Data):
            def __cat_dim__(self,key,value,*args): return super().__cat_dim__(key,value,*args)
            def __getitem__(self,key):
                calls.append(key)
                return super().__getitem__(key)
        class TorchGetter(TorchData):
            def __getitem__(self,key):
                wanted_calls.append(key)
                return super().__getitem__(key)
        actual=Batch.from_data_list([Getter(value=fr.tensor([1.]))])
        expected=TorchBatch.from_data_list([TorchGetter(value=torch.tensor([1.]))])
        calls.clear();wanted_calls.clear()
        self.close(actual['value'],expected['value'])
        self.assertEqual(calls,wanted_calls)

    def test_scalar_zero_offset_keeps_view_while_tensor_offset_owns_inverse(self):
        class Increment(Data):
            tensor_offset = False
            def __inc__(self,key,value,*args):
                return fr.tensor([1],dtype='int64') if self.tensor_offset else 1
        class TorchIncrement(TorchData):
            tensor_offset = False
            def __inc__(self,key,value,*args):
                return torch.tensor([1]) if self.tensor_offset else 1
        for shaped in (False,True):
            Increment.tensor_offset=TorchIncrement.tensor_offset=shaped
            actual=[Increment(value=fr.tensor([i+.2],requires_grad=True)) for i in range(2)]
            reference=[TorchIncrement(value=torch.tensor([i+.2],requires_grad=True)) for i in range(2)]
            batch,oracle=Batch.from_data_list(actual),TorchBatch.from_data_list(reference)
            first,second=batch.to_data_list();want_first,want_second=oracle.to_data_list()
            loss=first.value.square().sum()+second.value.square().sum()
            expected=want_first.value.square().sum()+want_second.value.square().sum()
            loss.backward();expected.backward()
            for a,b in zip(actual,reference):self.close(a.value.grad,b.value.grad)
            # Mutate the whole batch storage: Ferro intentionally rejects
            # writes through partial views, but a live view observes its owner.
            with fr.no_grad():batch.value.detach().fill_(9.)
            with torch.no_grad():oracle.value.detach().fill_(9.)
            self.close(batch.value,oracle.value)
            self.close(first.value,want_first.value)
            self.assertEqual(first.value.tolist()[0],9. if not shaped else actual[0].value.tolist()[0])
            with fr.no_grad():second.value.detach().fill_(8.)
            with torch.no_grad():want_second.value.detach().fill_(8.)
            self.close(batch.value,oracle.value)
            self.assertNotEqual(batch.value.tolist()[1],8.)

    def test_heterogeneous_relation_access_offsets_current_values_and_gradients(self):
        actual,reference=hetero_fixtures(True)
        batch,oracle=HeteroBatch.from_data_list(actual),TorchBatch.from_data_list(reference)
        self.assertIs(batch['a'],batch._nodes['a'])
        relation=batch.edge_types[0]
        self.assertIs(batch[relation],batch._edges[relation])
        batch['a'].x=batch['a'].x*1.3;oracle['a'].x=oracle['a'].x*1.3
        selected,wanted=batch[[2,0,2]],oracle[[2,0,2]]
        for a,b in zip(selected,wanted):self.hetero(a,b)
        for a,b in zip(batch[::-1],oracle[::-1]):self.hetero(a,b)
        self.hetero(batch[-1],oracle[batch.num_graphs-1])
        loss=sum((item['a'].x.square().sum() for item in selected),fr.tensor(0.))
        expected=sum(item['a'].x.square().sum() for item in wanted)
        loss.backward();expected.backward()
        for a,b in zip(actual,reference):
            if b['a'].x.grad is None:self.assertIsNone(a['a'].x.grad)
            else:self.close(a['a'].x.grad,b['a'].x.grad)

    def test_invalid_and_empty_selectors_preflight_before_field_reads(self):
        actual,_=fixtures()
        batch=Batch.from_data_list(actual)
        import ferro._graph_batch_select as implementation
        with patch.object(implementation,'_select',side_effect=AssertionError('selection before validation')):
            for index in (4,-5,True,fr.tensor(1.),fr.tensor(True,dtype='bool')):
                with self.assertRaises((IndexError,ValueError)):batch[index]
            for index in ([0,4],[0,-5],[0,True],fr.tensor([1.]),slice(None,None,0),object()):
                with self.assertRaises((IndexError,ValueError)):batch.index_select(index)
            huge=fr.tensor(0,dtype='int64').expand([sys.maxsize//8+1])
            with self.assertRaisesRegex(ValueError,'capacity'):batch.index_select(huge)
            for blank in (Batch(),HeteroBatch()):
                with self.assertRaisesRegex(ValueError,'from_data_list'):blank.get_example(0)
        blank=HeteroBatch()
        blank['new'].num_nodes=0
        self.assertEqual(blank.node_types,['new'])
        with patch.object(implementation,'_edge_parts',side_effect=AssertionError('read empty selection')):
            self.assertEqual(batch.index_select([]),[])
            self.assertEqual(batch[fr.tensor([False]*4,dtype='bool')],[])
            self.assertEqual(batch[3:1],[])
            self.assertEqual(batch.index_select(fr.tensor([],dtype='int64').reshape([1,0]).expand([10**12,0])),[])


if __name__=='__main__':unittest.main()
