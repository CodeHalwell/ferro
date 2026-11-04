"""Bool graph masks retain node universes, alignment and attribute gradients."""
import unittest

import torch
from torch_geometric.data import Data as ReferenceData, Batch as ReferenceBatch
from torch_geometric.utils import subgraph as reference_subgraph
import ferro as fr


class GraphMasks(unittest.TestCase):
    def test_subgraph_boolean_node_universe_and_aligned_edge_mask(self):
        for edge in (torch.tensor([[0, 1, 2, 1, 0], [1, 2, 0, 2, 1]]),
                     torch.tensor([[0], [1]]), torch.empty(2, 0, dtype=torch.int64)):
            for mask in ([True, True, False, True], [False] * 4, [True] * 4):
                attributes = torch.arange(edge.shape[1], dtype=torch.float32).requires_grad_()
                actual_attr = fr.tensor(attributes.tolist(), requires_grad=True)
                actual = fr.graph.subgraph(fr.tensor(mask), fr.tensor(edge.tolist(), dtype='int64'), actual_attr,
                                           relabel_nodes=True, return_edge_mask=True)
                expected = reference_subgraph(torch.tensor(mask), edge, attributes, relabel_nodes=True, return_edge_mask=True)
                self.assertEqual(actual[0].tolist(), expected[0].tolist())
                self.assertEqual(actual[1].tolist(), expected[1].tolist())
                self.assertEqual(actual[2].dtype, 'bool')
                self.assertEqual(actual[2].tolist(), expected[2].tolist())
                actual[1].sum().backward()
                expected[1].sum().backward()
                self.assertEqual(actual_attr.grad.tolist(), attributes.grad.tolist())
        with self.assertRaises(ValueError):
            fr.graph.subgraph(fr.tensor([True, False]), fr.graph.EdgeIndex(4, [], []))
        with self.assertRaises(ValueError):
            fr.graph.subgraph(fr.tensor([[True, False]]), fr.tensor([[], []], dtype='int64'))

    def test_data_batch_masks_match_pyg_and_round_trip_as_bool(self):
        graphs, references = [], []
        for n in (3, 0, 2):
            mask = [i % 2 == 0 for i in range(n)]
            graphs.append(fr.graph.Data(x=fr.ones(n, 1), edge_index=fr.tensor([[], []], dtype='int64'),
                                        train_mask=fr.tensor(mask, dtype='bool'), y=fr.tensor([n])))
            references.append(ReferenceData(x=torch.ones(n, 1), edge_index=torch.empty(2, 0, dtype=torch.int64),
                                            train_mask=torch.tensor(mask, dtype=torch.bool), y=torch.tensor([n])))
        actual, expected = fr.graph.Batch.from_data_list(graphs), ReferenceBatch.from_data_list(references)
        self.assertEqual(actual.train_mask.dtype, 'bool')
        self.assertEqual(actual.train_mask.tolist(), expected.train_mask.tolist())
        self.assertEqual(actual.ptr.tolist(), expected.ptr.tolist())
        for left, right in zip(actual.to_data_list(), references):
            self.assertEqual(left.train_mask.dtype, 'bool')
            self.assertEqual(left.train_mask.tolist(), right.train_mask.tolist())

    def test_batch_offsets_standard_link_indices(self):
        graphs = [fr.graph.Data(x=fr.ones(3, 2), edge_index=fr.tensor([[0], [1]]),
                                edge_label_index=fr.tensor([[0, 2], [2, 1]]),
                                edge_label=fr.tensor([1, 0]),
                                field_roles={'edge_label_index': 'graph', 'edge_label': 'graph'}) for _ in range(2)]
        batch = fr.graph.Batch.from_data_list(graphs)
        self.assertEqual(batch.edge_label_index.tolist(), [[0, 2, 3, 5], [2, 1, 5, 4]])
        self.assertEqual(batch.to_data_list()[1].edge_label_index.tolist(), graphs[1].edge_label_index.tolist())


if __name__ == '__main__':
    unittest.main()
