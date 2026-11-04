"""Public CPU edge attention against a dense, differentiable PyTorch oracle."""
import math
import sys
import unittest

import torch
import ferro as fr
from ferro.autograd import grad
from ferro.graph import EdgeIndex, edge_attention


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


def topology():
    return EdgeIndex(5, [0, 1, 1, 2, 3, 1, 0, 2], [1, 2, 2, 1, 3, 0, 2, 0])


def dense_reference(q, k, v, edges, bias=None):
    n = edges.num_nodes
    scores = (q @ k.T) / math.sqrt(q.shape[1])
    logits = scores[edges.dst, edges.src]
    if bias is not None:
        logits = logits + bias
    positions = torch.tensor([dst * n + src for src, dst in zip(edges.src, edges.dst)], dtype=torch.int64)
    weights = torch.zeros(n * n).scatter_add(0, positions, logits.exp()).reshape(n, n)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-30)
    return weights @ v


class GraphAttentionTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected, atol=2e-6, rtol=2e-5)

    def test_dense_values_and_all_operand_gradients(self):
        edges = topology()
        q = (torch.arange(15).reshape(5, 3) - 7) / 9
        k = (torch.arange(15).reshape(5, 3).flip(0) - 5) / 11
        v = (torch.arange(10).reshape(5, 2) - 4) / 7
        bias = (torch.arange(8) - 3) / 8
        probe = (torch.arange(10).reshape(5, 2).flip(1) - 5) / 9
        for with_bias in (False, True):
            with self.subTest(with_bias=with_bias):
                native = [tensor(x, True) for x in (q, k, v, bias)]
                reference = [x.clone().requires_grad_(True) for x in (q, k, v, bias)]
                result = edge_attention(*native[:3], edges, edge_bias=native[3] if with_bias else None)
                expected = dense_reference(*reference[:3], edges, reference[3] if with_bias else None)
                self.close(result, expected.detach())
                self.assertEqual(result.tolist()[4], [0., 0.])
                (result * tensor(probe)).sum().backward()
                (expected * probe).sum().backward()
                for actual, wanted in zip(native[:4 if with_bias else 3], reference):
                    self.close(actual.grad, wanted.grad)

    def test_noncontiguous_features_and_bias_preserve_gradients(self):
        edges = topology()
        raw = [torch.linspace(-0.7, 0.9, 15).reshape(3, 5), torch.linspace(0.8, -0.6, 15).reshape(3, 5),
               torch.linspace(-0.3, 0.7, 10).reshape(2, 5), torch.linspace(-0.8, 0.5, 16)]
        native = [tensor(x, True) for x in raw]
        reference = [x.clone().requires_grad_(True) for x in raw]
        result = edge_attention(*(x.transpose(0, 1) for x in native[:3]), edges, native[3][::2])
        expected = dense_reference(*(x.T for x in reference[:3]), edges, reference[3][::2])
        self.close(result, expected.detach())
        (result * result).sum().backward()
        (expected * expected).sum().backward()
        for actual, wanted in zip(native, reference):
            self.close(actual.grad, wanted.grad)

    def test_duplicate_entries_and_stable_per_destination_normalization(self):
        edges = EdgeIndex(3, [0, 0, 1], [2, 2, 2])
        q = tensor(torch.zeros(3, 1))
        v = tensor([[1.], [4.], [9.]], True)
        bias = tensor([0., math.log(2), math.log(3)], True)
        output = edge_attention(q, q, v, edges, bias)
        self.close(output, torch.tensor([[0.], [0.], [2.5]]))
        output.sum().backward()
        self.close(v.grad, torch.tensor([[.5], [.5], [0.]]))
        self.close(bias.grad, torch.tensor([-.25, -.5, .75]))
        shifted = edge_attention(q, q, v, edges, bias + 200)
        torch.testing.assert_close(values(shifted), values(output), atol=2e-5, rtol=2e-5)

    def test_permutations_and_disjoint_batch(self):
        edges = topology()
        q, k, v = [tensor((torch.arange(10).reshape(5, 2) - i) / 9) for i in (3, 5, 7)]
        bias = tensor(torch.arange(8) / 7)
        expected = edge_attention(q, k, v, edges, bias)
        order = [7, 2, 4, 0, 6, 3, 1, 5]
        permuted = EdgeIndex(5, [edges.src[e] for e in order], [edges.dst[e] for e in order])
        self.close(edge_attention(q, k, v, permuted, tensor(values(bias)[order])), values(expected))
        order = [2, 4, 0, 3, 1]
        inverse = {old: new for new, old in enumerate(order)}
        permuted = EdgeIndex(5, [inverse[src] for src in edges.src], [inverse[dst] for dst in edges.dst])
        self.close(edge_attention(*(tensor(values(x)[order]) for x in (q, k, v)), permuted, bias), values(expected)[order])
        union, _ = EdgeIndex.batch([EdgeIndex(0, [], []), edges, edges])
        joined = [tensor(torch.cat([values(x), values(x)])) for x in (q, k, v, bias)]
        self.close(edge_attention(*joined[:3], union, joined[3]), torch.cat([values(expected), values(expected)]))

    def test_empty_graphs_and_zero_width_values(self):
        for nodes, width, nonempty in ((3, 2, False), (0, 2, False), (3, 0, True)):
            with self.subTest(nodes=nodes, width=width, nonempty=nonempty):
                edges = EdgeIndex(nodes, [0, 1, 1], [1, 2, 2]) if nonempty else EdgeIndex(nodes, [], [])
                inputs = [tensor(torch.zeros(shape), True) for shape in ((nodes, 2), (nodes, 2), (nodes, width), (edges.num_edges,))]
                output = edge_attention(*inputs[:3], edges, inputs[3])
                self.close(output, torch.zeros(nodes, width))
                output.sum().backward()
                for x in inputs:
                    self.assertEqual(x.grad.shape, x.shape)
                    self.close(x.grad, torch.zeros(x.shape))

    def test_validation_capture_and_higher_order_boundaries(self):
        edges = EdgeIndex(2, [0, 1], [1, 1])
        q = tensor([[.2, -.3], [.4, .7]], True)
        v = tensor([[.5], [-.6]], True)
        bias = tensor([.1, -.2], True)
        for bad in (tensor(1.), tensor([1., 2.]), tensor([[1., 2.]]), tensor(torch.zeros(2, 0)), tensor(torch.zeros(2, 3))):
            for queries, keys in ((bad, q), (q, bad)):
                with self.assertRaises(ValueError):
                    edge_attention(queries, keys, v, edges)
        for bad in (tensor(0.), tensor(torch.zeros(2, 1)), tensor(torch.zeros(3))):
            with self.assertRaises(ValueError):
                edge_attention(q, q, v, edges, bad)
        for slot in range(4):
            args = [q, q, v, bias]
            args[slot] = fr.Tensor.from_i64([0] * math.prod(args[slot].shape), args[slot].shape)
            with self.assertRaises(ValueError):
                edge_attention(*args[:3], edges, args[3])
        huge = fr.Tensor([], [0, sys.maxsize // 4 + 1])
        with self.assertRaises(ValueError):
            edge_attention(huge, huge, fr.Tensor([], [0, 0]), EdgeIndex(0, [], []))
        for value in (float('inf'), float('nan')):
            with self.assertRaisesRegex(ValueError, 'finite'):
                edge_attention(q, q, v, edges, tensor([value, 0.]))
        for context in (fr.no_grad, fr.enable_grad):
            with context():
                mode = fr.is_grad_enabled()
                with self.assertRaisesRegex(ValueError, 'capture'):
                    fr.capture(lambda: edge_attention(q, q, v, edges, bias))
                self.assertEqual(fr.is_grad_enabled(), mode)
        output = edge_attention(q, q, v, edges, bias).sum()
        with self.assertRaisesRegex(ValueError, 'create_graph|unsupported'):
            grad(output, [q, v, bias], create_graph=True)
        self.assertIsNone(q.grad)
        self.assertIsNone(v.grad)
        self.assertEqual(len(grad(output, [q, v, bias])), 3)


if __name__ == '__main__':
    unittest.main()
