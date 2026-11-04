"""CPU public graph API against an independent dense PyTorch oracle."""
import tempfile
import unittest
import sys

import torch
import ferro as fr
from ferro.graph import EdgeIndex, GcnConv, scatter


def tensor(x, grad=False):
    x = torch.as_tensor(x, dtype=torch.float32)
    return fr.Tensor(x.reshape(-1).tolist(), list(x.shape)).requires_grad_(grad)


def values(x):
    return torch.tensor(x.tolist(), dtype=torch.float32).reshape(x.shape)


def topology():
    # Duplicate edges, duplicate self loops, a directed edge and an isolated node.
    return EdgeIndex(5, [0, 1, 1, 2, 1, 3, 3, 0], [1, 0, 2, 1, 2, 3, 3, 2])


def adjacency(edges, self_loops):
    a = torch.zeros(edges.num_nodes, edges.num_nodes)
    for src, dst in zip(edges.src, edges.dst):
        if not self_loops or src != dst:
            a[dst, src] += 1
    if self_loops:
        a.fill_diagonal_(1)
    degree = a.sum(1)
    inv = torch.zeros_like(degree)
    inv[degree > 0] = degree[degree > 0].rsqrt()
    return inv[:, None] * a * inv[None, :]


def layer(self_loops=True):
    model = GcnConv(in_features=3, out_features=2, self_loops=self_loops)
    with fr.no_grad():
        model.weight.tensor().copy_(tensor([[0.3, -0.4], [0.1, 0.7], [-0.2, 0.5]]))
        model.bias.tensor().copy_(tensor([0.1, -0.2]))
    return model


class GraphApiTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(values(actual), expected, atol=1e-5, rtol=1e-4)

    def test_topology_roundtrip_and_batch(self):
        edges = topology()
        recovered = EdgeIndex.from_tensor(edges.to_tensor(), edges.num_nodes)
        self.assertEqual(recovered.src, edges.src)
        self.assertEqual(recovered.dst, edges.dst)
        self.assertEqual(recovered.num_nodes, 5)
        self.assertEqual(recovered.num_edges, 8)
        copy = edges.src
        copy[0] = 4
        self.assertEqual(edges.src[0], 0)
        with self.assertRaises(AttributeError):
            edges.num_nodes = 3
        looped = edges.add_remaining_self_loops()
        self.assertEqual(looped.in_degree(), [2, 3, 4, 1, 1])
        self.close(looped.gcn_norm(), torch.tensor([
            adjacency(edges, True)[d, s] / sum(a == s and b == d for a, b in zip(looped.src, looped.dst))
            for s, d in zip(looped.src, looped.dst)]))
        union, owner = EdgeIndex.batch([EdgeIndex(0, [], []), edges, EdgeIndex(2, [], [])])
        self.assertEqual(union.num_nodes, 7)
        self.assertEqual(union.src, edges.src)
        self.assertEqual(owner, [1] * 5 + [2] * 2)
        empty, owner = EdgeIndex.batch([])
        self.assertEqual((empty.num_nodes, empty.num_edges, owner), (0, 0, []))

    def test_topology_validation(self):
        for args in ((3, [0], []), (3, [3], [0]), (3, [-1], [0]), (2**63, [], [])):
            with self.subTest(args=args), self.assertRaises((ValueError, OverflowError)):
                EdgeIndex(*args)
        for invalid in (tensor([[0, 1], [1, 0]]), fr.Tensor.from_i64([0, -1], [2, 1]),
                        fr.Tensor.from_i64([0, 3], [2, 1]), fr.Tensor.from_i64([0, 1], [1, 2])):
            with self.assertRaises(ValueError):
                EdgeIndex.from_tensor(invalid, 3)
        with self.assertRaises(ValueError):
            EdgeIndex.batch([EdgeIndex(sys.maxsize, [], [])])
        with self.assertRaises(ValueError):
            EdgeIndex.batch([EdgeIndex(sys.maxsize, [], []), EdgeIndex(sys.maxsize, [], [])])

    def test_scatter_values_and_gradients(self):
        ids = [2, 0, 2]
        data = torch.tensor([[1., -2.], [3., 4.], [5., -6.]])
        probe = torch.tensor([[0.2, 0.7], [0.1, 0.4], [-0.5, 0.3], [0.6, 0.9]])
        for reduction in ('sum', 'mean', 'max', 'min'):
            with self.subTest(reduction=reduction):
                x, reference = tensor(data, True), data.clone().requires_grad_(True)
                rows = []
                for i in range(4):
                    selected = reference[[j for j, group in enumerate(ids) if group == i]]
                    if selected.shape[0] == 0:
                        rows.append(reference.sum(0) * 0)
                    elif reduction == 'sum':
                        rows.append(selected.sum(0))
                    elif reduction == 'mean':
                        rows.append(selected.mean(0))
                    elif reduction == 'max':
                        rows.append(selected.amax(0))
                    else:
                        rows.append(selected.amin(0))
                expected = torch.stack(rows)
                actual = scatter(x, ids, 4, reduction)
                self.close(actual, expected.detach())
                (actual * tensor(probe)).sum().backward()
                (expected * probe).sum().backward()
                self.close(x.grad, reference.grad)
        for reduction in ('max', 'min'):
            x = tensor([2., 2.], True)
            scatter(x, [0, 0], 2, reduction).sum().backward()
            self.close(x.grad, torch.tensor([0.5, 0.5]))

    def test_gcn_values_and_all_gradients(self):
        edges = topology()
        data = (torch.arange(15).reshape(5, 3) - 6) / 7
        probe = (torch.arange(10).reshape(5, 2) - 4) / 9
        for loops in (True, False):
            with self.subTest(self_loops=loops):
                model = layer(loops)
                x = tensor(data, True)
                tx = data.clone().requires_grad_(True)
                tw = values(model.weight.tensor()).requires_grad_(True)
                tb = values(model.bias.tensor()).requires_grad_(True)
                expected = adjacency(edges, loops) @ (tx @ tw) + tb
                actual = model(x, edges)
                self.close(actual, expected.detach())
                (actual * tensor(probe)).sum().backward()
                (expected * probe).sum().backward()
                self.close(x.grad, tx.grad)
                self.close(model.weight.grad, tw.grad)
                self.close(model.bias.grad, tb.grad)
                self.close(model(tensor(data.T).transpose(0, 1), edges), expected.detach())

    def test_gcn_batch_and_node_permutation(self):
        model, edges = layer(), topology()
        data = (torch.arange(15).reshape(5, 3) - 6) / 7
        single = model(tensor(data), edges)
        union, owner = EdgeIndex.batch([edges, edges])
        self.close(model(tensor(torch.cat([data, data])), union), torch.cat([values(single)] * 2))
        self.assertEqual(owner, [0] * 5 + [1] * 5)
        perm = [3, 0, 4, 1, 2]
        inv = [perm.index(i) for i in range(5)]
        pe = EdgeIndex(5, [inv[i] for i in edges.src], [inv[i] for i in edges.dst])
        self.close(model(tensor(data[perm]), pe), values(single)[perm])

    def test_empty_graphs(self):
        for loops in (False, True):
            model = layer(loops)
            x = tensor(torch.empty(0, 3), True)
            out = model(x, EdgeIndex(0, [], []))
            self.assertEqual(out.shape, [0, 2])
            out.sum().backward()
            self.assertEqual(x.grad.shape, [0, 3])
            self.close(model.weight.grad, torch.zeros(3, 2))
            self.close(model.bias.grad, torch.zeros(2))
        for reduction in ('sum', 'mean', 'max', 'min'):
            self.close(scatter(tensor(torch.empty(0, 2)), [], 3, reduction), torch.zeros(3, 2))

    def test_invalid_inputs_and_capture_reject(self):
        model, edges = layer(), topology()
        for x in (tensor(torch.zeros(4, 3)), tensor(torch.zeros(5, 2)),
                  fr.Tensor.from_i64([0] * 15, [5, 3])):
            with self.assertRaises(ValueError):
                model(x, edges)
        for reduction in ('sum', 'mean', 'max', 'min'):
            with self.assertRaises(ValueError):
                scatter(fr.Tensor.from_i64([1, 2], [2]), [0, 0], 1, reduction)
            with self.assertRaises(ValueError):
                scatter(tensor([1., 2.]), [0, 2], 1, reduction)
        with self.assertRaises(ValueError):
            scatter(tensor([1.]), [0], 1, 'product')
        for width in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                GcnConv(in_features=width, out_features=2)
        with self.assertRaises(TypeError):
            GcnConv(in_features=3, out_features=2, self_loops=1)
        with self.assertRaises(AttributeError):
            model.self_loops = False
        with self.assertRaisesRegex(ValueError, 'capture'):
            fr.capture(lambda: model(tensor(torch.zeros(5, 3)), edges))

    def test_checkpoint_continuation_and_config_mismatch(self):
        model, edges, rng = layer(), topology(), fr.Generator(19)
        self.assertEqual([name for name, _ in model.named_parameters()], ['weight', 'bias'])
        opt = fr.optim.Adam(model.parameters(), lr=0.01)

        def step(model, optimizer, generator):
            optimizer.zero_grad()
            x = fr.Tensor.randn([5, 3], generator=generator)
            loss = model(x, edges).pow(2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()

        for _ in range(3):
            step(model, opt, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': opt}, generator=rng, step=3)
        expected = [step(model, opt, rng) for _ in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory)
            restored = fr.checkpoint.load(directory)
            target, target_rng = layer(), fr.Generator(999)
            target_opt = fr.optim.Adam(target.parameters(), lr=0.9)
            self.assertEqual(restored.restore(target, optimizers={'main': target_opt}, generator=target_rng), 3)
            self.assertEqual([step(target, target_opt, target_rng) for _ in range(4)], expected)
            for actual, wanted in zip(target.parameters(), model.parameters()):
                self.assertEqual(actual.tensor().tolist(), wanted.tensor().tolist())
            self.assertEqual(fr.Tensor.rand([4], generator=target_rng).tolist(), fr.Tensor.rand([4], generator=rng).tolist())
            incompatible = layer(False)
            incompatible_opt = fr.optim.Adam(incompatible.parameters(), lr=0.01)
            before = [p.tensor().tolist() for p in incompatible.parameters()]
            with self.assertRaises(ValueError):
                restored.restore(incompatible, optimizers={'main': incompatible_opt}, generator=fr.Generator(2))
            self.assertEqual([p.tensor().tolist() for p in incompatible.parameters()], before)


if __name__ == '__main__':
    unittest.main()
