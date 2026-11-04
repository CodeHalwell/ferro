"""Installed-wheel sparse GATConv against actual optional PyG reference."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.graph import EdgeIndex, GATConv, GATv2Conv

try:
    from torch_geometric.nn import GATConv as TorchGATConv
    from torch_geometric.nn import GATv2Conv as TorchGATv2Conv
except ImportError:
    TorchGATConv = None
    TorchGATv2Conv = None


def tensor(value, requires_grad=False):
    value = torch.as_tensor(value, dtype=torch.float32)
    return fr.Tensor(value.reshape(-1).tolist(), list(value.shape)).requires_grad_(requires_grad)


def edges(value):
    value = torch.as_tensor(value, dtype=torch.int64)
    return fr.Tensor.from_i64(value.reshape(-1).tolist(), list(value.shape))


def values(value):
    return torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape)


@unittest.skipUnless(TorchGATConv is not None, 'PyG is an optional test reference')
class PyGAttentionTests(unittest.TestCase):
    def pair(self, channels=3, **options):
        torch.manual_seed(7103)
        wanted = TorchGATConv(channels, 2, **options)
        actual = GATConv(channels, 2, **options)
        reference = dict(wanted.named_parameters())
        self.assertEqual(set(dict(actual.named_parameters())), set(reference))
        with fr.no_grad():
            for name, parameter in actual.named_parameters():
                parameter.tensor().copy_(tensor(reference[name].detach()))
        return actual, wanted

    def close(self, actual, wanted, atol=1e-5, rtol=1e-4):
        torch.testing.assert_close(values(actual), wanted.detach(), atol=atol, rtol=rtol)

    def compare(self, channels=3, heads=2, concat=True, loops=True, edge_dim=2, fill='mean',
                residual=True, target=True, size=None, empty=False, edge_vector=False):
        model, reference = self.pair(channels, heads=heads, concat=concat, add_self_loops=loops,
                                     edge_dim=edge_dim, fill_value=fill, residual=residual)
        generator = torch.Generator().manual_seed(513)
        source_width, target_width = (channels, channels) if isinstance(channels, int) else channels
        nsource, ntarget = size or (6, 6)
        tx = torch.randn(nsource, source_width, generator=generator).requires_grad_(True)
        sx = tensor(tx.detach(), True)
        pairs = isinstance(channels, tuple) or size is not None or not target
        if pairs and target:
            ty = torch.randn(ntarget, target_width, generator=generator).requires_grad_(True)
            sy = tensor(ty.detach(), True)
        else:
            ty = sy = None
        source = [0, 1, 1, 2, 3, 0, 4, 2] if not empty else []
        destination = [1, 1, 1, 2, 0, 3, 3, 0] if not empty else []
        source = [value % nsource for value in source]
        destination = [value % ntarget for value in destination]
        te = torch.tensor([source, destination], dtype=torch.int64)
        ta = torch.randn(len(source), edge_dim or 1, generator=generator) if edge_dim is not None else None
        if ta is not None and edge_vector:
            ta = ta.reshape(-1)
        if ta is not None:
            ta.requires_grad_(True)
        sa = None if ta is None else tensor(ta.detach(), True)
        native_inputs = (sx, sy) if pairs else sx
        torch_inputs = (tx, ty) if pairs else tx
        actual, (actual_edges, alpha) = model(native_inputs, edges(te), sa, size=size, return_attention_weights=True)
        wanted, (wanted_edges, wanted_alpha) = reference(torch_inputs, te, ta, size=size, return_attention_weights=True)
        self.close(actual, wanted)
        self.assertEqual(actual_edges.dtype, 'int64')
        self.assertEqual(actual_edges.tolist(), wanted_edges.tolist())
        self.close(alpha, wanted_alpha)
        probe = torch.randn(wanted.shape, generator=generator)
        probe_alpha = torch.randn(wanted_alpha.shape, generator=generator)
        loss = (actual * tensor(probe)).sum() + (alpha * tensor(probe_alpha)).sum() * .2
        reference_loss = (wanted * probe).sum() + (wanted_alpha * probe_alpha).sum() * .2
        loss.backward()
        reference_loss.backward()
        comparisons = [('source', sx, tx)]
        if sy is not None:
            comparisons.append(('target', sy, ty))
        if sa is not None:
            comparisons.append(('edge_attr', sa, ta))
        reference_parameters = dict(reference.named_parameters())
        comparisons.extend((name, parameter.tensor(), reference_parameters[name]) for name, parameter in model.named_parameters())
        for name, actual_input, reference_input in comparisons:
            with self.subTest(gradient=name):
                if reference_input.grad is None:
                    self.assertIsNone(actual_input.grad)
                else:
                    self.assertIsNotNone(actual_input.grad)
                    self.close(actual_input.grad, reference_input.grad)
        return model, reference

    def test_homogeneous_multihead_concat_mean_edge_features_and_residual(self):
        for heads, concat, edge_dim, loops, residual in ((1, True, None, True, False),
                                                       (3, True, 2, True, True),
                                                       (2, False, 2, True, True),
                                                       (2, True, None, False, False),
                                                       (2, False, 1, False, False)):
            with self.subTest(heads=heads, concat=concat, edge_dim=edge_dim, loops=loops, residual=residual):
                self.compare(heads=heads, concat=concat, edge_dim=edge_dim, loops=loops,
                             residual=residual, edge_vector=edge_dim == 1)

    def test_self_loop_fill_reductions_scalar_and_tensor(self):
        for fill in ('add', 'mean', 'min', 'max', 0.7, None):
            with self.subTest(fill=fill):
                self.compare(fill=fill)
        model, reference = self.pair(heads=2, edge_dim=2, fill_value=.0)
        fill = torch.tensor([.23, -.51], requires_grad=True)
        reference.fill_value = fill
        model.fill_value = tensor(fill.detach(), True)
        x = torch.tensor([[.2, -.5, .1], [.4, -.7, .6], [.9, .2, -.3]])
        edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
        attributes = torch.tensor([[.1, -.2], [.3, .6], [-.4, .7]])
        actual = model(tensor(x), edges(edge_index), tensor(attributes))
        wanted = reference(x, edge_index, attributes)
        self.close(actual, wanted)
        actual.sum().backward()
        wanted.sum().backward()
        self.close(model.fill_value.grad, fill.grad)

    def test_bipartite_separate_and_shared_transforms(self):
        for channels, target, size, loops, edge_dim in (((3, 4), True, (6, 4), True, 2),
                                                      ((3, 4), True, (4, 6), False, 2),
                                                      (3, True, (6, 4), True, 1),
                                                      ((3, 4), False, (6, 4), True, 2),
                                                      ((3, 4), False, None, False, None)):
            with self.subTest(channels=channels, target=target, size=size, loops=loops):
                self.compare(channels=channels, target=target, size=size, loops=loops, edge_dim=edge_dim)

    def test_empty_edges_nodes_and_zero_neighbor_outputs(self):
        for loops in (True, False):
            for edge_dim in (None, 2):
                with self.subTest(loops=loops, edge_dim=edge_dim):
                    self.compare(empty=True, loops=loops, edge_dim=edge_dim, residual=False)
        model, reference = self.pair(heads=2, residual=True)
        x = torch.zeros(0, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.int64)
        actual = model(tensor(x, True), edges(edge_index))
        wanted = reference(x, edge_index)
        self.close(actual, wanted)
        model, _ = self.pair(heads=2, add_self_loops=False, bias=False)
        large = tensor(torch.randn(4096, 3))
        output = model(large, edges([[0, 1, 1], [2, 2, 2]]))
        self.assertEqual(output.shape, [4096, 4])
        self.assertEqual(output.tolist()[4095], [0., 0., 0., 0.])

    def test_native_edges_disjoint_batches_permutations_and_return_flags(self):
        model, reference = self.pair(heads=2, add_self_loops=False)
        x = torch.tensor([[.2, -.5, .1], [.4, -.7, .6], [.9, .2, -.3]])
        topology = EdgeIndex(3, [0, 1, 1, 2], [1, 0, 0, 1])
        actual = model(tensor(x), topology)
        union, _ = EdgeIndex.batch([topology, topology])
        self.close(model(tensor(torch.cat([x, x])), union), torch.cat([values(actual), values(actual)]))
        permutation = [2, 0, 1]
        inverse = {old: new for new, old in enumerate(permutation)}
        reordered = EdgeIndex(3, [inverse[i] for i in topology.src], [inverse[i] for i in topology.dst])
        self.close(model(tensor(x[permutation]), reordered), values(actual)[permutation])
        te = torch.tensor([topology.src, topology.dst])
        for flag in (None, False, True):
            output = model(tensor(x), edges(te), return_attention_weights=flag)
            wanted = reference(x, te, return_attention_weights=flag)
            self.assertEqual(isinstance(output, tuple), isinstance(wanted, tuple))
            self.close(output[0] if isinstance(output, tuple) else output, wanted[0] if isinstance(wanted, tuple) else wanted)

    def test_checkpoint_preserves_weights_buffers_configuration_and_optimizer(self):
        model = GATConv(3, 2, heads=2, edge_dim=2, fill_value=tensor([.4, -.2]), residual=True)
        optimizer, rng = fr.optim.AdamW(model.parameters(), lr=.02), fr.Generator(7)
        x, edge_index, attributes = tensor([[.2, -.5, .1], [.4, -.7, .6]]), edges([[0, 1], [1, 0]]), tensor([[.2, .7], [-.3, .9]])
        model(x, edge_index, attributes).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            saved.save(Path(directory) / 'model')
            restored = GATConv(3, 2, heads=2, edge_dim=2, fill_value=tensor([0., 0.]), residual=True)
            restored_optimizer, restored_rng = fr.optim.AdamW(restored.parameters(), lr=.8), fr.Generator(999)
            step = fr.checkpoint.load_training(Path(directory) / 'model', restored, optimizers={'main': restored_optimizer}, generator=restored_rng)
            self.assertEqual(step, 1)
            self.assertEqual(model.fill_value.tolist(), restored.fill_value.tolist())
            self.assertEqual(model(x, edge_index, attributes).tolist(), restored(x, edge_index, attributes).tolist())
            for layer, opt in ((model, optimizer), (restored, restored_optimizer)):
                result = layer(x, edge_index, attributes)
                (result * result).mean().backward()
                opt.step()
                opt.zero_grad()
            for (name, parameter), (other_name, other_parameter) in zip(model.named_parameters(), restored.named_parameters()):
                self.assertEqual(name, other_name)
                self.assertEqual(parameter.tensor().tolist(), other_parameter.tensor().tolist())
            incompatible = GATConv(3, 2, heads=2, edge_dim=2, fill_value=tensor([0., 0.]), residual=True, negative_slope=.3)
            before = {name: p.tensor().tolist() for name, p in incompatible.named_parameters()}
            with self.assertRaises(ValueError):
                saved.restore(incompatible, optimizers={'main': fr.optim.AdamW(incompatible.parameters())}, generator=fr.Generator(1))
            self.assertEqual(before, {name: p.tensor().tolist() for name, p in incompatible.named_parameters()})

    def test_noncontiguous_features_topology_attributes_and_gradients(self):
        model, reference = self.pair(heads=2, edge_dim=2, residual=True)
        raw_x = torch.linspace(-.8, .9, 15).reshape(3, 5).requires_grad_(True)
        raw_attr = torch.linspace(-.9, .7, 12).reshape(2, 6).requires_grad_(True)
        edge_index = torch.tensor([[0, 1], [1, 0], [1, 0], [2, 3], [3, 4], [4, 2]], dtype=torch.int64)
        native_x, native_attr = tensor(raw_x.detach(), True), tensor(raw_attr.detach(), True)
        actual = model(native_x.transpose(0, 1), edges(edge_index).transpose(0, 1), native_attr.transpose(0, 1))
        wanted = reference(raw_x.T, edge_index.T, raw_attr.T)
        self.close(actual, wanted)
        (actual * actual).sum().backward()
        (wanted * wanted).sum().backward()
        self.close(native_x.grad, raw_x.grad)
        self.close(native_attr.grad, raw_attr.grad)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            self.close(parameter.grad, reference_parameters[name].grad)

    def test_rectangular_loop_fill_reduces_full_destination_space(self):
        model, reference = self.pair((3, 4), heads=2, edge_dim=2, add_self_loops=False)
        model.add_self_loops = True
        source = torch.linspace(-.6, .8, 9).reshape(3, 3).requires_grad_(True)
        target = torch.linspace(.9, -.7, 20).reshape(5, 4).requires_grad_(True)
        attributes = torch.tensor([[.2, -.4], [.5, .6], [-.7, .8]], requires_grad=True)
        topology = torch.tensor([[0, 1, 2], [4, 0, 1]])
        # PyG's automatic GAT loop fill uses min(Ns,Nt) as the reduction size.
        # Prepare its mathematically rectangular topology/attributes explicitly.
        loop_values = torch.stack([attributes[1], attributes[2], attributes.sum(0) * 0])
        prepared = torch.cat([topology, torch.tensor([[0, 1, 2], [0, 1, 2]])], 1)
        native_source, native_target, native_attributes = [tensor(x.detach(), True) for x in (source, target, attributes)]
        actual = model((native_source, native_target), edges(topology), native_attributes)
        wanted = reference((source, target), prepared, torch.cat([attributes, loop_values]))
        self.close(actual, wanted)
        (actual * actual).sum().backward()
        (wanted * wanted).sum().backward()
        for actual_input, wanted_input in ((native_source, source), (native_target, target), (native_attributes, attributes)):
            self.close(actual_input.grad, wanted_input.grad)

    def test_unsupported_options_and_invalid_inputs_reject(self):
        for options in ({'fill_value': 'mul'}, {'aggr': 'max'}, {'flow': 'target_to_source'}):
            with self.subTest(options=options), self.assertRaises(NotImplementedError):
                GATConv(3, 2, **options)
        for channels, options in ((-1, {}), ((3, -1), {}), (3, {'heads': 0}), (3, {'heads': True}),
                                  (3, {'edge_dim': 0}), (3, {'negative_slope': float('nan')})):
            with self.subTest(channels=channels, options=options), self.assertRaises((TypeError, ValueError)):
                GATConv(channels, 2, **options)
        model = GATConv(3, 2, edge_dim=2)
        x, topology = tensor(torch.ones(3, 3)), edges([[0, 1], [1, 2]])
        for bad in (tensor([[0., 1.], [1., 2.]]), edges([0, 1]), edges([[0, 3], [1, 2]]), edges([[0, -1], [1, 2]])):
            with self.subTest(edge_index=bad), self.assertRaises((TypeError, ValueError)):
                model(x, bad)
        with self.assertRaises(ValueError):
            model(x, topology, tensor(torch.ones(2, 3)))
        with self.assertRaises(ValueError):
            model(x, topology, size=(3, 2))
        with self.assertRaises(ValueError):
            model(fr.Tensor.from_i64([1] * 9, [3, 3]), topology)
        with self.assertRaises(TypeError):
            model(x, topology, return_attention_weights=1)
        for layer in (GATConv, GATv2Conv):
            with self.assertRaises(ValueError):
                layer(3, 2, heads=sys.maxsize)
            with self.assertRaises(ValueError):
                layer(3, 2, edge_dim=sys.maxsize)
            current = layer(3, 2)
            features = tensor(torch.ones(3, 3), True)
            with self.assertRaisesRegex(ValueError, 'capture'):
                fr.capture(lambda: current(features, topology))
            from ferro.autograd import grad
            with self.assertRaisesRegex(ValueError, 'create_graph|unsupported'):
                grad(current(features, topology).sum(), [features], create_graph=True)


@unittest.skipUnless(TorchGATv2Conv is not None, 'PyG is an optional test reference')
class PyGGATv2Tests(unittest.TestCase):
    def close(self, actual, wanted):
        torch.testing.assert_close(values(actual), wanted.detach(), atol=1e-5, rtol=1e-4)

    def pair(self, channels=3, out_channels=2, **options):
        torch.manual_seed(7139)
        model, reference = GATv2Conv(channels, out_channels, **options), TorchGATv2Conv(channels, out_channels, **options)
        parameters = dict(reference.named_parameters())
        self.assertEqual(set(dict(model.named_parameters())), set(parameters))
        with fr.no_grad():
            for name, parameter in model.named_parameters():
                parameter.tensor().copy_(tensor(parameters[name].detach()))
        return model, reference

    def compare(self, channels=3, heads=2, concat=True, edge_dim=2, share_weights=False,
                bias=True, residual=True, loops=True, empty=False):
        model, reference = self.pair(channels, heads=heads, concat=concat, edge_dim=edge_dim,
                                     share_weights=share_weights, bias=bias, residual=residual, add_self_loops=loops)
        generator = torch.Generator().manual_seed(185)
        paired = isinstance(channels, tuple)
        ns, nt = (6, 4) if paired else (6, 6)
        widths = channels if paired else (channels, channels)
        tx = torch.randn(ns, widths[0], generator=generator).requires_grad_(True)
        ty = torch.randn(nt, widths[1], generator=generator).requires_grad_(True) if paired else None
        sx, sy = tensor(tx.detach(), True), None if ty is None else tensor(ty.detach(), True)
        topology = torch.tensor([[0, 1, 1, 2, 3, 4, 2], [1, 1, 1, 2, 0, 3, 0]], dtype=torch.int64)
        if empty:
            topology = topology[:, :0]
        ta = torch.randn(topology.shape[1], edge_dim, generator=generator).requires_grad_(True) if edge_dim is not None else None
        sa = None if ta is None else tensor(ta.detach(), True)
        actual, (actual_edges, alpha) = model((sx, sy) if paired else sx, edges(topology), sa, return_attention_weights=True)
        wanted, (wanted_edges, wanted_alpha) = reference((tx, ty) if paired else tx, topology, ta, return_attention_weights=True)
        self.close(actual, wanted)
        self.close(alpha, wanted_alpha)
        self.assertEqual(actual_edges.tolist(), wanted_edges.tolist())
        probe = torch.randn(wanted.shape, generator=generator)
        attention_probe = torch.randn(wanted_alpha.shape, generator=generator)
        ((actual * tensor(probe)).sum() + (alpha * tensor(attention_probe)).sum() * .2).backward()
        ((wanted * probe).sum() + (wanted_alpha * attention_probe).sum() * .2).backward()
        pairs = [('source', sx, tx)]
        if paired:
            pairs.append(('target', sy, ty))
        if sa is not None:
            pairs.append(('edge_attr', sa, ta))
        parameters = dict(reference.named_parameters())
        pairs.extend((name, p.tensor(), parameters[name]) for name, p in model.named_parameters())
        for name, actual_input, wanted_input in pairs:
            with self.subTest(gradient=name):
                if wanted_input.grad is None:
                    self.assertIsNone(actual_input.grad)
                else:
                    self.assertIsNotNone(actual_input.grad)
                    self.close(actual_input.grad, wanted_input.grad)
        self.assertEqual(model.lin_l is model.lin_r, share_weights)
        for flag in (None, False):
            result = model((sx, sy) if paired else sx, edges(topology), sa, return_attention_weights=flag)
            self.assertIsInstance(result, fr.Tensor)

    def test_dynamic_attention_matches_actual_pyg_values_and_all_gradients(self):
        for options in ({}, {'heads': 3, 'concat': False}, {'edge_dim': None, 'bias': False, 'residual': False},
                        {'share_weights': True}, {'channels': (3, 4)},
                        {'channels': (3, 3), 'share_weights': True, 'concat': False},
                        {'empty': True, 'loops': False}, {'empty': True, 'loops': True}):
            with self.subTest(options=options):
                self.compare(**options)

    def test_query_can_reverse_neighbor_ranking(self):
        model, reference = self.pair((2, 2), add_self_loops=False, bias=False)
        with fr.no_grad(), torch.no_grad():
            for name in ('lin_l', 'lin_r'):
                getattr(model, name).weight.tensor().copy_(tensor(torch.eye(2)))
                getattr(reference, name).weight.copy_(torch.eye(2))
            model.att.tensor().copy_(tensor([[[1., -1.]]]))
            reference.att.copy_(torch.tensor([[[1., -1.]]]))
        source, target = torch.tensor([[2., 2.], [0., 0.]]), torch.tensor([[2., -4.], [-4., 2.]])
        topology = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
        actual, (_, alpha) = model((tensor(source), tensor(target)), edges(topology), return_attention_weights=True)
        wanted, (_, wanted_alpha) = reference((source, target), topology, return_attention_weights=True)
        self.close(actual, wanted)
        self.close(alpha, wanted_alpha)
        weights = alpha.tolist()
        self.assertGreater(weights[0][0], weights[1][0])
        self.assertLess(weights[2][0], weights[3][0])

    def test_zero_nodes_and_edge_vector_attributes(self):
        model, reference = self.pair(heads=2, edge_dim=1, residual=True)
        for count, index in ((0, torch.zeros(2, 0, dtype=torch.int64)),
                             (3, torch.tensor([[0, 1, 1, 2], [1, 0, 0, 1]]))):
            with self.subTest(count=count):
                features = torch.linspace(-.3, .7, count * 3).reshape(count, 3).requires_grad_(True)
                attributes = torch.linspace(-.2, .6, index.shape[1]).requires_grad_(True)
                native_features, native_attributes = tensor(features.detach(), True), tensor(attributes.detach(), True)
                actual = model(native_features, edges(index), native_attributes)
                wanted = reference(features, index, attributes)
                self.close(actual, wanted)
                actual.sum().backward()
                wanted.sum().backward()
                self.close(native_features.grad, features.grad)
                self.close(native_attributes.grad, attributes.grad)

    def test_shared_parameters_checkpoint_and_rejection_preserve_ties(self):
        model = GATv2Conv(3, 2, heads=2, share_weights=True, residual=True)
        optimizer, rng = fr.optim.AdamW(model.parameters(), lr=.02), fr.Generator(11)
        x, topology = tensor([[.2, -.1, .7], [.9, .3, -.4]]), edges([[0, 1], [1, 0]])
        model(x, topology).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        state = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=rng, step=1)
        with tempfile.TemporaryDirectory() as directory:
            state.save(directory)
            restored = GATv2Conv(3, 2, heads=2, share_weights=True, residual=True)
            other_optimizer = fr.optim.AdamW(restored.parameters(), lr=.9)
            fr.checkpoint.load_training(directory, restored, optimizers={'main': other_optimizer}, generator=fr.Generator(999))
            self.assertIs(restored.lin_l, restored.lin_r)
            self.assertEqual(model(x, topology).tolist(), restored(x, topology).tolist())
            for layer, opt in ((model, optimizer), (restored, other_optimizer)):
                output = layer(x, topology)
                (output * output).mean().backward()
                opt.step()
                opt.zero_grad()
            self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in restored.parameters()])
            different = GATv2Conv(3, 2, heads=2, share_weights=False, residual=True)
            with self.assertRaises(ValueError):
                state.restore(different, optimizers={'main': fr.optim.AdamW(different.parameters())}, generator=fr.Generator(1))
        with self.assertRaises(ValueError):
            GATv2Conv((3, 4), 2, share_weights=True)
        with self.assertRaises(ValueError):
            model((x, None), topology)
        with self.assertRaises(ValueError):
            model(x, topology, tensor([[.2], [.3]]))
        with self.assertRaises(ValueError):
            GATv2Conv(3, 2, dropout=1.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
