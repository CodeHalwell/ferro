"""Actual PyG GENConv composition, gradient and normalization-state references."""
import tempfile
import unittest
import sys
import torch
from torch_geometric.nn import GENConv as TorchGEN
import ferro as fr
from ferro.graph import GENConv
import test_pyg_kernels as helpers

native = helpers.native


class GENTests(unittest.TestCase):
    close = helpers.KernelTests.close
    copy = helpers.KernelTests.copy
    gradients = helpers.KernelTests.gradients

    def setUp(self):
        torch.manual_seed(1709)
        self.edge = torch.tensor([[0, 1, 2, 0, 3, 4, 1, 1, 2], [1, 2, 0, 3, 4, 0, 1, 2, 2]])

    def test_all_aggregators_conditional_maps_and_input_edge_parameter_gradients(self):
        cases = [('softmax', dict(learn_t=True)), ('softmax_sg', {}), ('powermean', dict(learn_p=True, p=1.3)),
                 ('power', dict(p=2.)), ('mean', {}), ('max', {}), (['mean', 'max'], {}),
                 (['mean', 'softmax'], dict(aggr_kwargs={'mode': 'attn', 'mode_kwargs': {'in_channels': 4, 'out_channels': 2}}))]
        for aggr, options in cases:
            with self.subTest(aggr=aggr, options=options):
                model = GENConv((3, 2), 4, aggr=aggr, edge_dim=2, norm='layer', num_layers=3, bias=True, msg_norm=True, learn_msg_scale=True, **options)
                reference = TorchGEN((3, 2), 4, aggr=aggr, edge_dim=2, norm='layer', num_layers=3, bias=True, msg_norm=True, learn_msg_scale=True, **options)
                self.copy(model, reference)
                x, target, attr = torch.randn(6, 3, requires_grad=True), torch.randn(5, 2, requires_grad=True), torch.randn(9, 2, requires_grad=True)
                nx, nt, na = native(x, True), native(target, True), native(attr, True)
                self.gradients(model, reference, model((nx, nt), native(self.edge), na), reference((x, target), self.edge, attr), [('source', nx, x), ('target', nt, target), ('edge_attr', na, attr)])

    def test_identity_maps_missing_targets_fixed_message_scale_and_empty_edges(self):
        for target, message_norm, layers, norm in ((True, False, 1, None), (True, True, 2, None), (False, False, 2, 'layer'), (False, True, 2, None)):
            model = GENConv(3, 3, aggr='softmax', msg_norm=message_norm, num_layers=layers, norm=norm)
            reference = TorchGEN(3, 3, aggr='softmax', msg_norm=message_norm, num_layers=layers, norm=norm)
            self.copy(model, reference)
            x = torch.randn(6, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, reference, model(nx if target else (nx, None), native(self.edge)),
                           reference(x if target else (x, None), self.edge), [('x', nx, x)])
        for norm in (None, 'layer'):
            model, reference = GENConv(3, 4, norm=norm), TorchGEN(3, 4, norm=norm)
            self.copy(model, reference)
            x = torch.empty(0, 3, requires_grad=True)
            nx, edge = native(x, True), torch.empty(2, 0, dtype=torch.int64)
            self.gradients(model, reference, model(nx, native(edge)), reference(x, edge), [('x', nx, x)])

    def test_batchnorm_existing_math_training_eval_reference(self):
        model, reference = GENConv(3, 4), TorchGEN(3, 4)
        self.copy(model, reference)
        for training in (True, False):
            model.train(training); reference.train(training)
            for parameter in model.parameters(): parameter.tensor().zero_grad()
            reference.zero_grad()
            x = torch.randn(6, 3, requires_grad=True)
            nx = native(x, True)
            self.gradients(model, reference, model(nx, native(self.edge)), reference(x, self.edge), [('x', nx, x)])
            self.close(model.mlp[1].running_mean, reference.mlp[1].running_mean)
            self.close(model.mlp[1].running_var, reference.mlp[1].running_var)
            self.assertEqual(model.mlp[1].num_batches_tracked.item(), reference.mlp[1].num_batches_tracked.item())

    def test_batchnorm_invalid_single_target_precedes_attention_rng_and_statistics(self):
        options = dict(aggr=['mean', 'sum'], aggr_kwargs={'mode': 'attn', 'mode_kwargs': {'in_channels': 4, 'out_channels': 4, 'dropout': .2}})
        model = GENConv((3, 2), 4, **options)
        stream = model.aggr_module.multihead_attn.attn_dropout.rng_state.tolist()
        mean = model.mlp[1].running_mean.tolist()
        with self.assertRaises(ValueError):
            model((fr.Tensor.ones([3, 3]), fr.Tensor.ones([1, 2])), fr.Tensor.from_i64([0, 1, 0, 0], [2, 2]))
        self.assertEqual(stream, model.aggr_module.multihead_attn.attn_dropout.rng_state.tolist())
        self.assertEqual(mean, model.mlp[1].running_mean.tolist())
        self.assertEqual(model.mlp[1].num_batches_tracked.item(), 0)
        model.mlp[1] = fr.nn.BatchNorm1d(8, track_running_stats=False)
        model.mlp[1].eval()
        with self.assertRaises(ValueError):
            model((fr.Tensor.ones([3, 3]), fr.Tensor.ones([1, 2])), fr.Tensor.from_i64([0, 1, 0, 0], [2, 2]))
        self.assertEqual(stream, model.aggr_module.multihead_attn.attn_dropout.rng_state.tolist())

    def test_default_batchnorm_empty_counter_and_reset_all_hidden_layers(self):
        model, oracle = GENConv(3, 4, num_layers=3), TorchGEN(3, 4, num_layers=3)
        self.copy(model, oracle)
        for nodes in (6, 0):
            for parameter in model.parameters(): parameter.tensor().zero_grad()
            oracle.zero_grad()
            x = torch.randn(nodes, 3, requires_grad=True)
            edge = self.edge if nodes else torch.empty(2, 0, dtype=torch.int64)
            nx = native(x, True)
            self.gradients(model, oracle, model(nx, native(edge)), oracle(x, edge), [('x', nx, x)])
            for index in (1, 5):
                self.close(model.mlp[index].running_mean, oracle.mlp[index].running_mean)
                self.close(model.mlp[index].running_var, oracle.mlp[index].running_var)
                self.assertEqual(model.mlp[index].num_batches_tracked.item(), oracle.mlp[index].num_batches_tracked.item())
        streams = [layer.rng_state.tolist() for layer in model.mlp if isinstance(layer, fr.nn.Dropout)]
        model.reset_parameters()
        for index in (1, 5):
            layer = model.mlp[index]
            self.assertEqual(layer.running_mean.tolist(), [0.] * 8)
            self.assertEqual(layer.running_var.tolist(), [1.] * 8)
            self.assertEqual(layer.num_batches_tracked.item(), 0)
            self.assertEqual(layer.weight.tensor().tolist(), [1.] * 8)
            self.assertEqual(layer.bias.tensor().tolist(), [0.] * 8)
        self.assertEqual(streams, [layer.rng_state.tolist() for layer in model.mlp if isinstance(layer, fr.nn.Dropout)])

    def test_default_batchnorm_saved_optimizer_counter_and_rng_continuation(self):
        def factory(momentum=.1):
            model = GENConv(3, 4, num_layers=3, bias=True)
            for layer in model.mlp:
                if isinstance(layer, fr.nn.BatchNorm1d): layer.momentum = momentum
            return model
        model, generator = factory(None), fr.Generator(197)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        edge = native(self.edge)
        def step(model, optimizer, generator):
            output = model(fr.Tensor.randn([6, 3], generator=generator), edge)
            loss = output.square().mean() + output.mean() * .1
            result = output.tolist(), loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return result
        step(model, optimizer, generator)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'gen': optimizer}, generator=generator, step=1)
        self.assertEqual(saved.tensors()['buffers.mlp.1.num_batches_tracked'].dtype, 'int64')
        with tempfile.TemporaryDirectory() as directory:
            saved.save(directory + '/state')
            saved = fr.checkpoint.load(directory + '/state')
        expected = [step(model, optimizer, generator) for _ in range(3)]
        other, rng = factory(None), fr.Generator(1)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.9)
        saved.restore(other, optimizers={'gen': other_opt}, generator=rng)
        self.assertEqual(expected, [step(other, other_opt, rng) for _ in range(3)])
        left = fr.checkpoint.snapshot_training(model, optimizers={'gen': optimizer}, generator=generator, step=4).tensors()
        right = fr.checkpoint.snapshot_training(other, optimizers={'gen': other_opt}, generator=rng, step=4).tensors()
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertEqual((left[name].dtype, left[name].shape, left[name].tolist()),
                             (right[name].dtype, right[name].shape, right[name].tolist()), name)
        changed = factory(.1)
        before = [(name, value.tolist()) for name, value in changed.named_buffers()]
        with self.assertRaises(ValueError):
            saved.restore(changed, optimizers={'gen': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [(name, value.tolist()) for name, value in changed.named_buffers()])

    def test_native_checkpoint_attention_dropout_parameter_aliases_and_config(self):
        def factory(norm='layer'):
            options = dict(aggr=['mean', 'softmax'], aggr_kwargs={'mode': 'attn', 'mode_kwargs': {'in_channels': 4, 'out_channels': 4, 'dropout': .2}})
            model = GENConv(3, 4, norm=norm, edge_dim=3, bias=True, msg_norm=True, learn_msg_scale=True, **options)
            model.lin_dst.weight = model.lin_src.weight
            return model
        model, rng = factory(), fr.Generator(157)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.002)
        edge = native(self.edge)
        def step(current, opt, generator):
            x = fr.Tensor.rand([6, 3], generator=generator)
            attrs = fr.Tensor.rand([9, 3], generator=generator)
            output = current(x, edge, attrs)
            loss = output.square().mean()
            values = output.tolist(), loss.item()
            loss.backward(); opt.step(); opt.zero_grad()
            return values
        step(model, optimizer, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'gen': optimizer}, generator=rng, step=1)
        expected = [step(model, optimizer, rng) for _ in range(3)]
        other, other_rng = factory(), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        saved.restore(other, optimizers={'gen': other_opt}, generator=other_rng)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertIs(other.lin_src.weight, other.lin_dst.weight)
        for left, right in zip(model.parameters(), other.parameters()): self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        self.assertEqual(model.aggr_module.multihead_attn.attn_dropout.rng_state.tolist(), other.aggr_module.multihead_attn.attn_dropout.rng_state.tolist())
        changed = factory(None)
        before = [parameter.tensor().tolist() for parameter in changed.parameters()]
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'gen': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertEqual(before, [parameter.tensor().tolist() for parameter in changed.parameters()])

    def test_aggregation_module_registration_override_and_reset(self):
        aggr = fr.graph.SoftmaxAggregation(.7, learn=True)
        with fr.no_grad(): aggr.t.tensor().copy_(fr.Tensor([2.], [1]))
        model = GENConv(3, 3, aggr=aggr, norm='layer', bias=True, msg_norm=True)
        self.assertIs(model.aggr_module, aggr)
        self.assertEqual(aggr.t.tensor().tolist(), [2.])
        with fr.no_grad():
            model.mlp[1].weight.tensor().copy_(fr.Tensor.zeros([6]))
            model.msg_norm.scale.tensor().copy_(fr.Tensor([2.], [1]))
        model.reset_parameters()
        self.assertAlmostEqual(aggr.t.tensor().item(), .7, places=6)
        self.assertEqual(model.mlp[1].weight.tensor().tolist(), [1.] * 6)
        self.assertEqual(model.msg_norm.scale.tensor().tolist(), [1.])
        model = GENConv(3, 3, aggr='softmax', learn_t=True, aggr_kwargs={'t': .4, 'learn': False}, norm=None)
        self.assertFalse(model.aggr_module.learn)
        self.assertEqual(model.aggr_module.t, .4)
        # Expansion is unused by a one-linear MLP, including its capacity checks.
        model, oracle = GENConv(3, 3, num_layers=1, expansion=sys.maxsize), TorchGEN(3, 3, num_layers=1, expansion=sys.maxsize)
        self.copy(model, oracle)
        x = torch.randn(6, 3, requires_grad=True)
        nx = native(x, True)
        self.gradients(model, oracle, model(nx, native(self.edge)), oracle(x, self.edge), [('x', nx, x)])

    def test_strict_feature_bounds_config_capacity_and_capture(self):
        for constructor in (lambda: GENConv(3, 0), lambda: GENConv(3, 4, num_layers=0), lambda: GENConv(3, 4, expansion=0),
                            lambda: GENConv(3, 4, eps=float('nan')), lambda: GENConv(3, 4, edge_dim=0),
                            lambda: GENConv(3, 4, num_layers=sys.maxsize)):
            with self.assertRaises((TypeError, ValueError)): constructor()
        for constructor in (lambda: GENConv(3, 4, aggr=None), lambda: GENConv(3, 4, norm='instance'), lambda: GENConv(3, 4, flow='target_to_source')):
            with self.assertRaises(NotImplementedError): constructor()
        model, x, edge = GENConv(3, 4, norm=None, edge_dim=2), fr.Tensor.ones([6, 3]), native(self.edge)
        for call in (lambda: model(x, edge, fr.Tensor.ones([8, 2])), lambda: model(x, edge, fr.Tensor.ones([9, 3])),
                     lambda: model(x, edge, size=(6, 7)), lambda: GENConv(3, 4, norm=None, msg_norm=True)((x, None), edge, size=(6, 7))):
            with self.assertRaises((TypeError, ValueError)): call()
        huge = fr.Tensor.ones([1, 1]).expand(sys.maxsize // 8 + 1, 1)
        with self.assertRaises(ValueError): GENConv(1, 1, norm=None)(huge, fr.Tensor.from_i64([], [2, 0]))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, edge))


if __name__ == '__main__':
    unittest.main(verbosity=2)
