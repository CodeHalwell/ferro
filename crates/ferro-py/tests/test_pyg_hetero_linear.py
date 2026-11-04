"""Actual PyG eager typed projections, complete gradients and native state."""
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.nn import HeteroLinear as TorchTyped, HeteroDictLinear as TorchDict
import ferro as fr
from ferro.graph import HeteroLinear, HeteroDictLinear
import test_pyg_modulated as helpers

native = helpers.native


class HeteroLinearTests(unittest.TestCase):
    close = helpers.ModulatedTests.close
    copy = helpers.ModulatedTests.copy
    gradients = helpers.ModulatedTests.gradients

    def setUp(self):
        torch.manual_seed(2309)
        self.types = torch.tensor([3, 0, 2, 3, 2, 0, 3])
        self.backend = patch('torch_geometric.backend.use_segment_matmul', False)
        self.backend.start(); self.addCleanup(self.backend.stop)

    def test_typed_sorted_unsorted_strided_missing_types_and_all_gradients(self):
        for sorted_, bias in ((False, True), (False, False), (True, True)):
            for rows in (7, 0):
                model, oracle = HeteroLinear(3, 4, 5, sorted_, bias=bias), TorchTyped(3, 4, 5, sorted_, bias=bias)
                self.copy(model, oracle)
                ids = self.types.sort().values if sorted_ else self.types
                if not rows: ids = torch.empty(0, dtype=torch.long)
                base = torch.randn(3, rows, requires_grad=True); nb = native(base, True)
                self.gradients(model, oracle, model(nb.transpose(0, 1), native(ids)), oracle(base.transpose(0, 1), ids), [('x', nb, base)])
                for missing in (1, 4): self.assertEqual(model.weight.tensor().grad.narrow(0, missing, 1).tolist(), [[[0.] * 4] * 3])

    def test_dictionary_mixed_width_missing_unknown_and_empty_fields_all_gradients(self):
        for bias in (False, True):
            for empty in (False, True):
                widths = dict(alpha=3, beta=2, gamma=4)
                model, oracle = HeteroDictLinear(widths, 5, bias=bias), TorchDict(widths, 5, bias=bias)
                self.copy(model, oracle)
                alpha = torch.randn(3, 0 if empty else 4, requires_grad=True)
                gamma = torch.randn(0 if empty else 2, 4, requires_grad=True)
                unknown = torch.randn(1, 3, requires_grad=True)
                na, ng, nu = native(alpha, True), native(gamma, True), native(unknown, True)
                actual = model(dict(unknown=nu, gamma=ng, alpha=na.transpose(0, 1)))
                expected = oracle(dict(unknown=unknown, gamma=gamma, alpha=alpha.transpose(0, 1)))
                self.assertEqual(list(actual), ['alpha', 'gamma'])
                self.assertEqual(list(actual), list(expected))
                self.gradients(model, oracle, fr.cat([value.flatten() for value in actual.values()], 0), torch.cat([value.flatten() for value in expected.values()]), [('alpha', na, alpha), ('gamma', ng, gamma), ('unknown', nu, unknown)])
                self.assertIsNone(model.lins['beta'].weight.tensor().grad)
                self.assertEqual(model({'unknown': object()}), {})
        for widths, keys in ((3, ['a', 'b']), ({'a': 3, 'b': 3}, ['b', 'a']), ({}, None)):
            model, oracle = HeteroDictLinear(widths, 4, keys), TorchDict(widths, 4, keys)
            self.assertEqual(model.in_channels, oracle.in_channels)
            self.assertEqual(list(model.types), list(oracle.types))
            self.assertEqual(model({}), {})
        declared = model.in_channels
        declared['outside'] = 99
        self.assertNotIn('outside', model.in_channels)

    def test_initializer_options_parameter_layout_bounds_reset_and_config(self):
        for initializer in (None, 'uniform', 'kaiming_uniform', 'glorot'):
            for bias_init in (None, 'zeros'):
                model = HeteroLinear(3, 4, 5, weight_initializer=initializer, bias_initializer=bias_init)
                oracle = TorchTyped(3, 4, 5, weight_initializer=initializer, bias_initializer=bias_init)
                self.assertEqual(model.weight.tensor().shape, list(oracle.weight.shape))
                bound = math.sqrt(6. / 7.) if initializer == 'glorot' else 3 ** -.5
                for layer in (model,):
                    self.assertTrue(all(abs(value) <= bound + 1e-7 for value in layer.weight.tensor().flatten().tolist()))
                    if bias_init == 'zeros': self.assertEqual(layer.bias.tensor().tolist(), [[0.] * 4] * 5)
                    else: self.assertTrue(any(value != 0. for value in layer.bias.tensor().flatten().tolist()))
                with fr.no_grad(): model.weight.tensor().copy_(fr.Tensor.ones([5, 3, 4]) * 17.)
                model.reset_parameters()
                self.assertTrue(all(abs(value) <= bound + 1e-7 for value in model.weight.tensor().flatten().tolist()))
                dictionary = HeteroDictLinear({'a': 3, 'b': 2}, 4, weight_initializer=initializer, bias_initializer=bias_init)
                for key, width in dictionary.in_channels.items():
                    limit = math.sqrt(6. / (width + 4)) if initializer == 'glorot' else width ** -.5
                    self.assertTrue(all(abs(value) <= limit + 1e-7 for value in dictionary.lins[key].weight.tensor().flatten().tolist()))
                    if bias_init == 'zeros': self.assertEqual(dictionary.lins[key].bias.tensor().tolist(), [0.] * 4)
        with self.assertRaises(AttributeError): model.is_sorted = True
        with self.assertRaises(AttributeError): dictionary.channel_spec = ()

    def test_many_dictionary_types_forward_matches_actual_unfused_pyg(self):
        widths = {str(i): 2 + i % 3 for i in range(12)}
        model, oracle = HeteroDictLinear(widths, 3), TorchDict(widths, 3)
        self.copy(model, oracle)
        inputs = {key: torch.randn(i % 4, width, requires_grad=True) for i, (key, width) in enumerate(reversed(list(widths.items())))}
        values = {key: native(value, True) for key, value in inputs.items()}
        actual, expected = model(values), oracle(inputs)
        self.assertEqual(list(actual), list(widths))
        self.gradients(model, oracle, fr.cat([value.flatten() for value in actual.values()], 0), torch.cat([value.flatten() for value in expected.values()]), [(key, values[key], value) for key, value in inputs.items()])

    def test_serialized_typed_and_dictionary_alias_optimizer_generator_continuation(self):
        for typed in (False, True):
            def factory(changed=False):
                if typed:
                    model = HeteroLinear(3, 4, 5, is_sorted=changed)
                    model.alias = model.weight
                else:
                    model = HeteroDictLinear({'a': 3, 'b': 3}, 4, bias_initializer='zeros' if changed else None)
                    model.lins['b'].weight = model.lins['a'].weight
                return model
            def step(model, optimizer, rng):
                if typed: output = model(fr.Tensor.randn([7, 3], generator=rng), native(self.types))
                else: output = fr.cat(list(model(dict(b=fr.Tensor.randn([3, 3], generator=rng), a=fr.Tensor.randn([4, 3], generator=rng))).values()), 0)
                loss = output.square().mean()
                values = output.tolist(), loss.item()
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                return values
            model, rng = factory(), fr.Generator(373)
            optimizer = fr.optim.AdamW(model.parameters(), lr=.003)
            step(model, optimizer, rng)
            saved = fr.checkpoint.snapshot_training(model, optimizers={'hetero': optimizer}, generator=rng, step=1)
            with tempfile.TemporaryDirectory() as directory:
                saved.save(directory + '/state'); saved = fr.checkpoint.load(directory + '/state')
            expected = [step(model, optimizer, rng) for _ in range(3)]
            other, other_rng = factory(), fr.Generator(1)
            other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
            saved.restore(other, optimizers={'hetero': other_opt}, generator=other_rng)
            if typed: self.assertIs(other.alias, other.weight)
            else: self.assertIs(other.lins['a'].weight, other.lins['b'].weight)
            self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
            left = fr.checkpoint.snapshot_training(model, optimizers={'hetero': optimizer}, generator=rng, step=4).tensors()
            right = fr.checkpoint.snapshot_training(other, optimizers={'hetero': other_opt}, generator=other_rng, step=4).tensors()
            self.assertEqual(set(left), set(right))
            for key in left: self.assertEqual((left[key].dtype, left[key].shape, left[key].tolist()), (right[key].dtype, right[key].shape, right[key].tolist()), key)
            changed = factory(True); before = [p.tensor().tolist() for p in changed.parameters()]
            with self.assertRaises(ValueError): saved.restore(changed, optimizers={'hetero': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
            self.assertEqual(before, [p.tensor().tolist() for p in changed.parameters()])

    def test_metadata_unknown_types_sorted_contract_capacity_and_capture(self):
        model, x = HeteroLinear(3, 4, 5), fr.Tensor.ones([7, 3])
        for ids in ([0] * 6, [0] * 6 + [5], [0] * 6 + [-1], fr.Tensor.ones([7])):
            with self.assertRaises((ValueError, TypeError)): model(x, ids)
        with self.assertRaises(ValueError): HeteroLinear(3, 4, 5, True)(x, native(self.types))
        with self.assertRaises(ValueError): model(fr.Tensor.ones([7, 2]), native(self.types))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, native(self.types)))
        for options in (dict(weight_initializer='bad'), dict(bias_initializer='bad'), dict(bias=1)):
            with self.assertRaises((ValueError, TypeError)): HeteroLinear(3, 4, 5, **options)
        with self.assertRaises(NotImplementedError): HeteroLinear(3, 4, 5, unknown=True)
        for width in (0, -1):
            with self.assertRaises(ValueError): HeteroLinear(width, 4, 5)
        for widths, types_ in ((3, None), ({'a': 3}, ['b']), ({'a.b': 3}, None), ({'a': -1}, None), (3, ['a', 'a'])):
            with self.assertRaises(ValueError): HeteroDictLinear(widths, 4, types_)
        dictionary = HeteroDictLinear({'a': 3, 'b': 2}, 4)
        with patch.object(dictionary.lins['a'], 'forward', side_effect=AssertionError('first field ran before invalid later field')):
            with self.assertRaises(ValueError): dictionary(dict(a=fr.Tensor.ones([2, 3]), b=fr.Tensor.ones([2, 3])))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: dictionary(dict(a=x)))
        with patch.object(fr.Tensor, 'zeros', side_effect=AssertionError('parameters allocated')):
            for count in (sys.maxsize // 8 + 1, sys.maxsize // 4 + 1):
                with self.assertRaisesRegex(ValueError, 'capacity'): HeteroLinear(1, 1, count)
        with patch('ferro._graph_hetero_linear.Linear', side_effect=AssertionError('parameters allocated')):
            with self.assertRaisesRegex(ValueError, 'capacity'): HeteroDictLinear({'a': 1, 'b': sys.maxsize // 4 + 1}, 1)


if __name__ == '__main__': unittest.main(verbosity=2)
