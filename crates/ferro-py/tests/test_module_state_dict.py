"""Tensor state dictionaries, storage aliases and stable-slot weight loading."""
from collections import OrderedDict
import unittest

import torch
import ferro as fr


class StateModel(fr.nn.Module):
    def build(self):
        self.weight = fr.nn.Parameter(fr.tensor([.4, -.7]))
        self.tied = self.weight
        self.register_buffer('precise', fr.tensor([1.123456789123], dtype='float64'))
        self.register_buffer('indices', fr.tensor([(1 << 60) + 3], dtype='int64'))
        self.register_buffer('mask', fr.tensor([True, False]))
        self.register_buffer('optional', None)
        self.register_buffer('temporary', fr.ones(1), persistent=False)
        self.norm = fr.nn.BatchNorm1d(2)


class TorchStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.4, -.7]))
        self.tied = self.weight
        self.register_buffer('precise', torch.tensor([1.123456789123], dtype=torch.float64))
        self.register_buffer('indices', torch.tensor([(1 << 60) + 3], dtype=torch.int64))
        self.register_buffer('mask', torch.tensor([True, False]))
        self.register_buffer('optional', None)
        self.register_buffer('temporary', torch.ones(1), persistent=False)
        self.norm = torch.nn.BatchNorm1d(2)


def native(value):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.'))


class ModuleStateDictTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for name, value in expected.items():
            self.assertEqual(actual[name].dtype, str(value.dtype).removeprefix('torch.'))
            self.assertEqual(actual[name].shape, list(value.shape))
            tensor = torch.tensor(actual[name].tolist(), dtype=value.dtype).reshape(value.shape)
            torch.testing.assert_close(tensor, value.detach(), atol=0, rtol=0)

    def test_keys_types_metadata_optional_and_nonpersistent_buffers_match_torch(self):
        actual, expected = StateModel().state_dict(), TorchStateModel().state_dict()
        self.same(actual, expected)
        self.assertEqual(actual._metadata, expected._metadata)
        self.assertNotIn('optional', actual)
        self.assertNotIn('temporary', actual)

    def test_shallow_detach_aliases_and_keep_vars(self):
        model, expected = StateModel(), TorchStateModel()
        values, reference = model.state_dict(), expected.state_dict()
        self.assertFalse(values['weight'].requires_grad)
        self.assertTrue(values['weight']._shares_storage(model.weight.tensor()))
        self.assertTrue(values['weight']._shares_storage(values['tied']))
        with fr.no_grad():
            values['weight'].fill_(2.)
        with torch.no_grad():
            reference['weight'].fill_(2.)
        self.same(model.state_dict(), expected.state_dict())
        self.assertTrue(model.state_dict(keep_vars=True)['weight'].requires_grad)
        self.assertIs(model.state_dict(keep_vars=True)['precise'], model.precise)

    def test_prefix_and_destination_preserve_order_and_metadata(self):
        model, expected = StateModel(), TorchStateModel()
        actual, reference = OrderedDict(marker='value'), OrderedDict(marker='value')
        self.assertIs(model.state_dict(destination=actual, prefix='net.'), actual)
        self.assertIs(expected.state_dict(destination=reference, prefix='net.'), reference)
        self.assertEqual(list(actual), list(reference))
        self.same(OrderedDict((k, v) for k, v in actual.items() if k != 'marker'),
                  OrderedDict((k, v) for k, v in reference.items() if k != 'marker'))

    def test_load_casts_preserves_parameters_gradients_and_buffer_identity(self):
        model, expected = StateModel(), TorchStateModel()
        model.weight.tensor().square().sum().backward()
        expected.weight.square().sum().backward()
        parameter, buffer, grad = model.weight, model.precise, model.weight.grad
        reference = expected.state_dict()
        reference['weight'] = torch.tensor([1.23456789123, -2.3456789123], dtype=torch.float64)
        reference['tied'] = reference['weight']
        reference['precise'] = torch.tensor([3.141592653589793], dtype=torch.float64)
        reference['indices'] = torch.tensor([(1 << 61) + 7], dtype=torch.int64)
        result = model.load_state_dict(OrderedDict((k, native(v)) for k, v in reference.items()))
        reference_result = expected.load_state_dict(reference)
        self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))
        self.assertEqual(str(result), str(reference_result))
        self.assertIs(model.weight, parameter)
        self.assertIs(model.precise, buffer)
        self.assertEqual(model.weight.grad._id(), grad._id())
        self.same(model.state_dict(), expected.state_dict())

    def test_strict_and_partial_keys_and_late_errors_reject_atomically(self):
        model, expected = StateModel(), TorchStateModel()
        partial = OrderedDict(weight=fr.tensor([.8, .9]), extra=fr.ones(1))
        result = model.load_state_dict(partial, strict=False)
        reference = expected.load_state_dict({'weight': torch.tensor([.8, .9]), 'extra': torch.ones(1)}, strict=False)
        self.assertEqual(result.missing_keys, reference.missing_keys)
        self.assertEqual(result.unexpected_keys, reference.unexpected_keys)
        before = OrderedDict((k, v.tolist()) for k, v in model.state_dict().items())
        bad = OrderedDict(model.state_dict())
        bad['weight'] = fr.zeros(2)
        bad['norm.running_var'] = fr.ones(3)
        for mapping, strict in ((partial, True), (bad, True), (bad, False)):
            with self.assertRaises(RuntimeError):
                model.load_state_dict(mapping, strict=strict)
            self.assertEqual(OrderedDict((k, v.tolist()) for k, v in model.state_dict().items()), before)
        with self.assertRaises(NotImplementedError):
            model.load_state_dict(model.state_dict(), assign=True)

    def test_saved_forward_versions_and_invalid_values(self):
        model = StateModel()
        output = model.weight.tensor().square().sum()
        values = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        values['weight'] = fr.ones(2)
        model.load_state_dict(values)
        with self.assertRaises((ValueError, RuntimeError)):
            output.backward()
        bad = OrderedDict(model.state_dict())
        bad['weight'] = [1., 2.]
        with self.assertRaisesRegex(RuntimeError, 'expected a Tensor'):
            model.load_state_dict(bad)
        with self.assertRaises(TypeError):
            model.load_state_dict([])

    def test_module_traversal_preserves_order_and_duplicate_controls(self):
        model, expected = StateModel(), TorchStateModel()
        model.again, expected.again = model.norm, expected.norm
        self.assertEqual([name for name, _ in model.named_children()], [name for name, _ in expected.named_children()])
        self.assertEqual(list(model.children()), [model.norm])
        for duplicate in (False, True):
            self.assertEqual([name for name, _ in model.named_modules(remove_duplicate=duplicate)],
                             [name for name, _ in expected.named_modules(remove_duplicate=duplicate)])
        self.assertEqual([name for name, _ in model.named_modules(prefix='net')],
                         [name for name, _ in expected.named_modules(prefix='net')])

    def test_dotted_getters_validate_registered_objects(self):
        model = StateModel()
        self.assertIs(model.get_submodule(''), model)
        self.assertIs(model.get_submodule('norm'), model.norm)
        self.assertIs(model.get_parameter('norm.weight'), model.norm.weight)
        self.assertIs(model.get_buffer('norm.running_mean'), model.norm.running_mean)
        self.assertIsNone(model.get_buffer('optional'))
        self.assertIs(model.get_buffer('temporary'), model.temporary)
        for method, name in ((model.get_submodule, 'weight'), (model.get_parameter, 'norm'),
                             (model.get_parameter, 'norm.running_mean'), (model.get_buffer, 'weight'),
                             (model.get_buffer, 'not_present'), (model.get_submodule, 'norm.not_present')):
            with self.assertRaises(AttributeError):
                method(name)

    def test_live_cross_destination_aliases_follow_sequential_copy_semantics(self):
        for dtype in ('float32', 'float64', 'int64', 'bool'):
            model, reference = fr.nn.Module(), torch.nn.Module()
            first = [[1, 2], [3, 4]] if dtype != 'bool' else [[True, False], [False, True]]
            second = [[5, 6], [7, 8]] if dtype != 'bool' else [[False, True], [True, False]]
            model.register_buffer('first', fr.tensor(first, dtype=dtype))
            model.register_buffer('second', fr.tensor(second, dtype=dtype))
            reference.register_buffer('first', torch.tensor(first, dtype=getattr(torch, dtype)))
            reference.register_buffer('second', torch.tensor(second, dtype=getattr(torch, dtype)))
            for transpose in (False, True):
                a = model.first.transpose(0, 1) if transpose else model.first
                b = reference.first.transpose(0, 1) if transpose else reference.first
                model.load_state_dict({'first': model.second, 'second': a})
                reference.load_state_dict({'first': reference.second, 'second': b})
                self.same(model.state_dict(), reference.state_dict())
