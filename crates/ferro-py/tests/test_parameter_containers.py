"""Parameter containers, registration, aliases and state against actual Torch."""
from collections import OrderedDict
import unittest

import torch
import ferro as fr


class ContainerModel(fr.nn.Module):
    def build(self):
        self.weights = fr.nn.ParameterList([fr.tensor([.4, -.7]), None, 2])
        self.extra = fr.nn.ParameterDict(OrderedDict([('_optional', None), ('bias', fr.tensor([.1, .2])), ('scale', .5)]))

    def forward(self, x):
        return x * self.weights[0].tensor() + self.extra['bias'].tensor()


class ParameterContainers(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(expected.shape), expected.detach(), atol=2e-6, rtol=2e-6)

    def assert_parameters(self, actual, expected):
        left, right = list(actual.named_parameters()), list(expected.named_parameters())
        self.assertEqual([name for name, _ in left], [name for name, _ in right])
        for (_, parameter), (_, reference) in zip(left, right):
            self.close(parameter.tensor(), reference)

    def test_list_converts_tensors_and_preserves_registered_aliases(self):
        source = fr.tensor([.2, .7])
        model = fr.nn.ParameterList([source, None, 3])
        reference = torch.nn.ParameterList([torch.tensor([.2, .7]), None, 3])
        self.assertFalse(source.requires_grad)
        self.assertIsInstance(model[0], fr.nn.Parameter)
        self.assertIsNone(model[1])
        self.assertEqual(model[2], reference[2])
        model.append(model[0])
        reference.append(reference[0])
        self.assertIs(model[0], model[-1])
        self.assertEqual(len(model), len(reference))
        self.assert_parameters(model, reference)
        self.assertEqual([name for name, _ in model.named_parameters(remove_duplicate=False)],
                         [name for name, _ in reference.named_parameters(remove_duplicate=False)])
        with fr.no_grad():
            model[0].tensor().copy_(fr.ones(2))
        self.assertEqual(source.tolist(), [.20000000298023224, .699999988079071])

    def test_list_slice_assignment_extension_and_errors(self):
        model = fr.nn.ParameterList(fr.tensor([float(index)]) for index in range(4))
        reference = torch.nn.ParameterList(torch.tensor([float(index)]) for index in range(4))
        selected, expected = model[::-2], reference[::-2]
        self.assertIs(selected[0], model[-1])
        self.assert_parameters(selected, expected)
        model[-1], reference[-1] = fr.tensor([7.]), torch.tensor([7.])
        model += [model[0], None]
        reference += [reference[0], None]
        self.assert_parameters(model, reference)
        for index in (-7, 6):
            with self.assertRaises(IndexError):
                model[index]
        with self.assertRaises(TypeError):
            model[0] = 3
        with self.assertRaises(TypeError):
            reference[0] = 3
        self.assert_parameters(model, reference)
        with self.assertRaises(TypeError):
            model.extend(fr.ones(2))
        self.assertIs(model.extend([]), model)
        with self.assertRaises(RuntimeError):
            model(fr.ones(1))

    def test_dict_order_matches_plain_mapping_and_ordered_updates(self):
        model = fr.nn.ParameterDict({'z': fr.tensor([3.]), 'a': fr.tensor([1.])})
        reference = torch.nn.ParameterDict({'z': torch.tensor([3.]), 'a': torch.tensor([1.])})
        self.assertEqual(list(model), list(reference))
        model.update(OrderedDict([('y', model['a']), ('b', fr.tensor([2.])), ('_private', None)]))
        reference.update(OrderedDict([('y', reference['a']), ('b', torch.tensor([2.])), ('_private', None)]))
        self.assertEqual(list(model.keys()), list(reference.keys()))
        self.assertEqual(list(reversed(model)), list(reversed(reference)))
        self.assertIs(model['y'], model['a'])
        self.assert_parameters(model, reference)
        self.assertEqual([key for key, _ in model.items()], [key for key, _ in reference.items()])
        self.assertEqual(len(list(model.values())), len(list(reference.values())))

    def test_dict_copy_pop_defaults_and_replacement(self):
        model = fr.nn.ParameterDict([('w', fr.tensor([.3])), ('missing', None)])
        reference = torch.nn.ParameterDict([('w', torch.tensor([.3])), ('missing', None)])
        copied, expected = model.copy(), reference.copy()
        self.assertIs(copied['w'], model['w'])
        self.assert_parameters(copied, expected)
        self.assertIs(model.setdefault('w', fr.tensor([9.])), model['w'])
        self.assertIs(model.setdefault('other', model['w']), model['w'])
        reference.setdefault('other', reference['w'])
        self.assertIsNone(model.pop('missing'))
        self.assertIsNone(reference.pop('missing'))
        self.assertEqual(model.popitem()[0], reference.popitem()[0])
        model['w'], reference['w'] = None, None
        model['w'], reference['w'] = fr.tensor([4.]), torch.tensor([4.])
        self.assert_parameters(model, reference)
        for key in ('', 'a.b', 'items'):
            with self.assertRaises((ValueError, KeyError)):
                model[key] = fr.tensor([1.])
        with self.assertRaises(TypeError):
            model['w'] = 4
        model.clear()
        self.assertEqual(len(model), 0)
        shared = fr.nn.Parameter(fr.ones(1))
        made = model.fromkeys(('a', 'b'), shared)
        self.assertIs(made['a'], made['b'])
        self.assertEqual(len(list(made.parameters())), 1)

    def test_optimizer_values_and_parameter_gradients_match_torch(self):
        model = ContainerModel()
        weight = torch.tensor([.4, -.7], requires_grad=True)
        bias = torch.tensor([.1, .2], requires_grad=True)
        optimizer = fr.optim.AdamW(model.parameters(), lr=.02)
        reference = torch.optim.AdamW([weight, bias], lr=.02)
        x = torch.tensor([[.2, 1.], [1.3, -.8]])
        value = fr.tensor(x.tolist())
        for _ in range(6):
            loss = model(value).square().mean()
            expected = (x * weight + bias).square().mean()
            loss.backward()
            expected.backward()
            self.close(model.weights[0].grad, weight.grad)
            self.close(model.extra['bias'].grad, bias.grad)
            optimizer.step()
            optimizer.zero_grad()
            reference.step()
            reference.zero_grad()
            self.close(model.weights[0].tensor(), weight)
            self.close(model.extra['bias'].tensor(), bias)

    def test_checkpoint_records_order_none_entries_and_private_values(self):
        model = ContainerModel()
        optimizer, generator = fr.optim.AdamW(model.parameters()), fr.Generator(7)
        model(fr.ones(2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
        target = ContainerModel()
        other = fr.optim.AdamW(target.parameters())
        saved.restore(target, optimizers={'main': other}, generator=fr.Generator(9))
        for left, right in zip(model.parameters(), target.parameters()):
            self.assertEqual(left.tensor().tolist(), right.tensor().tolist())
        target.extra['_optional'] = 3
        before = target.weights[0].tensor().tolist()
        with self.assertRaises(ValueError):
            saved.restore(target, optimizers={'main': other}, generator=fr.Generator(11))
        self.assertEqual(target.weights[0].tensor().tolist(), before)

    def test_trainer_locks_all_entries_and_releases_after_failure(self):
        from ferro.training import Trainer
        for update in (lambda model: model.weights.append(None),
                       lambda model: model.extra.setdefault('_new', None),
                       lambda model: model.extra.pop('_optional')):
            model = ContainerModel()
            before = (len(model.weights), list(model.extra))
            trainer = Trainer(model, train_step=lambda current, batch: update(current))
            with self.assertRaises(RuntimeError):
                trainer.fit([fr.ones(2)])
            self.assertEqual((len(model.weights), list(model.extra)), before)
            self.assertTrue(all(not module._structure_locked for module in model.modules()))

    def test_cleared_registered_slots_keep_kind_and_order(self):
        for dictionary in (False, True):
            if dictionary:
                model = fr.nn.ParameterDict([('a', fr.ones(1)), ('b', fr.ones(1))])
                reference = torch.nn.ParameterDict([('a', torch.ones(1)), ('b', torch.ones(1))])
                key = 'a'
            else:
                model = fr.nn.ParameterList([fr.ones(1), fr.ones(1)])
                reference = torch.nn.ParameterList([torch.ones(1), torch.ones(1)])
                key = 0
            model[key], reference[key] = None, None
            for value, wanted in ((3, 3), (fr.nn.Identity(), torch.nn.Identity())):
                with self.assertRaises(TypeError):
                    model[key] = value
                with self.assertRaises(TypeError):
                    reference[key] = wanted
            model[key], reference[key] = fr.ones(1), torch.ones(1)
            self.assert_parameters(model, reference)
        plain = fr.nn.ParameterList([None])
        plain[0] = 3
        self.assertEqual(plain[0], 3)
        former = fr.nn.ParameterList([fr.ones(1)])
        former[0] = None
        saved = fr.checkpoint.snapshot_training(former, optimizers={}, generator=fr.Generator(1))
        with self.assertRaises(ValueError):
            saved.restore(fr.nn.ParameterList([None]), optimizers={}, generator=fr.Generator(1))

    def test_nested_module_entries_follow_torch_parameter_order(self):
        model = fr.nn.ParameterList([fr.nn.Linear(1, 1), fr.ones(1)])
        reference = torch.nn.ParameterList([torch.nn.Linear(1, 1), torch.ones(1)])
        left, right = list(model.named_parameters()), list(reference.named_parameters())
        self.assertEqual([name for name, _ in left], [name for name, _ in right])
        model[0] = None
        reference[0] = None
        model[0] = fr.nn.Linear(1, 1)
        reference[0] = torch.nn.Linear(1, 1)
        self.assertEqual([name for name, _ in model.named_parameters()], [name for name, _ in reference.named_parameters()])
        for dictionary in (False, True):
            for clear in (False, True):
                if dictionary:
                    model = fr.nn.ParameterDict([('a', fr.nn.Identity()), ('b', fr.ones(1))])
                    reference = torch.nn.ParameterDict([('a', torch.nn.Identity()), ('b', torch.ones(1))])
                    key = 'a'
                else:
                    model = fr.nn.ParameterList([fr.nn.Identity(), fr.ones(1)])
                    reference = torch.nn.ParameterList([torch.nn.Identity(), torch.ones(1)])
                    key = 0
                if clear:
                    model[key], reference[key] = None, None
                model[key], reference[key] = fr.ones(1), torch.ones(1)
                self.assert_parameters(model, reference)


if __name__ == '__main__':
    unittest.main()
