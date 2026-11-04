"""Explicit registration, traversal and initializer callbacks against Torch."""
import unittest

import torch
import ferro as fr


def tree(library):
    root, child, shared = library.nn.Module(), library.nn.Module(), library.nn.Module()
    root.add_module('first', child)
    root.register_parameter('late', library.nn.Parameter(library.ones(2)))
    root.register_parameter('optional', None)
    root.register_module('pending', None)
    child.add_module('nested', shared)
    child.register_parameter('direct', library.nn.Parameter(library.ones(2)*2.))
    shared.register_parameter('weight', library.nn.Parameter(library.ones(2)*3.))
    root.add_module('tied', shared)
    root.register_buffer('_counter', library.tensor(7, dtype=library.int64))
    child.register_buffer('state', library.ones(2))
    shared.register_buffer('same', child.state)
    return root


class ModuleRegistrationTests(unittest.TestCase):
    def test_explicit_parameters_children_none_slots_and_replacement(self):
        actual, expected = tree(fr), tree(torch)
        for method in ('named_parameters', 'named_buffers', 'named_modules', 'named_children'):
            self.assertEqual([name for name, _ in getattr(actual, method)()], [name for name, _ in getattr(expected, method)()])
        actual.register_parameter('optional', fr.nn.Parameter(fr.ones(2)))
        expected.register_parameter('optional', torch.nn.Parameter(torch.ones(2)))
        actual.optional = None; expected.optional = None
        for value, reference in ((fr.ones(2), torch.ones(2)), (fr.nn.Module(), torch.nn.Module())):
            with self.assertRaises(TypeError): actual.optional = value
            with self.assertRaises(TypeError): expected.optional = reference
        self.assertIsNone(actual.optional)
        actual.pending = fr.nn.Module(); expected.pending = torch.nn.Module()
        actual.pending = None; expected.pending = None
        with self.assertRaises(TypeError): actual.pending = fr.ones(2)
        with self.assertRaises(TypeError): expected.pending = torch.ones(2)
        del actual.optional; del expected.optional
        actual.add_module('optional', fr.nn.Module()); expected.add_module('optional', torch.nn.Module())
        self.assertEqual([name for name, _ in actual.named_modules()], [name for name, _ in expected.named_modules()])

    def test_prefix_recurse_duplicate_and_direct_before_children_order(self):
        actual, expected = tree(fr), tree(torch)
        for prefix in ('', 'model', 'outer.'):
            for recurse in (False, True):
                for duplicate in (False, True):
                    for method in ('named_parameters', 'named_buffers'):
                        args = dict(prefix=prefix, recurse=recurse, remove_duplicate=not duplicate)
                        self.assertEqual([name for name, _ in getattr(actual, method)(**args)], [name for name, _ in getattr(expected, method)(**args)])
        self.assertEqual([name for name, _ in actual.named_parameters()], ['late', 'first.direct', 'first.nested.weight'])
        for recurse in (False, True):
            self.assertEqual(len(list(actual.parameters(recurse))), len(list(expected.parameters(recurse))))
            self.assertEqual(len(list(actual.buffers(recurse))), len(list(expected.buffers(recurse))))

    def test_state_categories_and_explicit_buffer_parameter_registration(self):
        actual, expected = fr.nn.Module(), torch.nn.Module()
        a, b = fr.nn.Parameter(fr.ones(2)), torch.nn.Parameter(torch.ones(2))
        actual.register_buffer('value', a); expected.register_buffer('value', b)
        self.assertEqual(list(dict(actual.named_parameters())), list(dict(expected.named_parameters())))
        actual.register_buffer('value', fr.zeros(2), persistent=False)
        expected.register_buffer('value', torch.zeros(2), persistent=False)
        self.assertEqual(list(actual.state_dict()), list(expected.state_dict()))
        actual.value = a; expected.value = b
        self.assertEqual(list(dict(actual.named_buffers())), list(dict(expected.named_buffers())))
        self.assertEqual(list(dict(actual.named_parameters())), list(dict(expected.named_parameters())))
        actual.child = fr.nn.Module(); expected.child = torch.nn.Module()
        actual.child = fr.nn.Parameter(fr.ones(2)); expected.child = torch.nn.Parameter(torch.ones(2))
        self.assertEqual(list(dict(actual.named_modules())), list(dict(expected.named_modules())))
        self.assertEqual(list(actual.state_dict()), list(expected.state_dict()))
        actual.register_parameter('_private', fr.nn.Parameter(fr.ones(1)))
        expected.register_parameter('_private', torch.nn.Parameter(torch.ones(1)))
        self.assertEqual(list(actual.state_dict()), list(expected.state_dict()))
        actual.early = fr.nn.Module(); expected.early = torch.nn.Module()
        actual.tail = fr.nn.Parameter(fr.ones(1)); expected.tail = torch.nn.Parameter(torch.ones(1))
        actual.early = fr.nn.Parameter(fr.ones(1)); expected.early = torch.nn.Parameter(torch.ones(1))
        self.assertEqual(list(dict(actual.named_parameters())), list(dict(expected.named_parameters())))

    def test_apply_matches_postorder_with_tied_children_and_grad_mode(self):
        actual, expected = tree(fr), tree(torch)
        a_names = {id(value): name for name, value in actual.named_modules()}
        b_names = {id(value): name for name, value in expected.named_modules()}
        a_calls, b_calls = [], []
        def update(library, names, calls):
            def callback(module):
                calls.append((names[id(module)], library.is_grad_enabled()))
                with library.no_grad():
                    for parameter in module.parameters(recurse=False):
                        parameter.fill_(len(calls)*.25)
            return callback
        self.assertIs(actual.apply(update(fr, a_names, a_calls)), actual)
        self.assertIs(expected.apply(update(torch, b_names, b_calls)), expected)
        self.assertEqual(a_calls, b_calls)
        for (name, value), (key, reference) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertEqual(name, key)
            torch.testing.assert_close(torch.tensor(value.tolist()), reference.detach())
        self.assertTrue(fr.is_grad_enabled())

    def test_invalid_names_types_cycles_and_structure_locks_are_prewrite(self):
        actual = tree(fr)
        before = (tuple(actual.__dict__), tuple(actual._registered), tuple(actual._buffers))
        for name in ('', 'bad.name', 3):
            for method, value in (('register_parameter', None), ('add_module', None), ('register_buffer', None)):
                with self.assertRaises((TypeError, ValueError)): getattr(actual, method)(name, value)
        with self.assertRaises(TypeError): actual.register_parameter('invalid', fr.ones(2))
        with self.assertRaises(TypeError): actual.add_module('invalid', fr.ones(2))
        with self.assertRaises(ValueError): actual.add_module('cycle', actual)
        with self.assertRaises(ValueError): actual.register_parameter('first', fr.nn.Parameter(fr.ones(2)))
        with self.assertRaises(ValueError): actual.register_buffer('late', fr.ones(2))
        self.assertEqual((tuple(actual.__dict__), tuple(actual._registered), tuple(actual._buffers)), before)
        actual._structure_locked = True
        for method, value in (('register_parameter', None), ('add_module', None), ('register_buffer', None)):
            with self.assertRaises((RuntimeError, ValueError)): getattr(actual, method)('new', value)
        self.assertNotIn('new', actual.__dict__)

    def test_checkpoint_binds_none_slot_categories_and_restores_stable_state(self):
        class State(fr.nn.Module):
            pass
        source, target = State(), State()
        for module in (source, target):
            module.register_parameter('optional', None)
            module.register_module('child', None)
            module.register_parameter('weight', fr.nn.Parameter(fr.ones(2)))
        with fr.no_grad(): source.weight.fill_(3.)
        saved = fr.checkpoint.snapshot_training(source, optimizers={}, generator=fr.Generator(7))
        identity = target.weight._id()
        saved.restore(target, optimizers={}, generator=fr.Generator(11))
        self.assertEqual(target.weight.tolist(), [3., 3.])
        self.assertEqual(target.weight._id(), identity)
        wrong = State()
        wrong.register_module('optional', None)
        wrong.register_module('child', None)
        wrong.register_parameter('weight', fr.nn.Parameter(fr.zeros(2)))
        with self.assertRaises(ValueError): saved.restore(wrong, optimizers={}, generator=fr.Generator(11))
        self.assertEqual(wrong.weight.tolist(), [0., 0.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
