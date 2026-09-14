"""CPU-only additional public package contracts; no mocks."""
import inspect
import unittest
from typing import ClassVar
import ferro
from ferro import nn
from ferro.training import Trainer


class AdditionalContracts(unittest.TestCase):
    def test_functional_higher_order_and_live_seed(self):
        from ferro.autograd import grad, vjp
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        weight = ferro.Tensor([3.0], [1]).requires_grad_(True)
        y = weight * x * x
        dx, = grad(y, [x], create_graph=True)
        dw, = grad(dx, [weight])
        self.assertEqual(dx.tolist(), [12.0])
        self.assertEqual(dw.tolist(), [4.0])
        self.assertIsNone(x.grad)
        seed = ferro.Tensor([5.0], [1]).requires_grad_(True)
        vx, = vjp(y, [x], seed, create_graph=True)
        ds, = grad(vx, [seed])
        self.assertEqual(ds.tolist(), [12.0])

    def test_higher_order_unsupported_is_explicit(self):
        from ferro.autograd import grad
        x = ferro.Tensor([2.0], [1]).requires_grad_(True)
        with self.assertRaisesRegex((ValueError, NotImplementedError), 'unsupported|create_graph'):
            grad(x.relu(), [x], create_graph=True)

    def test_registration_follows_mixed_assignment_order(self):
        class Mixed(nn.Module):
            def build(self):
                self.child = nn.Linear(1, 1)
                self.weight = nn.Parameter(ferro.Tensor.ones([1]))
        self.assertEqual(list(dict(Mixed().named_parameters())), ['child.weight', 'child.bias', 'weight'])

    def test_backward_with_seed(self):
        x = ferro.Tensor([2.0, 3.0], [2]).requires_grad_(True)
        (x * x).backward_with(ferro.Tensor([2.0, 4.0], [2]))
        self.assertEqual(x.grad.tolist(), [8.0, 24.0])

    def test_config_signature_and_defaults(self):
        class Config(nn.Module):
            width: int
            version: ClassVar[int] = 4
            _private: int = 8
        self.assertEqual(list(inspect.signature(Config).parameters), ['width'])
        c = Config(width=2)
        with self.assertRaises(AttributeError):
            c.width = 3
        with self.assertRaises(TypeError):
            class Bad(nn.Module):
                values: list = []
        with self.assertRaises(TypeError):
            class BadInit(nn.Module):
                def __init__(self):
                    pass

    def test_alias_paths_replacement_deletion_and_cycles(self):
        class Parent(nn.Module):
            def build(self):
                self.a = nn.Linear(1, 1)
                self.b = self.a
        m = Parent()
        self.assertEqual(len(list(m.named_parameters())), 2)
        self.assertEqual(len(list(m.named_parameters(remove_duplicate=False))), 4)
        with self.assertRaises(ValueError):
            m.a.parent = m
        del m.b
        old = m.a.weight
        m.a.weight = nn.Parameter(ferro.Tensor.ones([1, 1]))
        self.assertIsNot(list(m.parameters())[0], old)
        del m.a.weight
        self.assertEqual(list(dict(m.named_parameters())), ['a.bias'])

    def test_iterator_and_reporting_validation_before_update(self):
        m = nn.Linear(1, 1)
        step = lambda model, batch: {}
        with self.assertRaises(ValueError):
            Trainer(m, train_step=step).fit(iter([1]), epochs=2)
        with self.assertRaises(TypeError):
            Trainer(m, train_step=lambda m, b: 1).fit([1])
        self.assertTrue(m.training)
        with self.assertRaises(RuntimeError):
            Trainer(m, train_step=lambda m, b: setattr(m, 'child', nn.Linear(1, 1))).fit([1])
        self.assertFalse(m._structure_locked)

    def test_sgd_duplicate_native_parameters_and_input_ownership(self):
        from ferro.optim import SGD
        source = ferro.Tensor.ones([1])
        p = nn.Parameter(source)
        optimizer = SGD([p, p], lr=0.1)
        p.tensor().sum().backward()
        optimizer.step()
        self.assertAlmostEqual(p.tensor().item(), 0.9, places=6)
        self.assertEqual(source.item(), 1.0)

    def test_no_grad_does_not_disable_explicit_capture(self):
        x = ferro.Tensor.ones([1]).requires_grad_(True)
        with ferro.no_grad():
            y = ferro.capture(lambda: x * x)
            self.assertFalse(y.requires_grad)
            compiled = y.compile_fused()
        self.assertEqual(compiled.replay().tolist(), [1.0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
