"""Standard Module constructors coexist with annotation/build configuration."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr


class ConstructorModel(fr.nn.Module):
    weight: fr.nn.Parameter
    width: int

    def __init__(self, width, scale=1.0, *, bias=True):
        super().__init__()
        self.width, self.scale = width, scale
        self.weight = fr.nn.Parameter(fr.ones(width) * .4)
        self.bias = fr.nn.Parameter(fr.ones(width) * .1) if bias else None

    def forward(self, x):
        result = x * self.weight.tensor() * self.scale
        return result if self.bias is None else result + self.bias.tensor()


class TorchConstructorModel(torch.nn.Module):
    def __init__(self, width, scale=1.0, *, bias=True):
        super().__init__()
        self.width, self.scale = width, scale
        self.weight = torch.nn.Parameter(torch.ones(width) * .4)
        self.bias = torch.nn.Parameter(torch.ones(width) * .1) if bias else None

    def forward(self, x):
        result = x * self.weight * self.scale
        return result if self.bias is None else result + self.bias


class ModuleConstructorTests(unittest.TestCase):
    def same(self, actual, expected):
        actual = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(actual, expected.detach(), atol=1e-6, rtol=2e-6)

    def test_standard_constructor_registers_runtime_annotated_parameters_and_matches_training(self):
        for bias in (False, True):
            model, reference = ConstructorModel(3, 1.7, bias=bias), TorchConstructorModel(3, 1.7, bias=bias)
            self.assertEqual(inspect.signature(ConstructorModel), inspect.signature(TorchConstructorModel))
            self.assertEqual([name for name, _ in model.named_parameters()], [name for name, _ in reference.named_parameters()])
            optimizer = fr.optim.AdamW(model.parameters(), lr=.02)
            oracle = torch.optim.AdamW(reference.parameters(), lr=.02)
            x = torch.tensor([[.2, -.7, .4], [1.1, .3, -.6]])
            for _ in range(5):
                nx = fr.tensor(x.tolist(), requires_grad=True)
                tx = x.detach().requires_grad_(True)
                actual, expected = model(nx), reference(tx)
                self.same(actual, expected)
                actual.square().mean().backward()
                expected.square().mean().backward()
                self.same(nx.grad, tx.grad)
                for parameter, target in zip(model.parameters(), reference.parameters()):
                    self.same(parameter.grad, target.grad)
                optimizer.step(); oracle.step()
                optimizer.zero_grad(); oracle.zero_grad()
                for parameter, target in zip(model.parameters(), reference.parameters()):
                    self.same(parameter.tensor(), target)

    def test_custom_constructor_inheritance_and_nested_registration(self):
        class Derived(ConstructorModel):
            multiplier: float

            def __init__(self, width, multiplier=2.):
                super().__init__(width, bias=False)
                self.multiplier = multiplier
                self.child = fr.nn.Linear(width, width)

        class Inherited(Derived):
            pass

        model = Inherited(2, 3.)
        self.assertEqual(str(inspect.signature(Inherited)), '(width, multiplier=2.0)')
        self.assertEqual([name for name, _ in model.named_parameters()], ['weight', 'child.weight', 'child.bias'])
        model.multiplier = 4.
        self.assertEqual(model.multiplier, 4.)
        model.eval()
        self.assertFalse(model.child.training)

    def test_annotation_build_path_keeps_required_defaults_and_readonly_config(self):
        class Built(fr.nn.Module):
            width: int
            scale: float = 2.

            def build(self):
                self.weight = fr.nn.Parameter(fr.ones(self.width))

        self.assertEqual(str(inspect.signature(Built)), '(*, width, scale=2.0)')
        model = Built(width=3)
        self.assertEqual(model.weight.tensor().shape, [3])
        self.assertEqual(model.scale, 2.)
        with self.assertRaises(AttributeError):
            model.width = 4
        with self.assertRaises(TypeError):
            Built()

    def test_custom_constructor_attributes_do_not_become_implicit_configuration(self):
        class WithAttributes(fr.nn.Module):
            cache: list = []
            size: int

            def __init__(self, size):
                super().__init__()
                self.size = size
                self.cache = []

        left, right = WithAttributes(2), WithAttributes(3)
        left.cache.append(1)
        left.size = 4
        self.assertEqual((left.cache, right.cache, left.size), ([1], [], 4))

    def test_missing_super_and_invalid_base_arguments_reject(self):
        class Missing(fr.nn.Module):
            def __init__(self):
                self.weight = fr.nn.Parameter(fr.ones(1))

        class Invalid(fr.nn.Module):
            def __init__(self):
                super().__init__(unsupported=True)

        with self.assertRaisesRegex(AttributeError, 'super'):
            Missing()
        with self.assertRaises(TypeError):
            Invalid()

    def test_constructor_models_have_exact_training_checkpoint_continuation(self):
        model = ConstructorModel(2, .7)
        optimizer, generator = fr.optim.SGD(model.parameters(), lr=.02, momentum=.8), fr.Generator(7)
        x = fr.tensor([[.2, -.3], [1.1, .4]])
        def step(module, opt):
            loss = module(x).square().mean()
            loss.backward(); opt.step(); opt.zero_grad()
            return loss.tolist()
        step(model, optimizer)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
        target = ConstructorModel(2, .7)
        other = fr.optim.SGD(target.parameters(), lr=.02, momentum=.8)
        saved.restore(target, optimizers={'main': other}, generator=fr.Generator(11))
        for _ in range(4):
            self.assertEqual(step(model, optimizer), step(target, other))
            self.assertEqual([p.tensor().tolist() for p in model.parameters()], [p.tensor().tolist() for p in target.parameters()])

    def test_standard_constructor_fresh_process_resume_and_configuration_guard(self):
        model = ConstructorModel(2, .7)
        optimizer, generator = fr.optim.AdamW(model.parameters(), lr=.02), fr.Generator(7)
        x = fr.tensor([[.2, -.3], [1.1, .4]])
        def step(module, opt):
            loss = module(x).square().mean()
            loss.backward(); opt.step(); opt.zero_grad()
            return loss.tolist()
        step(model, optimizer)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer}, generator=generator)
        mismatch = ConstructorModel(2, .8)
        with self.assertRaises((ValueError, RuntimeError)):
            saved.restore(mismatch, optimizers={'main': fr.optim.AdamW(mismatch.parameters(), lr=.02)}, generator=fr.Generator(1))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'state')
            saved.save(path)
            expected = {'losses': [step(model, optimizer) for _ in range(4)],
                        'parameters': [p.tensor().tolist() for p in model.parameters()]}
            code = """import json,sys
import ferro as fr
sys.path.insert(0,sys.argv[1])
from test_module_constructors import ConstructorModel
model=ConstructorModel(2,.7)
optimizer=fr.optim.AdamW(model.parameters(),lr=.02)
fr.checkpoint.load(sys.argv[2]).restore(model,optimizers={'main':optimizer},generator=fr.Generator(11))
x=fr.tensor([[.2,-.3],[1.1,.4]])
losses=[]
for _ in range(4):
 loss=model(x).square().mean();losses.append(loss.tolist());loss.backward();optimizer.step();optimizer.zero_grad()
print(json.dumps({'losses':losses,'parameters':[p.tensor().tolist() for p in model.parameters()]}))
"""
            result = subprocess.run([sys.executable, '-c', code, str(Path(__file__).parent), path],
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout), expected)
