"""Public AdamW trajectories with intermittent and frozen parameter gradients."""
import math
import unittest

import ferro as fr
import torch


class IntermittentParameters(fr.nn.Module):
    def build(self):
        self.matrix = fr.nn.Parameter(fr.Tensor([0.8, -0.4, 0.2, 1.1], [2, 2]))
        self.bias = fr.nn.Parameter(fr.Tensor([-0.3, 0.7], [2]))
        self.late = fr.nn.Parameter(fr.Tensor([0.6, -0.9, 0.1], [3]))


class OptimizerSkipTests(unittest.TestCase):
    def test_adamw_skipped_frozen_and_delayed_gradients_match_torch(self):
        model = IntermittentParameters()
        parameters = dict(model.named_parameters())
        reference = {name: torch.nn.Parameter(torch.tensor(p.tensor().tolist(), dtype=torch.float32))
                     for name, p in parameters.items()}
        options = {'lr': 0.035, 'betas': (0.7, 0.9), 'eps': 1e-6, 'weight_decay': 0.17}
        optimizer = fr.optim.AdamW(model.parameters(), **options)
        torch_optimizer = torch.optim.AdamW(list(reference.values()), foreach=False, **options)
        rng = fr.Generator(17)
        schedule = (
            ((), ()),
            (('matrix',), ()),
            (('matrix', 'bias'), ()),
            (('matrix', 'late'), ('matrix',)),
            (('matrix', 'bias', 'late'), ('matrix',)),
            (('matrix',), ('matrix',)),
            ((), ('matrix',)),
            (('matrix', 'late'), ()),
            (('bias',), ()),
            (('matrix', 'bias', 'late'), ('late',)),
            (('late',), ()),
        )
        for step, (used, frozen) in enumerate(schedule):
            optimizer.zero_grad()
            torch_optimizer.zero_grad(set_to_none=True)
            for name, parameter in parameters.items():
                parameter.trainable = name not in frozen
                reference[name].requires_grad_(name not in frozen)
            loss, torch_loss = None, None
            for index, name in enumerate(used):
                tensor = parameters[name].tensor()
                count = math.prod(tensor.shape)
                coefficients = [(-1.) ** (j + step) * (0.3 + 0.07 * (j + index)) for j in range(count)]
                targets = [0.2 * ((step + j) % 3 - 1) for j in range(count)]
                residual = tensor * fr.Tensor(coefficients, list(tensor.shape)) - fr.Tensor(targets, list(tensor.shape))
                torch_residual = reference[name] * torch.tensor(coefficients).reshape(tensor.shape) - torch.tensor(targets).reshape(tensor.shape)
                term, torch_term = (residual * residual).sum(), (torch_residual * torch_residual).sum()
                loss = term if loss is None else loss + term
                torch_loss = torch_term if torch_loss is None else torch_loss + torch_term
            if loss is not None:
                loss.backward()
            if torch_loss is not None and torch_loss.requires_grad:
                torch_loss.backward()
            # Ferro freezing is an optimizer filter, while forward derivatives
            # remain available. Exercise an actual hidden gradient, not omission.
            for name in set(used).intersection(frozen):
                self.assertIsNotNone(parameters[name].tensor().grad)
                self.assertIsNone(parameters[name].grad)
            optimizer.step()
            torch_optimizer.step()
            optimizer.zero_grad()
            state = fr.checkpoint.snapshot_training(model, optimizers={'main': optimizer},
                                                    generator=rng, step=step + 1).tensors()
            with self.subTest(step=step, used=used, frozen=frozen):
                expected_steps = []
                for index, (name, parameter) in enumerate(parameters.items()):
                    expected = reference[name]
                    torch.testing.assert_close(torch.tensor(parameter.tensor().tolist()), expected.detach(),
                                               atol=1e-6, rtol=1e-5, msg='parameter ' + name)
                    torch_state = torch_optimizer.state.get(expected, {})
                    expected_steps.append(float(torch_state['step']) if torch_state else 0.)
                    for native_name, torch_name in (('m', 'exp_avg'), ('v', 'exp_avg_sq')):
                        actual = state[f'optim.main.{native_name}.{index}']
                        wanted = torch_state.get(torch_name, torch.zeros_like(expected)).reshape(-1)
                        self.assertEqual(actual.shape, [expected.numel()])
                        torch.testing.assert_close(torch.tensor(actual.tolist()), wanted,
                                                   atol=1e-6, rtol=1e-5, msg=native_name + ' for ' + name)
                self.assertEqual(state['optim.main.t'].shape, [len(parameters)])
                self.assertEqual(state['optim.main.t'].tolist(), expected_steps)


if __name__ == '__main__':
    unittest.main()
