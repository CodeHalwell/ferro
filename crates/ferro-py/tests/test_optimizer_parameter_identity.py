"""Parameter Tensor value equality cannot replace optimizer slot identity."""
import unittest
import torch
import ferro as fr


class OptimizerParameterIdentityTests(unittest.TestCase):
    def test_equal_value_replacement_is_rejected_before_native_mutation(self):
        for values in (1., [[1., 2.], [3., 4.]]):
            with self.subTest(values=values):
                parameter = fr.nn.Parameter(fr.tensor(values))
                self.assertIsInstance(parameter, fr.Tensor)
                replacement = fr.nn.Parameter(fr.tensor(values))
                optimizer = fr.optim.SGD([parameter], lr=.1)
                parameter.square().sum().backward()
                config = optimizer._native.configurations()
                optimizer.param_groups[0]['params'] = [replacement]
                optimizer.param_groups[0]['lr'] = .2
                with self.assertRaisesRegex(ValueError, 'membership/order'):
                    optimizer.step()
                self.assertEqual(optimizer._native.configurations(), config)
                self.assertEqual(parameter.tolist(), replacement.tolist())
                self.assertIsNotNone(parameter.grad)
                self.assertIsNone(replacement.grad)

    def test_distinct_equal_value_matrix_groups_add_and_update_like_torch(self):
        values = [[1., 2.], [3., 4.]]
        parameters = [fr.nn.Parameter(fr.tensor(values)) for _ in range(2)]
        expected = [torch.nn.Parameter(torch.tensor(values)) for _ in range(2)]
        optimizer = fr.optim.SGD([parameters[0]], lr=.1, momentum=.5)
        oracle = torch.optim.SGD([expected[0]], lr=.1, momentum=.5)
        optimizer.add_param_group({'params': [parameters[1]], 'lr': .2})
        oracle.add_param_group({'params': [expected[1]], 'lr': .2})
        with self.assertRaisesRegex(ValueError, 'multiple groups'):
            optimizer.add_param_group({'params': [parameters[0]]})
        for _ in range(3):
            optimizer.zero_grad(); oracle.zero_grad()
            for parameter, target in zip(parameters, expected):
                parameter.square().sum().backward(); target.square().sum().backward()
            optimizer.step(); oracle.step()
            for parameter, target in zip(parameters, expected):
                torch.testing.assert_close(torch.tensor(parameter.tolist()), target.detach(), atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
