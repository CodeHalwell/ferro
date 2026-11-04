"""Actual PyG MessageNorm references through public native vector norms."""
import sys
import unittest

import torch
from torch_geometric.nn import MessageNorm as TorchMessageNorm
import ferro as fr
from ferro.graph import MessageNorm


def native(value, grad=False):
    return fr.Tensor(value.detach().reshape(-1).tolist(), list(value.shape)).requires_grad_(grad)


class MessageNormTests(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=4e-5, rtol=5e-4)

    def compare(self, learned, x, msg, p):
        model, reference = MessageNorm(learned), TorchMessageNorm(learned)
        with fr.no_grad(): model.scale.tensor().copy_(fr.Tensor([.7], [1]))
        with torch.no_grad(): reference.scale.fill_(.7)
        nx, nm = native(x, True), native(msg, True)
        actual, expected = model(nx, nm, p), reference(x, msg, p)
        self.close(actual, expected)
        probe = torch.randn(expected.shape)
        (actual * native(probe)).sum().backward(); (expected * probe).sum().backward()
        for left, right in ((nx, x), (nm, msg), (model.scale.tensor(), reference.scale)):
            if right.grad is None: self.assertIsNone(left.grad)
            else: self.close(left.grad, right.grad)

    def test_all_norm_orders_learned_fixed_scale_and_all_gradients(self):
        torch.manual_seed(1613)
        for p in (0., .5, 1., 2., 3., -2., float('inf'), -float('inf')):
            for learned in (False, True):
                with self.subTest(p=p, learned=learned):
                    x = (torch.rand(4, 3) * 2 + .2).requires_grad_(True)
                    msg = (torch.rand(4, 5) * 2 + .3).requires_grad_(True)
                    self.compare(learned, x, msg, p)

    def test_zero_norms_empty_rows_widths_and_extremum_ties(self):
        for rows, source_width, message_width in ((0, 3, 2), (3, 0, 2), (3, 2, 0), (3, 2, 2)):
            for p in (0., 1., 2.):
                self.compare(True, torch.zeros(rows, source_width, requires_grad=True),
                             torch.zeros(rows, message_width, requires_grad=True), p)
        self.compare(True, torch.tensor([[2., -2., 1.]], requires_grad=True),
                     torch.tensor([[3., -3., 1., -1.]], requires_grad=True), float('inf'))

    def test_scale_freeze_unfreeze_and_reset(self):
        model, reference = MessageNorm(False), TorchMessageNorm(False)
        model.scale.trainable = True
        reference.scale.requires_grad_(True)
        x, msg = torch.tensor([[1., 2.]], requires_grad=True), torch.tensor([[2., 1.]], requires_grad=True)
        nx, nm = native(x, True), native(msg, True)
        model(nx, nm).sum().backward(); reference(x, msg).sum().backward()
        self.close(model.scale.tensor().grad, reference.scale.grad)
        with fr.no_grad(): model.scale.tensor().copy_(fr.Tensor([4.], [1]))
        model.reset_parameters()
        self.assertEqual(model.scale.tensor().tolist(), [1.])

    def test_native_optimizer_checkpoint_flags_and_configuration(self):
        model, rng = MessageNorm(True), fr.Generator(139)
        opt = fr.optim.AdamW(model.parameters(), lr=.01)
        def step(current, optimizer, generator):
            x = fr.Tensor.rand([4, 3], generator=generator)
            msg = fr.Tensor.rand([4, 5], generator=generator)
            loss = current(x, msg).square().mean()
            value = loss.item()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return value
        step(model, opt, rng)
        saved = fr.checkpoint.snapshot_training(model, optimizers={'message': opt}, generator=rng, step=1)
        expected = [step(model, opt, rng) for _ in range(3)]
        other, other_rng = MessageNorm(True), fr.Generator(999)
        other_opt = fr.optim.AdamW(other.parameters(), lr=.8)
        other.scale.trainable = False
        saved.restore(other, optimizers={'message': other_opt}, generator=other_rng)
        self.assertTrue(other.scale.trainable)
        self.assertEqual(expected, [step(other, other_opt, other_rng) for _ in range(3)])
        self.assertEqual(model.scale.tensor().tolist(), other.scale.tensor().tolist())
        changed = MessageNorm(False)
        with self.assertRaises(ValueError): saved.restore(changed, optimizers={'message': fr.optim.AdamW(changed.parameters())}, generator=fr.Generator(1))
        self.assertFalse(changed.scale.trainable)

    def test_strict_shapes_types_capacity_and_capture(self):
        with self.assertRaises(TypeError): MessageNorm(1)
        with self.assertRaises(NotImplementedError): MessageNorm(device='cuda:0')
        x, msg = fr.Tensor.ones([3, 2]), fr.Tensor.ones([3, 4])
        model = MessageNorm()
        for call in (lambda: model(x, fr.Tensor.ones([2, 4])), lambda: model(x, msg, float('nan')), lambda: model(x, msg, True),
                     lambda: model(fr.Tensor.ones([3]), msg), lambda: model(fr.Tensor.from_f64([1.] * 6, [3, 2]), msg)):
            with self.assertRaises((ValueError, TypeError)): call()
        huge = fr.Tensor.ones([1, 1]).expand(1, sys.maxsize // 4 + 1)
        with self.assertRaises(ValueError): model(huge, fr.Tensor.ones([1, 2]))
        with self.assertRaises(NotImplementedError): fr.capture(lambda: model(x, msg))


if __name__ == '__main__':
    unittest.main(verbosity=2)
