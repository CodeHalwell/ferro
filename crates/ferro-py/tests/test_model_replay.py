import unittest
import ferro
import torch


class ModelReplayTests(unittest.TestCase):
    def test_layout_norm_and_backward(self):
        self.check_layout_norm_backward("cpu")

    def test_cuda_layout_norm_backward(self):
        if not ferro.cuda_is_available() or not ferro.cuda_init(0):
            self.skipTest("CUDA runtime unavailable")
        self.check_layout_norm_backward("cuda")

    def check_layout_norm_backward(self, device):
        values = torch.arange(24, dtype=torch.float32).sin().reshape(3, 8)
        x = ferro.Tensor(values.flatten().tolist(), [3, 8]).to(device).requires_grad_(True)
        w = ferro.Tensor([1.1] * 8, [8]).to(device).requires_grad_(True)
        b = ferro.Tensor([0.1] * 8, [8]).to(device).requires_grad_(True)
        tx = values.clone().requires_grad_(True)
        tw = torch.full((8,), 1.1).requires_grad_(True)
        tb = torch.full((8,), 0.1).requires_grad_(True)
        y = x.layer_norm(w, b).reshape([3, 2, 4]).transpose(0, 1).reshape([2, 12])
        ty = torch.nn.functional.layer_norm(tx, (8,), tw, tb).reshape(3, 2, 4).transpose(0, 1).reshape(2, 12)
        torch.testing.assert_close(torch.tensor(y.tolist()).reshape(2, 12), ty, rtol=1e-5, atol=1e-6)
        (y * y).sum().backward()
        (ty * ty).sum().backward()
        for f, t in [(x, tx), (w, tw), (b, tb)]:
            torch.testing.assert_close(torch.tensor(f.grad.tolist()).reshape(t.shape), t.grad, rtol=2e-3, atol=2e-5)

    def test_full_attention_and_mlp_current_input(self):
        for device in ['cpu', 'cuda']:
            if device == 'cuda' and (not ferro.cuda_is_available() or not ferro.cuda_init(0)):
                continue
            x = ferro.Tensor([i * 0.03 - 0.2 for i in range(24)], [3, 8]).to(device)
            w = ferro.Tensor([0.3 if i % 9 == 0 else 0.02 for i in range(64)], [8, 8]).to(device)
            g = ferro.Tensor([1.1] * 8, [8]).to(device)
            b = ferro.Tensor([0.1] * 8, [8]).to(device)
            def forward():
                z = x.layer_norm(g, b)
                q = (z.matmul(w) + b).reshape([3, 2, 4]).transpose(0, 1).reshape([2, 3, 4])
                k = q.transpose(1, 2).reshape([2, 4, 3])
                a = q.bmm(k).softmax(-1).bmm(q).transpose(0, 1).reshape([3, 8])
                y = x + a.matmul(w)
                return y + (y.layer_norm(g, b).matmul(w) + b).gelu().matmul(w)
            root = ferro.capture(forward)
            graph = root.compile_fused()
            self.assertGreaterEqual(graph.num_steps, 20)
            x.copy_(ferro.Tensor([0.7 - i * 0.04 for i in range(24)], [3, 8]))
            got = graph.replay()
            self.assertFalse(got.requires_grad)
            self.assertEqual(got.fusion_launches(), (0, 0))
            torch.testing.assert_close(torch.tensor(got.tolist()), torch.tensor(forward().tolist()), rtol=2e-4, atol=2e-5)
            self.assertNotEqual(got.tolist(), root.tolist())
            w.copy_(ferro.Tensor([0.1 if i % 9 == 0 else -0.03 for i in range(64)], [8, 8]))
            torch.testing.assert_close(torch.tensor(graph.replay().tolist()), torch.tensor(forward().tolist()), rtol=2e-4, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
