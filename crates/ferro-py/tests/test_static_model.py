import os
import unittest
import ferro
import torch

class StaticModelTests(unittest.TestCase):
    def test_static_attention_fresh_values_and_snapshots(self):
        if not ferro.cuda_is_available() or not ferro.cuda_init(0):
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                self.fail("CUDA required")
            self.skipTest("CUDA unavailable")
        def tensor(t):
            return ferro.Tensor(t.flatten().tolist(), list(t.shape)).to("cuda")
        tx = torch.arange(24, dtype=torch.float32).sin().reshape(3,8)
        tw = torch.arange(64, dtype=torch.float32).cos().reshape(8,8)*0.1
        x,w = tensor(tx),tensor(tw)
        g,b = tensor(torch.ones(8)),tensor(torch.zeros(8))
        def forward():
            z=x.layer_norm(g,b)
            q=(z.matmul(w)+b).reshape([3,2,4]).transpose(0,1)
            a=q.bmm(q.transpose(1,2)).softmax(-1).bmm(q).transpose(0,1).reshape([3,8])
            y=x+a.matmul(w)
            return y+(y.layer_norm(g,b).matmul(w)+b).gelu().matmul(w)
        root=ferro.capture(forward)
        compiled=root.compile_fused()
        graph=compiled.prepare_static()
        graph.replay()
        old=graph.snapshot()
        old_values=old.tolist()
        for i in range(3):
            x.copy_(tensor(tx+i*0.1))
            w.copy_(tensor(tw+i*0.01))
            graph.replay()
            actual=graph.snapshot()
            torch.testing.assert_close(torch.tensor(actual.tolist()),torch.tensor(forward().tolist()),rtol=2e-4,atol=2e-5)
            self.assertEqual(old.tolist(),old_values)
            self.assertFalse(actual.requires_grad)
        self.assertEqual(graph.replay_count,4)
        del graph
        self.assertEqual(old.tolist(),old_values)

if __name__ == "__main__": unittest.main()
