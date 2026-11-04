import inspect
import unittest

import torch
import ferro as fr

torch.set_num_threads(1)


class SGDDefaultsTests(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.shape, list(expected.shape))
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        torch.testing.assert_close(torch.tensor(actual.tolist(), dtype=expected.dtype), expected.detach(),
                                   rtol=0, atol=2e-7 if expected.dtype == torch.float32 else 0)

    def test_default_lr_and_parameter_order_match_actual_torch(self):
        actual = inspect.signature(fr.optim.SGD).parameters
        reference = inspect.signature(torch.optim.SGD).parameters
        self.assertEqual(list(actual), list(reference))
        for name in actual:
            self.assertEqual(actual[name].kind, reference[name].kind)
            self.assertEqual(actual[name].default, reference[name].default)
        for dtype in ('float32', 'float64'):
            with self.subTest(dtype=dtype):
                p = fr.nn.Parameter(fr.tensor([.5, -.25, 1.], dtype=dtype))
                q = torch.nn.Parameter(torch.tensor([.5, -.25, 1.], dtype=getattr(torch, dtype)))
                a, b = fr.optim.SGD([p]), torch.optim.SGD([q])
                for values in ([.25, -.5, 1.], [2., .125, -1.], [-.75, 1., .5]):
                    p.grad = fr.tensor(values, dtype=dtype)
                    q.grad = torch.tensor(values, dtype=q.dtype)
                    a.step(); b.step(); self.same(p, q)
                self.assertEqual(a.param_groups[0]['lr'], b.param_groups[0]['lr'])

    def test_per_group_lr_and_full_positional_options_follow_reference(self):
        for dtype in ('float32', 'float64'):
            for options in (None, (.03, .8, .2, .1, False), (.02, .9, 0., .04, True)):
                with self.subTest(dtype=dtype, options=options):
                    ps = [fr.nn.Parameter(fr.tensor([.5, -.25], dtype=dtype)) for _ in range(2)]
                    qs = [torch.nn.Parameter(torch.tensor([.5, -.25], dtype=getattr(torch, dtype))) for _ in range(2)]
                    if options is None:
                        a = fr.optim.SGD([{'params': [ps[0]], 'lr': .015}, {'params': [ps[1]], 'lr': .04}])
                        b = torch.optim.SGD([{'params': [qs[0]], 'lr': .015}, {'params': [qs[1]], 'lr': .04}])
                    else:
                        a, b = fr.optim.SGD(ps, *options), torch.optim.SGD(qs, *options)
                    for values in ([.5, -.25], [-.75, .125], [1., -2.]):
                        for p, q in zip(ps, qs):
                            p.grad = fr.tensor(values, dtype=dtype)
                            q.grad = torch.tensor(values, dtype=q.dtype)
                        a.step(); b.step()
                        for p, q in zip(ps, qs): self.same(p, q)

    def test_disabled_execution_flags_work_and_active_unsupported_flags_reject(self):
        for dtype in ('float32', 'float64'):
            p = fr.nn.Parameter(fr.tensor([.5], dtype=dtype))
            q = torch.nn.Parameter(torch.tensor([.5], dtype=getattr(torch, dtype)))
            a = fr.optim.SGD([p], foreach=False, fused=False, differentiable=False)
            b = torch.optim.SGD([q], foreach=False, fused=False, differentiable=False)
            p.grad = fr.tensor([.25], dtype=dtype); q.grad = torch.tensor([.25], dtype=q.dtype)
            a.step(); b.step(); self.same(p, q)
            identity, before = p._id(), p.tolist()
            for name in ('foreach', 'fused', 'differentiable'):
                with self.assertRaises(NotImplementedError): fr.optim.SGD([p], **{name: True})
                with self.assertRaises(TypeError): fr.optim.SGD([p], **{name: 'invalid'})
                self.assertEqual(p._id(), identity)
                self.assertEqual(p.tolist(), before)


if __name__ == '__main__':
    unittest.main()
