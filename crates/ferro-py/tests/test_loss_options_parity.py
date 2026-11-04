"""Loss options, elementwise reductions and native adjoints against Torch."""
import unittest
import torch
import ferro as fr
from ferro.nn import functional as F


def native(value):
    return fr.tensor(value.detach().tolist(), dtype=str(value.dtype).removeprefix('torch.'), requires_grad=value.requires_grad)


class LossCompositions(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual is None, expected is None)
        if expected is None:
            return
        actual = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        tolerance = 3e-6 if expected.dtype == torch.float32 else 2e-10
        torch.testing.assert_close(actual, expected.detach(), atol=tolerance, rtol=tolerance, equal_nan=True)

    def compare(self, name, data, options):
        actual = [native(value) for value in data]
        result = getattr(F, name)(*actual, **options)
        expected = getattr(torch.nn.functional, name)(*data, **options)
        self.same(result, expected)
        result.sum().backward()
        expected.sum().backward()
        for a, b in zip(actual, data):
            self.same(a.grad, b.grad)

    def test_elementwise_losses_all_reductions_and_broadcast_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for reduction in ('none', 'sum', 'mean'):
                for name, extra in [('l1_loss', {}), ('huber_loss', {'delta': .7}), ('smooth_l1_loss', {'beta': .4}), ('smooth_l1_loss', {'beta': 0.}), ('soft_margin_loss', {}), ('hinge_embedding_loss', {'margin': .7})]:
                    with self.subTest(dtype=dtype, reduction=reduction, name=name, extra=extra):
                        a = torch.tensor([[-.3, .7, 1.3], [1.2, -.8, .2]], dtype=dtype, requires_grad=True)
                        b = torch.tensor([[.2, -.7, .6]], dtype=dtype, requires_grad=True)
                        if name == 'soft_margin_loss':
                            b = b.detach()
                        if name == 'hinge_embedding_loss':
                            b = torch.tensor([[1., -1., 2.]], dtype=dtype, requires_grad=True)
                        self.compare(name, [a, b], dict(extra, reduction=reduction))

    def test_kl_and_poisson_options(self):
        for dtype in (torch.float32, torch.float64):
            for reduction in ('none', 'sum', 'mean', 'batchmean'):
                for log_target in (False, True):
                    with self.subTest(dtype=dtype, reduction=reduction, log_target=log_target):
                        x = torch.tensor([[-.3, -.7], [-1.1, -.2]], dtype=dtype, requires_grad=True)
                        t = torch.tensor([[0., .4], [.2, .6]], dtype=dtype, requires_grad=True)
                        if log_target:
                            t = torch.tensor([[-.4, -.7], [-1.2, -.3]], dtype=dtype, requires_grad=True)
                        self.compare('kl_div', [x, t], dict(reduction=reduction, log_target=log_target))
            for log_input in (False, True):
                for full in (False, True):
                    with self.subTest(dtype=dtype, log_input=log_input, full=full):
                        x = torch.tensor([[.3, 1.2], [.7, 1.8]], dtype=dtype, requires_grad=True)
                        t = torch.tensor([[0., 1.], [2., 4.]], dtype=dtype, requires_grad=True)
                        self.compare('poisson_nll_loss', [x, t], dict(log_input=log_input, full=full, reduction='none'))

    def test_ranking_cosine_and_triplet(self):
        for dtype in (torch.float32, torch.float64):
            for reduction in ('none', 'sum', 'mean'):
                a = torch.tensor([[.2, -.4], [.7, .3]], dtype=dtype, requires_grad=True)
                b = torch.tensor([[-.6, .5], [.2, .9]], dtype=dtype, requires_grad=True)
                t = torch.tensor([[1., -1.], [1., -1.]], dtype=dtype, requires_grad=True)
                self.compare('margin_ranking_loss', [a, b, t], dict(margin=.3, reduction=reduction))
                a = torch.tensor([[.2, -.4], [.7, .3]], dtype=dtype, requires_grad=True)
                b = torch.tensor([[-.6, .5], [.2, .9]], dtype=dtype, requires_grad=True)
                t = torch.tensor([1., -1.], dtype=dtype, requires_grad=True)
                self.compare('cosine_embedding_loss', [a, b, t], dict(margin=.3, reduction=reduction))
                for swap in (False, True):
                    a = torch.tensor([[.2, -.4], [.7, .3]], dtype=dtype, requires_grad=True)
                    b = torch.tensor([[-.6, .5], [.2, .9]], dtype=dtype, requires_grad=True)
                    c = torch.tensor([[.4, .8], [-.5, .7]], dtype=dtype, requires_grad=True)
                    self.compare('triplet_margin_loss', [a, b, c], dict(margin=.7, p=1.7, swap=swap, reduction=reduction))

    def test_cosine_epsilon_and_hinge_boundary_adjoints(self):
        for dtype in (torch.float32, torch.float64):
            a = torch.tensor([[0., 0.], [1e-6, 0.]], dtype=dtype, requires_grad=True)
            b = a.detach().clone().requires_grad_(True)
            t = torch.ones(2, dtype=dtype)
            self.compare('cosine_embedding_loss', [a, b, t], dict(reduction='none'))
            a = torch.tensor([1., 0., -.3, 1.2], dtype=dtype, requires_grad=True)
            t = torch.tensor([-1., 1., 0., 2.], dtype=dtype, requires_grad=True)
            self.compare('hinge_embedding_loss', [a, t], dict(reduction='none'))

    def test_weighted_l1_and_huber_use_distinct_mean_normalizations(self):
        for dtype in (torch.float32, torch.float64):
            for name in ('l1_loss', 'huber_loss'):
                for reduction in ('none', 'sum', 'mean'):
                    x = torch.tensor([[.2, -.4], [.8, .7]], dtype=dtype, requires_grad=True)
                    t = torch.tensor([[.7, .9], [.3, .1]], dtype=dtype, requires_grad=True)
                    w = torch.tensor([[.3, 1.4], [.7, 2.]], dtype=dtype, requires_grad=True)
                    a, b, c = [native(v) for v in (x, t, w)]
                    actual = getattr(F, name)(a, b, weight=c, reduction=reduction)
                    expected = getattr(torch.nn.functional, name)(x, t, weight=w, reduction=reduction)
                    self.same(actual, expected)
                    actual.sum().backward()
                    expected.sum().backward()
                    for actual_value, reference in zip((a, b, c), (x, t, w)):
                        self.same(actual_value.grad, reference.grad)

    def test_empty_reductions_and_invalid_hyperparameters(self):
        for dtype in (torch.float32, torch.float64):
            for name in ('l1_loss', 'huber_loss', 'smooth_l1_loss', 'soft_margin_loss'):
                for reduction in ('none', 'sum', 'mean'):
                    x = torch.empty(0, 3, dtype=dtype, requires_grad=True)
                    t = torch.empty(0, 3, dtype=dtype)
                    self.compare(name, [x, t], dict(reduction=reduction))
        x = fr.ones(2, 3)
        with self.assertRaises(ValueError):
            F.huber_loss(x, x, delta=float('nan'))
        with self.assertRaises(ValueError):
            F.smooth_l1_loss(x, x, beta=float('nan'))
        with self.assertRaises(ValueError):
            F.triplet_margin_loss(x, fr.ones(3), x)
        with self.assertRaises(RuntimeError):
            F.soft_margin_loss(x, fr.ones(2, 3, requires_grad=True))

    def test_infinite_huber_threshold_and_zero_target_kl(self):
        for dtype in (torch.float32, torch.float64):
            for delta in (float('inf'), 1e40):
                x = torch.tensor([.2, -.4], dtype=dtype, requires_grad=True)
                t = torch.tensor([.7, .9], dtype=dtype, requires_grad=True)
                self.compare('huber_loss', [x, t], dict(delta=delta, reduction='none'))
            x = torch.tensor([float('inf'), float('nan'), -float('inf')], dtype=dtype, requires_grad=True)
            t = torch.zeros(3, dtype=dtype, requires_grad=True)
            self.compare('kl_div', [x, t], dict(reduction='none'))

    def test_modules_route_unreduced_options(self):
        x = torch.tensor([[.2, .7], [.4, .9]])
        t = torch.tensor([[.3, .8], [.1, .6]])
        cases = [
            ('L1Loss', {}), ('HuberLoss', {'delta': .7}), ('SmoothL1Loss', {'beta': .4}),
            ('KLDivLoss', {'log_target': True}),
            ('PoissonNLLLoss', {'log_input': False, 'full': True, 'eps': .01}),
            ('SoftMarginLoss', {}), ('HingeEmbeddingLoss', {'margin': .7}),
        ]
        for name, options in cases:
            with self.subTest(name=name):
                actual = getattr(fr.nn, name)(reduction='none', **options)
                expected = getattr(torch.nn, name)(reduction='none', **options)
                self.same(actual(native(x), native(t)), expected(x, t))
        a = torch.tensor([[.2, -.4], [.7, .3]])
        b = torch.tensor([[-.6, .5], [.2, .9]])
        c = torch.tensor([[.4, .8], [-.5, .7]])
        actual = fr.nn.TripletMarginLoss(margin=.7, p=1.7, swap=True, reduction='none')
        expected = torch.nn.TripletMarginLoss(margin=.7, p=1.7, swap=True, reduction='none')
        self.same(actual(native(a), native(b), native(c)), expected(a, b, c))


