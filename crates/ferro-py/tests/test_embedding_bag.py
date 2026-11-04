"""Actual Torch dense EmbeddingBag values, adjoints, state and strict boundaries."""
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from ferro.nn import functional as F
import test_embedding_options as embedding_helpers

native = embedding_helpers.native


class EmbeddingBagTests(unittest.TestCase):
    same = embedding_helpers.EmbeddingOptionsTests.same

    def test_modes_offsets_fixed_bags_padding_and_all_dense_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ('sum', 'mean', 'max'):
                for padding in (None, 0, -1):
                    for fixed in (False, True):
                        for scale in ((False, True) if mode != 'max' else (False,)):
                            with self.subTest(dtype=dtype, mode=mode, padding=padding, fixed=fixed, scale=scale):
                                w = torch.tensor([[.2, -.7, 1.1], [.8, .1, -.3], [-.5, 1.3, .2], [1.7, -.4, .5]], dtype=dtype, requires_grad=True)
                                ids = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3])
                                if fixed: ids = ids.reshape(2, 4)
                                offsets = None if fixed else torch.tensor([0, 3, 3, 6, 8])
                                nw = native(w, True)
                                options = dict(mode=mode, padding_idx=padding, scale_grad_by_freq=scale, include_last_offset=True)
                                actual = F.embedding_bag(native(ids), nw, None if fixed else native(offsets), **options)
                                expected = torch.nn.functional.embedding_bag(ids, w, offsets, **options)
                                self.same(actual, expected)
                                seed = torch.arange(expected.numel(), dtype=dtype).reshape(expected.shape) * .13 + .2
                                (actual * native(seed)).sum().backward(); (expected * seed).sum().backward()
                                self.same(nw.grad, w.grad)

    def test_live_per_sample_weights_frequency_and_strided_operands(self):
        for dtype in (torch.float32, torch.float64):
            for scale in (False, True):
                for fixed in (False, True):
                    w = torch.arange(12, dtype=dtype).reshape(3, 4).t().detach().requires_grad_(True)
                    ids = torch.tensor([[0, 9, 0, 9, 0, 9], [1, 9, 2, 9, 2, 9]])[:, ::2]
                    samples = torch.tensor([[.2, 9., .4, 9., .7, 9.], [-.3, 9., 1.2, 9., .6, 9.]], dtype=dtype)[:, ::2].detach().requires_grad_(True)
                    nw = native(w.t()).transpose(0, 1).detach().requires_grad_(True)
                    ni = native(torch.tensor([[0, 9, 0, 9, 0, 9], [1, 9, 2, 9, 2, 9]]))[:, ::2]
                    np = native(samples, True)
                    if not fixed: ids, ni, samples_, np_ = ids.reshape(-1), ni.reshape([-1]), samples.reshape(-1), np.reshape([-1])
                    else: samples_, np_ = samples, np
                    offsets = None if fixed else torch.tensor([0, 3, 3])
                    no = None if fixed else native(torch.tensor([0, 99, 3, 99, 3, 99]))[::2]
                    options = dict(mode='sum', padding_idx=0, scale_grad_by_freq=scale)
                    actual = F.embedding_bag(ni, nw, no, per_sample_weights=np_, **options)
                    expected = torch.nn.functional.embedding_bag(ids, w, offsets, per_sample_weights=samples_, **options)
                    self.same(actual, expected)
                    seed = torch.arange(expected.numel(), dtype=dtype).reshape(expected.shape) * .2 + .1
                    (actual * native(seed)).sum().backward(); (expected * seed).sum().backward()
                    self.same(nw.grad, w.grad); self.same(np.grad, samples.grad)

    def test_current_torch_sorted_ordinal_frequency_rule_is_explicit(self):
        ids = fr.tensor([0, 0, 0, 1, 2, 2]); offsets = fr.tensor([0, 3])
        weight = fr.ones(3, 1, requires_grad=True)
        F.embedding_bag(ids, weight, offsets, mode='sum', scale_grad_by_freq=True).sum().backward()
        self.same(weight.grad, torch.tensor([[1.], [1./3.], [2./3.]]))

    def test_max_first_winner_ties_nan_infinity_and_signed_zero(self):
        for dtype in (torch.float32, torch.float64):
            for order in ([0, 1, 2, 3], [3, 2, 1, 0]):
                w = torch.tensor([[float('nan'), 4., -float('inf'), -0.], [2., float('nan'), -float('inf'), 0.], [3., 5., -float('inf'), 0.], [3., 5., -float('inf'), 0.]], dtype=dtype, requires_grad=True)
                nw = native(w, True); ids = torch.tensor(order); offsets = torch.tensor([0, 4])
                a = F.embedding_bag(native(ids), nw, native(offsets), mode='max')
                b = torch.nn.functional.embedding_bag(ids, w, offsets, mode='max')
                self.same(a, b)
                self.assertEqual(torch.signbit(torch.tensor(a.tolist()))[0, 3].item(), torch.signbit(b)[0, 3].item())
                a.sum().backward(); b.sum().backward(); self.same(nw.grad, w.grad)

    def test_empty_bags_zero_channels_and_empty_fixed_batch(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ('sum', 'mean', 'max'):
                for shape, ids, offsets, last in (((3, 2), [], [], False), ((0, 3), [], [0, 0], False), ((3, 0), [1, 2], [0, 1, 2], True), ((3, 2), [], [0], True), ((3, 2), [], [0, 0, 0], False)):
                    w = torch.empty(shape, dtype=dtype, requires_grad=True); nw = native(w, True)
                    ids, offsets = torch.tensor(ids, dtype=torch.int64), torch.tensor(offsets, dtype=torch.int64)
                    a = F.embedding_bag(native(ids), nw, native(offsets), mode=mode, include_last_offset=last)
                    b = torch.nn.functional.embedding_bag(ids, w, offsets, mode=mode, include_last_offset=last)
                    self.same(a, b); a.sum().backward(); b.sum().backward(); self.same(nw.grad, w.grad)
                w = torch.ones(3, 2, dtype=dtype, requires_grad=True); nw = native(w, True)
                a = F.embedding_bag(fr.zeros(0, 2, dtype='int64'), nw, mode=mode)
                b = torch.nn.functional.embedding_bag(torch.empty(0, 2, dtype=torch.int64), w, mode=mode)
                self.same(a, b); a.sum().backward(); b.sum().backward(); self.same(nw.grad, w.grad)

    def test_renorm_all_orders_repeated_calls_and_invalid_metadata_nonmutation(self):
        for dtype in (torch.float32, torch.float64):
            for order in (0., 1., 2., 3., -2., float('inf'), -float('inf')):
                w = torch.tensor([[3., 4.], [-2., .5], [0., 0.]], dtype=dtype, requires_grad=True); nw = native(w, True)
                ids, offsets = torch.tensor([0, 0, 1, 2]), torch.tensor([0, 2, 4])
                options = dict(mode='mean', padding_idx=0, max_norm=1., norm_type=order)
                a = F.embedding_bag(native(ids), nw, native(offsets), **options)
                b = torch.nn.functional.embedding_bag(ids, w, offsets, **options)
                c = F.embedding_bag(native(ids), nw, native(offsets), **options)
                d = torch.nn.functional.embedding_bag(ids, w, offsets, **options)
                self.same(a, b); self.same(c, d); self.same(nw, w)
                (a.sum()+c.sum()).backward(); (b.sum()+d.sum()).backward(); self.same(nw.grad, w.grad)
        weight = fr.tensor([[3., 4.], [1., 2.]])
        before = weight.tolist()
        for offsets in (fr.tensor([], dtype='int64'), fr.tensor([1]), fr.tensor([0, -1]), fr.tensor([0, 2, 1]), fr.tensor([0, 4]), fr.tensor(0), fr.tensor([0.])):
            with self.assertRaises((ValueError, TypeError)):
                F.embedding_bag(fr.tensor([0, 1, 0]), weight, offsets, max_norm=1.)
            self.assertEqual(weight.tolist(), before)
        with self.assertRaises(ValueError): F.embedding_bag(fr.tensor([0, 1]), weight, fr.tensor([0, 1]), max_norm=1., include_last_offset=True)
        self.assertEqual(weight.tolist(), before)

    def test_constructor_pretrained_alias_flags_padding_reset_and_signatures(self):
        self.assertEqual(list(inspect.signature(fr.nn.EmbeddingBag).parameters), list(inspect.signature(torch.nn.EmbeddingBag).parameters))
        self.assertEqual(list(inspect.signature(fr.nn.EmbeddingBag.from_pretrained).parameters), list(inspect.signature(torch.nn.EmbeddingBag.from_pretrained).parameters))
        self.assertEqual(list(inspect.signature(F.embedding_bag).parameters), list(inspect.signature(torch.nn.functional.embedding_bag).parameters))
        model = fr.nn.EmbeddingBag(4, 3, padding_idx=-1, mode='sum')
        self.assertEqual(model.padding_idx, 3); self.assertEqual(model.weight.tensor().tolist()[3], [0., 0., 0.])
        identity = model.weight.tensor()._id(); parameter = model.weight
        model.reset_parameters(); self.assertIs(model.weight, parameter); self.assertEqual(model.weight.tensor()._id(), identity)
        self.assertEqual(model.weight.tensor().tolist()[3], [0., 0., 0.])
        for freeze in (False, True):
            w = torch.tensor([[.2, .7], [.5, -.3], [1.2, .4]], requires_grad=True); nw = native(w, True)
            model = fr.nn.EmbeddingBag.from_pretrained(nw, freeze=freeze, mode='sum', padding_idx=0)
            oracle = torch.nn.EmbeddingBag.from_pretrained(w, freeze=freeze, mode='sum', padding_idx=0)
            ids, offsets = torch.tensor([0, 1, 1, 2]), torch.tensor([0, 2])
            a, b = model(native(ids), native(offsets)), oracle(ids, offsets)
            self.same(a, b); self.assertEqual(a.requires_grad, b.requires_grad)
            self.assertTrue(model.weight.tensor()._shares_storage(nw)); self.assertTrue(model.weight.trainable)
            if not freeze: a.sum().backward(); b.sum().backward(); self.same(model.weight.grad, oracle.weight.grad)
            self.assertIsNone(nw.grad)
            with fr.no_grad(): nw.copy_(fr.ones(3, 2))
            with torch.no_grad(): w.copy_(torch.ones(3, 2))
            self.same(model(native(ids), native(offsets)), oracle(ids, offsets))

    def test_version_guards_metadata_samples_and_weight_value_dependencies(self):
        for mode in ('sum', 'mean', 'max'):
            for changed in ('weight', 'ids', 'offsets'):
                nw, ids, offsets = fr.ones(3, 2, requires_grad=True), fr.tensor([0, 1, 2]), fr.tensor([0, 2])
                result = F.embedding_bag(ids, nw, offsets, mode=mode)
                with fr.no_grad():
                    if changed == 'weight': nw.copy_(fr.zeros(3, 2))
                    elif changed == 'ids': ids.copy_(fr.tensor([0, 1, 2]))
                    else: offsets.copy_(fr.tensor([0, 2]))
                if changed == 'weight': result.sum().backward(); self.assertIsNotNone(nw.grad)
                else:
                    with self.assertRaises((ValueError, RuntimeError)): result.sum().backward()
                    self.assertIsNone(nw.grad)
        for changed in ('weight', 'samples'):
            w, p = fr.ones(3, 2, requires_grad=True), fr.ones(3, requires_grad=True)
            result = F.embedding_bag(fr.tensor([0, 1, 2]), w, fr.tensor([0, 2]), mode='sum', per_sample_weights=p)
            with fr.no_grad(): (w if changed == 'weight' else p).copy_(fr.zeros(3, 2) if changed == 'weight' else fr.zeros(3))
            with self.assertRaises((ValueError, RuntimeError)): result.sum().backward()
            self.assertIsNone(w.grad); self.assertIsNone(p.grad)

    def test_dtype_shape_options_capture_higher_order_and_capacity_boundaries(self):
        w, ids, offsets = fr.ones(3, 2), fr.tensor([0, 1]), fr.tensor([0])
        calls = (lambda:F.embedding_bag(fr.tensor([0., 1.]), w, offsets), lambda:F.embedding_bag(fr.tensor(0), w, offsets),
                 lambda:F.embedding_bag(ids, fr.ones(3, 2, dtype='int64'), offsets), lambda:F.embedding_bag(ids, w),
                 lambda:F.embedding_bag(ids.reshape([1,2]), w, offsets), lambda:F.embedding_bag(ids, w, offsets, mode='bad'),
                 lambda:F.embedding_bag(ids, w, offsets, per_sample_weights=fr.ones(2)),
                 lambda:F.embedding_bag(ids, w, offsets, mode='sum', per_sample_weights=fr.ones(2,dtype='float64')),
                 lambda:F.embedding_bag(ids, w, offsets, mode='sum', per_sample_weights=fr.ones(1)),
                 lambda:F.embedding_bag(ids, w, offsets, mode='max',scale_grad_by_freq=True),
                 lambda:F.embedding_bag(ids, w, offsets, sparse=True), lambda:fr.nn.EmbeddingBag(3,2,dtype='int64'))
        for call in calls:
            with self.assertRaises((ValueError, TypeError, NotImplementedError)): call()
        with self.assertRaises((ValueError, NotImplementedError)):fr.capture(lambda:F.embedding_bag(ids,w,offsets))
        leaf = w.detach().requires_grad_(True)
        with self.assertRaises((ValueError, RuntimeError)):fr.autograd.grad(F.embedding_bag(ids,leaf,offsets).sum(),(leaf,),create_graph=True)
        huge = sys.maxsize // 8 + 1
        with self.assertRaisesRegex(ValueError, 'capacity'):
            F.embedding_bag(fr.tensor([[0]]).expand(huge,1),fr.ones(1,1))
        virtual = fr.zeros(huge,0,requires_grad=True)
        result = F.embedding_bag(fr.tensor([[0]]).expand(huge,1),virtual,mode='max')
        self.assertEqual(result.shape,[huge,0]);result.sum().backward();self.assertEqual(virtual.grad.shape,[huge,0])

    def test_fresh_process_checkpoint_options_aliases_rng_and_optimizer_suffix(self):
        script = r'''
import json,sys
import ferro as fr
source=fr.tensor([[.2,.7],[.5,-.3],[1.2,.4]])
model=fr.nn.EmbeddingBag.from_pretrained(source,freeze=False,padding_idx=0,max_norm=1.,mode='sum',scale_grad_by_freq=True,include_last_offset=True)
model.register_buffer('source',source)
optimizer=fr.optim.AdamW(model.parameters(),lr=.02);rng=fr.Generator(71)
def step():
    ids=fr.tensor([0,1,1,2]);offsets=fr.tensor([0,2,4])
    target=fr.Tensor.rand([2,2],generator=rng)
    loss=(model(ids,offsets)-target).square().sum();value=loss.item()
    loss.backward();optimizer.step();optimizer.zero_grad();return value
if sys.argv[1]=='save':
    step();fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=rng,step=1).save(sys.argv[2])
else:fr.checkpoint.load(sys.argv[2]).restore(model,optimizers={'main':optimizer},generator=rng)
losses=[step() for _ in range(3)]
tensors=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=rng,step=4).tensors()
print(json.dumps({'losses':losses,'alias':model.weight.tensor()._shares_storage(source),'tensors':{k:[v.dtype,v.shape,v.tolist()] for k,v in tensors.items()}},sort_keys=True))
'''
        env = os.environ.copy();env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory() as directory:
            results = [subprocess.run([sys.executable,'-c',script,mode,directory+'/state'],env=env,capture_output=True,text=True,check=True).stdout for mode in ('save','load')]
        self.assertEqual(json.loads(results[0]),json.loads(results[1]));self.assertTrue(json.loads(results[0])['alias'])
        original=fr.nn.EmbeddingBag(3,2,mode='sum');saved=fr.checkpoint.snapshot_training(original,optimizers={},generator=fr.Generator(1))
        target=fr.nn.EmbeddingBag(3,2,mode='mean');before=target.weight.tensor().tolist()
        with self.assertRaises(ValueError):saved.restore(target,optimizers={},generator=fr.Generator(2))
        self.assertEqual(target.weight.tensor().tolist(),before)


if __name__ == '__main__': unittest.main(verbosity=2)
