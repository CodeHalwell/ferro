"""Restore portable pre-migration builtin aggregator checkpoints without schema changes."""
import hashlib
import json
from pathlib import Path
import unittest

import ferro as fr


class AggregationCompatibilityTests(unittest.TestCase):
    def test_frozen16_builtin_family_checkpoints_restore_exact_typed_state(self):
        fixture = Path(__file__).parent / 'fixtures' / 'aggregation_v16'
        manifest = json.loads((fixture / 'manifest.json').read_text())
        self.assertEqual(manifest['format'], 'ferro-aggregation-pre-public-base-v16')
        self.assertEqual(manifest['aggregation_source_sha256'], '0d0e387ecebec131ad0a5fa0dd7a4d57cda4200993892740e63a07f073064b4d')
        self.assertEqual(manifest['native_sha256'], 'dacc9eff465b7d0ab3cd6a1a201d2cec1041181e4dfda6a330ebf15617668811')
        files = {path.relative_to(fixture).as_posix(): path for path in fixture.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        self.assertEqual(set(files), set(manifest['files']))
        for name, path in files.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest['files'][name], name)
        factories={
            'sum':lambda:fr.graph.SumAggregation(),
            'mean':lambda:fr.graph.MeanAggregation(),
            'min':lambda:fr.graph.MinAggregation(),
            'max':lambda:fr.graph.MaxAggregation(),
            'var':lambda:fr.graph.VarAggregation(),
            'std':lambda:fr.graph.StdAggregation(semi_grad=True),
            'softmax':lambda:fr.graph.SoftmaxAggregation(t=.7,learn=True,channels=3),
            'power':lambda:fr.graph.PowerMeanAggregation(p=1.2,learn=True,channels=3),
            'multi':lambda:fr.graph.MultiAggregation(['sum','mean'],mode='proj',mode_kwargs=dict(in_channels=3,out_channels=4)),
            'lstm':lambda:fr.graph.LSTMAggregation(3,4,num_layers=2,dropout=.2),
            'gru':lambda:fr.graph.GRUAggregation(3,4,num_layers=2,dropout=.2),
            'sort':lambda:fr.graph.SortAggregation(2),
            'quantile':lambda:fr.graph.QuantileAggregation([.25,.7]),
            'median':lambda:fr.graph.MedianAggregation(),
            'set2set':lambda:fr.graph.Set2Set(3,2,num_layers=2,dropout=.2),
            'set_transformer':lambda:fr.graph.SetTransformerAggregation(4,num_seed_points=2,heads=2,dropout=.2),
            'multiset':lambda:fr.graph.GraphMultisetTransformer(4,2,heads=2,dropout=.2),
            'deep_sets':lambda:fr.graph.DeepSetsAggregation(fr.nn.Sequential(fr.nn.Linear(3,3),fr.nn.Tanh()),fr.nn.Linear(3,4)),
            'attentional':lambda:fr.graph.AttentionalAggregation(fr.nn.Linear(3,1),fr.nn.Linear(3,3)),
            'degree':lambda:fr.graph.DegreeScalerAggregation(['sum','mean'],['identity','linear'],fr.Tensor([1.,2.,1.],[3]),train_norm=True),
            'variance_preserving':lambda:fr.graph.VariancePreservingAggregation(),
        }
        self.assertEqual(len(factories), 21)
        self.assertEqual(list(factories), manifest['families'])
        for name, factory in factories.items():
            with self.subTest(family=name):
                saved = fr.checkpoint.load(str(fixture / name))
                expected = saved.tensors()
                model = factory()
                params = list(model.parameters())
                optimizers = {'opt': fr.optim.AdamW(params, lr=.8)} if params else {}
                rng = fr.Generator(1)
                saved.restore(model, optimizers=optimizers, generator=rng)
                actual = fr.checkpoint.snapshot_training(model, optimizers=optimizers, generator=rng, step=0).tensors()
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    self.assertEqual((actual[key].dtype, actual[key].shape, actual[key].tolist()),
                                     (expected[key].dtype, expected[key].shape, expected[key].tolist()), key)
                for _, child in model.named_modules(): self.assertFalse(child.training)
                if params: self.assertFalse(params[0].trainable)
                self.assertEqual(fr.Tensor.rand([4], generator=rng).tolist(), fr.Tensor.rand([4], generator=fr.Generator(911)).tolist())


if __name__ == '__main__': unittest.main(verbosity=2)
