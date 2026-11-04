"""Ordinary optimizer closures can update validated live group settings."""
import struct
import unittest

import torch
import ferro as fr
import test_adaptive_optimizers as adaptive_tests


KINDS=('SGD','Adam','AdamW','RMSprop','Adagrad','Adamax','Adadelta','ASGD','Rprop','RAdam','NAdam','Adafactor')


def native_state(optimizer):
    return optimizer._native.configurations(),[(name,value.shape,value.tolist()) for name,value in optimizer._native.state_tensors()]


class OptimizerClosureConfigurationTests(unittest.TestCase):
    compare_state=adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_each_ordinary_family_reads_successful_closure_settings_and_matches_torch(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                params=[fr.nn.Parameter(fr.tensor(values)) for values in ([.4,-.6],[.7])]
                targets=[torch.nn.Parameter(torch.tensor(values)) for values in ([.4,-.6],[.7])]
                options=dict(lr=.017)
                if kind in ('SGD','RMSprop'):options['momentum']=.6
                optimizer=getattr(fr.optim,kind)([{'params':[p]} for p in params],**options)
                reference=getattr(torch.optim,kind)([{'params':[p]} for p in targets],**options,foreach=False)
                for index in range(7):
                    returned=[]
                    def settings(groups):
                        for gi,group in enumerate(groups):
                            group['lr']=.02+.003*index+.007*gi
                            if 'momentum' in group:group['momentum']=.6+.03*index
                            if 'betas' in group:group['betas']=(.7+.02*index,group['betas'][1])
                    def closure():
                        optimizer.zero_grad();settings(optimizer.param_groups)
                        loss=sum(p.tensor().square().sum()*(.3+.02*index)+p.tensor().sum()*.03 for p in params)
                        loss.backward();returned.append(loss);return loss
                    def oracle_closure():
                        reference.zero_grad();settings(reference.param_groups)
                        loss=sum(p.square().sum()*(.3+.02*index)+p.sum()*.03 for p in targets)
                        loss.backward();return loss
                    with fr.no_grad():actual=optimizer.step(closure)
                    expected=reference.step(oracle_closure)
                    self.assertIs(actual,returned[0]);self.assertAlmostEqual(actual.item(),expected.item(),places=5)
                    # Inspect before state_dict(), which itself synchronizes options.
                    for group,native in zip(optimizer.param_groups,optimizer._native.configurations()):
                        config=optimizer._config(group)
                        rounded=list(struct.unpack('<'+'f'*len(config),struct.pack('<'+'f'*len(config),*config)))
                        self.assertEqual(native,rounded)
                    for parameter,target in zip(params,targets):
                        torch.testing.assert_close(torch.tensor(parameter.tensor().tolist()),target.detach(),atol=4e-6,rtol=6e-5)
                        torch.testing.assert_close(torch.tensor(parameter.grad.tolist()),target.grad,atol=4e-6,rtol=6e-5)
                    self.compare_state(optimizer,reference)

    def test_initial_invalid_configuration_rejects_before_invoking_closure(self):
        parameter=fr.nn.Parameter(fr.ones(2));optimizer=fr.optim.AdamW([parameter],lr=.01)
        parameter.tensor().square().sum().backward();optimizer.step();optimizer.zero_grad()
        before=native_state(optimizer);values=parameter.tensor().tolist()
        optimizer.param_groups[0]['amsgrad']=True
        with self.assertRaisesRegex(ValueError,'amsgrad'):
            optimizer.step(lambda:self.fail('initial preflight must precede the closure'))
        self.assertEqual(native_state(optimizer),before);self.assertEqual(parameter.tensor().tolist(),values)

    def test_invalid_late_closure_settings_reject_before_any_native_update(self):
        for kind,key,value in (('RMSprop','momentum',.9),('RMSprop','centered',True),
                               ('AdamW','amsgrad',True),('SGD','dampening',.1),('SGD','lr',-1.)):
            with self.subTest(kind=kind,key=key):
                params=[fr.nn.Parameter(fr.ones(2)),fr.nn.Parameter(fr.ones(1))]
                optimizer=getattr(fr.optim,kind)([{'params':[p]} for p in params],lr=.01)
                sum(p.tensor().square().sum() for p in params).backward();optimizer.step();optimizer.zero_grad()
                before=native_state(optimizer);values=[p.tensor().tolist() for p in params]
                def closure():
                    optimizer.param_groups[0]['lr']=.02
                    optimizer.param_groups[-1][key]=value
                    loss=sum(p.tensor().square().sum() for p in params);loss.backward();return loss
                with self.assertRaises(ValueError):optimizer.step(closure)
                self.assertEqual(native_state(optimizer),before)
                self.assertEqual([p.tensor().tolist() for p in params],values)
                self.assertEqual(optimizer.param_groups[-1][key],value)
                self.assertTrue(all(p.grad is not None for p in params))
