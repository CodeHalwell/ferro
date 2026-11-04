"""Public optimizer gradients follow Torch while raw Rust steps remain consuming."""
import unittest
import torch
import ferro as fr
from ferro.training import Trainer
from test_optimizer_groups_schedulers import plain
import test_adaptive_optimizers as adaptive_tests
import test_lbfgs_optimizer as lbfgs_tests


CASES = [
    ('SGD', dict(momentum=.6, weight_decay=.03)),
    ('Adam', dict(betas=(.7,.9), weight_decay=.03, amsgrad=True)),
    ('AdamW', dict(betas=(.7,.9), weight_decay=.03, amsgrad=True)),
    ('RMSprop', dict(alpha=.8, momentum=.3, centered=True, weight_decay=.02)),
    ('Adagrad', dict(initial_accumulator_value=.2, lr_decay=.04, weight_decay=.01)),
    ('Adamax', dict(betas=(.7,.9), weight_decay=.02)),
    ('Adadelta', dict(rho=.8, weight_decay=.02)),
    ('ASGD', dict(lambd=.03, t0=2., alpha=.7, weight_decay=.02)),
    ('Rprop', dict(etas=(.5,1.2), step_sizes=(.001,.1))),
    ('RAdam', dict(betas=(.7,.8), weight_decay=.04)),
    ('NAdam', dict(betas=(.7,.9), momentum_decay=.01, weight_decay=.02)),
    ('Adafactor', dict(beta2_decay=-.8, eps=(None,.001), weight_decay=.02)),
]


def setup(kind, options):
    values = [[[.6,-.4,.7],[-.2,.8,.1]], [.3,-.5]]
    parameters = [fr.nn.Parameter(fr.tensor(value)) for value in values]
    reference = [torch.nn.Parameter(torch.tensor(value)) for value in values]
    groups = lambda params: [{'params':[params[0]],'lr':.03},{'params':[params[1]],'lr':.01,'maximize':True}]
    optimizer = getattr(fr.optim, kind)(groups(parameters), lr=.02, **options)
    oracle = getattr(torch.optim, kind)(groups(reference), lr=.02, **options, foreach=False)
    return parameters, reference, optimizer, oracle


class OptimizerGradientRetentionTests(unittest.TestCase):
    compare_state = adaptive_tests.AdaptiveOptimizerTests.compare_state
    compare = lbfgs_tests.LBFGSTests.compare

    def close(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
        else:
            self.assertIsNotNone(actual)
            torch.testing.assert_close(torch.tensor(actual.tolist()).reshape(actual.shape), expected.detach(), atol=4e-6, rtol=6e-5)

    def test_all_elementwise_families_repeated_steps_keep_same_gradients_and_match_torch(self):
        for name, options in CASES:
            with self.subTest(name=name):
                parameters, reference, optimizer, oracle = setup(name, options)
                for index in range(6):
                    optimizer.zero_grad(); oracle.zero_grad(set_to_none=True)
                    for pi,(parameter,target) in enumerate(zip(parameters,reference)):
                        if pi == 1 and index % 3 == 0: continue
                        scale,shift=.2+.03*index,.1-.02*pi
                        (parameter.tensor().square().sum()*scale+parameter.tensor().sum()*shift).backward()
                        (target.square().sum()*scale+target.sum()*shift).backward()
                    identities=[None if p.grad is None else p.grad._id() for p in parameters]
                    for _ in range(2):
                        optimizer.step(); oracle.step()
                        for parameter,target,identity in zip(parameters,reference,identities):
                            self.close(parameter.tensor(),target); self.close(parameter.grad,target.grad)
                            self.assertEqual(None if parameter.grad is None else parameter.grad._id(),identity)
                        self.compare_state(optimizer,oracle)

    def test_zero_gradient_slots_update_decay_and_moments_while_none_slots_skip(self):
        for name,options in CASES:
            with self.subTest(name=name):
                parameters,reference,optimizer,oracle=setup(name,options)
                for parameter,target in zip(parameters,reference):
                    parameter.tensor().square().sum().backward();target.square().sum().backward()
                optimizer.step();oracle.step()
                identities=[p.grad._id() for p in parameters]
                optimizer.zero_grad(set_to_none=False);oracle.zero_grad(set_to_none=False)
                for parameter,target,identity in zip(parameters,reference,identities):
                    self.close(parameter.grad,target.grad);self.assertEqual(parameter.grad._id(),identity)
                optimizer.step();oracle.step();self.compare_state(optimizer,oracle)
                for parameter,target in zip(parameters,reference):self.close(parameter.tensor(),target)
                optimizer.zero_grad();oracle.zero_grad(set_to_none=True)
                before=plain(optimizer.state_dict());values=[p.tensor().tolist() for p in parameters]
                optimizer.step();oracle.step();self.compare_state(optimizer,oracle)
                self.assertEqual(plain(optimizer.state_dict()),before)
                self.assertEqual([p.tensor().tolist() for p in parameters],values)
                self.assertTrue(all(p.grad is None for p in parameters))

    def test_frozen_hidden_gradient_is_retained_but_zero_grad_false_clears_its_slot(self):
        parameter=fr.nn.Parameter(fr.tensor([.5,-.7]));optimizer=fr.optim.AdamW([parameter],lr=.02)
        parameter.trainable=False
        parameter.tensor().square().sum().backward()
        self.assertIsNone(parameter.grad)
        held=parameter.tensor().grad;identity=held._id();values=held.tolist()
        before=plain(optimizer.state_dict());optimizer.step()
        self.assertEqual(parameter.tensor().grad._id(),identity)
        self.assertEqual(parameter.tensor().grad.tolist(),values)
        self.assertEqual(plain(optimizer.state_dict()),before)
        optimizer.zero_grad(set_to_none=False)
        self.assertEqual(parameter.tensor().grad._id(),identity)
        self.assertEqual(parameter.tensor().grad.tolist(),[0.,0.])
        optimizer.zero_grad();self.assertIsNone(parameter.tensor().grad)

    def test_lbfgs_fixed_and_search_leave_last_closure_gradients_like_torch(self):
        # A well-conditioned objective: on Rosenbrock a few-ulp change in Torch's float32 kernels (which vary by host ISA) moves the strong-Wolfe path past tolerance.
        for search,max_eval in ((None,8),('strong_wolfe',8),('strong_wolfe',1)):
            with self.subTest(search=search,max_eval=max_eval):
                parameter=fr.nn.Parameter(fr.tensor([-1.2,1.]));target=torch.nn.Parameter(torch.tensor([-1.2,1.]))
                settings=dict(lr=.8,max_iter=4,max_eval=max_eval,history_size=4,line_search_fn=search,tolerance_grad=0.,tolerance_change=0.)
                optimizer=fr.optim.LBFGS([parameter],**settings);oracle=torch.optim.LBFGS([target],**settings)
                for _ in range(3):
                    identities=[]
                    def closure():
                        optimizer.zero_grad();p=parameter.tensor()
                        loss=(p[0]-1.).square()+(p[1]+.5).square()*2.+(p[0]*p[1]).square()*.5
                        loss.backward();identities.append(parameter.grad._id());return loss
                    def reference_closure():
                        oracle.zero_grad();loss=(target[0]-1.).square()+(target[1]+.5).square()*2.+(target[0]*target[1]).square()*.5
                        loss.backward();return loss
                    with fr.no_grad(),torch.no_grad():
                        actual=optimizer.step(closure);expected=oracle.step(reference_closure)
                    self.compare(actual,expected);self.compare(parameter.tensor(),target.detach())
                    self.compare(parameter.grad,target.grad);self.assertEqual(parameter.grad._id(),identities[-1])
                    self.compare(optimizer.state_dict()['state'],oracle.state_dict()['state'])

    def test_checkpoint_requires_explicit_clearing_after_public_step(self):
        model=fr.nn.Linear(2,1);optimizer=fr.optim.AdamW(model.parameters());generator=fr.Generator(7)
        model(fr.ones(2,2)).square().sum().backward();optimizer.step()
        gradients=[p.grad._id() for p in model.parameters()]
        with self.assertRaisesRegex(ValueError,'pending gradients'):
            fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)
        saved=optimizer.state_dict();optimizer.load_state_dict(saved)
        self.assertEqual([p.grad._id() for p in model.parameters()],gradients)
        optimizer.zero_grad();checkpoint=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)
        target=fr.nn.Linear(2,1);other=fr.optim.AdamW(target.parameters());checkpoint.restore(target,optimizers={'main':other},generator=fr.Generator(99))
        self.assertEqual(plain(optimizer.state_dict()),plain(other.state_dict()))
        self.assertTrue(all(p.grad is None for p in target.parameters()))

    def test_capture_rejects_before_closure_configuration_parameter_or_gradient_changes(self):
        for kind in ('SGD', 'AdamW', 'RMSprop', 'LBFGS'):
            with self.subTest(kind=kind):
                parameter=fr.nn.Parameter(fr.tensor([.4,-.7]))
                optimizer=getattr(fr.optim,kind)([parameter],lr=.02)
                parameter.tensor().square().sum().backward()
                before=plain(optimizer.state_dict());values=parameter.tensor().tolist()
                gradient=parameter.tensor().grad;identity=gradient._id()
                optimizer.param_groups[0]['lr']=.1
                native_config=optimizer._native.configurations()
                def captured():
                    optimizer.step(lambda:self.fail('capture must reject before calling a closure'))
                    return parameter.tensor()
                with self.assertRaises(NotImplementedError):fr.capture(captured)
                self.assertEqual(optimizer._native.configurations(),native_config)
                self.assertEqual(parameter.tensor().tolist(),values)
                self.assertEqual(parameter.tensor().grad._id(),identity)
                self.assertEqual(parameter.tensor().grad.tolist(),gradient.tolist())
                optimizer.param_groups[0]['lr']=.02
                self.assertEqual(plain(optimizer.state_dict()),before)

    def test_managed_trainer_clears_at_complete_update_boundary(self):
        model=fr.nn.Linear(2,1);optimizer=fr.optim.SGD(model.parameters(),lr=.03)
        trainer=Trainer(model,optimizer=optimizer,objective=lambda net,x:net(x).square().mean())
        reports=trainer.fit([fr.ones(3,2),fr.ones(2,2)],epochs=2)
        self.assertEqual(len(reports),4)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=fr.Generator(11),step=4)


if __name__=='__main__':unittest.main()
