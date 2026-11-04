"""Strong-Wolfe trials, complete history and fresh-process native restoration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
import test_lbfgs_optimizer as fixed_tests
from test_optimizer_groups_schedulers import checkpoint_digest, plain


class Model(fr.nn.Module):
    def build(self):
        self.point=fr.nn.Parameter(fr.Tensor([-1.2,1.],[2]))


def objective(point,bias=0.):
    return (point[1]-point[0].square()).square()*100.+(1.-point[0]).square()+point[0]*bias


def setup(search=True,**kwargs):
    model=Model();options=dict(lr=.8,max_iter=4,max_eval=8,tolerance_grad=0.,tolerance_change=0.,history_size=4,line_search_fn='strong_wolfe' if search else None)
    options.update(kwargs)
    return model,fr.optim.LBFGS(model.parameters(),**options)


def step(model,optimizer,bias=0.,generator=None):
    trace=[]
    def closure():
        optimizer.zero_grad()
        noise=0. if generator is None else fr.Tensor.randn([],generator=generator).item()*1e-7
        loss=objective(model.point.tensor(),bias+noise)
        trace.append([model.point.tensor().tolist(),loss.item()]);loss.backward();return loss
    loss=optimizer.step(closure)
    return loss.item(),trace


def suffix(model,optimizer,scheduler,generator,start):
    trace=[]
    for index in range(start,7):
        bias=fr.Tensor.randn([],generator=generator).item()*.0002
        trace.append(step(model,optimizer,bias,generator));scheduler.step()
    optimizer.zero_grad()
    return dict(trace=trace,state=plain(optimizer.state_dict()),scheduler=scheduler.state_dict(),
                checkpoint=checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'fit':optimizer},generator=generator,step=7)),
                draws=fr.Tensor.randn([7],generator=generator).tolist())


def worker(directory):
    path=Path(directory);model,optimizer=setup(False)
    scheduler=fr.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.8);generator=fr.Generator(999)
    checkpoint=fr.checkpoint.load(path/'training');checkpoint.restore(model,optimizers={'fit':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((path/'scheduler.json').read_text()))
    (path/'result.json').write_text(json.dumps(suffix(model,optimizer,scheduler,generator,checkpoint.step),allow_nan=False))


class StrongWolfeTests(unittest.TestCase):
    compare=fixed_tests.LBFGSTests.compare

    def test_rosenbrock_every_trial_history_and_parameter_matches_torch(self):
        model,optimizer=setup();parameter=torch.nn.Parameter(torch.tensor([-1.2,1.]))
        oracle=torch.optim.LBFGS([parameter],lr=.8,max_iter=4,max_eval=8,tolerance_grad=0.,tolerance_change=0.,history_size=4,line_search_fn='strong_wolfe')
        for _ in range(3):
            actual,trace=step(model,optimizer);expected_trace=[]
            def closure():
                oracle.zero_grad();loss=objective(parameter)
                expected_trace.append([parameter.detach().tolist(),loss.item()]);loss.backward();return loss
            expected=oracle.step(closure)
            self.compare(actual,expected.item());self.compare(trace,expected_trace,'line search trials')
            self.compare(model.point.tensor(),parameter.detach(),'accepted point')
            self.compare(optimizer.state_dict()['state'],oracle.state_dict()['state'])
        self.assertLess(objective(model.point.tensor()).item(),3.)

    def test_first_step_independently_satisfies_armijo_and_curvature(self):
        model,optimizer=setup(lr=1.,max_iter=1,max_eval=25)
        initial=torch.tensor([-1.2,1.],requires_grad=True);initial_loss=objective(initial);gradient=torch.autograd.grad(initial_loss,initial)[0]
        step(model,optimizer)
        final=torch.tensor(model.point.tensor().tolist(),requires_grad=True);final_loss=objective(final);final_gradient=torch.autograd.grad(final_loss,final)[0]
        t=optimizer.state_dict()['state'][0]['t'];direction=-gradient;slope=gradient.dot(direction).item()
        self.assertLessEqual(final_loss.item(),initial_loss.item()+1e-4*t*slope)
        self.assertLessEqual(abs(final_gradient.dot(direction).item()),-.9*slope)
        torch.testing.assert_close(final.detach(),initial.detach()+t*direction,atol=2e-6,rtol=2e-6)

    def test_budget_and_early_exit_match_torch(self):
        for options in (dict(max_eval=1),dict(tolerance_grad=1000.),dict(max_iter=1,max_eval=2)):
            with self.subTest(options=options):
                model,optimizer=setup(**options);parameter=torch.nn.Parameter(torch.tensor([-1.2,1.]))
                settings=dict(lr=.8,max_iter=4,max_eval=8,tolerance_grad=0.,tolerance_change=0.,history_size=4,line_search_fn='strong_wolfe');settings.update(options)
                oracle=torch.optim.LBFGS([parameter],**settings)
                for _ in range(2):
                    actual,trace=step(model,optimizer);expected_trace=[]
                    def closure():
                        oracle.zero_grad();loss=objective(parameter);expected_trace.append([parameter.detach().tolist(),loss.item()]);loss.backward();return loss
                    expected=oracle.step(closure)
                    self.compare(actual,expected.item());self.compare(trace,expected_trace)
                    self.compare(model.point.tensor(),parameter.detach());self.compare(optimizer.state_dict()['state'],oracle.state_dict()['state'])

    def test_zero_lr_repeated_search_preserves_parameters_despite_torch_reference_bug(self):
        model,optimizer=setup(lr=0.);parameter=torch.nn.Parameter(torch.tensor([-1.2,1.]))
        oracle=torch.optim.LBFGS([parameter],lr=0.,max_iter=4,max_eval=8,tolerance_grad=0.,tolerance_change=0.,history_size=4,line_search_fn='strong_wolfe')
        expected_trace=[]
        def closure():
            oracle.zero_grad();loss=objective(parameter)
            expected_trace.append([parameter.detach().tolist(),loss.item()]);loss.backward();return loss
        actual,trace=step(model,optimizer);expected=oracle.step(closure)
        self.compare(actual,expected.item());self.compare(trace,expected_trace)
        self.compare(optimizer.state_dict()['state'],oracle.state_dict()['state'])
        before=model.point.tensor().tolist()
        # Torch 2.13's second call passes a Python zero to cubic interpolation,
        # which divides by zero. Ferro retains a finite zero-step update.
        with self.assertRaises(ZeroDivisionError):oracle.step(closure)
        again,trials=step(model,optimizer)
        self.assertEqual(model.point.tensor().tolist(),before)
        self.assertEqual(again,actual)
        self.assertTrue(all(point==before and loss==actual for point,loss in trials))
        self.assertEqual(optimizer.state_dict()['state'][0]['t'],0.)
        json.dumps(plain(optimizer.state_dict()),allow_nan=False)

    def test_mode_checkpoint_migration_and_failed_trial_baseline_restore(self):
        for search in (False,True):
            source,optimizer=setup(search,max_iter=1);step(source,optimizer)
            optimizer.zero_grad()
            saved=fr.checkpoint.snapshot_training(source,optimizers={'fit':optimizer},generator=fr.Generator(7))
            target,other=setup(not search,max_iter=1);saved.restore(target,optimizers={'fit':other},generator=fr.Generator(9))
            self.assertEqual(plain(optimizer.state_dict()),plain(other.state_dict()))
            self.assertEqual(step(source,optimizer),step(target,other))
        model,optimizer=setup();initial=model.point.tensor().tolist();calls=[]
        def broken_trial():
            optimizer.zero_grad();calls.append(model.point.tensor().tolist())
            if len(calls)>1:raise LookupError('trial failure')
            loss=objective(model.point.tensor());loss.backward();return loss
        with self.assertRaisesRegex(LookupError,'trial failure'):optimizer.step(broken_trial)
        self.assertNotEqual(calls[0],calls[1]);self.assertEqual(model.point.tensor().tolist(),initial)
        optimizer.zero_grad();optimizer.state_dict()

    def test_exact_fresh_process_after_search_restores_history_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);model,optimizer=setup();generator=fr.Generator(43)
            scheduler=fr.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.8)
            for _ in range(3):
                step(model,optimizer,fr.Tensor.randn([],generator=generator).item()*.0002,generator);scheduler.step()
            optimizer.zero_grad()
            fr.checkpoint.save_training(path/'training',model,optimizers={'fit':optimizer},generator=generator,step=3)
            (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
            expected=json.loads(json.dumps(suffix(model,optimizer,scheduler,generator,3)))
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(child.returncode,0,child.stdout);self.assertEqual(json.loads((path/'result.json').read_text()),expected)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        from test_lbfgs_line_search import worker as run_worker
        run_worker(sys.argv[2])
    else:unittest.main(module='test_lbfgs_line_search')
