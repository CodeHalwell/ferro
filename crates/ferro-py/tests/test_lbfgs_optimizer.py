"""Independent Torch closure/history parity and complete-step native restarts."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
from test_optimizer_groups_schedulers import checkpoint_digest, plain


class Model(fr.nn.Module):
    def build(self):
        self.a = fr.nn.Parameter(fr.Tensor([.9, -.6], [2]))
        self.b = fr.nn.Parameter(fr.Tensor([.7], []))
        self.empty = fr.nn.Parameter(fr.Tensor([], [0]))


def objective(a, b, index, shift=0.):
    loss = a.square().sum() * .7 + a.pow(4).sum() * .1 + a.sum() * (-.2 + .025 * index + shift)
    if index % 5 != 3:
        loss = loss + b.square() * 1.1 + a.sum() * b * .15
    return loss


def setup():
    model = Model()
    optimizer = fr.optim.LBFGS(model.parameters(), lr=.4, max_iter=3, max_eval=4,
                              tolerance_grad=0., tolerance_change=0., history_size=3)
    return model, optimizer


def run_step(model, optimizer, index, shift=0., generator=None):
    calls = []
    def closure():
        optimizer.zero_grad()
        noise = fr.Tensor.randn([], generator=generator).item() * .0001 if generator is not None else 0.
        loss = objective(model.a.tensor(), model.b.tensor(), index, shift + noise)
        calls.append(loss.item()); loss.backward()
        return loss
    result = optimizer.step(closure)
    return result.item(), calls


def suffix(model, optimizer, scheduler, generator, start):
    trace = []
    for index in range(start, 9):
        shift = fr.Tensor.randn([], generator=generator).item() * .003
        trace.append(run_step(model, optimizer, index, shift, generator)); scheduler.step()
    optimizer.zero_grad()
    return dict(trace=trace, checkpoint=checkpoint_digest(fr.checkpoint.snapshot_training(model, optimizers={'fit': optimizer}, generator=generator, step=9)),
                state=plain(optimizer.state_dict()), scheduler=scheduler.state_dict(), draws=fr.Tensor.randn([7], generator=generator).tolist())


def worker(directory):
    path = Path(directory)
    model, optimizer = setup()
    scheduler = fr.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.8)
    generator = fr.Generator(999)
    checkpoint = fr.checkpoint.load(path / 'state')
    checkpoint.restore(model, optimizers={'fit': optimizer}, generator=generator)
    scheduler.load_state_dict(json.loads((path / 'scheduler.json').read_text()))
    (path / 'result.json').write_text(json.dumps(suffix(model, optimizer, scheduler, generator, checkpoint.step), allow_nan=False))


class LBFGSTests(unittest.TestCase):
    def compare(self, actual, expected, path='state'):
        if isinstance(actual, fr.Tensor):
            actual = torch.tensor(actual.tolist()).reshape(actual.shape)
        if isinstance(expected, torch.Tensor):
            actual = torch.as_tensor(actual)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-3, msg=path)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            for key in expected:
                self.compare(actual[key], expected[key], path + '.' + str(key))
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected), path)
            for index, (a, b) in enumerate(zip(actual, expected)):
                self.compare(a, b, path + '.' + str(index))
        elif isinstance(expected, float):
            self.assertAlmostEqual(float(actual), expected, delta=3e-5 + abs(expected)*2e-3, msg=path)
        else:
            self.assertEqual(actual, expected, path)

    def test_multistep_closure_history_parameters_and_missing_gradient_match_torch(self):
        model, optimizer = setup()
        params = [torch.nn.Parameter(torch.tensor(p.tensor().tolist()).reshape(p.tensor().shape)) for p in model.parameters()]
        oracle = torch.optim.LBFGS(params, lr=.4, max_iter=3, max_eval=4, tolerance_grad=0., tolerance_change=0., history_size=3)
        self.compare(optimizer.state_dict()['state'], oracle.state_dict()['state'])
        for index in range(9):
            actual_loss, actual_calls = run_step(model, optimizer, index)
            calls = []
            def closure():
                oracle.zero_grad()
                loss = objective(params[0], params[1], index)
                calls.append(loss.item()); loss.backward(); return loss
            expected_loss = oracle.step(closure)
            self.compare(actual_calls, calls, 'closure evaluations')
            self.compare(actual_loss, expected_loss.item(), 'returned initial loss')
            for p, tp in zip(model.parameters(), params):
                self.compare(p.tensor(), tp.detach(), 'parameters')
                if tp.grad is None:
                    self.assertIsNone(p.grad)
                else:
                    self.compare(p.grad, tp.grad, 'last closure gradient')
            self.compare(optimizer.state_dict()['state'], oracle.state_dict()['state'])
        saved = optimizer.state_dict(); before = plain(saved)
        run_step(model, optimizer, 9)
        self.assertEqual(plain(saved), before)
        optimizer.load_state_dict(saved)
        self.assertEqual(plain(optimizer.state_dict()), before)

    def test_zero_gradient_early_exit_and_one_iteration_history_match_torch(self):
        for max_iter in (1, 4):
            model = Model(); opt = fr.optim.LBFGS(model.parameters(), max_iter=max_iter, history_size=2)
            params = [torch.nn.Parameter(torch.tensor(p.tensor().tolist()).reshape(p.tensor().shape)) for p in model.parameters()]
            oracle = torch.optim.LBFGS(params, max_iter=max_iter, history_size=2)
            for zero in (True, False, True):
                def closure():
                    opt.zero_grad(); loss = model.a.tensor().sum() * 0. if zero else model.a.tensor().square().sum()
                    loss.backward(); return loss
                def reference():
                    oracle.zero_grad(); loss = params[0].sum() * 0. if zero else params[0].square().sum()
                    loss.backward(); return loss
                self.compare(opt.step(closure).item(), oracle.step(reference).item())
                self.compare(opt.state_dict()['state'], oracle.state_dict()['state'])
                opt.load_state_dict(opt.state_dict())

    def test_closure_grad_mode_identity_reentry_and_error_release(self):
        model, optimizer = setup(); first=[]
        def closure():
            optimizer.zero_grad()
            with self.assertRaises(RuntimeError): optimizer.state_dict()
            with self.assertRaises(RuntimeError): optimizer.step(lambda: None)
            with self.assertRaises(RuntimeError): fr.checkpoint.snapshot_training(model, optimizers={'fit': optimizer}, generator=fr.Generator(0))
            value=model.a.tensor().square().sum(); value.backward();first.append(value);return value
        with fr.no_grad():
            returned=optimizer.step(closure)
        self.assertIs(returned,first[0])
        def broken():
            optimizer.zero_grad(); raise LookupError('closure sentinel')
        with self.assertRaisesRegex(LookupError,'closure sentinel'): optimizer.step(broken)
        optimizer.zero_grad(); optimizer.state_dict()
        for value in (3., fr.Tensor([1.], [1]), fr.Tensor([float('nan')], [])):
            with self.assertRaises((TypeError,ValueError)):
                optimizer.step(lambda:value)
        run_step(model,optimizer,0)

    def test_evaluation_budget_zero_lr_and_stopping_conditions_match_torch(self):
        cases = (dict(max_iter=5, max_eval=2), dict(max_iter=5, max_eval=1), dict(lr=0.), dict(tolerance_grad=20.),
                 dict(tolerance_change=20.), dict(max_iter=1), dict(history_size=1))
        for options in cases:
            with self.subTest(options=options):
                parameter=fr.nn.Parameter(fr.Tensor([.7,-.3],[2]))
                reference=torch.nn.Parameter(torch.tensor([.7,-.3]))
                settings=dict(lr=.3,max_iter=4,max_eval=5,history_size=3)
                settings.update(options)
                optimizer=fr.optim.LBFGS([parameter],**settings)
                oracle=torch.optim.LBFGS([reference],**settings)
                for _ in range(3):
                    calls=[]; expected_calls=[]
                    def closure():
                        optimizer.zero_grad();value=parameter.tensor().square().sum()
                        calls.append(value.item());value.backward();return value
                    def target():
                        oracle.zero_grad();value=reference.square().sum()
                        expected_calls.append(value.item());value.backward();return value
                    self.compare(optimizer.step(closure).item(),oracle.step(target).item())
                    self.compare(calls,expected_calls)
                    self.compare(parameter.tensor(),reference.detach())
                    self.compare(optimizer.state_dict()['state'],oracle.state_dict()['state'])

    def test_state_dict_and_checkpoint_corruption_are_atomic(self):
        from test_training_checkpoint import rewrite_checkpoint
        model, optimizer=setup(); generator=fr.Generator(19)
        for index in range(3):run_step(model,optimizer,index)
        optimizer.zero_grad()
        before=checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'fit':optimizer},generator=generator))
        for fault in ('unknown','counter','rho','history','config','alpha'):
            state=optimizer.state_dict();payload=state['state'][0]
            if fault=='unknown':payload['extra']=1
            elif fault=='counter':payload['n_iter']=-1
            elif fault=='rho':payload['ro'][-1]=fr.Tensor([float('nan')],[])
            elif fault=='history':payload['old_dirs'][-1]=fr.Tensor.ones([99])
            elif fault=='config':state['param_groups'][0]['history_size']=4
            else:payload['al'][-1]=None
            with self.subTest(fault=fault), self.assertRaises(ValueError):optimizer.load_state_dict(state)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'fit':optimizer},generator=generator)),before)
        saved=fr.checkpoint.snapshot_training(model,optimizers={'fit':optimizer},generator=generator)
        fields={'optim.fit.alpha_valid':fr.Tensor([1.,.5,1.],[3]), 'optim.fit.unexpected':fr.Tensor([1.],[]),
                'optim.fit.old_stps':fr.Tensor.ones([3,99]), 'scalars.optim_meta.fit.g0.line_search_fn':fr.Tensor([2.,0.,0.,0.],[4])}
        for name,value in fields.items():
            with tempfile.TemporaryDirectory() as directory:
                malformed=rewrite_checkpoint(saved,directory,{name:value})
                with self.assertRaises(ValueError):malformed.restore(model,optimizers={'fit':optimizer},generator=generator)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'fit':optimizer},generator=generator)),before)

    def test_fresh_process_restores_history_scheduler_model_rng_and_next_closures(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);model,optimizer=setup();generator=fr.Generator(37)
            scheduler=fr.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.8)
            for index in range(4):
                run_step(model,optimizer,index,fr.Tensor.randn([],generator=generator).item()*.003,generator);scheduler.step()
            optimizer.zero_grad()
            fr.checkpoint.save_training(path/'state',model,optimizers={'fit':optimizer},generator=generator,step=4)
            (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
            expected=json.loads(json.dumps(suffix(model,optimizer,scheduler,generator,4)))
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(child.returncode,0,child.stdout)
            self.assertEqual(json.loads((path/'result.json').read_text()),expected)

    def test_validation_before_closure_and_single_group_boundaries(self):
        model,optimizer=setup()
        for options in (dict(max_iter=0),dict(max_eval=1.5),dict(history_size=0),dict(tolerance_grad=-1),dict(lr=float('inf'))):
            with self.assertRaises(ValueError):fr.optim.LBFGS(model.parameters(),**options)
        with self.assertRaises(NotImplementedError):fr.optim.LBFGS(model.parameters(),line_search_fn='backtracking')
        with self.assertRaises(ValueError):fr.optim.LBFGS([{'params':[model.a]},{'params':[model.b]}])
        with self.assertRaises(ValueError):optimizer.add_param_group({'params':[model.a]})
        with self.assertRaises(ValueError):fr.optim.LBFGS([model.empty])
        grouped=fr.optim.LBFGS([{'params':list(model.parameters()),'history_size':2}],history_size=5)
        self.assertEqual(grouped.param_groups[0]['history_size'],2)
        run_step(model,grouped,0)
        grouped.load_state_dict(grouped.state_dict())
        model.a.trainable=False
        with self.assertRaises(ValueError):optimizer.step(lambda:self.fail('must preflight frozen params'))
        model.a.trainable=True
        self.assertEqual(optimizer.state_dict()['state'],{})


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        from test_lbfgs_optimizer import worker as run_worker
        run_worker(sys.argv[2])
    else:unittest.main(module='test_lbfgs_optimizer')
