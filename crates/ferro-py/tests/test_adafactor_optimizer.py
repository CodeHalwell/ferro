"""Actual Torch factored state, grouped update and fresh-process continuation."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import ferro as fr
import test_adaptive_optimizers as adaptive_tests
from test_adaptive_optimizers import update, continuation
from test_optimizer_groups_schedulers import checkpoint_digest, plain


class Model(fr.nn.Module):
    def build(self):
        for name, shape in (('scalar', []), ('vector', [3]), ('matrix', [2, 3]), ('batched', [2, 2, 3])):
            setattr(self, name, fr.nn.Parameter(fr.Tensor([.6-.07*i for i in range(math.prod(shape))], shape)))


def specifications(parameters, wrong=False):
    a,b,c,d=parameters
    groups=[{'params':[a,c],'lr':.05},{'params':[b,d],'lr':.02,'beta2_decay':-.6,'eps':(1e-5,.002),'d':2.,'maximize':True}]
    if wrong:
        groups[0].update(lr=.4,eps=(.1,.1));groups[1]['lr']=.3
    return groups,dict(lr=.03,beta2_decay=-.7,eps=(None,.001),d=1.3,weight_decay=.03)


def setup(wrong=False):
    model=Model();groups,options=specifications(list(model.parameters()),wrong)
    return model,fr.optim.Adafactor(groups,**options)


def worker(directory):
    path=Path(directory);model,optimizer=setup(True)
    scheduler=fr.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=8,eta_min=.001)
    generator=fr.Generator(999);checkpoint=fr.checkpoint.load(path/'training')
    checkpoint.restore(model,optimizers={'main':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((path/'scheduler.json').read_text()))
    (path/'result.json').write_text(json.dumps(continuation(model,optimizer,scheduler,generator,checkpoint.step),allow_nan=False))


class AdafactorTests(unittest.TestCase):
    compare_state=adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_all_factor_shapes_and_values_match_torch_across_groups_skips_and_freezes(self):
        model,optimizer=setup();params=list(model.parameters())
        others=[torch.nn.Parameter(torch.tensor(p.tensor().tolist()).reshape(p.tensor().shape)) for p in params]
        groups,options=specifications(others);oracle=torch.optim.Adafactor(groups,**options,foreach=False)
        self.compare_state(optimizer,oracle)
        for step in range(35):
            optimizer.zero_grad();oracle.zero_grad(set_to_none=True)
            if step==11:optimizer.param_groups[0]['lr']=oracle.param_groups[0]['lr']=.04
            if step==19:optimizer.param_groups[0]['eps']=oracle.param_groups[0]['eps']=(2e-6,.001)
            if step==24:optimizer.param_groups[0]['eps']=oracle.param_groups[0]['eps']=(None,.001)
            for pi,(p,tp) in enumerate(zip(params,others)):
                p.trainable=not(pi==1 and step%5==2)
                if (step+pi)%4==0:continue
                scale,shift=.2+.03*pi+.017*step,-.15+.04*pi
                (p.tensor().square().sum()*scale+p.tensor().sum()*shift).backward()
                (tp.square().sum()*scale+tp.sum()*shift).backward()
                if not p.trainable:tp.grad=None
            optimizer.step();oracle.step()
            self.compare_state(optimizer,oracle)
            for p,tp in zip(params,others):
                torch.testing.assert_close(torch.tensor(p.tensor().tolist()).reshape(p.tensor().shape),tp.detach(),atol=4e-6,rtol=6e-5)
        optimizer.zero_grad()
        state=optimizer.state_dict()['state']
        # Native storage factors genuinely reduce the matrix/tensor state.
        self.assertEqual(state[1]['row_var'].shape,[2,1]);self.assertEqual(state[1]['col_var'].shape,[1,3])
        self.assertEqual(state[3]['row_var'].shape,[2,2,1]);self.assertEqual(state[3]['col_var'].shape,[2,1,3])
        before=plain(optimizer.state_dict());saved=optimizer.state_dict()
        update(model,optimizer,fr.Generator(3),37)
        self.assertEqual(plain(saved),before)
        optimizer.load_state_dict(saved);self.assertEqual(plain(optimizer.state_dict()),before)

    def test_relative_step_cap_zero_gradient_decay_and_epsilon_boundaries_match_torch(self):
        options=(dict(lr=1.2,beta2_decay=0.,eps=(.001,0.),d=1.),
                 dict(lr=0.,beta2_decay=-1.,eps=(None,.001),d=3.),
                 dict(lr=.3,beta2_decay=-.4,eps=(0.,.002),d=1.))
        for config in options:
            with self.subTest(config=config):
                p=fr.nn.Parameter(fr.Tensor([.7,-.2,.4,-.5],[2,2]));tp=torch.nn.Parameter(torch.tensor([[.7,-.2],[.4,-.5]]))
                opt=fr.optim.Adafactor([p],weight_decay=.07,**config);oracle=torch.optim.Adafactor([tp],weight_decay=.07,foreach=False,**config)
                for index in range(8):
                    oracle.zero_grad();opt.zero_grad()
                    weight=0. if index==0 and config['eps'][0]!=0. else .2+index*.03
                    (p.tensor().sum()*weight).backward();(tp.sum()*weight).backward();opt.step();oracle.step()
                    self.compare_state(opt,oracle)
                    torch.testing.assert_close(torch.tensor(p.tensor().tolist()),tp.detach(),atol=4e-6,rtol=6e-5)
        p=fr.nn.Parameter(fr.Tensor([.7],[1]));tp=torch.nn.Parameter(torch.tensor([.7]))
        opt=fr.optim.Adafactor([p],eps=(0.,0.));oracle=torch.optim.Adafactor([tp],eps=(0.,0.),foreach=False)
        (p.tensor().sum()*0.).backward();(tp.sum()*0.).backward();opt.step();oracle.step()
        torch.testing.assert_close(torch.tensor(p.tensor().tolist()),tp.detach(),equal_nan=True)

    def test_added_group_lazy_state_matches_torch(self):
        a=fr.nn.Parameter(fr.Tensor([.3],[1]));b=fr.nn.Parameter(fr.Tensor([.7,-.2],[1,2]))
        ta=torch.nn.Parameter(torch.tensor([.3]));tb=torch.nn.Parameter(torch.tensor([[.7,-.2]]))
        opt=fr.optim.Adafactor([a]);oracle=torch.optim.Adafactor([ta],foreach=False)
        opt.add_param_group({'params':[b],'eps':(.02,.03)});oracle.add_param_group({'params':[tb],'eps':(.02,.03)})
        self.compare_state(opt,oracle)
        for p,tp in ((a,ta),(b,tb)):(p.tensor().square().sum()).backward();tp.square().sum().backward()
        opt.step();oracle.step();self.compare_state(opt,oracle)

    def test_state_and_late_group_checkpoint_corruption_reject_atomically(self):
        from test_training_checkpoint import rewrite_checkpoint
        model,optimizer=setup();generator=fr.Generator(7)
        for i in range(4):update(model,optimizer,generator,i)
        before=checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator))
        for fault in ('shape','counter','unknown','eps','inactive'):
            saved=optimizer.state_dict()
            if fault=='shape':saved['state'][3]['col_var']=fr.Tensor.ones([99])
            elif fault=='counter':saved['state'][3]['step']=fr.Tensor([-.5],[])
            elif fault=='unknown':saved['state'][3]['extra']=fr.Tensor([1.],[])
            elif fault=='eps':saved['param_groups'][1]['eps']=(float('nan'),.001)
            else:saved['state'][3]['variance']=fr.Tensor.ones([2,2,3])
            with self.subTest(fault=fault),self.assertRaises(ValueError):optimizer.load_state_dict(saved)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)),before)
        saved=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)
        fields={'optim.main.g1.col_var.1':fr.Tensor.ones([99]),'optim.main.g1.t.1':fr.Tensor([float('inf')],[]),
                'optim.main.extra':fr.Tensor([1.],[]),'scalars.optim_meta.main.g0.eps.0':fr.Tensor([1.,0.,0.,0.],[4])}
        for name,value in fields.items():
            with tempfile.TemporaryDirectory() as directory:
                malformed=rewrite_checkpoint(saved,directory,{name:value})
                with self.assertRaises(ValueError):malformed.restore(model,optimizers={'main':optimizer},generator=generator)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)),before)

    def test_fresh_process_restores_factored_history_none_epsilon_groups_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);model,optimizer=setup();generator=fr.Generator(43)
            scheduler=fr.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=8,eta_min=.001)
            for index in range(7):update(model,optimizer,generator,index);scheduler.step()
            fr.checkpoint.save_training(path/'training',model,optimizers={'main':optimizer},generator=generator,step=7)
            (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
            expected=continuation(model,optimizer,scheduler,generator,7)
            env=dict(os.environ);env.pop('PYTHONPATH',None)
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            self.assertEqual(child.returncode,0,child.stdout)
            self.assertEqual(json.loads((path/'result.json').read_text()),expected)

    def test_invalid_configuration_and_empty_active_gradient_preflight(self):
        model,optimizer=setup()
        for options in (dict(beta2_decay=.1),dict(beta2_decay=float('-inf')),dict(eps=(-1.,.01)),dict(eps=(None,-.1)),dict(d=.99),dict(maximize=1)):
            with self.assertRaises(ValueError):fr.optim.Adafactor(model.parameters(),**options)
        p=fr.nn.Parameter(fr.Tensor([.7],[1]));empty=fr.nn.Parameter(fr.Tensor([],[0,3]));opt=fr.optim.Adafactor([p,empty])
        before=plain(opt.state_dict());value=p.tensor().tolist()
        p.tensor().sum().backward();empty.tensor().sum().backward()
        with self.assertRaises(ValueError):opt.step()
        self.assertEqual(plain(opt.state_dict()),before);self.assertEqual(p.tensor().tolist(),value)
        opt.zero_grad();empty.trainable=False;p.tensor().sum().backward();opt.step()
        self.assertEqual(set(opt.state_dict()['state']),{0})
        huge=fr.nn.Parameter(fr.Tensor([],[1<<62,0]))
        with self.assertRaisesRegex(ValueError,'factor exceeds'):
            fr.optim.Adafactor([huge])


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        from test_adafactor_optimizer import worker as run_worker
        run_worker(sys.argv[2])
    else:unittest.main(module='test_adafactor_optimizer')
