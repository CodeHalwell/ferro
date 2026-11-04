"""Independent Torch rates, mutable group rates and validated scheduler resumes."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import torch
import ferro as fr
from test_optimizer_groups_schedulers import Model, checkpoint_digest, plain

CASES = (('MultiStepLR', dict(milestones=[0, 3, 3, 7], gamma=.6)),
         ('ConstantLR', dict(factor=.3, total_iters=4)),
         ('LinearLR', dict(start_factor=.2, end_factor=.8, total_iters=5)),
         ('PolynomialLR', dict(total_iters=6, power=1.7)))


def setup(name, options):
    model=Model();a,b,c=list(model.parameters())
    optimizer=fr.optim.SGD([{'params':[a,c],'lr':.08},{'params':[b],'lr':.025}],lr=.1,momentum=.6)
    scheduler=getattr(fr.optim.lr_scheduler,name)(optimizer,**options)
    return model,optimizer,scheduler


def update(model,optimizer,generator):
    optimizer.zero_grad();loss=None
    for parameter in model.parameters():
        term=(parameter.tensor()+fr.Tensor.randn(parameter.tensor().shape,generator=generator)*.01).square().sum()
        loss=term if loss is None else loss+term
    loss.backward();optimizer.step();optimizer.zero_grad()
    return loss.item()


def suffix(model,optimizer,scheduler,generator,start):
    trace=[]
    for _ in range(start,10):
        trace.append([update(model,optimizer,generator),scheduler.get_last_lr()]);scheduler.step()
    return dict(trace=trace,state=plain(optimizer.state_dict()),scheduler=scheduler.state_dict(),
                checkpoint=checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=10)),
                draws=fr.Tensor.randn([5],generator=generator).tolist())


def worker(name,directory):
    path=Path(directory);options=dict(CASES)[name]
    model,optimizer,scheduler=setup(name,options);generator=fr.Generator(999)
    checkpoint=fr.checkpoint.load(path/'training')
    checkpoint.restore(model,optimizers={'main':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((path/'scheduler.json').read_text()))
    (path/'result.json').write_text(json.dumps(suffix(model,optimizer,scheduler,generator,checkpoint.step),allow_nan=False))


class ExtendedSchedulerTests(unittest.TestCase):
    def assert_rates(self, scheduler, target):
        self.assertEqual(scheduler.last_epoch,target.last_epoch)
        for actual,expected in zip(scheduler.get_last_lr(),target.get_last_lr()):
            self.assertAlmostEqual(actual,expected,places=13)

    def test_all_four_schedules_match_torch_automatic_and_explicit_epochs(self):
        for name,options in CASES:
            for explicit in (False,True):
                with self.subTest(name=name,explicit=explicit),warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _,optimizer,scheduler=setup(name,options)
                    p,q=torch.nn.Parameter(torch.tensor(1.)),torch.nn.Parameter(torch.tensor(2.))
                    target_optimizer=torch.optim.SGD([{'params':[p],'lr':.08},{'params':[q],'lr':.025}],lr=.1,momentum=.6)
                    target=getattr(torch.optim.lr_scheduler,name)(target_optimizer,**options)
                    self.assert_rates(scheduler,target)
                    epochs=[1,2,3,5,6,7,9,12] if explicit else [None]*12
                    for step,epoch in enumerate(epochs):
                        if step==2:
                            optimizer.param_groups[1]['lr']*=1.4;target_optimizer.param_groups[1]['lr']*=1.4
                        optimizer.step();target_optimizer.step()
                        scheduler.step(epoch);target.step(epoch);self.assert_rates(scheduler,target)

    def test_initial_resume_epoch_and_boundary_factors_match_torch(self):
        cases=CASES+(('MultiStepLR',dict(milestones=[],gamma=0.)),('ConstantLR',dict(factor=1.,total_iters=1)),
                     ('LinearLR',dict(start_factor=1.,end_factor=0.,total_iters=2)),
                     ('LinearLR',dict(start_factor=.2,end_factor=0.,total_iters=4)),('PolynomialLR',dict(total_iters=1,power=0.)))
        for name,options in cases:
            with self.subTest(name=name,options=options),warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p=fr.nn.Parameter(fr.Tensor([1.],[1]));optimizer=fr.optim.SGD([{'params':[p],'lr':.03,'initial_lr':.07}],lr=.1)
                tp=torch.nn.Parameter(torch.tensor([1.]));target_optimizer=torch.optim.SGD([{'params':[tp],'lr':.03,'initial_lr':.07}],lr=.1)
                scheduler=getattr(fr.optim.lr_scheduler,name)(optimizer,last_epoch=2,**options)
                target=getattr(torch.optim.lr_scheduler,name)(target_optimizer,last_epoch=2,**options)
                self.assert_rates(scheduler,target)
                for _ in range(5):
                    optimizer.step();target_optimizer.step();scheduler.step();target.step();self.assert_rates(scheduler,target)

    def test_state_is_owned_and_invalid_load_or_options_leave_rates_unchanged(self):
        for name,options in CASES:
            model,optimizer,scheduler=setup(name,options)
            scheduler.step();saved=scheduler.state_dict();owned=copy.deepcopy(saved)
            scheduler.step();self.assertEqual(saved,owned)
            before=copy.deepcopy(scheduler.state_dict());rates=[g['lr'] for g in optimizer.param_groups]
            for key,value in (('last_epoch',-2),('base_lrs',[.1]),('_step_count',1.5),('_last_lr',[float('nan'),0.])):
                malformed=copy.deepcopy(saved);malformed[key]=value
                with self.assertRaises(ValueError):scheduler.load_state_dict(malformed)
                self.assertEqual(scheduler.state_dict(),before);self.assertEqual([g['lr'] for g in optimizer.param_groups],rates)
            scheduler.load_state_dict(saved)
            self.assertEqual(scheduler.state_dict(),saved);self.assertEqual([g['lr'] for g in optimizer.param_groups],rates)
        _,optimizer,_=setup(*CASES[0]);rates=[g['lr'] for g in optimizer.param_groups]
        bad=(('MultiStepLR',dict(milestones=[1,.5])),('ConstantLR',dict(factor=0)),('LinearLR',dict(start_factor=0)),
             ('PolynomialLR',dict(total_iters=0)),('PolynomialLR',dict(power=float('nan'))))
        for name,options in bad:
            with self.assertRaises(ValueError):getattr(fr.optim.lr_scheduler,name)(optimizer,**options)
            self.assertEqual([g['lr'] for g in optimizer.param_groups],rates)

    def test_each_schedule_exact_fresh_process_continuation(self):
        for name,options in CASES:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);model,optimizer,scheduler=setup(name,options);generator=fr.Generator(43)
                for _ in range(4):update(model,optimizer,generator);scheduler.step()
                fr.checkpoint.save_training(path/'training',model,optimizers={'main':optimizer},generator=generator,step=4)
                (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
                expected=suffix(model,optimizer,scheduler,generator,4)
                env=dict(os.environ);env.pop('PYTHONPATH',None)
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name,str(path)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                self.assertEqual(child.returncode,0,child.stdout)
                self.assertEqual(json.loads((path/'result.json').read_text()),expected)


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3])
    else:unittest.main()
