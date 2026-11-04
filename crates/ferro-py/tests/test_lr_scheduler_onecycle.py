"""Actual Torch one-cycle trajectories and transactional state tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
import torch
import ferro as fr
import test_adaptive_optimizers as adaptive_tests
from test_optimizer_groups_schedulers import plain,checkpoint_digest


class Model(fr.nn.Module):
    def build(self):
        self.a=fr.nn.Parameter(fr.tensor([.4,-.6]));self.b=fr.nn.Parameter(fr.tensor([.7]))


def options(kind):
    if kind=='SGD':return dict(lr=.01,momentum=.8,weight_decay=.01)
    if kind=='RMSprop':return dict(lr=.01,momentum=.8,alpha=.9,centered=True,eps=1e-5)
    if kind=='Adagrad':return dict(lr=.01,lr_decay=.01)
    return dict(lr=.01,betas=(.8,.91),weight_decay=.01)


def setup(kind='SGD',reference=False,strategy='cos',three=False,cycle=True):
    if reference:
        model=[torch.nn.Parameter(torch.tensor([.4,-.6])),torch.nn.Parameter(torch.tensor([.7]))];params=model;api=torch.optim
    else:
        model=Model();params=list(model.parameters());api=fr.optim
    optimizer=getattr(api,kind)([{'params':[p]} for p in params],**options(kind))
    scheduler=api.lr_scheduler.OneCycleLR(optimizer,max_lr=[.05,.035],total_steps=24,pct_start=.35,
        anneal_strategy=strategy,three_phase=three,cycle_momentum=cycle,
        base_momentum=[.7,.6],max_momentum=[.93,.87])
    return model,optimizer,scheduler


def update(model,optimizer,scheduler,generator,index):
    optimizer.zero_grad();loss=None
    for pi,p in enumerate(model.parameters()):
        term=(p.tensor()+fr.randn(*p.tensor().shape,generator=generator)*.01).square().sum()*(.2+.03*index+.1*pi)
        loss=term if loss is None else loss+term
    loss.backward();optimizer.step();optimizer.zero_grad();scheduler.step()
    return loss.item(),scheduler.get_last_lr()


def suffix(model,optimizer,scheduler,generator,start):
    trace=[update(model,optimizer,scheduler,generator,i) for i in range(start,20)]
    cp=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=20)
    return dict(trace=trace,checkpoint=checkpoint_digest(cp),optimizer=plain(optimizer.state_dict()),scheduler=scheduler.state_dict(),draws=fr.randn(9,generator=generator).tolist())


def worker(kind,directory):
    path=Path(directory);model,optimizer,scheduler=setup(kind,strategy='linear',three=True);generator=fr.Generator(999)
    checkpoint=fr.checkpoint.load(path/'training')
    checkpoint.restore(model,optimizers={'main':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((path/'scheduler.json').read_text()))
    (path/'actual.json').write_text(json.dumps(suffix(model,optimizer,scheduler,generator,checkpoint.step),allow_nan=False))


class OneCycleSchedulerTests(unittest.TestCase):
    compare_state=adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_actual_torch_all_phases_rates_momenta_parameters_and_state(self):
        for kind in ('SGD','AdamW','RMSprop'):
            for strategy in ('linear','cos'):
                for three in (False,True):
                    with self.subTest(kind=kind,strategy=strategy,three=three),warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        model,optimizer,scheduler=setup(kind,strategy=strategy,three=three)
                        params,reference,oracle=setup(kind,True,strategy,three)
                        self.assertEqual(scheduler.state_dict(),oracle.state_dict());self.compare_state(optimizer,reference)
                        for step in range(23):
                            optimizer.zero_grad();reference.zero_grad()
                            for pi,(p,tp) in enumerate(zip(model.parameters(),params)):
                                value=.2+.03*step+.1*pi
                                (p.tensor().square().sum()*value+p.tensor().sum()*.1).backward()
                                (tp.square().sum()*value+tp.sum()*.1).backward()
                            optimizer.step();reference.step()
                            for p,tp in zip(model.parameters(),params):
                                torch.testing.assert_close(torch.tensor(p.tensor().tolist()),tp.detach(),atol=5e-6,rtol=7e-5)
                            scheduler.step();oracle.step()
                            self.assertEqual(scheduler.state_dict(),oracle.state_dict());self.compare_state(optimizer,reference)
                        self.assertEqual(scheduler.get_lr(),oracle.get_lr());self.compare_state(optimizer,reference)

    def test_no_momentum_inferred_budget_explicit_epochs_and_resumed_constructor(self):
        for kind in ('SGD','Adagrad'):
            with self.subTest(kind=kind),warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model=Model();optimizer=getattr(fr.optim,kind)(model.parameters(),**options(kind))
                params=[torch.nn.Parameter(torch.tensor(p.tensor().tolist())) for p in model.parameters()]
                reference=getattr(torch.optim,kind)(params,**options(kind))
                kwargs=dict(max_lr=.03,epochs=3,steps_per_epoch=8,cycle_momentum=False,pct_start=.4)
                scheduler=fr.optim.lr_scheduler.OneCycleLR(optimizer,**kwargs)
                oracle=torch.optim.lr_scheduler.OneCycleLR(reference,**kwargs)
                for epoch in (None,4,8,9,13,19,23,24):
                    scheduler.step(epoch);oracle.step(epoch)
                    self.assertEqual(scheduler.state_dict(),oracle.state_dict());self.compare_state(optimizer,reference)
                before=scheduler.state_dict();state=plain(optimizer.state_dict())
                with self.assertRaises(ValueError):scheduler.step()
                self.assertEqual(scheduler.state_dict(),before);self.assertEqual(plain(optimizer.state_dict()),state)
                resumed=fr.optim.lr_scheduler.OneCycleLR(optimizer,last_epoch=5,**kwargs)
                target=torch.optim.lr_scheduler.OneCycleLR(reference,last_epoch=5,**kwargs)
                self.assertEqual(resumed.state_dict(),target.state_dict());self.compare_state(optimizer,reference)

    def test_exact_fresh_process_restores_every_group_metadata_field_and_future_draws(self):
        for kind in ('SGD','AdamW'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);model,optimizer,scheduler=setup(kind,strategy='linear',three=True);generator=fr.Generator(19)
                for index in range(7):update(model,optimizer,scheduler,generator,index)
                fr.checkpoint.save_training(path/'training',model,optimizers={'main':optimizer},generator=generator,step=7)
                (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
                expected=json.loads(json.dumps(suffix(model,optimizer,scheduler,generator,7)))
                child=subprocess.run([sys.executable,__file__,'--worker',kind,str(path)],capture_output=True,text=True,timeout=60)
                self.assertEqual(child.returncode,0,child.stdout+child.stderr)
                self.assertEqual(json.loads((path/'actual.json').read_text()),expected)

    def test_state_owns_phase_dicts_and_rejects_late_corruption_without_mutation(self):
        _,optimizer,scheduler=setup('AdamW',three=True)
        for _ in range(6):scheduler.step()
        saved=scheduler.state_dict();before=copy.deepcopy(saved)
        _,other,restored=setup('AdamW',three=True)
        current=restored.state_dict();options_before=plain(other.state_dict())
        faults=[]
        for transform in (lambda state:state['_schedule_phases'][-1].update(end_lr='unknown'),
                          lambda state:state['_schedule_phases'][-1].update(end_step=float('nan')),
                          lambda state:state['_schedule_phases'][1].update(end_step=3),
                          lambda state:state.update(last_epoch=25),lambda state:state.update(use_beta1=False)):
            bad=copy.deepcopy(saved);transform(bad);faults.append(bad)
        for bad in faults:
            with self.assertRaises((TypeError,ValueError)):restored.load_state_dict(bad)
            self.assertEqual(restored.state_dict(),current);self.assertEqual(plain(other.state_dict()),options_before)
        restored.load_state_dict(saved);self.assertEqual(restored.state_dict(),saved)
        self.assertEqual(plain(other.state_dict()),options_before)
        saved['_schedule_phases'][0]['end_step']=99
        self.assertEqual(restored.state_dict(),before)

    def test_negative_linear_terminal_rate_is_explicit_atomic_boundary(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model=Model();optimizer=fr.optim.SGD(model.parameters(),lr=.1,momentum=.9)
            parameter=torch.nn.Parameter(torch.ones(1));reference=torch.optim.SGD([parameter],lr=.1,momentum=.9)
            scheduler=fr.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.1,total_steps=10,anneal_strategy='linear')
            oracle=torch.optim.lr_scheduler.OneCycleLR(reference,max_lr=.1,total_steps=10,anneal_strategy='linear')
            for _ in range(9):
                self.assertEqual(scheduler.get_last_lr(),oracle.get_last_lr());scheduler.step();oracle.step()
            self.assertEqual(scheduler.get_last_lr(),oracle.get_last_lr())
            before=scheduler.state_dict();state=plain(optimizer.state_dict());native=optimizer._native.configurations()
            oracle.step();self.assertLess(reference.param_groups[0]['lr'],0.)
            with self.assertRaisesRegex(ValueError,'negative extrapolation'):scheduler.step()
            self.assertEqual(scheduler.state_dict(),before);self.assertEqual(plain(optimizer.state_dict()),state)
            self.assertEqual(optimizer._native.configurations(),native)

    def test_invalid_construction_rolls_back_metadata_native_settings_and_groups(self):
        model=Model();optimizer=fr.optim.AdamW(model.parameters(),lr=.03)
        state=plain(optimizer.state_dict());native=optimizer._native.configurations()
        cases=[dict(total_steps=0),dict(pct_start=float('nan')),dict(three_phase=1),dict(max_momentum=1.),
               dict(anneal_strategy='unknown'),dict(div_factor=0),dict(final_div_factor=0),
               dict(total_steps=10,pct_start=.1),dict(last_epoch=3),dict(cycle_momentum=1)]
        for kwargs in cases:
            options=dict(max_lr=.1,total_steps=24);options.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):
                fr.optim.lr_scheduler.OneCycleLR(optimizer,**options)
            self.assertEqual(plain(optimizer.state_dict()),state);self.assertEqual(optimizer._native.configurations(),native)
        optimizer=fr.optim.RMSprop(model.parameters(),lr=.03,momentum=0.)
        sum(p.tensor().square().sum() for p in model.parameters()).backward()
        optimizer.step();optimizer.zero_grad()
        state=plain(optimizer.state_dict());native=optimizer._native.configurations()
        with self.assertRaisesRegex(ValueError,'momentum presence'):
            fr.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.1,total_steps=24)
        self.assertEqual(plain(optimizer.state_dict()),state);self.assertEqual(optimizer._native.configurations(),native)


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        from test_lr_scheduler_onecycle import worker
        worker(sys.argv[2],sys.argv[3])
    else:unittest.main(verbosity=2)
