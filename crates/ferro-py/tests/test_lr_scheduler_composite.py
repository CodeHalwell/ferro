"""Actual Torch composition, owning staged loads, and process continuation."""
import copy
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest
import warnings
import torch
import ferro as fr



class Factor:
    def __init__(self, value):self.value,self.epochs,self.fail=value,[],False
    def __call__(self, epoch):
        self.epochs.append(epoch)
        if self.fail:raise LookupError('callback sentinel')
        return self.value ** (epoch % 3)


def setup(reference=False):
    params=[torch.nn.Parameter(torch.tensor([1.])),torch.nn.Parameter(torch.tensor([2.]))] if reference else [fr.nn.Parameter(fr.tensor([1.])),fr.nn.Parameter(fr.tensor([2.]))]
    api=torch.optim if reference else fr.optim
    opt=api.SGD([{'params':[params[0]],'lr':.05},{'params':[params[1]],'lr':.02}],lr=.1,momentum=.6)
    return opt,api.lr_scheduler


def construct(optimizer,api,name):
    if name=='chain':return api.ChainedScheduler([api.ConstantLR(optimizer,factor=.3,total_iters=4),api.ExponentialLR(optimizer,gamma=.8)])
    if name=='nested':
        child=api.ExponentialLR(optimizer,gamma=.9)
        inner=api.ChainedScheduler([child,child])
        return api.ChainedScheduler([inner,api.StepLR(optimizer,step_size=3,gamma=.7)])
    if name=='warm_chain':return api.ChainedScheduler([api.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2),api.ExponentialLR(optimizer,gamma=.9)])
    if name=='sequence':return api.SequentialLR(optimizer,[api.LinearLR(optimizer,start_factor=.2,total_iters=3),api.CosineAnnealingLR(optimizer,T_max=5,eta_min=.001)],[3])
    if name=='warm_sequence':return api.SequentialLR(optimizer,[api.LinearLR(optimizer,start_factor=.3,total_iters=3),api.CosineAnnealingWarmRestarts(optimizer,T_0=4,T_mult=2)],[3])
    if name=='warm_first':return api.SequentialLR(optimizer,[api.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2),api.ExponentialLR(optimizer,gamma=.8)],[4])
    if name=='zero':return api.SequentialLR(optimizer,[api.ConstantLR(optimizer,factor=.4,total_iters=3),api.ExponentialLR(optimizer,gamma=.9),api.StepLR(optimizer,step_size=2,gamma=.8)],[0,0])
    if name=='callbacks':return api.SequentialLR(optimizer,[api.LambdaLR(optimizer,[Factor(.9),Factor(.8)]),api.MultiplicativeLR(optimizer,[Factor(.9),Factor(.8)])],[3])
    if name=='resume':return api.SequentialLR(optimizer,[api.ConstantLR(optimizer,factor=.2,total_iters=2),api.ExponentialLR(optimizer,gamma=.9)],[2],last_epoch=4)
    raise ValueError(name)


class Model(fr.nn.Module):
    def build(self):
        self.a=fr.nn.Parameter(fr.tensor([.4,-.6]))
        self.b=fr.nn.Parameter(fr.tensor([.7]))


def training_setup(name):
    model=Model()
    optimizer=fr.optim.SGD([{'params':[model.a],'lr':.05},{'params':[model.b],'lr':.02}],lr=.1,momentum=.6)
    return model,optimizer,construct(optimizer,fr.optim.lr_scheduler,name)


def update(model,optimizer,scheduler,generator,index):
    optimizer.zero_grad()
    a=model.a.tensor()+fr.randn(2,generator=generator)*.01
    b=model.b.tensor()+fr.randn(1,generator=generator)*.01
    loss=a.square().sum()+b.square().sum()*(1.+index*.02)
    loss.backward();optimizer.step();optimizer.zero_grad();scheduler.step()
    return loss.item(),scheduler.get_last_lr()


def suffix(model,optimizer,scheduler,generator,start):
    trace=[update(model,optimizer,scheduler,generator,index) for index in range(start,10)]
    checkpoint=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=10)
    digest=hashlib.sha256()
    for name,value in sorted(checkpoint.tensors().items()):
        digest.update(json.dumps([name,value.dtype,value.shape]).encode())
        data=value.reshape([value.numel()]).tolist()
        digest.update(struct.pack('<'+'f'*len(data),*data))
    return {'trace':trace,'checkpoint':digest.hexdigest(),'scheduler':scheduler.state_dict(),'draws':fr.randn(9,generator=generator).tolist()}


def worker(name,directory):
    directory=Path(directory);model,optimizer,scheduler=training_setup(name);generator=fr.Generator(999)
    saved=fr.checkpoint.load(directory/'training');saved.restore(model,optimizers={'main':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((directory/'scheduler.json').read_text()))
    (directory/'actual.json').write_text(json.dumps(suffix(model,optimizer,scheduler,generator,saved.step),allow_nan=False))


class CompositeSchedulerTests(unittest.TestCase):
    def test_actual_torch_rates_and_every_nested_state_at_all_boundaries(self):
        for name in ('chain','nested','warm_chain','sequence','warm_sequence','warm_first','zero','callbacks','resume'):
            with self.subTest(name=name),warnings.catch_warnings():
                warnings.simplefilter('ignore')
                optimizer,api=setup();reference,target=setup(True)
                scheduler=construct(optimizer,api,name);oracle=construct(reference,target,name)
                self.assertEqual(scheduler.state_dict(),oracle.state_dict())
                for epoch in range(13):
                    optimizer.step();reference.step()
                    scheduler.step();oracle.step()
                    self.assertEqual(scheduler.get_last_lr(),oracle.get_last_lr(),(name,epoch))
                    self.assertEqual(scheduler.state_dict(),oracle.state_dict(),(name,epoch))

    def test_load_stages_all_nested_and_callable_state_without_rate_changes(self):
        for name in ('nested','callbacks'):
            with self.subTest(name=name):
                optimizer,api=setup();scheduler=construct(optimizer,api,name)
                for _ in range(5):scheduler.step()
                saved=scheduler.state_dict();owned=copy.deepcopy(saved)
                scheduler.step();self.assertEqual(saved,owned)
                other,target=setup();restored=construct(other,target,name)
                before=restored.state_dict();rates=[g['lr'] for g in other.param_groups]
                bad=copy.deepcopy(saved)
                bad['_schedulers'][-1]['_last_lr']=[float('nan'),.02]
                with self.assertRaises(ValueError):restored.load_state_dict(bad)
                self.assertEqual(restored.state_dict(),before)
                self.assertEqual([g['lr'] for g in other.param_groups],rates)
                restored.load_state_dict(saved)
                self.assertEqual(restored.state_dict(),saved)
                self.assertEqual([g['lr'] for g in other.param_groups],rates)
                saved['_last_lr'][0]=.99
                self.assertNotEqual(restored.state_dict(),saved)

    def test_fresh_process_full_training_and_nested_callback_schedule_suffix(self):
        for name in ('chain','callbacks'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                directory=Path(directory);model,optimizer,scheduler=training_setup(name);generator=fr.Generator(7)
                for index in range(3):update(model,optimizer,scheduler,generator,index)
                fr.checkpoint.save_training(directory/'training',model,optimizers={'main':optimizer},generator=generator,step=3)
                (directory/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
                expected=json.loads(json.dumps(suffix(model,optimizer,scheduler,generator,3)))
                child=subprocess.run([sys.executable,__file__,'--worker',name,str(directory)],capture_output=True,text=True,timeout=60)
                self.assertEqual(child.returncode,0,child.stdout+child.stderr)
                self.assertEqual(json.loads((directory/'actual.json').read_text()),expected)

    def test_validation_rejects_unsupported_phase_initialization_before_rate_changes(self):
        optimizer,api=setup();a=api.ExponentialLR(optimizer,.9);b=api.StepLR(optimizer,2)
        before=[g['lr'] for g in optimizer.param_groups]
        for call in (lambda:api.ChainedScheduler([]),lambda:api.ChainedScheduler([a],optimizer=setup()[0]),
                     lambda:api.SequentialLR(optimizer,[a,b],[]),lambda:api.SequentialLR(optimizer,[a,b],[-1]),
                     lambda:api.SequentialLR(optimizer,[a,b],[1.2])):
            with self.assertRaises((ValueError,TypeError)):call()
            self.assertEqual([g['lr'] for g in optimizer.param_groups],before)
        warm=api.ChainedScheduler([api.ExponentialLR(optimizer,.9)])
        before=[g['lr'] for g in optimizer.param_groups]
        with self.assertRaises(NotImplementedError):api.SequentialLR(optimizer,[a,warm],[2])
        self.assertEqual([g['lr'] for g in optimizer.param_groups],before)

    def test_failed_initial_callback_restores_existing_children_and_optimizer_rates(self):
        optimizer,api=setup();factor=Factor(.8)
        a=api.LambdaLR(optimizer,factor);b=api.ExponentialLR(optimizer,.7)
        factor.fail=True
        before=[a.state_dict(),b.state_dict()]
        rates=[g['lr'] for g in optimizer.param_groups];native=optimizer._native.configurations()
        with self.assertRaisesRegex(LookupError,'callback sentinel'):
            api.SequentialLR(optimizer,[a,b],[3])
        self.assertEqual([a.state_dict(),b.state_dict()],before)
        self.assertEqual([g['lr'] for g in optimizer.param_groups],rates)
        self.assertEqual(optimizer._native.configurations(),native)
        self.assertIs(a.lr_lambdas[0],factor)


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        from test_lr_scheduler_composite import worker
        worker(sys.argv[2],sys.argv[3])
    else:unittest.main(verbosity=2)
