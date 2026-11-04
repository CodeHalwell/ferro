"""Actual Torch LR/momentum cycling, strict metadata and process resume."""
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


class Scale:
    def __init__(self,gamma):self.gamma,self.seen=gamma,[]
    def __call__(self,value):self.seen.append(value);return self.gamma**value


def options(kind):
    if kind=='SGD':return dict(lr=.01,momentum=.8,weight_decay=.01)
    if kind=='RMSprop':return dict(lr=.01,momentum=.8,alpha=.9,centered=True,eps=1e-5)
    return dict(lr=.01,betas=(.8,.91),weight_decay=.01)


class Model(fr.nn.Module):
    def build(self):
        self.a=fr.nn.Parameter(fr.tensor([.4,-.6]));self.b=fr.nn.Parameter(fr.tensor([.7]))


def setup(kind='SGD',reference=False,mode='triangular'):
    if reference:
        params=[torch.nn.Parameter(torch.tensor([.4,-.6])),torch.nn.Parameter(torch.tensor([.7]))]
        api=torch.optim
    else:
        model=Model();params=list(model.parameters());api=fr.optim
    optimizer=getattr(api,kind)([{'params':[p]} for p in params],**options(kind))
    kwargs=dict(base_lr=[.001,.003],max_lr=[.05,.035],step_size_up=2.5,step_size_down=3.5,
                base_momentum=[.7,.6],max_momentum=[.93,.87])
    if mode=='custom':kwargs.update(scale_fn=Scale(.97),scale_mode='iterations')
    else:kwargs.update(mode=mode,gamma=.98)
    scheduler=api.lr_scheduler.CyclicLR(optimizer,**kwargs)
    return (params if reference else model),optimizer,scheduler


def update(model,optimizer,scheduler,generator,index):
    optimizer.zero_grad();loss=None
    for pi,p in enumerate(model.parameters()):
        term=(p.tensor()+fr.randn(*p.tensor().shape,generator=generator)*.01).square().sum()*(.2+.03*index+.1*pi)
        loss=term if loss is None else loss+term
    loss.backward();optimizer.step();optimizer.zero_grad();scheduler.step()
    return loss.item(),scheduler.get_last_lr()


def suffix(model,optimizer,scheduler,generator,start):
    trace=[update(model,optimizer,scheduler,generator,i) for i in range(start,15)]
    cp=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=15)
    return dict(trace=trace,checkpoint=checkpoint_digest(cp),optimizer=plain(optimizer.state_dict()),scheduler=scheduler.state_dict(),draws=fr.randn(9,generator=generator).tolist())


def worker(kind,path):
    path=Path(path);model,optimizer,scheduler=setup(kind,mode='custom');generator=fr.Generator(999)
    cp=fr.checkpoint.load(path/'training');cp.restore(model,optimizers={'main':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((path/'scheduler.json').read_text()))
    (path/'actual.json').write_text(json.dumps(suffix(model,optimizer,scheduler,generator,cp.step),allow_nan=False))


class CyclicSchedulerTests(unittest.TestCase):
    compare_state=adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_actual_torch_group_rates_momentum_parameters_and_every_state(self):
        for kind in ('SGD','AdamW','RMSprop'):
            for mode in ('triangular','triangular2','exp_range','custom'):
                with self.subTest(kind=kind,mode=mode),warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model,optimizer,scheduler=setup(kind,mode=mode)
                    params,reference,oracle=setup(kind,True,mode)
                    self.assertEqual(scheduler.state_dict(),oracle.state_dict());self.compare_state(optimizer,reference)
                    for step in range(19):
                        optimizer.zero_grad();reference.zero_grad()
                        for pi,(p,tp) in enumerate(zip(model.parameters(),params)):
                            value=.2+.03*step+.1*pi
                            (p.tensor().square().sum()*value+p.tensor().sum()*.1).backward()
                            (tp.square().sum()*value+tp.sum()*.1).backward()
                        optimizer.step();reference.step()
                        for p,tp in zip(model.parameters(),params):
                            torch.testing.assert_close(torch.tensor(p.tensor().tolist()),tp.detach(),atol=5e-6,rtol=7e-5)
                        epoch=step+2 if step%4==2 else None
                        scheduler.step(epoch);oracle.step(epoch)
                        self.assertEqual(scheduler.state_dict(),oracle.state_dict());self.compare_state(optimizer,reference)
                    self.assertEqual(scheduler.get_lr(),oracle.get_lr());self.compare_state(optimizer,reference)

    def test_callback_load_is_owning_atomic_and_keeps_identity_without_changing_options(self):
        _,optimizer,scheduler=setup('AdamW',mode='custom')
        for _ in range(4):scheduler.step()
        saved=scheduler.state_dict();owned=copy.deepcopy(saved);scheduler.step();self.assertEqual(saved,owned)
        _,other,restored=setup('AdamW',mode='custom');callback=restored._scale_fn_custom
        before=restored.state_dict();rates=plain(other.state_dict())
        for key,value in (('max_lrs',[.1,float('nan')]),('use_beta1',False),('_scale_fn_custom',None)):
            bad=copy.deepcopy(saved);bad[key]=value
            with self.assertRaises((TypeError,ValueError)):restored.load_state_dict(bad)
            self.assertEqual(restored.state_dict(),before);self.assertEqual(plain(other.state_dict()),rates)
        restored.load_state_dict(saved);self.assertIs(restored._scale_fn_custom,callback)
        self.assertEqual(restored.state_dict(),saved);self.assertEqual(plain(other.state_dict()),rates)
        saved['_scale_fn_custom']['seen'].append(999);self.assertNotEqual(restored.state_dict(),saved)

    def test_fresh_process_continuation_preserves_momentum_options_metadata_and_rng(self):
        for kind in ('SGD','AdamW'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);model,optimizer,scheduler=setup(kind,mode='custom');generator=fr.Generator(19)
                for index in range(5):update(model,optimizer,scheduler,generator,index)
                fr.checkpoint.save_training(path/'training',model,optimizers={'main':optimizer},generator=generator,step=5)
                (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
                expected=json.loads(json.dumps(suffix(model,optimizer,scheduler,generator,5)))
                result=subprocess.run([sys.executable,__file__,'--worker',kind,str(path)],capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(json.loads((path/'actual.json').read_text()),expected)

    def test_constructor_and_failed_step_preflight_leave_groups_native_state_and_counters_unchanged(self):
        model=Model();optimizer=fr.optim.AdamW(model.parameters(),lr=.03)
        before=plain(optimizer.state_dict());native=optimizer._native.configurations()
        for options in ({'step_size_up':0},{'cycle_momentum':1},{'max_momentum':1.},
                        {'base_lr':[.1,.2,.3]},{'scale_mode':'invalid','scale_fn':lambda value:1.},
                        {'scale_fn':lambda value:float('nan')}):
            kwargs=dict(base_lr=.01,max_lr=.03);kwargs.update(options)
            with self.assertRaises((ValueError,TypeError)):
                fr.optim.lr_scheduler.CyclicLR(optimizer,**kwargs)
            self.assertEqual(plain(optimizer.state_dict()),before);self.assertEqual(optimizer._native.configurations(),native)
        callback=lambda epoch:1. if epoch==0 else float('nan')
        scheduler=fr.optim.lr_scheduler.CyclicLR(optimizer,base_lr=.01,max_lr=.03,scale_fn=callback,scale_mode='iterations')
        before=plain(optimizer.state_dict());state=scheduler.state_dict();native=optimizer._native.configurations()
        with self.assertRaises(ValueError):scheduler.step()
        self.assertEqual(scheduler.state_dict(),state);self.assertEqual(plain(optimizer.state_dict()),before)
        self.assertEqual(optimizer._native.configurations(),native)
        for initial,maximum in ((0.,.9),(.9,0.)):
            model=Model();optimizer=fr.optim.RMSprop(model.parameters(),lr=.03,momentum=initial)
            sum(p.tensor().square().sum() for p in model.parameters()).backward()
            optimizer.step();optimizer.zero_grad()
            before=plain(optimizer.state_dict());native=optimizer._native.configurations()
            with self.assertRaisesRegex(ValueError,'momentum presence'):
                fr.optim.lr_scheduler.CyclicLR(optimizer,.01,.03,base_momentum=0.,max_momentum=maximum)
            self.assertEqual(plain(optimizer.state_dict()),before);self.assertEqual(optimizer._native.configurations(),native)
        scheduler=fr.optim.lr_scheduler.CyclicLR(optimizer,.01,.03,step_size_up=1,base_momentum=0.,max_momentum=.9)
        before=plain(optimizer.state_dict());state=scheduler.state_dict();native=optimizer._native.configurations()
        with self.assertRaisesRegex(ValueError,'momentum presence'):scheduler.step()
        self.assertEqual(scheduler.state_dict(),state);self.assertEqual(plain(optimizer.state_dict()),before)
        self.assertEqual(optimizer._native.configurations(),native)

    def test_complete_absent_legacy_metadata_extension_and_late_faults_are_atomic(self):
        extras=('max_lr','min_lr','base_momentum','max_momentum')
        def snapshot(model,opt,rng,transform=lambda values:values):
            params,buffers,scalars,opts,generator,_=fr.checkpoint._state(model,{'main':opt},rng)
            native=fr._native._TrainingCheckpoint.snapshot(params,buffers,transform(scalars),opts,generator,3)
            return fr.checkpoint.TrainingCheckpoint(native)
        for kind in ('SGD','AdamW'):
            model=Model();optimizer=getattr(fr.optim,kind)([{'params':[model.a]},{'params':[model.b]}],**options(kind));generator=fr.Generator(19)
            for p in model.parameters():p.tensor().square().sum().backward()
            optimizer.step();optimizer.zero_grad();fr.randn(5,generator=generator)
            expected=snapshot(model,optimizer,generator)
            legacy=snapshot(model,optimizer,generator,lambda entries:[(name,value) for name,value in entries if not any('.'+key in name for key in extras)])
            target=Model();other=getattr(fr.optim,kind)([{'params':[target.a]},{'params':[target.b]}],**options(kind));rng=fr.Generator(999)
            legacy.restore(target,optimizers={'main':other},generator=rng)
            self.assertEqual(checkpoint_digest(snapshot(target,other,rng)),checkpoint_digest(expected))
            before=checkpoint_digest(snapshot(target,other,rng))
            prefix='optim_meta.main.g1.max_momentum'
            faults=[lambda entries:[(name,value) for name,value in entries if name!=prefix+'.present'],
                    lambda entries:[(name,value) for name,value in entries if not any(name.startswith('optim_meta.main.g1.'+key) for key in extras)],
                    lambda entries:[(name,2 if name==prefix+'.present' else value) for name,value in entries],
                    lambda entries:[(name,0x3ff0000000000000 if name==prefix else value) for name,value in entries],
                    lambda entries:entries+[('optim_meta.main.g1.unknown_scheduler_field',0)]]
            for transform in faults:
                bad=snapshot(model,optimizer,generator,transform)
                with self.assertRaises(ValueError):bad.restore(target,optimizers={'main':other},generator=rng)
                self.assertEqual(checkpoint_digest(snapshot(target,other,rng)),before)

    def test_composite_load_preserves_custom_scale_identity_and_stages_later_children(self):
        _,optimizer,scheduler=setup('AdamW',mode='custom')
        chain=fr.optim.lr_scheduler.ChainedScheduler([scheduler,fr.optim.lr_scheduler.StepLR(optimizer,3,gamma=.8)])
        for _ in range(5):chain.step()
        saved=chain.state_dict()
        _,other,restored=setup('AdamW',mode='custom')
        target=fr.optim.lr_scheduler.ChainedScheduler([restored,fr.optim.lr_scheduler.StepLR(other,3,gamma=.8)])
        callback=restored._scale_fn_custom;before=target.state_dict();options=plain(other.state_dict())
        bad=copy.deepcopy(saved)
        bad['_schedulers'][0]['_scale_fn_custom']['seen'].append(999)
        bad['_schedulers'][1]['_last_lr']=[float('nan'),.01]
        with self.assertRaises(ValueError):target.load_state_dict(bad)
        self.assertIs(restored._scale_fn_custom,callback)
        self.assertEqual(target.state_dict(),before);self.assertEqual(plain(other.state_dict()),options)
        other.load_state_dict(optimizer.state_dict());target.load_state_dict(saved)
        self.assertIs(restored._scale_fn_custom,callback);self.assertEqual(target.state_dict(),saved)
        for _ in range(4):
            chain.step();target.step()
            self.assertEqual(chain.state_dict(),target.state_dict())
            self.assertEqual(plain(optimizer.state_dict()),plain(other.state_dict()))

    def test_scheduler_commits_preserve_initialized_native_configuration_guards(self):
        cases=(('RMSprop','momentum',.9),('RMSprop','centered',True),
               ('Adam','amsgrad',True),('AdamW','amsgrad',True),('SGD','dampening',.1))
        for kind,key,value in cases:
            for cyclic in (False,True):
                with self.subTest(kind=kind,key=key,cyclic=cyclic):
                    model=Model();optimizer=getattr(fr.optim,kind)([{'params':[p]} for p in model.parameters()],lr=.03)
                    sum(p.tensor().square().sum() for p in model.parameters()).backward()
                    optimizer.step();optimizer.zero_grad()
                    scheduler=(fr.optim.lr_scheduler.CyclicLR(optimizer,.01,.03,cycle_momentum=False) if cyclic
                               else fr.optim.lr_scheduler.StepLR(optimizer,1,gamma=.5))
                    optimizer.param_groups[-1][key]=value
                    state=scheduler.state_dict();groups=[dict(group) for group in optimizer.param_groups]
                    configs=optimizer._native.configurations()
                    buffers=lambda:[(name,tensor.shape,tensor.tolist()) for name,tensor in optimizer._native.state_tensors()]
                    saved_buffers=buffers();parameters=[p.tensor().tolist() for p in model.parameters()]
                    with self.assertRaises(ValueError):optimizer._sync()
                    with self.assertRaises(ValueError):scheduler.step()
                    self.assertEqual(scheduler.state_dict(),state);self.assertEqual(optimizer.param_groups,groups)
                    self.assertEqual(optimizer._native.configurations(),configs);self.assertEqual(buffers(),saved_buffers)
                    self.assertEqual([p.tensor().tolist() for p in model.parameters()],parameters)


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        from test_lr_scheduler_cyclic import worker
        worker(sys.argv[2],sys.argv[3])
    else:unittest.main(verbosity=2)
