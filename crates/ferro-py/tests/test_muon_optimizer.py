"""Actual Torch Muon values, BF16 updates, groups and complete native restart."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
import ferro as fr
import test_adaptive_optimizers as adaptive_tests
from test_optimizer_groups_schedulers import plain,checkpoint_digest


def owned(value):
    if isinstance(value,fr.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {key:owned(item) for key,item in value.items()}
    if isinstance(value,list):return [owned(item) for item in value]
    if isinstance(value,tuple):return tuple(owned(item) for item in value)
    return value


class Model(fr.nn.Module):
    def build(self):
        self.a=fr.nn.Parameter(fr.tensor([[.4,-.6,.3],[.1,.7,-.8]]))
        self.b=fr.nn.Parameter(fr.tensor([[.7,.2],[-.5,.3],[.1,-.4]]))


def setup(reference=False):
    if reference:
        model=[torch.nn.Parameter(torch.tensor([[.4,-.6,.3],[.1,.7,-.8]])),
               torch.nn.Parameter(torch.tensor([[.7,.2],[-.5,.3],[.1,-.4]]))]
        params=model;api=torch.optim
    else:
        model=Model();params=list(model.parameters());api=fr.optim
    groups=[dict(params=[params[0]],lr=.002,momentum=.95,nesterov=True,adjust_lr_fn=None),
            dict(params=[params[1]],lr=.003,momentum=.83,nesterov=False,adjust_lr_fn='match_rms_adamw',ns_coefficients=(3.2,-4.1,1.9))]
    return model,api.Muon(groups)


def update(model,optimizer,scheduler,generator,index):
    optimizer.zero_grad();loss=None
    for pi,p in enumerate(model.parameters()):
        if (index+pi)%5==2:continue
        term=(p.tensor()+fr.randn(*p.tensor().shape,generator=generator)*.03).square().sum()*(.3+.02*index+.1*pi)
        loss=term if loss is None else loss+term
    loss.backward();optimizer.step();optimizer.zero_grad();scheduler.step()
    return loss.item(),scheduler.get_last_lr()


def suffix(model,optimizer,scheduler,generator,start):
    trace=[update(model,optimizer,scheduler,generator,index) for index in range(start,15)]
    checkpoint=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator,step=15)
    return dict(trace=trace,checkpoint=checkpoint_digest(checkpoint),optimizer=plain(optimizer.state_dict()),
                scheduler=scheduler.state_dict(),draws=fr.randn(9,generator=generator).tolist())


def worker(directory):
    path=Path(directory);model,optimizer=setup();generator=fr.Generator(999)
    scheduler=fr.optim.lr_scheduler.CyclicLR(optimizer,[.0005,.001],[.003,.004],step_size_up=3,
        base_momentum=[.7,.6],max_momentum=[.95,.83])
    checkpoint=fr.checkpoint.load(path/'state');checkpoint.restore(model,optimizers={'main':optimizer},generator=generator)
    scheduler.load_state_dict(json.loads((path/'scheduler.json').read_text()))
    (path/'actual.json').write_text(json.dumps(suffix(model,optimizer,scheduler,generator,checkpoint.step),allow_nan=False))


class MuonTests(unittest.TestCase):
    compare_state=adaptive_tests.AdaptiveOptimizerTests.compare_state

    def test_actual_torch_multistep_parameters_momentum_groups_skips_and_zero_gradients(self):
        for steps in (0,1,5):
            with self.subTest(steps=steps):
                model,optimizer=setup();params,oracle=setup(True)
                for group in optimizer.param_groups:group['ns_steps']=steps
                for group in oracle.param_groups:group['ns_steps']=steps
                for iteration in range(25):
                    optimizer.zero_grad();oracle.zero_grad()
                    for pi,(p,tp) in enumerate(zip(model.parameters(),params)):
                        if (iteration+pi)%5==2:continue
                        if iteration%7==3:
                            actual=p.tensor().sum()*0.;expected=tp.sum()*0.
                        else:
                            actual=p.tensor().square().sum()*(.2+.03*iteration+pi*.1)+p.tensor().sum()*.07
                            expected=tp.square().sum()*(.2+.03*iteration+pi*.1)+tp.sum()*.07
                        actual.backward();expected.backward()
                    optimizer.step();oracle.step()
                    self.compare_state(optimizer,oracle)
                    for p,tp in zip(model.parameters(),params):
                        torch.testing.assert_close(torch.tensor(p.tensor().tolist()),tp.detach(),atol=3e-5,rtol=3e-4)
                        if tp.grad is None:self.assertIsNone(p.grad)
                        else:torch.testing.assert_close(torch.tensor(p.grad.tolist()),tp.grad,atol=3e-5,rtol=3e-4)

    def test_shared_parameter_leaves_follow_torch_sequential_update_order(self):
        for grouped in (False,True):
            with self.subTest(grouped=grouped):
                storage=fr.tensor([[.4,-.6,.3],[.1,.7,-.8]])
                target=torch.tensor(storage.tolist())
                params=[fr.nn.Embedding.from_pretrained(storage,freeze=False).weight for _ in range(2)]
                expected=[torch.nn.Parameter(target) for _ in range(2)]
                groups=[dict(params=[p]) for p in params] if grouped else params
                targets=[dict(params=[p]) for p in expected] if grouped else expected
                optimizer=fr.optim.Muon(groups,lr=.003,weight_decay=.07)
                oracle=torch.optim.Muon(targets,lr=.003,weight_decay=.07)
                for index in range(12):
                    optimizer.zero_grad();oracle.zero_grad()
                    for pi,(p,tp) in enumerate(zip(params,expected)):
                        if (index+pi)%5==1:continue
                        (p.tensor().square().sum()*(.3+.1*pi)+p.tensor().sum()*(.07*index-.1*pi)).backward()
                        (tp.square().sum()*(.3+.1*pi)+tp.sum()*(.07*index-.1*pi)).backward()
                    optimizer.step();oracle.step();self.compare_state(optimizer,oracle)
                    torch.testing.assert_close(torch.tensor(storage.tolist()),target,atol=3e-5,rtol=3e-4)

    def test_zero_matrices_empty_rows_and_hidden_frozen_gradients(self):
        for shape in ((1,1),(0,3),(4,2)):
            with self.subTest(shape=shape):
                parameter=fr.nn.Parameter(fr.zeros(*shape));target=torch.nn.Parameter(torch.zeros(shape))
                optimizer=fr.optim.Muon([parameter]);oracle=torch.optim.Muon([target])
                parameter.tensor().square().sum().backward();target.square().sum().backward()
                optimizer.step();oracle.step();self.compare_state(optimizer,oracle)
                self.assertEqual(parameter.tensor().tolist(),target.detach().tolist())
                optimizer.zero_grad();parameter.trainable=False
                parameter.tensor().sum().backward();raw=parameter.tensor().grad
                before=plain(optimizer.state_dict());values=parameter.tensor().tolist()
                optimizer.step()
                self.assertEqual(plain(optimizer.state_dict()),before);self.assertEqual(parameter.tensor().tolist(),values)
                self.assertEqual(parameter.tensor().grad._id(),raw._id())

    def test_fresh_process_complete_state_groups_schedule_rng_and_next_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);model,optimizer=setup();generator=fr.Generator(19)
            scheduler=fr.optim.lr_scheduler.CyclicLR(optimizer,[.0005,.001],[.003,.004],step_size_up=3,
                base_momentum=[.7,.6],max_momentum=[.95,.83])
            for index in range(6):update(model,optimizer,scheduler,generator,index)
            fr.checkpoint.save_training(path/'state',model,optimizers={'main':optimizer},generator=generator,step=6)
            (path/'scheduler.json').write_text(json.dumps(scheduler.state_dict(),allow_nan=False))
            expected=json.loads(json.dumps(suffix(model,optimizer,scheduler,generator,6)))
            child=subprocess.run([sys.executable,__file__,'--worker',str(path)],capture_output=True,text=True,timeout=60)
            self.assertEqual(child.returncode,0,child.stdout+child.stderr)
            self.assertEqual(json.loads((path/'actual.json').read_text()),expected)

    def test_owning_state_and_late_public_or_native_corruption_are_atomic(self):
        model,optimizer=setup()
        sum(p.tensor().square().sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad()
        saved=optimizer.state_dict();before=plain(saved)
        sum(p.tensor().sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad()
        self.assertEqual(plain(saved),before);optimizer.load_state_dict(saved);self.assertEqual(plain(optimizer.state_dict()),before)
        for transform in (lambda state:state['param_groups'][-1].update(ns_steps=100),
                          lambda state:state['param_groups'][-1].update(ns_coefficients=(1.,2.)),
                          lambda state:state['state'][1].update(momentum_buffer=fr.ones(6)),
                          lambda state:state['state'][1].update(step=fr.tensor(1.))):
            bad=owned(saved);transform(bad)
            with self.assertRaises((TypeError,ValueError)):optimizer.load_state_dict(bad)
            self.assertEqual(plain(optimizer.state_dict()),before)
        config=optimizer._packed_config(optimizer.param_groups)
        for name,value in (('g1.initialized.0',fr.tensor(.5)),('g1.momentum_buffer.0',fr.ones(5)),('unknown',fr.tensor(1.))):
            arrays=dict(optimizer._native.state_tensors());arrays[name]=value
            with self.assertRaises(ValueError):optimizer._native.restore_state(config,list(arrays.items()))
            self.assertEqual(plain(optimizer.state_dict()),before)

    def test_checkpoint_scalar_or_late_native_corruption_preserves_every_target(self):
        model,optimizer=setup();generator=fr.Generator(7)
        sum(p.tensor().square().sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad()
        before=checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator))
        for key,replacement in (('optim_meta.main.g1.adjust_lr_fn',3),('optim_meta.main.g1.ns_steps',0x4059000000000000),
                                ('optim_meta.main.g1.ns_coefficients.2',0x7ff8000000000000)):
            params,buffers,scalars,opts,rng,_=fr.checkpoint._state(model,{'main':optimizer},generator)
            scalars=[(name,replacement if name==key else value) for name,value in scalars]
            native=fr._native._TrainingCheckpoint.snapshot(params,buffers,scalars,opts,rng,0)
            bad=fr.checkpoint.TrainingCheckpoint(native)
            with self.assertRaises(ValueError):bad.restore(model,optimizers={'main':optimizer},generator=generator)
            self.assertEqual(checkpoint_digest(fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)),before)

    def test_validation_capacity_and_group_addition_before_state_changes(self):
        for tensor in (fr.ones(3),fr.ones(2,3,4),fr.ones(2,0),fr.zeros(0,0)):
            with self.assertRaises(ValueError):fr.optim.Muon([fr.nn.Parameter(tensor)])
        parameter=fr.nn.Parameter(fr.ones(2,3))
        for options in (dict(ns_steps=100),dict(ns_steps=-1),dict(ns_steps=2.5),dict(nesterov=1),
                        dict(ns_coefficients=(1.,float('nan'),2.)),dict(adjust_lr_fn='unsupported'),dict(eps=3.39e38)):
            with self.assertRaises((TypeError,ValueError)):fr.optim.Muon([parameter],**options)
        reference=torch.nn.Parameter(torch.ones(2,3));reference.grad=torch.ones_like(reference)
        with self.assertRaisesRegex(RuntimeError,'BFloat16.*overflow'):torch.optim.Muon([reference],eps=3.39e38).step()
        model,optimizer=setup();before=plain(optimizer.state_dict())
        with self.assertRaises(ValueError):optimizer.add_param_group({'params':[model.a]})
        self.assertEqual(plain(optimizer.state_dict()),before)
        added=fr.nn.Parameter(fr.ones(2,2));optimizer.add_param_group({'params':[added],'nesterov':False,'ns_steps':1})
        self.assertEqual(len(optimizer.param_groups),3);self.assertEqual(optimizer.state_dict()['state'],{})
        before=plain(optimizer.state_dict())
        model.b.tensor()._set_leaf_grad(model.a.tensor().detach().transpose(0,1))
        with self.assertRaisesRegex(ValueError,'share parameter storage'):optimizer.step()
        self.assertEqual(plain(optimizer.state_dict()),before)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        from test_muon_optimizer import worker
        worker(sys.argv[2])
    else:unittest.main(verbosity=2)
