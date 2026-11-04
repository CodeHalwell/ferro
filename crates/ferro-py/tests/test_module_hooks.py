"""Actual Torch local Module hook contracts with native Ferro tensor graphs."""
import unittest
import warnings

import torch
import ferro as fr


class Affine(fr.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = fr.nn.Parameter(fr.tensor([.4, -.2]))

    def forward(self, x, *, scale=1.):
        return x*self.weight*scale


class TorchAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.4, -.2]))

    def forward(self, x, *, scale=1.):
        return x*self.weight*scale


def plain(value):
    if isinstance(value, (fr.Tensor, torch.Tensor)):
        return value.detach().tolist()
    if isinstance(value, dict): return {k: plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


class ModuleHooksTests(unittest.TestCase):
    def test_order_kwargs_args_replacements_and_real_parameter_gradients(self):
        records=[]
        for library, kind in ((fr,Affine),(torch,TorchAffine)):
            model=kind();events=[]
            def first(module,args):
                events.append(('pre-first',plain(args)))
                return args[0]+.3
            def keyword(module,args,kwargs):
                events.append(('pre-keyword',plain(args),plain(kwargs)))
                return (args[0]*2.,),dict(scale=kwargs.get('scale',1.)+.5)
            def tail(module,args,result):
                events.append(('post-tail',plain(args),plain(result)))
                return result.square()
            def head(module,args,kwargs,result):
                events.append(('post-head',plain(kwargs),plain(result)))
                return result+.2
            model.register_forward_pre_hook(first)
            model.register_forward_pre_hook(keyword,prepend=True,with_kwargs=True)
            model.register_forward_hook(tail)
            model.register_forward_hook(head,prepend=True,with_kwargs=True)
            x=library.tensor([.2,-.7],requires_grad=True)
            output=model(x,scale=1.3)
            output.sum().backward()
            records.append((events,plain(output),plain(x.grad),plain(model.weight.grad)))
        self.assertEqual(records[0][0],records[1][0])
        for a,b in zip(records[0][1:],records[1][1:]):
            torch.testing.assert_close(torch.tensor(a),torch.tensor(b),atol=1e-7,rtol=1e-6)

    def test_context_removal_snapshot_iteration_and_registration_during_callbacks(self):
        results=[]
        for kind in (Affine,TorchAffine):
            model=kind();events=[];handles=[]
            def first(module,args):
                events.append('first');handles[1].remove()
                module.register_forward_hook(lambda *args:events.append('from-pre'))
            def second(module,args):events.append('second')
            handles.extend((model.register_forward_pre_hook(first),model.register_forward_pre_hook(second)))
            model.register_forward_hook(lambda *args:events.append('post'))
            library=fr if kind is Affine else torch
            model(library.ones(2));one=list(events);events.clear()
            handles[0].remove();handles[0].remove()
            with model.register_forward_hook(lambda *args:events.append('context')) as handle:
                self.assertIsInstance(handle.id,int)
                model(library.ones(2))
            two=list(events);events.clear();model(library.ones(2))
            results.append((one,two,list(events)))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[0][0],['first','second','post','from-pre'])

    def test_new_post_hook_is_visible_when_registered_from_pre_hook(self):
        records=[]
        for library,kind in ((fr,Affine),(torch,TorchAffine)):
            model=kind();events=[]
            def pre(module,args):
                events.append('pre')
                module.register_forward_hook(lambda *args:events.append('new'))
            model.register_forward_pre_hook(pre)
            model(library.ones(2));records.append(events)
        self.assertEqual(records[0],records[1])
        self.assertEqual(records[0],['pre','new'])

    def test_pre_hook_validation_and_always_call_primary_error_preservation(self):
        for failure in ('pre','forward','post'):
            records=[]
            for library,kind in ((fr,Affine),(torch,TorchAffine)):
                model=kind();events=[]
                if failure=='pre':
                    model.register_forward_pre_hook(lambda *args:(_ for _ in ()).throw(ValueError('primary')))
                elif failure=='forward':
                    model.forward=lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('primary'))
                else:
                    model.register_forward_hook(lambda *args:(_ for _ in ()).throw(ValueError('primary')))
                model.register_forward_hook(lambda module,args,result:events.append(('always',plain(result))),always_call=True)
                def broken(*args):
                    events.append(('broken',))
                    raise RuntimeError('secondary')
                model.register_forward_hook(broken,always_call=True)
                model.register_forward_hook(lambda module,args,kwargs,result:events.append(('last',plain(kwargs))),always_call=True,with_kwargs=True)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    with self.assertRaisesRegex(ValueError,'primary'):model(library.ones(2),scale=2.)
                records.append((events,[(type(w.message).__name__,'secondary' in str(w.message)) for w in caught]))
            self.assertEqual(records[0],records[1])
        for library,kind in ((fr,Affine),(torch,TorchAffine)):
            model=kind();model.register_forward_pre_hook(lambda *args:3,with_kwargs=True)
            with self.assertRaisesRegex(RuntimeError,'pre-hook'):model(library.ones(2))

    def test_failing_always_hook_runs_once_and_forward_callable_is_captured_first(self):
        records=[]
        for library,kind in ((fr,Affine),(torch,TorchAffine)):
            model=kind();events=[]
            def fail(*args):events.append('fail');raise RuntimeError('hook')
            model.register_forward_hook(fail,always_call=True)
            model.register_forward_hook(lambda *args:events.append('later'),always_call=True)
            with self.assertRaisesRegex(RuntimeError,'hook'):model(library.ones(2))
            records.append(list(events))
            model=kind()
            def change(module,args):module.forward=lambda value: value*7.
            model.register_forward_pre_hook(change)
            first=model(library.ones(2));second=model(library.ones(2))
            self.assertEqual(plain(first),plain(library.tensor([.4,-.2])))
            self.assertEqual(plain(second),[7.,7.])
        self.assertEqual(records[0],records[1]);self.assertEqual(records[0],['fail','later'])



    def test_no_grad_capture_and_base_initialization_guards(self):
        outputs=[]
        for library,kind in ((fr,Affine),(torch,TorchAffine)):
            raw=kind.__new__(kind)
            with self.assertRaises(AttributeError):raw.register_forward_pre_hook(lambda *args:None)
            model=kind()
            model.register_forward_pre_hook(lambda module,args:(args[0]+1.,))
            model.register_forward_hook(lambda module,args,result:result*2.)
            with library.no_grad():
                value=model(library.tensor([.3,.7],requires_grad=True))
            self.assertFalse(value.requires_grad)
            outputs.append(plain(value))
        self.assertEqual(outputs[0],outputs[1])
        model=Affine()
        model.register_forward_pre_hook(lambda module,args:(args[0]+1.,))
        model.register_forward_hook(lambda module,args,result:result*2.)
        x=fr.tensor([.3,.7])
        with fr.no_grad():
            captured=fr.capture(lambda:model(x)).compile_fused()
        torch.testing.assert_close(torch.tensor(captured.replay().tolist()),torch.tensor(outputs[0]))

    def test_structured_replacements_keep_real_tensor_graphs(self):
        records=[]
        for library,kind in ((fr,Affine),(torch,TorchAffine)):
            model=kind()
            model.register_forward_hook(lambda module,args,result:dict(value=result*3.,aux=(result.detach(),)))
            model.register_forward_hook(lambda module,args,result:None)
            x=library.tensor([1.,2.],requires_grad=True)
            result=model(x)
            self.assertFalse(result['aux'][0].requires_grad)
            result['value'].sum().backward()
            records.append((plain(result),plain(x.grad),plain(model.weight.grad)))
        self.assertEqual(records[0],records[1])

    def test_checkpoint_callbacks_are_recreated_in_fresh_process(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            model,optimizer,generator=setup()
            suffix(model,optimizer,generator,0,3)
            fr.checkpoint.save_training(path/'state',model,optimizers={'main':optimizer},generator=generator,step=3)
            expected=json.loads(json.dumps(suffix(model,optimizer,generator,3,8)))
            child=subprocess.run([sys.executable,__file__,'--worker',str(path)],capture_output=True,text=True,timeout=45)
            self.assertEqual(child.returncode,0,child.stdout+child.stderr)
            self.assertEqual(json.loads((path/'actual.json').read_text()),expected)


def install_hooks(model):
    model.register_forward_pre_hook(lambda module,args:(args[0]+.2,))
    model.register_forward_hook(lambda module,args,result:result.square()+.1)


def setup():
    model=Affine();install_hooks(model)
    return model,fr.optim.AdamW(model.parameters(),lr=.02),fr.Generator(7)


def suffix(model,optimizer,generator,start,end):
    losses=[]
    for _ in range(start,end):
        optimizer.zero_grad()
        loss=model(fr.randn(2,generator=generator)).sum()
        loss.backward();optimizer.step();optimizer.zero_grad();losses.append(loss.item())
    return dict(losses=losses,model=plain(model.state_dict()),optimizer=plain(optimizer.state_dict()),draws=fr.randn(5,generator=generator).tolist())


def worker(directory):
    import json
    from pathlib import Path
    path=Path(directory);model=Affine();optimizer=fr.optim.AdamW(model.parameters(),lr=.02);generator=fr.Generator(100)
    saved=fr.checkpoint.load(path/'state')
    saved.restore(model,optimizers={'main':optimizer},generator=generator)
    assert not model.__dict__.get('_forward_hooks') and not model.__dict__.get('_forward_pre_hooks')
    install_hooks(model)
    (path/'actual.json').write_text(json.dumps(suffix(model,optimizer,generator,saved.step,8)))


if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        from test_module_hooks import worker
        worker(sys.argv[2])
    else:
        unittest.main(module='test_module_hooks',verbosity=2)
