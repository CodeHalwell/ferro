"""Module and optimizer clearing share native parameter slots, as in Torch."""
import unittest
import torch
import ferro as fr


class Model(fr.nn.Module):
    def build(self):
        self.child=fr.nn.Linear(2,1)
        self.alias=self.child
        self.extra=fr.nn.Parameter(fr.tensor([.3,-.7]))
        self.register_buffer('floating',fr.tensor([.5,-1.],dtype='float64').requires_grad_(True))


class Oracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child=torch.nn.Linear(2,1)
        self.alias=self.child
        self.extra=torch.nn.Parameter(torch.tensor([.3,-.7]))
        self.register_buffer('floating',torch.tensor([.5,-1.],dtype=torch.float64,requires_grad=True))


def setup():
    model,oracle=Model(),Oracle()
    # Registration order differs for ordinary Torch modules, so bind by name.
    parameters=dict(model.named_parameters())
    with torch.no_grad():
        for name,p in oracle.named_parameters():p.copy_(torch.tensor(parameters[name].tensor().tolist()))
    for name,p in model.named_parameters():p.tensor().square().sum().backward()
    for name,p in oracle.named_parameters():p.square().sum().backward()
    model.floating.square().sum().backward();oracle.floating.square().sum().backward()
    return model,oracle


class ModuleZeroGradTests(unittest.TestCase):
    def test_nested_tied_parameters_keep_gradient_handles_or_clear_slots_like_torch(self):
        for mode in (False,True):
            with self.subTest(set_to_none=mode):
                model,oracle=setup()
                parameters=dict(model.named_parameters());targets=dict(oracle.named_parameters())
                identities={name:p.tensor().grad._id() for name,p in parameters.items()}
                buffer=model.floating.grad;buffer_id=buffer._id()
                self.assertIs(model.alias,model.child)
                self.assertEqual(len(parameters),3)
                self.assertIsNone(model.zero_grad(set_to_none=mode))
                oracle.zero_grad(set_to_none=mode)
                for name,p in parameters.items():
                    if mode:self.assertIsNone(p.tensor().grad);self.assertIsNone(targets[name].grad)
                    else:
                        self.assertEqual(p.tensor().grad._id(),identities[name])
                        torch.testing.assert_close(torch.tensor(p.tensor().grad.tolist()),targets[name].grad,atol=0,rtol=0)
                self.assertEqual(model.floating.grad._id(),buffer_id)
                torch.testing.assert_close(torch.tensor(model.floating.grad.tolist(),dtype=torch.float64),oracle.floating.grad,atol=0,rtol=0)
                model.zero_grad(set_to_none=mode);oracle.zero_grad(set_to_none=mode)
                self.assertEqual(model.floating.grad.tolist(),[1.,-2.])

    def test_frozen_hidden_gradients_and_optimizer_clearing_share_exact_slots(self):
        model,oracle=setup();parameter=model.extra;reference=oracle.extra
        parameter.trainable=False;reference.requires_grad_(False)
        parameter.tensor().square().sum().backward()
        identity=parameter.tensor().grad._id();values=parameter.tensor().tolist()
        self.assertIsNone(parameter.grad)
        optimizer=fr.optim.SGD(model.parameters(),lr=.1)
        target=torch.optim.SGD(oracle.parameters(),lr=.1)
        optimizer.zero_grad(set_to_none=False);target.zero_grad(set_to_none=False)
        self.assertEqual(parameter.tensor().grad._id(),identity)
        self.assertEqual(parameter.tensor().grad.tolist(),reference.grad.tolist())
        self.assertEqual(parameter.tensor().tolist(),values)
        parameter.tensor().sum().backward()
        reference.grad=torch.ones_like(reference)
        model.zero_grad(set_to_none=False);oracle.zero_grad(set_to_none=False)
        self.assertEqual(parameter.tensor().grad.tolist(),reference.grad.tolist())
        model.zero_grad();oracle.zero_grad()
        self.assertIsNone(parameter.tensor().grad);self.assertIsNone(reference.grad)
        self.assertEqual(model.floating.grad.tolist(),[1.,-2.])

    def test_invalid_mode_preserves_slots_and_buffer_gradients_remain_pending_for_checkpoint(self):
        model,oracle=setup()
        before=[(p.tensor().grad._id(),p.tensor().grad.tolist()) for p in model.parameters()]
        for mode in (None,0,1,'false'):
            with self.assertRaises(TypeError):model.zero_grad(set_to_none=mode)
            self.assertEqual([(p.tensor().grad._id(),p.tensor().grad.tolist()) for p in model.parameters()],before)
        optimizer=fr.optim.AdamW(model.parameters());generator=fr.Generator(13)
        model.zero_grad()
        with self.assertRaisesRegex(ValueError,'pending gradients'):
            fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)
        model.floating.zero_grad()
        saved=fr.checkpoint.snapshot_training(model,optimizers={'main':optimizer},generator=generator)
        self.assertEqual(saved.tensors()['buffers.floating'].dtype,'float64')


if __name__=='__main__':unittest.main(verbosity=2)
