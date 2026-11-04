"""Native anti-symmetric residual graph dynamics with registered message passing."""
import math

from ._native import Tensor, no_grad
from ._tensor_api import eye
from ._graph_conditioned import _eager
from ._graph_conv import GCNConv, _configure, _features, _flags, _positive
from ._graph_utils import _capacity, _edge_parts
from . import nn


class AntiSymmetricConv(nn.Module):
    """Iterated x += epsilon * act(x @ (W-W.T-gamma*eye).T + phi(x)).

    CPU float32 matrices have an explicit positive feature width, finite
    coefficients and a nonnegative integer iteration count. The default
    phi is GCNConv with no bias; supplied native Modules are registered directly
    and reset at construction as in PyG. Their extra args, kwargs and internal
    side effects remain their own contracts. Activations are native Modules,
    supported names, or None. Zero iterations return the identical input without
    calling phi or inspecting topology. Lazy widths and capture are unsupported.
    """
    _builtin_init = True
    in_channels: int
    num_iters: int
    epsilon: float
    gamma: float
    act_name: object

    def __init__(self, in_channels, phi=None, num_iters=1, epsilon=.1, gamma=.1,
                 act='tanh', act_kwargs=None, bias=True):
        _positive(in_channels, 'in_channels'); _flags(bias=bias)
        if type(num_iters) is not int or num_iters < 0: raise ValueError('num_iters must be a nonnegative integer')
        for value, name in ((epsilon, 'epsilon'), (gamma, 'gamma')):
            if type(value) not in (int, float) or not math.isfinite(value): raise ValueError(name + ' must be finite')
        if phi is not None and (not isinstance(phi, nn.Module) or not callable(getattr(phi, 'reset_parameters', None))):
            raise TypeError('phi must be a native Module with reset_parameters')
        if act_kwargs is not None and type(act_kwargs) is not dict: raise TypeError('act_kwargs must be a dictionary or None')
        _capacity((in_channels, in_channels), 8)
        activations = dict(relu=nn.ReLU, tanh=nn.Tanh, gelu=nn.GELU, silu=nn.SiLU, elu=nn.ELU, leaky_relu=nn.LeakyReLU)
        if act is None or isinstance(act, nn.Module):
            if act_kwargs: raise ValueError('act_kwargs require an activation name')
            activation, name = act, None if act is None else 'module'
        elif isinstance(act, str) and act in activations:
            activation, name = activations[act](**(act_kwargs or {})), act
        else: raise NotImplementedError('activation must be a supported name, native Module, or None')
        _configure(self, in_channels=in_channels, num_iters=num_iters, epsilon=float(epsilon), gamma=float(gamma), act_name=name)
        self.W = nn.Parameter(Tensor.zeros([in_channels, in_channels]))
        self.register_buffer('eye', eye(in_channels))
        self.phi = GCNConv(in_channels, in_channels, bias=False) if phi is None else phi
        self.act = activation
        self.bias = nn.Parameter(Tensor.zeros([in_channels])) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            self.W.tensor().copy_((Tensor.rand([self.in_channels, self.in_channels]) * 2 - 1) * self.in_channels ** -.5)
        self.phi.reset_parameters()
        if self.bias is not None:
            with no_grad(): self.bias.tensor().copy_(Tensor.zeros([self.in_channels]))

    def forward(self, x, edge_index, *args, **kwargs):
        _eager(); _features(x, 'x', self.in_channels); _capacity(x.shape)
        if self.num_iters == 0: return x
        _edge_parts(edge_index, num_nodes=x.shape[0])
        weight = self.W.tensor() - self.W.tensor().transpose(0, 1) - self.eye * self.gamma
        for _ in range(self.num_iters):
            message = self.phi(x, edge_index, *args, **kwargs)
            _features(message, 'phi output', self.in_channels)
            if message.shape[0] != x.shape[0]: raise ValueError('phi must preserve the number of nodes')
            value = x.matmul(weight.transpose(0, 1)) + message
            if self.bias is not None: value = value + self.bias.tensor()
            if self.act is not None:
                value = self.act(value)
                _features(value, 'activation output', self.in_channels)
                if value.shape[0] != x.shape[0]: raise ValueError('activation must preserve the number of nodes')
            x = x + value * self.epsilon
        return x
