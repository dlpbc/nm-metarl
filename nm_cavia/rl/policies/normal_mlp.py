import math
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from policies.policy import Policy, weight_init
from policies.nmlinear import NMLinear


class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """

    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def forward_analysismode(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        activations = [] # NOTE
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
            activations.append(output) # NOTE
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        activations.append(mu) # NOTE
        activations.append(scale.view(1, -1)) # NOTE

        return Normal(loc=mu, scale=scale), torch.cat(activations, dim=1) # NOTE

class CaviaMLPPolicy(Policy, nn.Module):
    """CAVIA network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces (eg.
    `HalfCheetahDir`).
    """

    def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(CaviaMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes
        self.add_module('layer{0}'.format(1), nn.Linear(layer_sizes[0] + num_context_params, layer_sizes[1]))
        for i in range(2, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        # if no parameters are given, use the standard ones
        if params is None:
            params = OrderedDict(self.named_parameters())

        # concatenate context parameters to input
        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)

        # forward through FC Layer
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """

        # take the gradient wrt the context params
        grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]

        # set correct computation graph
        if not first_order:
            self.context_params = self.context_params - step_size * grads
        else:
            self.context_params = self.context_params - step_size * grads.detach()

        return OrderedDict(self.named_parameters())

    def reset_context(self):
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

    def forward_analysismode(self, input, params=None):

        # if no parameters are given, use the standard ones
        if params is None:
            params = OrderedDict(self.named_parameters())

        # concatenate context parameters to input
        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)

        activations = [] # NOTE
        activations.append(self.context_params.expand(input.shape[:-1] + self.context_params.shape))
        # forward through FC Layer
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
            activations.append(output) # NOTE

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        activations.append(mu) # NOTE
        activations.append(scale.view(1, -1)) # NOTE

        return Normal(loc=mu, scale=scale), torch.cat(activations, dim=1) # NOTE

class NMNormalMLPPolicy(Policy):
    """Policy network based on a neuromodulated multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """

    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6, nm_size=5, nm_gate='hard'):
        super(NMNormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.nm_size = nm_size
        assert nm_gate in ['hard', 'soft'], '`nm_gate` should be \'hard\' or \'soft\''
        self.nm_gate = nm_gate

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            NMLinear(layer_sizes[i - 1], layer_sizes[i], nm_size, nm_gate))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output_std = F.linear(output,
                              weight=params['layer{0}.std.weight'.format(i)],
                              bias=params['layer{0}.std.bias'.format(i)])
            mod_features = F.relu(F.linear(output,
                                weight=params['layer{0}.in_nm.weight'.format(i)],
                                bias=params['layer{0}.in_nm.bias'.format(i)]))
            sign_ = torch.tanh(F.linear(mod_features, 
                                weight=params['layer{0}.out_nm.weight'.format(i)],
                                bias=params['layer{0}.out_nm.bias'.format(i)]))
            if self.nm_gate == 'hard':
                sign_ = torch.sign(sign_)
                sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
            output = self.nonlinearity(output_std * sign_)
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def forward_analysismode(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        activations = [] # NOTE
        for i in range(1, self.num_layers):
            output_std = F.linear(output,
                              weight=params['layer{0}.std.weight'.format(i)],
                              bias=params['layer{0}.std.bias'.format(i)])
            mod_features = F.relu(F.linear(output,
                                weight=params['layer{0}.in_nm.weight'.format(i)],
                                bias=params['layer{0}.in_nm.bias'.format(i)]))
            sign_ = torch.tanh(F.linear(mod_features, 
                                weight=params['layer{0}.out_nm.weight'.format(i)],
                                bias=params['layer{0}.out_nm.bias'.format(i)]))
            if self.nm_gate == 'hard':
                sign_ = torch.sign(sign_)
                sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
            sign_copy = copy.deepcopy(sign_) # NOTE
            output = self.nonlinearity(output_std * sign_)
            activations.append(output_std) # NOTE
            activations.append(mod_features) # NOTE
            activations.append(sign_copy) # NOTE
            activations.append(output) # NOTE
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        activations.append(mu) # NOTE
        activations.append(scale.view(1, -1)) # NOTE

        return Normal(loc=mu, scale=scale), torch.cat(activations, dim=1) # NOTE

class NMCaviaMLPPolicy(Policy, nn.Module):
    """CAVIA network based on a multi-layer perceptron (MLP), with a
    `Normal` distribution output, with trainable standard deviation. This
    policy network can be used on tasks with continuous action spaces (eg.
    `HalfCheetahDir`).
    """

    def __init__(self, input_size, output_size, device, hidden_sizes=(), num_context_params=10,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6, nm_size=5, nm_gate='hard'):
        super(NMCaviaMLPPolicy, self).__init__(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        assert nm_gate in ['hard', 'soft'], '`nm_gate` should be \'hard\' or \'soft\''
        self.nm_gate = nm_gate
        self.nm_size = nm_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1
        self.context_params = []

        layer_sizes = (input_size,) + hidden_sizes
        self.add_module('layer{0}'.format(1), NMLinear(layer_sizes[0] + num_context_params, layer_sizes[1], nm_size, nm_gate))
        for i in range(2, self.num_layers):
            self.add_module('layer{0}'.format(i), NMLinear(layer_sizes[i - 1], layer_sizes[i], nm_size, nm_gate))

        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        # if no parameters are given, use the standard ones
        if params is None:
            params = OrderedDict(self.named_parameters())

        # concatenate context parameters to input
        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)

        # forward through FC Layer
        for i in range(1, self.num_layers):
            output_std = F.linear(output,
                              weight=params['layer{0}.std.weight'.format(i)],
                              bias=params['layer{0}.std.bias'.format(i)])
            mod_features = F.relu(F.linear(output,
                                weight=params['layer{0}.in_nm.weight'.format(i)],
                                bias=params['layer{0}.in_nm.bias'.format(i)]))
            sign_ = torch.tanh(F.linear(mod_features, 
                                weight=params['layer{0}.out_nm.weight'.format(i)],
                                bias=params['layer{0}.out_nm.bias'.format(i)]))
            if self.nm_gate == 'hard':
                sign_ = torch.sign(sign_)
                sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
            output = self.nonlinearity(output_std * sign_)

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)

    def update_params(self, loss, step_size, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """

        # take the gradient wrt the context params
        grads = torch.autograd.grad(loss, self.context_params, create_graph=not first_order)[0]

        # set correct computation graph
        if not first_order:
            self.context_params = self.context_params - step_size * grads
        else:
            self.context_params = self.context_params - step_size * grads.detach()

        return OrderedDict(self.named_parameters())

    def reset_context(self):
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True).to(self.device)

    def forward_analysismode(self, input, params=None):

        # if no parameters are given, use the standard ones
        if params is None:
            params = OrderedDict(self.named_parameters())

        # concatenate context parameters to input
        output = torch.cat((input, self.context_params.expand(input.shape[:-1] + self.context_params.shape)),
                           dim=len(input.shape) - 1)

        activations = [] # NOTE
        activations.append(self.context_params.expand(input.shape[:-1] + self.context_params.shape))
        # forward through FC Layer
        for i in range(1, self.num_layers):
            output_std = F.linear(output,
                              weight=params['layer{0}.std.weight'.format(i)],
                              bias=params['layer{0}.std.bias'.format(i)])
            mod_features = F.relu(F.linear(output,
                                weight=params['layer{0}.in_nm.weight'.format(i)],
                                bias=params['layer{0}.in_nm.bias'.format(i)]))
            sign_ = torch.tanh(F.linear(mod_features, 
                                weight=params['layer{0}.out_nm.weight'.format(i)],
                                bias=params['layer{0}.out_nm.bias'.format(i)]))
            if self.nm_gate == 'hard':
                sign_ = torch.sign(sign_)
                sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
            sign_copy = copy.deepcopy(sign_)
            output = self.nonlinearity(output_std * sign_)
            activations.append(output_std) # NOTE
            activations.append(mod_features) # NOTE
            activations.append(sign_copy) # NOTE
            activations.append(output) # NOTE

        # last layer outputs mean; scale is a learned param independent of the input
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        activations.append(mu) # NOTE
        activations.append(scale.view(1, -1)) # NOTE

        return Normal(loc=mu, scale=scale), torch.cat(activations, dim=1) # NOTE
