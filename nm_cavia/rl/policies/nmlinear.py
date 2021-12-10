#-*- coding: utf-8 -*-
import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
import torch.nn as nn

class NMLinear(Module):
    ''' neuromodulator that inverts sign of neural activity (weighted sum of input) '''
    def __init__(self, in_features, out_features, nm_features, bias=True, gating='soft'):
        super(NMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nm_features = nm_features
        self.in_nm_act = F.relu # NOTE hardcoded activation function
        self.out_nm_act = torch.tanh # NOTE hardcoded activation function
        assert gating in ['hard', 'soft'], '`gating` should be \'hard\' or \'soft\''
        self.gating = gating

        self.std = nn.Linear(in_features, out_features, bias=bias)
        self.in_nm = nn.Linear(in_features, nm_features, bias=bias)
        self.out_nm = nn.Linear(nm_features, out_features, bias=bias)

    def forward(self, data, params=None):
        output = self.std(data)
        mod_features = self.in_nm_act(self.in_nm(data))
        sign_ = self.out_nm_act(self.out_nm(mod_features))
        if self.gating == 'hard':
            sign_ = torch.sign(sign_)
            sign_[sign_ == 0.] = 1. # a zero value should have sign of 1. and not 0.
        output *= sign_
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, nm_features={}'.format(self.in_features,\
                self.out_features, self.nm_features)
