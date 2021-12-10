# -*- coding: utf-8 -*-
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.nmlinear import NMLinearReduced


class M(nn.Module):
    def __init__(self, insize, outsize):
        super(M, self).__init__()
        self.fc1 = nn.Linear(insize, 4)
        self.fc2 = nn.Linear(4, outsize)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NM(nn.Module):
    def __init__(self, insize, outsize):
        super(NM, self).__init__()
        self.fc1 = NMLinearReduced(insize, 4, 3)
        self.fc2 = nn.Linear(4, outsize)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    m = M(5, 3)
    n = NM(5, 3)

    # fc2
    n.fc2.weight.data.copy_(m.fc2.weight.data)
    n.fc2.bias.data.copy_(m.fc2.bias.data)

    # fc1
    n.fc1.std.weight.data.copy_(m.fc1.weight.data)
    n.fc1.std.bias.data.copy_(m.fc1.bias.data)

    print('linear model')
    pprint(dict(m.named_parameters()))
    print('\nneuromodulated linear model')
    pprint(dict(n.named_parameters()))

    x = torch.zeros(6, 5)
    x[0] = torch.rand(5)
    x[1] = x[0] + torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
    x[2] = torch.FloatTensor(5).uniform_(-1., 1.)
    x[3] = x[2] + torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
    x[4] = torch.FloatTensor(5).uniform_(-5., 5.)
    x[5] = x[4] + torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
    print()
    print(x)
    my = m(x)
    ny = n(x)

    print()
    print(my)

    print()
    print(ny)

if __name__ == '__main__':
    main()

