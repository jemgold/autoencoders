from torch import nn
import numpy as np


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def flatten(x):
    return x.view(x.size(0), -1)


Flatten = Lambda(flatten)


def gaussian_weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.constant(m.bias, 0.1)
