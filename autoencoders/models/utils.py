import torch
from torch import nn
import numpy as np

# Layers


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def flatten(x):
    return x.view(x.size(0), -1)


Flatten = Lambda(flatten)

# Inits


def gaussian_weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.constant(m.bias, 0.1)


def to_one_hot(y, batch_size=8, n_classes=1):
    y_one_hot = torch.zeros(batch_size, n_classes)

    return y_one_hot.scatter_(1, y.unsqueeze(1), 1)
