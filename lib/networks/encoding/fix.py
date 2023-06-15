import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from lib.config import cfg
import torch.nn.functional as F
eps = 1e-6

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        W = kwargs['W']
        D = kwargs['D']
        input_ch = kwargs['input_dim']
        output_ch = kwargs['output_dim']
        init_val = kwargs['init_val']

    def forward(self, x):
        output = torch.ones_like(x) * -1.
        output[x>= (56/172)] = 0.
        output[x>= (109/172)] = 1.
        return output
