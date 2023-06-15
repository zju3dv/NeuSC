import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from lib.config import cfg
import torch.nn.functional as F
eps = 1e-6
from scipy import signal

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.freq = kwargs['freq']

    def forward(self, x):
        t = x.detach().cpu().numpy()
        for i in range(self.freq - 1):
            x = torch.cat([x, torch.from_numpy(signal.square(2 * np.pi * t * (2**i))).float().to(x.device)], dim=-1)
        return x
