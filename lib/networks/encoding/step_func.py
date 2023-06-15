import numpy as np
import torch
import torch.nn as nn
from lib.config import cfg
import torch.nn.functional as F
eps = 1e-6

class Encoder(nn.Module):
    def __init__(self, hard_forward=True, **kwargs):
        super().__init__()
        input_ch = kwargs['input_dim']
        output_ch = kwargs['output_dim']
        init_val = kwargs['init_val']
        self.hard_forward = hard_forward
        self.mean = nn.Parameter(torch.rand(output_ch))
        self.beta = nn.Parameter(torch.ones(output_ch) * init_val)
        self.input_ch = input_ch

    def forward(self, x):
        mean = self.mean[None]
        beta = self.beta[None]
        output = x - mean
        msk = output <= 0.
        output[msk] = 0.5 * torch.exp(output[msk] / torch.clamp_min(torch.abs(beta.repeat(len(msk), 1)[msk]), 1e-3))
        output[~msk] = 1 - 0.5 * torch.exp(- output[~msk] / torch.clamp_min(torch.abs(beta.repeat(len(msk), 1)[~msk]), 1e-3))
        if self.hard_forward:
            msk = output <= 0.5
            output[msk] = 0. + output[msk] - output[msk].detach()
            output[~msk] = 1. + output[~msk] - output[~msk].detach()
        return output
