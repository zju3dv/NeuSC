import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from lib.config import cfg
eps = 1e-6

class MRL(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        base_res = kwargs['base_resolution']
        per_level_scale = kwargs['per_level_scale']
        level = kwargs['level']
        level_dim = kwargs['level_dim']
        self.level_dim = level_dim
        self.level = level

        device = torch.device('cuda:{}'.format(cfg.local_rank))

        self.offsets = [0]
        self.scales = []
        for i in range(level):
            res = int(base_res *  per_level_scale**i )
            self.scales.append(res)
            n_entrys =  (res + 8)
            self.offsets.append(self.offsets[-1] + n_entrys)

        self.offsets_pos = torch.tensor([[0.],
                                     [1.]]).float().to(device)



        self.data = nn.Parameter(torch.zeros((self.offsets[-1], level_dim)))
        std = 1e-4
        self.data.data.uniform_(-std, std)
        self.out_dim = 1 + level_dim * level
        self.scales = torch.tensor(self.scales).to(device).float()
        self.offsets = torch.tensor(np.array(self.offsets)).to(device).long()

    def forward(self, x):
        x = torch.clamp(x, 0., 1.)

        inputs = x
        inputs = inputs[None].repeat(self.level, 1, 1)
        float_xyz = inputs * self.scales[:, None, None]
        int_xyz = (float_xyz[:, :, None] + self.offsets_pos[None, None]).long()
        offset_xyz = float_xyz - int_xyz[:, :, 0]
        ind = torch.zeros_like(int_xyz[..., 0])
        ind = int_xyz[..., 0]
        ind = ind.reshape(self.level, -1)
        ind += self.offsets[:-1, None]
        ind = ind.reshape(-1)
        val = torch.gather(self.data, 0, ind[:, None].repeat(1, self.level_dim))
        val = val.reshape(self.level, -1, 2, self.level_dim)
        weights_xyz = torch.clamp((1 - self.offsets_pos[None, None]) + (2 * self.offsets_pos[None, None] - 1.) * offset_xyz[:, :, None], min=0., max=1.)
        weights_xyz = weights_xyz[..., 0]
        val = (weights_xyz[..., None] * val).sum(dim=-2)
        val = val.permute(1, 0, 2).reshape(-1, self.level*self.level_dim)
        return torch.cat([x, val], dim=-1)
