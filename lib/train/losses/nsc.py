import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg
import math
import numpy as np
from lib.train.losses.vgg_perceptual_loss import VGGPerceptualLoss
from math import sqrt
import os

EPS = 1.0e-6


def outer(
    t0_starts: torch.Tensor,
    t0_ends: torch.Tensor,
    t1_starts: torch.Tensor,
    t1_ends: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    """Faster version of
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64
    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), right=True) - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), right=True)
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer


def lossfun_outer(
    t: torch.Tensor,  # [..., "num_samples+1"],
    w: torch.Tensor,  # [..., "num_samples"],
    t_env: torch.Tensor,  # [..., "num_samples+1"],
    w_env: torch.Tensor,  # [..., "num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80
    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram (from proposal model)
        w_env: weights that should upper bound the inner (t,w) histogram (from proposal model)
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader=None):
        super(NetworkWrapper, self).__init__()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.train_loader = train_loader
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)

        self.weightKL = 1e-5
        self.weightRec = 1e-3
        self.weightMS = 1e-6
        self.annealing = ExponentialAnnealingWeight(max=0.05, min=0.006, k=0.001)

        self.step = 0


    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02

        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_size, loss_focus_digit


    def mipnerf_reg(self, output, level):
        z_vals = output[f'z_vals_{level}'] / 20

        w = output[f'weights_{level}'][0]
        m = (z_vals[0, ..., :-1] + z_vals[0, ..., 1:]) / 2.
        interval = z_vals[0, ..., 1:] - z_vals[0, ..., :-1]


        loss_uni = (1/3) * (interval * w.pow(2)).sum(dim=-1).mean()

        wm = w*m
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)
        loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
        loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
        loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()

        return loss_uni, loss_bi

    def forward(self, batch):
        output = self.net(batch)
        scalar_stats = {}
        loss = 0

        # c_l = 0.5 * ((1 - output['mask'].detach()) * (output['rgb_0'] - batch['rgb']) ** 2).mean()
        f_l = 0.5 * (( 1 - output['mask'] ) * (output['rgb_1'] - batch['rgb'])**2).mean()

        r_ms, r_md = self.mask_regularize(output['mask'], self.annealing.getWeight(0.), 0.)

        not_sky = (batch['semantics'] != 2)
        if 'content_rgb_1' in output and self.training and not_sky.sum() > 0:
            content_loss = 0.5 * ((1 - output['mask'][not_sky].detach()) * (output['content_rgb_1'][not_sky] - batch['rgb'][not_sky]) ** 2).mean()
            loss += content_loss * 0.5
            scalar_stats.update({'content_mse': content_loss})
        loss += f_l
        scalar_stats.update({'color_mse': f_l})

        loss += (r_ms + r_md)

        if 'z_vals_1' in output and 'weights_1' in output:
            reg_1, reg_2 = self.mipnerf_reg(output, 1)
            scalar_stats.update({'reg_uni': reg_1})
            scalar_stats.update({'reg_bi': reg_2})
            loss += (reg_1 + reg_2) * 1e-3
        # print(os.getpid(), batch['step'])
        if 'semantics' in batch and (batch['semantics'] == 2).sum() >= 1 and batch['step'] < 10000 and cfg.network.envmap:
            alpha = output[f'weights_0'].sum(dim=-1)
            reg_alpha = alpha[batch['semantics'] == 2].mean() * 0.05
            reg_alpha += (1 - alpha[batch['semantics'] != 2].mean()) * 0.05
            loss += reg_alpha
            scalar_stats.update({'reg_sky_coarse': reg_alpha})

            alpha = output[f'weights_1'].sum(dim=-1)
            reg_alpha = alpha[batch['semantics'] == 2].mean() * 0.05
            reg_alpha += (1 - alpha[batch['semantics'] != 2].mean()) * 0.05
            loss += reg_alpha
            scalar_stats.update({'reg_sky_fine_vis': reg_alpha})

        if (output['mask'][..., 0] < 0.5).sum() > 0:
            psnr = -10. * torch.log(self.color_crit(output['rgb_1'][output['mask'][..., 0]<0.5].detach(), batch['rgb'][output['mask'][..., 0]<0.5]).detach()) / math.log(10)
            scalar_stats.update({'psnr_vis': psnr})
            if 'content_rgb_1' in output and self.training:
                psnr = -10. * torch.log(self.color_crit(output['content_rgb_1'][output['mask'][..., 0]<0.5].detach(), batch['rgb'][output['mask'][..., 0]<0.5]).detach()) / math.log(10)
                scalar_stats.update({'psnr_content_vis': psnr})

        if 'weights_0' in output and 'weights_1' in output and 'z_vals_0' in output and 'z_vals_1' in output:
            w = output['weights_1'][0].detach()
            c = output['z_vals_1'][0]
            wp = output['weights_0'][0]
            cp = output['z_vals_0'][0]
            loss_item = lossfun_outer(c, w, cp, wp).mean()
            loss += loss_item
            scalar_stats.update({'hist': loss_item})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))
