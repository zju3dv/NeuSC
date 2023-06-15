import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg as global_cfg
from lib.config import cfg


class NeRF(nn.Module):
    def __init__(self, input_ch, input_ch_views, input_ch_time, cfg, app_input_ch=None, skips=[4]):
        """
        """
        super(NeRF, self).__init__()
        D, W, D_V = cfg.D, cfg.W, cfg.D_V
        D_A = cfg.D_A
        D_T = cfg.D_T
        W_2 = W // 2

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_view_dir = global_cfg.network.use_view_dir

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.time_linears = nn.ModuleList([nn.Linear(
            W + input_ch_time + app_input_ch, W_2)] + [nn.Linear(W_2, W_2) for i in range(D_T)])
        self.time_linear = nn.Linear(W_2, W)

        if self.use_view_dir:
            self.views_linears = nn.ModuleList(
                [nn.Linear(input_ch_views + W, W_2)] + [nn.Linear(W_2, W_2) for i in range(D_V)])
        else:
            self.views_linears = nn.ModuleList(
                [nn.Linear(W, W_2)] + [nn.Linear(W_2, W_2) for i in range(D_V)])

        self.apps_linears = nn.ModuleList([nn.Linear(
            W + global_cfg.dim_app_emb + 3, W_2)] + [nn.Linear(W_2, W_2) for i in range(D_A)])
        self.app_linear = nn.Sequential(nn.Linear(W_2, 3), nn.Sigmoid())
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Sequential(nn.Linear(W_2, 3), nn.Sigmoid())

    def forward(self, x, geo_time=None, xyz_app_encoding=None):
        dim = global_cfg.dim_app_emb
        input_pts, input_views, app, input_times = torch.split(
            x, [self.input_ch, self.input_ch_views, dim, self.input_ch_time], dim=-1)
        h = input_pts

        # backbone
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        h = torch.cat([h, xyz_app_encoding], dim=-1)
        h = torch.cat([h, input_times], dim=-1)

        # temporal
        for i, l in enumerate(self.time_linears):
            h = self.time_linears[i](h)
            h = F.relu(h)
        h = self.time_linear(h)
        h = F.relu(h)
        feat = h

        # viewdir
        if global_cfg.network.use_view_dir:
            h = torch.cat([h, input_views], -1)
        else:
            h = h
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)

        # illumination
        h = torch.cat([feat, app, rgb], -1)
        for i, l in enumerate(self.apps_linears):
            h = self.apps_linears[i](h)
            h = F.relu(h)
        app_rgb = self.app_linear(h)

        content_rgb = rgb
        rgb = app_rgb
        outputs = torch.cat([rgb, alpha, content_rgb], -1)
        return outputs


class NeRF_guide(nn.Module):

    def __init__(self, input_ch, cfg, skips=[4]):
        super(NeRF_guide, self).__init__()
        D, W = cfg.D, cfg.W
        self.input_ch = input_ch
        self.skips = skips
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.alpha_linear = nn.Linear(W, 1)

    def forward(self, x):
        input_pts = x#[:, :self.input_ch]
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        return alpha
