import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.nsc.mlp import NeRF, NeRF_guide
from lib.networks.nsc.modules import MaskNet, EnvNet
from lib.networks.nsc.encoding import get_encoder
from lib.networks.nsc import utils
from lib.config import cfg
import trimesh
if cfg.query_octree:
    import kaolin.render.spc as spc_render
    from kaolin.ops.spc import unbatched_query
    import wisp.ops.spc as wisp_spc_ops
    import kaolin as kal
import os
import trimesh


# TODO: hard coding
N_anneal_steps = 2500
N_anneal_slope = 10.
bias = lambda x, b: (b*x) / ((b-1)*x + 1)

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.cascade_samples = cfg.cascade_samples
        
        # sample net 
        self.xyz_encoder, input_ch = get_encoder(net_cfg.xyz_encoder)
        self.sample_net = NeRF_guide(input_ch, net_cfg.nerf)
        
        # fine net 
        self.dir_encoder, input_ch_views = get_encoder(net_cfg.dir_encoder)
        self.time_encoder, input_ch_time = get_encoder(net_cfg.time_encoder)
        self.app_embeddings = nn.Embedding(cfg.n_emb, cfg.dim_app_emb)
        self.app_xyz_encoder, app_input_ch = get_encoder(net_cfg.app_xyz_encoder)
        self.net_fine = NeRF(input_ch, input_ch_views, input_ch_time, net_cfg.nerf, app_input_ch)
        
        self.uv_encoder, input_ch_uv = get_encoder(net_cfg.uv_encoder)
        self.mask_net = MaskNet(input_ch_uv=input_ch_uv)

        if net_cfg.envmap:
            mesh = trimesh.load(os.path.join(cfg.workspace, cfg.test_dataset.data_root, cfg.scene, cfg.octree_ply_path))
            points = np.array(mesh.vertices)
            min_point, max_point = points.min(axis=0), points.max(axis=0)
            center_point = (min_point + max_point)/2.
            self.register_buffer('center_point', torch.from_numpy(center_point).float())
            self.r = np.linalg.norm(max_point - min_point)/2. * 1.5
            self.env_net = EnvNet()

        # apperance parametric encoding
        mesh = trimesh.load(os.path.join(cfg.workspace, cfg.test_dataset.data_root, cfg.scene, cfg.octree_ply_path))
        points = np.array(mesh.vertices)
        min_point, max_point = points.min(axis=0), points.max(axis=0)
        self.register_buffer('min_point', torch.from_numpy(min_point).float())
        self.register_buffer('max_point', torch.from_numpy(max_point).float())

    def forward_network(self, xyz, xyz_dir, network, emb=None, input_time=None):
        N_rays, N_samples = xyz.shape[:2]
        xyz, xyz_dir = xyz.reshape(-1, xyz.shape[-1]), xyz_dir.reshape(-1, xyz_dir.shape[-1])
        xyz_encoding = self.xyz_encoder(xyz)
        dir_encoding = self.dir_encoder(xyz_dir)
        xyz_app_encoding = self.app_xyz_encoder(xyz, torch.cat([self.min_point, self.max_point]))
        net_input = torch.cat([xyz_encoding, dir_encoding], dim=-1)
        emb = emb.reshape(-1, emb.shape[-1])
        net_input = torch.cat([net_input, emb], dim=-1)
        input_time = input_time.reshape(-1, input_time.shape[-1])
        input_time_ = self.time_encoder(input_time)
        net_input = torch.cat([net_input, input_time_], dim=-1)
        net_output = network(net_input, xyz_app_encoding=xyz_app_encoding)
        return net_output.reshape(N_rays, N_samples, -1)
    
    def forward_network_prop(self, xyz):
        return self.sample_net(self.xyz_encoder(xyz))
    
    def transient_layer(self, output, rays, batch):
        uv = rays[:, 10:12]
        emb_id = rays[:, 9].long()
        mask = self.mask_net(self.uv_encoder(uv), emb_id)
        output.update({'mask': mask})
        
    def sky_layer(self, output, rays, batch):
        emb_id = rays[:, 9].long()
        view_dir = utils.get_sphere_uv(rays, self.center_point, self.r)
        sky_color = self.env_net(view_dir, emb_id, batch)
        fg_alpha = output['weights_1'].sum(dim=-1, keepdim=True)
        # We found that the sky layer is hard to optimize unless we multiply the fg_rgb with fg_alpha:
        # rgb = rgb * fg_alpha + sky_color * (1 - fg_alpha)
        # After countless experimental attempts and analyses, we further found that the function of the product is akin to the random background loss in Instant NGP.
        # We replace it with the following correct formulation:
        # rgb = rgb * (1 + fg_alpha - fg_alpha.detach()) + sky_color * (1 - fg_alpha)
        # The formulation is correct, but it eventually produces some minor artifacts.
        # So we insist on using the incorrect formulation, as we found this to yield the best results. Furthermore, because fg_alpha usually becomes either 0 or 1 after training, the impact of the error is not significant.
        output['rgb_1'] = output['rgb_1'] * fg_alpha + sky_color * (1-fg_alpha)
        if self.training:
            output['content_rgb_1'] = output['content_rgb_1'] * fg_alpha  + sky_color.detach() * (1-fg_alpha)
        else:
            output['content_rgb_1'] = output['content_rgb_1'] + 1-fg_alpha

    def render_rays(self, rays, batch):
        rays_o, rays_d, near, far  = rays[:, :3], rays[:, 3:6], rays[:, 6], rays[:, 7]
        viewdir = rays_d / rays_d.norm(dim=-1, keepdim=True)
        near = torch.clamp_min(near, 1e-8)
        
        # proposal points 
        z_vals = utils.sample_along_ray(near, far, self.cascade_samples[0], self.training)
        xyz = rays_o[:, None] + rays_d[:, None] * (z_vals[:, :-1, None] + z_vals[:, 1:, None]) / 2.
        # raw = self.sample_net(self.xyz_encoder(xyz))
        raw = self.forward_network_prop(xyz)
        ret = utils.raw2weights(raw, z_vals, rays_d)
        # for hist loss
        if self.training:
            ret['z_vals'] = z_vals

        # for evaluating points
        output = {}
        for key in ret:
            output[key + '_0'] = ret[key]
        weights = ret['weights'].clone().detach()
        z_vals_mid = z_vals

        # proposal points
        step = batch['step'] if 'step' in batch else 0
        train_frac = np.clip(step / N_anneal_steps, 0, 1)
        if self.training:
            anneal = bias(train_frac, N_anneal_slope)
            annealed_weights = torch.pow(weights, anneal)
        else:
            annealed_weights = weights
        z_samples = utils.sample_pdf(z_vals_mid, annealed_weights, self.cascade_samples[1] + 1, det=True)
        z_vals = z_samples

        xyz = rays_o[:, None] + rays_d[:, None] * (z_vals[:, 1:, None] + z_vals[:, :-1, None]) / 2.
        xyz_dir = viewdir[:, None].repeat(1, self.cascade_samples[1], 1)
        emb_id = rays[..., 9].long()
        emb = self.app_embeddings(emb_id)[:, None].repeat(1, xyz.shape[1], 1)
        input_time = rays[..., 8:9][:, None].repeat(1, xyz.shape[1], 1)
        raw = self.forward_network(xyz, xyz_dir, self.net_fine, emb, input_time)
        ret = utils.raw2outputs(raw, z_vals, rays_d, cfg.white_bkgd)
        if self.training:
            ret.update({'z_vals': z_vals})
        for key in ret:
            output[key + '_1'] = ret[key]
        output['rgb'] = output['rgb_1']
        
        # transient objects 
        self.transient_layer(output, rays, batch)
        # sky
        self.sky_layer(output, rays, batch)
        return output


    def batchify_rays(self, rays, batch):
        all_ret = {}
        chunk = cfg.chunk_size
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret
                
    def forward(self, batch):
        B, N_rays, C = batch['rays'].shape
        ret = self.batchify_rays(batch['rays'].reshape(-1, C), batch)
        output = {k:ret[k].reshape(B, N_rays, -1) for k in ret}
        return output
