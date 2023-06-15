import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.nsc.network import Network as NSCNetwork
from lib.networks.nsc import utils
from kaolin.ops.spc import unbatched_query
import kaolin as kal
import trimesh
from lib.config import cfg
import os
from lib.utils.py_utils import load_function_from_file
return_interested = load_function_from_file(cfg.render_filter, 'return_interested')
return_ground = load_function_from_file(cfg.render_filter, 'return_ground')

class Network(NSCNetwork):
    def __init__(self,):
        super(Network, self).__init__()
        # build octree
        level = 3
        num_sample_points = 1000000
        mesh = trimesh.load(os.path.join(cfg.workspace, cfg.test_dataset.data_root, cfg.scene, cfg.query_octree_ply_path))
        try:
            points, _ = trimesh.sample.sample_surface(mesh, count=num_sample_points)
        except:
            points = np.array(mesh.vertices)
        min_point, max_point = points.min(axis=0), points.max(axis=0)
        self.scale = (max_point - min_point).max().item() * (0.5 + 1e-6)
        self.shift = (min_point + max_point) / 2.
        points = points - self.shift
        points = points / self.scale
        points = torch.from_numpy(points).cuda().float()
        self.spc = kal.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(points, level)
        self.shift = torch.from_numpy(self.shift).float().cuda()
        self.level = level
        
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
        
        net_output = torch.zeros_like(net_input[..., :7])
        net_output[..., 3] = -100.
        xyz_norm = (xyz - self.shift[None]) / self.scale
        xyz_ret = unbatched_query(self.spc.octrees, self.spc.exsum, xyz_norm, self.level)
        xyz_mask = torch.logical_and(xyz_ret != -1, return_interested(xyz))
        if xyz_mask.sum() > 0:
            net_output[xyz_mask] = network(net_input[xyz_mask], xyz_app_encoding=xyz_app_encoding[xyz_mask])
        return net_output.reshape(N_rays, N_samples, -1)
    
    def forward_network_prop(self, xyz):
        N_rays, N_samples = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        
        raw = torch.ones_like(xyz[..., :1]).float() * -100.
        xyz_norm = (xyz - self.shift[None]) / self.scale
        xyz_ret = unbatched_query(self.spc.octrees, self.spc.exsum, xyz_norm, self.level)
        xyz_mask = torch.logical_and(xyz_ret != -1, return_interested(xyz))
        if xyz_mask.sum() > 0:
            raw[xyz_mask] = self.sample_net(self.xyz_encoder(xyz[xyz_mask]))
        return raw.reshape(N_rays, N_samples, 1)
    
    def transient_layer(self, output, rays, batch):
        pass
    
    def sky_layer(self, output, rays, batch):
        emb_id = rays[:, 9].long()
        view_dir = utils.get_sphere_uv(rays, self.center_point, self.r)
        sky_color = self.env_net(view_dir, emb_id, batch)
        
        ground = return_ground(rays[..., 3:6])
        if ground.sum() > 0:
            sky_color[ground] = 1.
        fg_alpha = output['weights_1'].sum(dim=-1, keepdim=True)
        output['rgb_1'] = output['rgb_1'] * fg_alpha + sky_color * (1-fg_alpha)
        if self.training:
            output['content_rgb_1'] = output['content_rgb_1'] * fg_alpha  + sky_color.detach() * (1-fg_alpha)
        else:
            output['content_rgb_1'] = output['content_rgb_1'] + 1-fg_alpha