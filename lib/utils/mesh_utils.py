import numpy as np
import torch
from tqdm import tqdm
import trimesh
from skimage import measure
from lib.config import cfg
import os
import torch.nn.functional as F

def scnerf(network):
    N = cfg.resolution
    level = cfg.level
    bbox_mesh_path = os.path.join(cfg.workspace, cfg.test_dataset.data_root, cfg.scene, cfg.octree_ply_path)
    bbox_mesh = trimesh.load(bbox_mesh_path)
    vertices = np.array(bbox_mesh.vertices)
    bbox = np.concatenate([vertices.min(axis=0)-2., vertices.max(axis=0)+2.])

    def queryfn(x, network, fine=True):
        input_pts = network.xyz_encoder(x)
        h = input_pts
        net = network.net_fine if fine else network.net
        with torch.no_grad():
            for i, l in enumerate(net.pts_linears):
                h = net.pts_linears[i](h)
                h = F.relu(h)
                if i in net.skips:
                    h = torch.cat([input_pts, h], -1)
            alpha = net.alpha_linear(h)
            return alpha

    bbox = np.array(bbox).reshape((2, 3))

    voxel_grid_origin = np.mean(bbox, axis=0)
    volume_size = bbox[1] - bbox[0]
    s = volume_size[0]

    overall_index = np.arange(0, N ** 3, 1).astype(np.int)
    xyz = np.zeros([N ** 3, 3])

    # transform first 3 columns
    # to be the x, y, z index
    xyz[:, 2] = overall_index % N
    xyz[:, 1] = (overall_index / N) % N
    xyz[:, 0] = ((overall_index / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    xyz[:, 0] = (xyz[:, 0] * (volume_size[0]/(N-1))) + bbox[0][0]
    xyz[:, 1] = (xyz[:, 1] * (volume_size[1]/(N-1))) + bbox[0][1]
    xyz[:, 2] = (xyz[:, 2] * (volume_size[2]/(N-1))) + bbox[0][2]

    xyz = torch.from_numpy(xyz).float()

    batch_size = 65536
    density = []
    for i in tqdm(range(N ** 3 // batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        density.append(queryfn(xyz[start: end].cuda(), network, cfg.extract_fine)[..., 0].detach().cpu())

    density = torch.cat(density, dim=-1)
    density = density.view(N, N, N)
    vertices, faces, normals, _ = measure.marching_cubes(-density.numpy(), level=cfg.level, spacing=[float(v) / N for v in volume_size])
    vertices += bbox[:1]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    ply_path = os.path.join(cfg.result_dir, '{}.ply'.format('fine' if cfg.extract_fine else 'coarse'))
    os.system('mkdir -p {}'.format(os.path.dirname(ply_path)))
    print(ply_path)
    mesh.export(ply_path)


