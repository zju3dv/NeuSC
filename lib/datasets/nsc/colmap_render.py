import torch.utils.data as data
import numpy as np
import os
from lib.config import cfg
from lib.utils.mask_utils import get_label_id_mapping
from lib.utils.data_utils import load_image_bytes, load_image_from_bytes, load_npz_encode_bytes
from lib.utils.parallel_utils import parallel_execution
import datetime
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join
import cv2
from lib.utils.rend_utils import gen_render_queue
import trimesh

# TODO: hard coding
transient_objects = ['bicycle', 'minibike']





class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        # 1. generate render_queue
        # 2. load_pose
        # 3. ixt, emb, near_fars
        
        # parse basic information
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        data_root = os.path.join(os.environ['workspace'], data_root)
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.scene = scene
        self.split = split

        # epoch
        self.epoch = 0

        # precaching
        MAX_IMG_SIZE = 5120
        X, Y = np.meshgrid(np.arange(MAX_IMG_SIZE), np.arange(MAX_IMG_SIZE))
        X, Y = X + 0.5, Y + 0.5  # shift pixels
        self.XYZ = np.stack([X, Y, np.ones_like(X)],
                            axis=-1).astype(np.float32)
        

        self.metas = gen_render_queue(cfg.render_json)
        self.load_exts(**kwargs)   
        self.load_ixt(**kwargs)
        self.compute_near_fars(**kwargs)
        
    def load_exts(self, **kwargs):
        self.exts = []
        b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        transforms = np.load(cfg.render_path)
        for transform in transforms:
            self.exts.append(np.linalg.inv(transform @ b2c).astype(np.float32))
        
    def load_ixt(self, **kwargs):
        fov, h, w, emb_id = cfg.render_fhwe 
        ixt = np.eye(3).astype(np.float32)
        ixt[0][2], ixt[1][2] = w/2., h/2.
        focal = max(ixt[0][2] / np.tan(.5 * fov / 180 * np.pi), ixt[1][2] / np.tan(.5 * fov / 180 * np.pi))
        ixt[0][0], ixt[1][1] = focal, focal
        self.ixt = ixt
        self.emb_id = emb_id
        self.h = h
        self.w = w
        
    def compute_near_fars(self, **kwargs):
        mesh = trimesh.load(os.path.join(cfg.workspace, cfg.test_dataset.data_root, cfg.scene, cfg.octree_ply_path))
        points = np.array(mesh.vertices).astype(np.float32)
        def compute_near_far(points, ext, ixt, h, w):
            xyz = points @ ext[:3, :3].T + ext[:3, 3:].T
            xyz = xyz @ ixt.T 
            uv = xyz[:, :2] / xyz[:, 2:]
            mask_w = np.logical_and(uv[:, 0] >= 0., uv[:, 0]<= w)
            mask_h = np.logical_and(uv[:, 1] >= 0., uv[:, 1]<= h)
            mask = np.logical_and(mask_w, mask_h)
            xyz = xyz[mask]
            min_v, max_v = np.percentile(xyz[:, 2], 0.01), np.percentile(xyz[:, 2], 99.99)
            near_far = np.array([min_v, max_v]).astype(np.float32)
            return near_far
        self.near_fars = parallel_execution(
            [points for ext in self.exts],
            [ext for ext in self.exts],
            [self.ixt for ext in self.exts],
            [self.h for ext in self.exts],
            [self.w for ext in self.exts],
            action = compute_near_far,
            num_processes=32,
            print_progress=True,
            sequential=False,
            async_return=False,
            desc = 'Computing near far'
        )
            
    def __getitem__(self, index):
        # load meta
        date, view_id = self.metas[index]
        near_far = self.near_fars[view_id]
        ext = self.exts[view_id]
        s_date = datetime.date(*cfg.start_date)
        e_date = datetime.date(*cfg.end_date)
        time = ((date - s_date).days) / ((e_date - s_date).days)
        
        emb_id = self.emb_id
        ixt = self.ixt.copy()
        h, w = self.h, self.w
        if self.input_ratio != 1:
            h, w = h * self.input_ratio, w * self.input_ratio
            ixt[:2] *= self.input_ratio
        h, w = int(h), int(w)
        XYZ = self.XYZ[:h, :w].reshape(-1, 3).copy()
        
        # N x 8, rays_o, rays_d, near_far, in
        rays = self.get_rays(XYZ, ixt, ext, near_far)
        uv = XYZ[:, :2].copy()
        uv[:, 0] /= w
        uv[:, 1] /= h
        time = time * np.ones((rays.shape[0], 1))
        emb_id = emb_id * np.ones((rays.shape[0], 1))
        rays = np.concatenate([rays, time, emb_id, uv],
                              axis=-1).astype(np.float32)
        ret = {'rays': rays}
        ret['meta'] = {'idx': index, 'h': h, 'w': w, 'emb_id': emb_id[0].item(), 'date': [date.year, date.month, date.day]}
        return ret

    def __len__(self):
        return len(self.metas)
    
    def get_rays(self, xy, ixt, ext, near_far):
        c2w = np.linalg.inv(ext)
        XYZ = np.stack([(xy[..., 0]-ixt[0][2])/ixt[0][0], (xy[..., 1] -
                       ixt[1][2])/ixt[1][1], np.ones_like(xy[..., 0])], axis=-1)
        rays_d = (XYZ[..., None, :] * c2w[:3, :3]).sum(-1)
        rays = np.concatenate((c2w[:3, 3][None].repeat(len(
            rays_d), axis=0), rays_d, near_far[None].repeat(len(rays_d), axis=0)), axis=-1)
        return rays.astype(np.float32)
