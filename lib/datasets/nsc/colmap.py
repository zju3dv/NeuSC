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

# TODO: hard coding
transient_objects = ['bicycle', 'minibike']


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
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

        self.load_metas(**kwargs)
        self.load_data(**kwargs)

    def load_metas(self, **kwargs):
        cam_dict = np.load(
            join(self.data_root, 'annots/cam_dict.npy'), allow_pickle=True).item()
        self.cam_dict = cam_dict
        s_date = datetime.date(*cfg.start_date)
        e_date = datetime.date(*cfg.end_date)

        # filter
        trashes = []
        interests = []
        if os.path.exists(os.path.join(self.data_root, 'annots/trash.txt')):
            trashes.extend([line.strip() for line in open(
                os.path.join(self.data_root, 'annots/trash.txt')).readlines()])
        if os.path.exists(os.path.join(self.data_root, 'annots/{}'.format(kwargs['ann_file']))):
            interests.extend([line.strip() for line in open(os.path.join(
                self.data_root + '/annots', kwargs['ann_file'])).readlines()])

        # build metas

        self.metas = []
        self.img_paths = []
        self.times = []
        for k in tqdm(cam_dict):
            img_name = cam_dict[k]['img_path']
            img_path = os.path.join(
                self.data_root, 'dense', 'images', img_name)
            if len(interests) != 0 and img_name not in interests and os.path.basename(img_name) not in interests:
                continue
            if os.path.basename(img_path).split('_')[1] in trashes or os.path.basename(img_path) in trashes:
                continue
            if os.path.basename(img_path).split('_')[0] == 'none':
                continue
            time = [int(item) for item in os.path.basename(
                img_path).split('_')[0].split('-')]
            year, month, day = time
            date = datetime.date(*(year, month, day))
            if (date - s_date).days < 0 or (date-e_date).days > 0:
                continue
            # date = datetime.date(*(year, month, day))
            s_date = datetime.date(*cfg.start_date)
            e_date = datetime.date(*cfg.end_date)
            t = ((date - s_date).days) / ((e_date - s_date).days)
            self.metas.append(k)
            self.img_paths.append(img_path)
            self.times.append(t)

        self.global_metas = self.metas
        # The number of images should be less than the number of learnable per-image embeddings
        assert(len(self.global_metas) < cfg.n_emb) 

        b, e, s = kwargs['input_sample']
        e = len(self.metas) if e == -1 else e
        self.metas = self.metas[b:e:s]
        self.img_paths = self.img_paths[b:e:s]
        self.times = self.times[b:e:s]

    def load_data(self, **kwargs):
        if self.split not in ['test', 'train']:
            return
        self.img_buffers = parallel_execution(self.img_paths, action=load_image_bytes, num_processes=32,
                                              print_progress=True, sequential=False, async_return=False, desc='preload bytes for images')
        seg_paths = [join(self.data_root, 'semantic_maps', os.path.basename(
            img_path).replace('.jpg', '.npz')) for img_path in self.img_paths]
        self.seg_buffers = parallel_execution(seg_paths, action=load_npz_encode_bytes, num_processes=32,
                                              print_progress=True, sequential=False, async_return=False, desc='preload bytes for semantic maps')

    def __getitem__(self, index):
        # load meta
        key = self.metas[index]
        time = self.times[index]
        img = load_image_from_bytes(self.img_buffers[index])
        emb_id = self.global_metas.index(key)
        ext = self.cam_dict[key]['ext']
        ixt = self.cam_dict[key]['ixt'].copy()
        near_far = self.cam_dict[key]['near_far']

        if self.input_ratio != 1:
            img = cv2.resize(
                img, (int(img.shape[1] * self.input_ratio), int(img.shape[0] * self.input_ratio)))
            ixt[:2] *= self.input_ratio

        h, w = img.shape[:2]

        XYZ = self.XYZ[:h, :w].reshape(-1, 3).copy()
        rgb = img.reshape(-1, 3)

        # training needs sample rays
        if self.split == 'train':
            # filter person and car
            seg = load_image_from_bytes(
                self.seg_buffers[index], normalize=False)[..., 0]
            msk = np.logical_and(seg != get_label_id_mapping()[
                                 'person'], seg != get_label_id_mapping()['car'])
            # for object in transient_objects:
            # msk = np.logical_and(msk, seg != get_label_id_mapping()[object])
            # DEBUG
            # import matplotlib.pyplot as plt
            # plt.imsave('test_rgb.jpg', img)
            # plt.imsave('test_seg.jpg', seg)
            # plt.imsave('test_msk.jpg', msk)
            # import ipdb; ipdb.set_trace()

            nonzero = msk.reshape(-1).nonzero()[0]
            ids = np.random.choice(nonzero, size=cfg.num_pixels, replace=False)
            seg = seg.reshape(-1)[ids]

            XYZ = XYZ[ids]
            rgb = rgb[ids]

        # N x 8, rays_o, rays_d, near_far, in
        rays = self.get_rays(XYZ, ixt, ext, near_far)
        uv = XYZ[:, :2].copy()
        uv[:, 0] /= w
        uv[:, 1] /= h
        time = time * np.ones((rays.shape[0], 1))
        emb_id = emb_id * np.ones((rays.shape[0], 1))
        rays = np.concatenate([rays, time, emb_id, uv],
                              axis=-1).astype(np.float32)

        ret = {'rays': rays, 'rgb': rgb}
        if self.split == 'train':
            ret.update({'semantics': seg})
        ret['meta'] = {'idx': index, 'name': os.path.basename(self.cam_dict[key]['img_path']), 'h': h, 'w': w, 'emb_id': emb_id[0].item(),
                       'img_path': self.cam_dict[key]['img_path']}
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
