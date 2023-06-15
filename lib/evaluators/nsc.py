import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import imageio
from lib.utils import img_utils, vis_utils
import cv2


class Evaluator:

    def __init__(self,):
        self.psnrs = []
        self.ssims = []

        if cfg.clear_result:
            os.system('rm -rf ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/overview')
        os.system('mkdir -p ' + cfg.result_dir + '/detail')

    def evaluate(self, output, batch):
        B, N_rays = batch['rays'].shape[:2]
        output['rgb_1']  = torch.clamp(output['rgb_1'], min=0., max=1.)
        for b in range(B):
            gt_rgb = batch['rgb'][b].reshape(-1, 3).detach().cpu().numpy()
            pred_rgb = output['rgb_1'][b].detach().cpu().numpy()
            self.psnrs.append(psnr(pred_rgb, gt_rgb, data_range=1.))
            
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            gt_rgb = batch['rgb'][b].reshape(h, w, 3).detach().cpu().numpy()
            pred_rgb_fine = output['rgb_1'][b].reshape(h, w, 3).detach().cpu().numpy()
            self.ssims.append(ssim(gt_rgb, pred_rgb_fine, channel_axis=-1, data_range=1.))
            
            if cfg.save_result:
                save_path = os.path.join(cfg.result_dir, 'detail/view{:06d}'.format(batch['meta']['idx'][b].item()))
                save_path = save_path + '_{}.jpg'
                save_imgs = []
                save_path = os.path.join(cfg.result_dir, 'overview', 'view{:06d}_'.format(batch['meta']['idx'][b].item()) + batch['meta']['img_path'][b].split('/')[-1])
                save_imgs.extend([gt_rgb, pred_rgb_fine])
                if 'mask' in output:
                    mask = output['mask'][b].reshape(h, w)[..., None].repeat(1,1,3).detach().cpu().numpy()
                    save_imgs.append(mask)
                if 'content_rgb_1' in output:
                    content_rgb = output['content_rgb_1'][b].reshape(h, w, 3).detach().cpu().numpy()
                    save_imgs.append(content_rgb)
                depth_0 = output['depth_0'][0].reshape(h, w).detach().cpu().numpy()
                depth_1 = output['depth_1'][0].reshape(h, w).detach().cpu().numpy()
                d_min, d_max  = np.stack([depth_0, depth_1]).min(), np.stack([depth_0, depth_1]).max()
                depth_0 = (depth_0 - d_min) / (d_max - d_min)
                depth_1 = (depth_1 - d_min) / (d_max - d_min)
                save_imgs.extend([depth_0[..., None].repeat(3, -1), depth_1[..., None].repeat(3, -1)])
                save_img = vis_utils.merge([ (img * 255.).astype(np.uint8) for img in save_imgs])
                imageio.imwrite(save_path, save_img)

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        ret.update({'ssim': np.mean(self.ssims)})
        print(ret)
        self.psnrs = []
        self.ssims = []
        return ret
