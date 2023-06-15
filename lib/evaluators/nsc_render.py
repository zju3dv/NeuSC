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
from lib.utils.rend_utils import draw_text
from lib.utils.data_utils import save_img as save_img_func
import time
import cv2


class Evaluator:

    def __init__(self,):
        if cfg.clear_result:
            os.system('rm -rf ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/overview')
        os.system('mkdir -p ' + cfg.result_dir + '/detail')

    def evaluate(self, output, batch):
        B, N_rays = batch['rays'].shape[:2]
        output['rgb_1']  = torch.clamp(output['rgb_1'], min=0., max=1.)
        for b in range(B):
            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()
            pred_rgb_fine = output['rgb_1'][b].reshape(h, w, 3).detach().cpu().numpy()
            if cfg.save_result:
                save_path = os.path.join(cfg.result_dir, 'detail/view{:06d}'.format(batch['meta']['idx'][b].item()))
                save_path = save_path + '_{}.png'
                save_img_func(save_path.format('rgb'), pred_rgb_fine)
                
                save_imgs = []
                save_imgs.extend([pred_rgb_fine])
                if 'content_rgb_1' in output:
                    content_rgb = output['content_rgb_1'][b].reshape(h, w, 3).detach().cpu().numpy()
                    
                    year = batch['meta']['date'][0][b]
                    month = batch['meta']['date'][1][b]
                    day = batch['meta']['date'][2][b]
                    rs = cfg.test_dataset.input_ratio
                    text_img = draw_text((content_rgb.copy() * 255).astype(np.uint8), 
                                         '{:04d}-{:02d}-{:02d}'.format(year, month, day), 
                                         pos=(w-int(450*rs), int(10*rs)), 
                                         font=cv2.FONT_HERSHEY_COMPLEX, 
                                         font_scale=int(2 * rs + 0.9999), 
                                         text_color=(255,255,255), 
                                         font_thickness=int(3 * rs + 0.9999), 
                                         text_color_bg=(64, 64, 64))
                    save_img_func(save_path.format('content'), text_img)
                    save_imgs.append(content_rgb)

    def summarize(self):
        time.sleep(10)
        # Sleep for 10 seconds to wait for all the images to be written to disk 
        cmd = 'ffmpeg -loglevel error -y -framerate {} -i {}/detail/view%06d_content.png -c:v libx264 -pix_fmt yuv420p {}/content.mp4'.format(cfg.render_fps, cfg.result_dir, cfg.result_dir)
        print(cmd)
        os.system(cmd)
        cmd = 'ffmpeg -loglevel error -y -framerate {} -i {}/detail/view%06d_rgb.png -c:v libx264 -pix_fmt yuv420p {}/rgb.mp4'.format(cfg.render_fps, cfg.result_dir, cfg.result_dir)
        os.system(cmd)
        ret = {}
        print(ret)
        return ret
