import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cv2

def main(vdir):
    img_path_list = sorted(os.listdir(vdir))
    img_list = []
    for img_path in img_path_list:
        img_list.append((np.array(imageio.imread(os.path.join(vdir, img_path)))/255.).astype(np.float32))


    # grid
    h, w = img_list[0].shape[:2]
    win_size = 16

    vis_h, vis_w = h // win_size * (win_size+2), w // win_size * (win_size+2)
    vis_img = np.ones((vis_h, vis_w, 3))
    for i in range(h//win_size):
        for j in range(w//win_size):
            vis_img[i*(win_size+2):(i+1)*(win_size+2)-2, j*(win_size+2):(j+1)*(win_size+2)-2] = \
                    img_list[0][i*win_size:(i+1)*win_size, j*win_size:(j+1)*win_size]

    imgs_np = np.array(img_list)
    def ps(x):
        return x*win_size
    def pe(x):
        return (x+1)*win_size

    patches = []
    for i in range(h//win_size):
        patches_ = []
        for j in range(w//win_size):
            patches_.append(Patch(imgs_np[:, ps(i):pe(i), ps(j):pe(j)], img_path_list, win_size))
        patches.append(patches_)

    patches = process_patches(patches)

    imgs = []
    for time in img_path_list:
        img = vis_patches(patches, time.split('.')[0], win_size)
        year, month, day = [int(item) for item in time.split('.')[0].split('-')]
        h, w = img.shape[:2]
        img = cv2.putText(img, '{:04d}-{:02d}-{:02d}'.format(year, month, day),(w//2-100, h//2-100),cv2.FONT_HERSHEY_COMPLEX,1,(1.,1.,1.),2)
        imgs.append(img)
    imageio.mimwrite('test.mp4', imgs, fps=4)

def vis_patches(patches, time, win_size):
    h, w = len(patches) * win_size, len(patches[0]) * win_size
    img = np.zeros((h, w, 3))
    for i in range(h//win_size):
        for j in range(w//win_size):
            img[i*win_size:(i+1)*win_size, j*win_size:(j+1)*win_size] = patches[i][j].gettime(time)
    # plt.imshow(img)
    # plt.show()
    return img.copy()

min_num = 5
min_score = 0.8
same_patch_score = 0.9

class Patch:
    def __init__(self, patches, time_list, win_size):
        cov = np.corrcoef(patches.reshape(len(time_list), -1))
        self.cov = cov
        self.patches_color = patches
        patch_remain = np.ones(len(time_list)).astype(np.bool)

        self.win_size = win_size
        self.time_list = []
        self.patches = []
        def get_patch(score, in_patch, patch_remain):
            same_patch = list(set((score[in_patch] > same_patch_score).nonzero()[1]))
            in_patch[same_patch] = True
            if (patch_remain[in_patch] == False).all():
                return
            patch_remain[in_patch] = False
            get_patch(score, in_patch, patch_remain)

        while patch_remain.sum() > 0:
            in_patch = np.arange(len(patch_remain))  == patch_remain.nonzero()[0][0]
            get_patch(cov, in_patch, patch_remain)
            if in_patch.sum() < 5 and patch_remain.sum() < 5:
                break
            if in_patch.sum() >= 5:
                begin_time = datetime.date(*[int(item) for item in time_list[in_patch.nonzero()[0][0]].split('.')[0].split('-')])
                end_time = datetime.date(*[int(item) for item in time_list[in_patch.nonzero()[0][-1]].split('.')[0].split('-')])
                self.patches.append((in_patch, patches[in_patch].mean(axis=0), [begin_time, end_time]))
        self.mean_patch_color = np.mean(patches, axis=0)

    def gettime(self, time):
        if len(self.patches) == 0:
            return self.mean_patch_color
        elif len(self.patches) == 1:
            return self.patches[0][1]
        else:
            time = datetime.date(*[int(item) for item in time.split('-')])
            result_high = []
            result = []
            for patch in self.patches:
                if (time - patch[2][0]).days > 0 and (time - patch[2][1]).days < 0:
                    result_high.append(((patch[2][1] - patch[2][0]).days, patch))
                else:
                    result.append(np.min(np.abs([(time-patch[2][0]).days, (time-patch[2][1]).days])))
            if len(result_high) != 0:
                return result_high[np.array([result_high[i][0] for i in range(len(result_high)) ]).argmin()][1][1]
            else:
                return self.patches[np.array(result).argmin()][1]


def process_patches(patches):
    return patches
    patches_np = np.array(patches)
    patches_len = np.zeros(patches_np.shape)
    for i in range(patches_len.shape[0]):
        for j in range(patches_len.shape[1]):
            patches_len[i, j] = patches_np[i, j].patches.__len__()

    start_patch = np.zeros_like(patches_len)
    start_patch[0, 0] = True
    start_edges =

    while True:
        __import__('ipdb').set_trace()
