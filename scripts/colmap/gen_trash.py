import os
import sys
sys.path.append('.')
import argparse
from lib.utils.colmap.read_write_model import read_model
from tqdm import tqdm
import numpy as np
from lib.utils.mask_utils import get_label_id_mapping
transient_objects = ['person', 'car', 'bicycle', 'minibike', 'tree']

def main(data_root):
    cameras, images, _ = read_model(os.path.join(data_root, 'dense/sparse'))
    
    trashes = []
    os.system(f'rm -rf {data_root}/trash')
    os.system(f'mkdir -p {data_root}/trash')

    if os.path.exists(os.path.join(data_root, 'train.txt')):
        interests = [item.strip() for item in open(os.path.join(data_root, 'train.txt')).readlines()]
    else:
        interests = None
    for im_id in tqdm(images):
        image = images[im_id]
        if interests is not None and image.name not in interests:
            continue
        cam = cameras[image.camera_id]
        min_dim = min(cam.height, cam.width)
        if ((image.point3D_ids != -1).sum() <= 300 and (image.point3D_ids != -1).mean() < 0.1) or min_dim <= 128:
            trashes.append(os.path.basename(image.name))
            os.system('cp {} {}'.format(os.path.join(data_root, 'dense/images', image.name), os.path.join(data_root, 'trash', f'{(image.point3D_ids != -1).sum()}_{image.point3D_ids.__len__()}_' + os.path.basename(image.name))))
        else:
            if os.path.exists(os.path.join(data_root, 'semantic_maps')):
                seg = np.load(os.path.join(data_root, 'semantic_maps', os.path.basename(image.name)[:-4] + '.npz'))['arr_0'] # [..., 0] != 0)ip
                mask = np.ones_like(seg, dtype=np.bool)
                mask[seg==get_label_id_mapping()['person']] = False
                mask[seg==get_label_id_mapping()['car']] = False
                
                if mask.mean() < 0.5: # if the ratio of transient objects is too large, 
                    trashes.append(os.path.basename(image.name))
                    os.system('cp {} {}'.format(os.path.join(data_root, 'dense/images', image.name), os.path.join(data_root, 'trash', f'not-interested_' + os.path.basename(image.name))))
            # if os.path.exists(os.path.join(output_path, 'mask_sphere')):
            #     mask = (imageio.imread(os.path.join(output_path, 'mask_sphere', os.path.basename(image.name))) != 0)
            #     if mask.sum() < 2048:
            #         trashes.append(os.path.basename(image.name))
            #         os.system('cp {} {}'.format(os.path.join(input_path, 'images', image.name), os.path.join(output_path, 'trash', f'not-in-sphere_' + os.path.basename(image.name))))
    open(data_root + '/trash.txt', 'w').writelines([item + '\n' for item in trashes])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    args = parser.parse_args()
    main(args.data_root)