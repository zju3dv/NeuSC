import sys
import argparse
import os
import glob
import pandas as pd
sys.path.append('.')
from lib.utils.colmap.read_write_model import read_images_binary
from mmseg.apis import inference_segmentor, init_segmentor
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import imageio

transient_objects = ['person', 'car', 'bicycle', 'minibike', 'tree']
label_id_mapping_ade20k = {'airplane': 90,
 'animal': 126,
 'apparel': 92,
 'arcade machine': 78,
 'armchair': 30,
 'ashcan': 138,
 'awning': 86,
 'bag': 115,
 'ball': 119,
 'bannister': 95,
 'bar': 77,
 'barrel': 111,
 'base': 40,
 'basket': 112,
 'bathtub': 37,
 'bed ': 7,
 'bench': 69,
 'bicycle': 127,
 'blanket': 131,
 'blind': 63,
 'boat': 76,
 'book': 67,
 'bookcase': 62,
 'booth': 88,
 'bottle': 98,
 'box': 41,
 'bridge': 61,
 'buffet': 99,
 'building': 1,
 'bulletin board': 144,
 'bus': 80,
 'cabinet': 10,
 'canopy': 106,
 'car': 20,
 'case': 55,
 'ceiling': 5,
 'chair': 19,
 'chandelier': 85,
 'chest of drawers': 44,
 'clock': 148,
 'coffee table': 64,
 'column': 42,
 'computer': 74,
 'conveyer belt': 105,
 'counter': 45,
 'countertop': 70,
 'cradle': 117,
 'crt screen': 141,
 'curtain': 18,
 'cushion': 39,
 'desk': 33,
 'dirt track': 91,
 'dishwasher': 129,
 'door': 14,
 'earth': 13,
 'escalator': 96,
 'fan': 139,
 'fence': 32,
 'field': 29,
 'fireplace': 49,
 'flag': 149,
 'floor': 3,
 'flower': 66,
 'food': 120,
 'fountain': 104,
 'glass': 147,
 'grandstand': 51,
 'grass': 9,
 'hill': 68,
 'hood': 133,
 'house': 25,
 'hovel': 79,
 'kitchen island': 73,
 'lake': 128,
 'lamp': 36,
 'land': 94,
 'light': 82,
 'microwave': 124,
 'minibike': 116,
 'mirror': 27,
 'monitor': 143,
 'mountain': 16,
 'ottoman': 97,
 'oven': 118,
 'painting': 22,
 'palm': 72,
 'path': 52,
 'person': 12,
 'pier': 140,
 'pillow': 57,
 'plant': 17,
 'plate': 142,
 'plaything': 108,
 'pole': 93,
 'pool table': 56,
 'poster': 100,
 'pot': 125,
 'radiator': 146,
 'railing': 38,
 'refrigerator': 50,
 'river': 60,
 'road': 6,
 'rock': 34,
 'rug': 28,
 'runway': 54,
 'sand': 46,
 'sconce': 134,
 'screen': 130,
 'screen door': 58,
 'sculpture': 132,
 'sea': 26,
 'seat': 31,
 'shelf': 24,
 'ship': 103,
 'shower': 145,
 'sidewalk': 11,
 'signboard': 43,
 'sink': 47,
 'sky': 2,
 'skyscraper': 48,
 'sofa': 23,
 'stage': 101,
 'stairs': 53,
 'stairway': 59,
 'step': 121,
 'stool': 110,
 'stove': 71,
 'streetlight': 87,
 'swimming pool': 109,
 'swivel chair': 75,
 'table': 15,
 'tank': 122,
 'television receiver': 89,
 'tent': 114,
 'toilet': 65,
 'towel': 81,
 'tower': 84,
 'trade name': 123,
 'traffic light': 136,
 'tray': 137,
 'tree': 4,
 'truck': 83,
 'van': 102,
 'vase': 135,
 'wall': 0,
 'wardrobe': 35,
 'washer': 107,
 'water': 21,
 'waterfall': 113,
 'windowpane': 8}
def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to run')

    return parser.parse_args()

# deeplabv3 config file path
config_file = 'configs/3rdparty/deeplabv3_config/deeplabv3_r101-d8_512x512_160k_ade20k.py'
checkpoint_file = '{}/3rdparty/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'.format(os.environ['workspace'])


def get_mask(result):
    mask = np.ones_like(result)
    for label_name in transient_objects:
        mask[label_id_mapping_ade20k[label_name] == result] = False
    return np.logical_not(mask).astype(np.uint8)

if __name__ == '__main__':
    args = get_opts()

    os.makedirs(os.path.join(args.root_dir, f'semantic_maps'), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, f'mask_vis'), exist_ok=True)

    print(f'Preparing semantic maps for {args.root_dir.split("/")[-1]} set...')

    image_paths = os.listdir(os.path.join(args.root_dir, 'dense/images'))

    # build the DeepLabv3 model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=f'cuda:{args.gpu}')


    for root, dirs, files in os.walk(os.path.join(args.root_dir, 'dense/images')):
        print(root)
        for file in tqdm(files):
            if file[0] == '.':
                continue
            img_path = os.path.join(root, file)
            img = Image.open(os.path.join(img_path))
            img_w, img_h = img.size
            img = np.array(img)
            if max(img.shape) > 1500:
                h, w = img.shape[:2]
                tar_h, tar_w = int(h*0.5), int(w*0.5)
                result = inference_segmentor(model, cv2.resize(img, (tar_w, tar_h), interpolation=cv2.INTER_AREA))[0] # (H, W)
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                result = inference_segmentor(model, img)[0] # (H, W)
            result = result.copy()
            mask = get_mask(result)
            msk = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.6, msk, 0.4, 0)
            imageio.imwrite(os.path.join(args.root_dir, 'mask_vis', os.path.basename(img_path)), img)
            image_name = os.path.basename(img_path).split('.')[0]
            np.savez_compressed(os.path.join(
                args.root_dir, f'semantic_maps/{image_name}.npz'), result)

