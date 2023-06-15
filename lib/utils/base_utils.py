from __future__ import annotations
import pickle
import os
import numpy as np
import cv2
import time
from termcolor import colored
import importlib
import torch.distributed as dist
import math
import trimesh

from copy import copy
from typing import Mapping, TypeVar, Union, Iterable, Callable, Dict
# these are generic type vars to tell mapping to accept any type vars when creating a type
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type

# TODO: move this to engine implementation
# TODO: this is a special type just like Config
# ? However, dotdict is a general purpose data passing object, instead of just designed for config
# The only reason we defined those special variables are for type annotations
# If removed, all will still work flawlessly, just no editor annotation for output, type and meta


def return_dotdict(func: Callable):
    def inner(*args, **kwargs):
        return dotdict(func(*args, **kwargs))
    return inner


class dotdict(dict, Dict[KT, VT]):
    """
    This is the default data passing object used throughout the codebase
    Main function: dot access for dict values & dict like merging and updates

    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = make_dotdict() or d = make_dotdict{'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct: Dict = None, **kwargs):
        dct = copy(dct)  # avoid modifying the original dict, use super's copy to avoid recursion

        # Handle different arguments
        if dct is None:
            dct = kwargs
        elif isinstance(dct, Mapping):
            dct.update(kwargs)
        else:
            super().update(dct, **kwargs)
            return

        # Recursive updates
        for k, v in dct.items():
            if k in self:

                # Handle type conversions
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == 'True'
                    else:
                        dct[k] = target_type(v)

                if isinstance(v, dict):
                    self[k].update(v)  # recursion from here
                else:
                    self[k] = v
            else:
                if isinstance(v, dict):
                    self[k] = dotdict(v)  # recursion?
                else:
                    self[k] = v
        return self

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    copy = return_dotdict(dict.copy)
    fromkeys = return_dotdict(dict.fromkeys)

    # def __hash__(self):
    #     # return hash(''.join([str(self.values().__hash__())]))
    #     return super(dotdict, self).__hash__()

    # def __init__(self, *args, **kwargs):
    #     super(dotdict, self).__init__(*args, **kwargs)

    """
    Uncomment following lines and 
    comment out __getattr__ = dict.__getitem__ to get feature:
    
    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)
    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    # FIXME: not working typing hinting code
    __getattr__: Callable[..., 'torch.Tensor'] = __getitem__  # type: ignore # overidden dict.__getitem__
    __getattribute__: Callable[..., 'torch.Tensor']  # type: ignore
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # TODO: better ways to programmically define these special variables?

    @property
    def meta(self) -> dotdict:
        # Special variable used for storing cpu tensor in batch
        if 'meta' not in self:
            self.meta = dotdict()
        return self.__getitem__('meta')

    @meta.setter
    def meta(self, meta):
        self.__setitem__('meta', meta)

    @property
    def output(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing output tensor in batch
        if 'output' not in self:
            self.output = dotdict()
        return self.__getitem__('output')

    @output.setter
    def output(self, output):
        self.__setitem__('output', output)

    @property
    def type(self) -> str:  # late annotation needed for this
        # Special entry for type based construction system
        return self.__getitem__('type')

    @type.setter
    def type(self, type):
        self.__setitem__('type', type)


class default_dotdict(dotdict):
    def __init__(self, default_type=object, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        dict.__setattr__(self, 'default_type', default_type)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (AttributeError, KeyError) as e:
            super().__setitem__(key, dict.__getattribute__(self, 'default_type')())
            return super().__getitem__(key)


context = dotdict()  # a global context object. Forgot why I did this. TODO: remove this

def get_aabb(ply_path, scale=1.5):
    mesh = trimesh.load(ply_path)
    points = np.array(mesh.vertices)
    min_point, max_point = points.min(axis=0), points.max(axis=0)
    return min_point, max_point - min_point

def get_center_r(ply_path, scale=1.5):
    mesh = trimesh.load(ply_path)
    points = np.array(mesh.vertices)
    min_point, max_point = points.min(axis=0), points.max(axis=0)
    center_point = (min_point + max_point)/2.
    r = np.linalg.norm(max_point - min_point)/2. * scale
    return center_point, r

class perf_timer:
    def __init__(self, msg="Elapsed time: {}s", logf=lambda x: print(colored(x, 'yellow')), sync_cuda=True, use_ms=False, disabled=False):
        self.logf = logf
        self.msg = msg
        self.sync_cuda = sync_cuda
        self.use_ms = use_ms
        self.disabled = disabled

        self.loggedtime = None

    def __enter__(self,):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.logtime(self.msg)

    def logtime(self, msg=None, logf=None):
        if self.disabled:
            return
        # SAME CLASS, DIFFERENT FUNCTIONALITY, is this good?
        # call the logger for timing code sections
        if self.sync_cuda:
            torch.cuda.synchronize()

        # always remember current time
        prev = self.loggedtime
        self.loggedtime = time.perf_counter()

        # print it if we've remembered previous time
        if prev is not None and msg:
            logf = logf or self.logf
            diff = self.loggedtime-prev
            diff *= 1000 if self.use_ms else 1
            logf(msg.format(diff))

        return self.loggedtime

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bbox_2d(bbox, K, RT):
    pts = np.array([[bbox[0, 0], bbox[0, 1], bbox[0, 2]],
                    [bbox[0, 0], bbox[0, 1], bbox[1, 2]],
                    [bbox[0, 0], bbox[1, 1], bbox[0, 2]],
                    [bbox[0, 0], bbox[1, 1], bbox[1, 2]],
                    [bbox[1, 0], bbox[0, 1], bbox[0, 2]],
                    [bbox[1, 0], bbox[0, 1], bbox[1, 2]],
                    [bbox[1, 0], bbox[1, 1], bbox[0, 2]],
                    [bbox[1, 0], bbox[1, 1], bbox[1, 2]],
                    ])
    pts_2d = project(pts, K, RT)
    return [pts_2d[:, 0].min(), pts_2d[:, 1].min(), pts_2d[:, 0].max(), pts_2d[:, 1].max()]


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def load_object(module_name, module_args, **extra_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**extra_args, **module_args)
    return obj



def get_indices(length):
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    num_samples = int(math.ceil(length * 1.0 / num_replicas))
    total_size = num_samples * num_replicas
    indices = np.arange(length).tolist()
    indices += indices[: (total_size - len(indices))]
    offset = num_samples * rank
    indices = indices[offset:offset+num_samples]
    return indices



