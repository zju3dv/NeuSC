# copy from EasyVolumetricVideo
# Author: Zhen Xu https://github.com/dendenxu

from typing import Callable, List, Dict
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
# from lib.utils.console_utils import *
from threading import Thread

def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper

def parallel_execution(*args, action: Callable, num_processes=32, print_progress=False, sequential=False, async_return=False, desc=None, **kwargs):
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        # TODO: Support all types of iterable
        action_kwargs = {key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == length else kwargs[key]) for key in kwargs}
        return action_args, action_kwargs

    if not sequential:
        # Create ThreadPool
        pool = ThreadPool(processes=num_processes)

        # Spawn threads
        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        if not async_return:
            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())  # will sync the corresponding thread
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), desc=desc, disable=not print_progress):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results
