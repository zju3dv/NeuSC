from .yacs import CfgNode as CN
import argparse
import os
import numpy as np
from . import yacs

cfg = CN()

cfg.workspace = os.environ['workspace']
print('Workspace: ', cfg.workspace)

cfg.save_result = False
cfg.clear_result = False
cfg.save_tag = 'default'
# module
cfg.train_dataset_module = 'lib.datasets.dtu.neus'
cfg.test_dataset_module = 'lib.datasets.dtu.neus'
cfg.val_dataset_module = 'lib.datasets.dtu.neus'
cfg.network_module = 'lib.neworks.neus.neus'
cfg.loss_module = 'lib.train.losses.neus'
cfg.evaluator_module = 'lib.evaluators.neus'

# experiment name
cfg.exp_name = 'gitbranch_hello'
cfg.exp_name_tag = ''
cfg.pretrain = ''

# network
cfg.distributed = False

# task
cfg.task = 'hello'

# gpus
cfg.gpus = list(range(1))
# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.save_latest_ep = 1000
cfg.eval_ep = 1
cfg.log_interval = 1

cfg.task_arg = CN()
# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = 'default'
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({})
cfg.train.shuffle = True
cfg.train.eps = 1e-8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 5e-4
cfg.train.weight_decay = 0.
cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [
                         80, 120, 200, 240], 'gamma': 0.5})
cfg.train.batch_size = 4

# test
cfg.test = CN()
cfg.test.batch_size = 1
cfg.test.collator = 'default'
cfg.test.epoch = -1
cfg.test.num_workers = 0
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({})

# trained model
cfg.trained_model_dir = os.path.join(os.environ['workspace'], 'trained_model')
cfg.clean_tag = 'debug'

# recorder
cfg.record_dir = os.path.join(os.environ['workspace'], 'record')

# result
cfg.result_dir = os.path.join(os.environ['workspace'], 'result')

# evaluation
cfg.skip_eval = False
cfg.fix_random = False


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')
    # assign the gpus
    if -1 not in cfg.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(
            [str(gpu) for gpu in cfg.gpus])

    if len(cfg.exp_name_tag) != 0:
        cfg.exp_name += ('_' + cfg.exp_name_tag)
    cfg.exp_name = cfg.exp_name.replace('gitbranch', os.popen(
        'git describe --all').readline().strip()[6:])
    cfg.exp_name = cfg.exp_name.replace('gitcommit', os.popen(
        'git describe --tags --always').readline().strip())
    print('EXP NAME: ', cfg.exp_name)
    
    cfg.trained_model_dir = os.path.join(
        cfg.trained_model_dir, cfg.task, cfg.scene, cfg.exp_name)
    cfg.record_dir = os.path.join(
        cfg.record_dir, cfg.task, cfg.scene, cfg.exp_name)
    cfg.result_dir = os.path.join(
        cfg.result_dir, cfg.task, cfg.scene, cfg.exp_name, cfg.save_tag)
    cfg.local_rank = args.local_rank
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        cfg[module.replace('_module', '_path')] = cfg[module].replace(
            '.', '/') + '.py'


def make_cfg(args):
    def merge_cfg(cfg_file, cfg):
        with open(cfg_file, 'r') as f:
            current_cfg = yacs.load_cfg(f)
        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            cfg.merge_from_other_cfg(current_cfg)
        else:
            cfg.merge_from_other_cfg(current_cfg)
        print(cfg_file)
        return cfg
    cfg_ = merge_cfg(args.cfg_file, cfg)
    try:
        index = args.opts.index('other_opts')
        cfg_.merge_from_list(args.opts[:index])
    except:
        cfg_.merge_from_list(args.opts)
    parse_cfg(cfg_, args)
    return cfg_


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
