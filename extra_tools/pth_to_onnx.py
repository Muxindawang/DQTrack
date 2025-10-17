# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import torch
import mmdet
import mmcv
from mmcv import Config, DictAction
from mmdet3d.models import build_model

from functools import partial



import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--show', action='store_true', help='show results')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    
    cfg.gpu_ids = [args.gpu_id]

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)



    cfg.model.train_cfg = None
    # build the model and load checkpoint
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.eval()
    model.cuda()

    img = torch.randn(1, 3, 900, 1600, device='cuda')

    timestamp = torch.tensor([0.0], device='cuda')

    # 2. 固定的img_metas（双层嵌套）
    fixed_img_metas = {
                'lidar2img': torch.eye(4, device='cuda').unsqueeze(0),
                'img_shape': torch.tensor([900, 1600], device='cuda'),
                'pad_shape': torch.tensor([900, 1600], device='cuda'),
                'scene_token': ['fcbccedd61424f1b85dcbf8f897f9754'],
            }


    # 创建包装类
    original_forward = model.forward

    def new_forward(img, timestamp):
        return original_forward(
            img=img,
            timestamp=[timestamp],
            return_loss=False,
            img_metas=[fixed_img_metas],
            points=None
        )

    model.forward = new_forward

    # 导出ONNX模型
    torch.onnx.export(
        model,
        args=(img, timestamp),
        f="DQTrack.onnx",
        input_names=['input_img', 'timestamp'],
        output_names=["bbox"],
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=True,
        opset_version=14
    )


if __name__ == '__main__':
    main()
