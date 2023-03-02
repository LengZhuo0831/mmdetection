# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)
from tqdm import tqdm
from mmdet.attack import mmAttacker, Noisier
import matplotlib.pyplot as plt
import cv2
import time
import mmcv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def reverse_transform(image, image_meta):
    # reverse nomarlize
    if 'img_norm_cfg' in image_meta:
        norm_cfg = image_meta['img_norm_cfg']
        image = mmcv.imdenormalize(image,mean=norm_cfg['mean'],std=norm_cfg['std'],to_bgr=False)

    if 'ori_shape' in image_meta:
        h,w,c = image_meta['ori_shape']
        image = mmcv.imresize(image,(w,h))
    # clamp
    image = np.clip(image,0,255)
    return image.astype(np.uint8)

def attack_detector(model,
                   dataset,
                   cfg,
                   distributed=False):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        shuffle=False,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loader = [build_dataloader(ds, **train_loader_cfg) for ds in dataset][0]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    for params in model.parameters():
        params.requires_grad=False

    # FGSM
    # attacker = mmAttacker(model, eps=0.04)
    attacker = Noisier(model, 50)
    torch.cuda.empty_cache()
    img_save_path = '/data/Projects/attack/adv_images/noisy50'
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    for i, data_batch in tqdm(enumerate(data_loader)):
        adv_img = attacker.attack(data_batch)
        save_img = adv_img.permute(0,2,3,1).detach().cpu().numpy()[0]
        img_meta = data_batch['img_metas']._data[0][0]
        save_img = reverse_transform(save_img, img_meta)
        file_name = img_meta['ori_filename']
        full_file_path = img_meta['filename']
        save_name = os.path.join(img_save_path,file_name)
        save_name2 = save_name[:-4]+'ori.jpg'
        plt.imsave(save_name,save_img)
        # os.system(f"cp {full_file_path} {save_name2}")
        # break

    # os.system(f"cd /data/Projects/attack/mmdetection; python tools/test.py configs/ssd/ssd512_coco.py /data/Projects/attack/pretrain_weights/mmdet/ssd512_coco_20210803_022849-0a47a1ca.pth --eval bbox")
    return
