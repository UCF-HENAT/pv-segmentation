from mmengine.runner import Runner
from mmengine import Config
import os.path as osp
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS
import numpy as np
from PIL import Image
import mmengine
import argparse
import datetime




def train_test(args):
    
    if args.model == 'mask2former':
        config_path = '/pv-segmentation/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'
        checkpoint_path = '/pv-segmentation/models/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth'
    elif args.model == 'deeplabv3+':
        config_path = '/pv-segmentation/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
        checkpoint_path = '/pv-segmentation/models/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
    elif args.model == 'unet':
        config_path = '/pv-segmentation/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
        checkpoint_path = '/pv-segmentation/models/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth'
    
    
    classes = ('bg', 'pv')
    palette = [[0, 0, 0], [255, 255, 255]]


    @DATASETS.register_module()
    class MyCustomDataset(BaseSegDataset):
        METAINFO = dict(classes=classes, palette=palette)

        def __init__(self, **kwargs):
            super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

    # Pick dataset and image/label path
    data_root = f'datasets/{args.dataset}'

    img_dir = 'img'
    ann_dir = 'mask'

    # convert dataset annotation to semantic segmentation map
    for file in mmengine.scandir(osp.join(data_root, ann_dir), suffix='.png'):
        seg_img = Image.open(osp.join(data_root, ann_dir, file)).convert('P')
        seg_img.putpalette(np.array(palette, dtype=np.uint8))
        seg_img.save(osp.join(data_root, ann_dir, file))
        
    # Load the config file        
    cfg = Config.fromfile(config_path)

    # Load the pretrained weights
    cfg.load_from = checkpoint_path

    now = datetime.datetime.now()
    
    # Set up working dir to save files and logs
    cfg.work_dir = f'./work_dirs/{args.model}/{args.dataset}/'

    cfg.crop_size = (512, 512)  # Change this: desired crop size
    cfg.model.data_preprocessor.size = cfg.crop_size

    # Change this: set number of classes
    cfg.model.decode_head.num_classes = 2
    cfg.num_classes = 2

    cfg.dataset_type = 'MyCustomDataset'  # Name of the dataset you want to use
    cfg.data_root = data_root  # Directory in which you have images/ and labels/ folders

    # To output all evaluation metrics
    cfg.val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
    cfg.test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(
            type='RandomChoiceResize',
            scales=[
                256, 307, 358, 409, 460, 512, 563, 614, 665, 716, 768, 819, 870,
                921, 972, 1024
            ],
            resize_type='ResizeShortestEdge',
            max_size=2048),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='PackSegInputs')
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 512), keep_ratio=True),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='PackSegInputs')
    ]

    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.data_prefix = dict(
        img_path=img_dir, seg_map_path=ann_dir)
    cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'  # file names for train set
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path=img_dir, seg_map_path=ann_dir)
    cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'  # file names for val set
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline

    cfg.test_dataloader.dataset.type = cfg.dataset_type
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.data_prefix = dict(
        img_path=img_dir, seg_map_path=ann_dir)
    cfg.test_dataloader.dataset.ann_file = 'splits/test.txt'  # file names for test set
    cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline

    
    # Numbers for 50 epochs
    interval = {'google': 8513, 'ign': 4918, 'PV01': 412, 'PV03': 1477, 'PV08': 488}
    total_images_x_num_epochs = {'google': 425650, 'ign': 245900, 'PV01': 20600, 'PV03': 73850, 'PV08': 24400}

    cfg.default_hooks.logger.interval = interval[args.dataset]
    cfg.train_cfg.val_interval = interval[args.dataset]
    cfg.train_cfg.max_iters = total_images_x_num_epochs[args.dataset]
    cfg.default_hooks.checkpoint.interval = total_images_x_num_epochs[args.dataset]
        
    cfg['randomness'] = dict(seed=0)

    weights = [1.0] * cfg.num_classes
    weights.append(0.1) # Expected if you look at original config, need 0.1 as last item

    if args.model == 'mask2former':
        cfg.model.decode_head.loss_cls["class_weight"] = [1.0] * cfg.num_classes + [0.1]

    print(f'Config:\n{cfg.pretty_text}')

    runner = Runner.from_cfg(cfg)
    runner.train()
    runner.test()
    
    print('Training and testing successful')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mask2former', help='model to evaluate')
    parser.add_argument('--dataset', type=str, default='PV01',  help='dataset to evaluate')
    args = parser.parse_args()
    
    train_test(args)