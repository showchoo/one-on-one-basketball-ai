"""YOLOX Nano, one-class basketball detector experiment.

Apache-2.0 implementation: https://github.com/Megvii-BaseDetection/YOLOX
Model weights and datasets require independent redistribution/license checks.
Run only after ml/export_verified_coco.py passes its session separation checks.
"""
import os

import torch.nn as nn
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.25
        self.num_classes = 1
        self.depthwise = True
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (16, 24)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.35
        self.enable_mixup = False
        self.data_dir = os.environ.get("BALL_COCO_DIR", "ml_data/COCO")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.max_epoch = 80
        self.no_aug_epochs = 12
        self.data_num_workers = 2
        self.eval_interval = 1
        self.exp_name = "ball_yolox_nano"

    def get_model(self, sublinear=False):
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        def init_yolo(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
