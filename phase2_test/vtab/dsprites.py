from copyreg import pickle
import os
import os.path
from typing import Any, Callable, List, Optional, Union, Tuple

from PIL import Image

import numpy as np
import math
from torch import scalar_tensor

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

_TRAIN_SPLIT_PERCENT = 80
_VAL_SPLIT_PERCENT = 10

class DspritesOrientation(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_classes: int = 16,
        type: str = 'all'
    ) -> None:
        super().__init__(os.path.join(root, "dsprites"), transform=transform, target_transform=target_transform)
        self.filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        self.path = os.path.join(root, 'dsprites', self.filename)
        self.original_classes = 40
        group_factor = math.floor(self.original_classes / num_classes)
        data = np.load(self.path, allow_pickle=True)
        origin_imgs = data['imgs']
        tgt = data['latents_classes']
        num_objects = tgt.shape[0]
        labels = []
        imgs = []

        train_count = (num_objects * _TRAIN_SPLIT_PERCENT) // 100
        trainval_count = (num_objects * (_TRAIN_SPLIT_PERCENT + _VAL_SPLIT_PERCENT)) // 100
        for i in range(num_objects):
            labels.append(math.floor(tgt[i, 3] / group_factor))
            img = origin_imgs[i:i+1, :, :] * 255
            img = np.concatenate((img, img, img), axis=0)
            imgs.append(img)

        train_imgs = imgs[:train_count]
        train_labels = labels[:train_count]
        val_imgs = imgs[train_count:trainval_count]
        val_labels = labels[train_count:trainval_count]
        test_imgs = imgs[trainval_count:]
        test_labels = labels[trainval_count:]

        if type == 'train1000':
            self.imgs = train_imgs[:800] + val_imgs[:200]
            self.labels = train_labels[:800] + val_labels[:200]
        if type == 'train800':
            self.imgs = train_imgs[:800]
            self.labels = train_labels[:800]
        if type == 'val200':
            self.imgs = val_imgs[:200]
            self.labels = val_labels[:200]
        if type == 'all':
            if split == 'train':
                self.imgs = train_imgs
                self.labels = train_labels
            elif split == 'val':
                self.imgs = val_imgs
                self.labels = val_labels
            elif split == 'test':
                self.imgs = test_imgs
                self.labels = test_labels
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Any:
        img, label = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, label

class DspritesXLocation(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        num_classes: int = 16,
        type: str = 'all'
    ) -> None:
        super().__init__(os.path.join(root, "dsprites"), transform=transform, target_transform=target_transform)
        self.filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        self.path = os.path.join(root, 'dsprites', self.filename)
        self.original_classes = 32
        group_factor = math.floor(self.original_classes / num_classes)
        data = np.load(self.path, allow_pickle=True)
        origin_imgs = data['imgs']
        tgt = data['latents_classes']
        num_objects = tgt.shape[0]
        labels = []
        imgs = []

        train_count = (num_objects * _TRAIN_SPLIT_PERCENT) // 100
        trainval_count = (num_objects * (_TRAIN_SPLIT_PERCENT + _VAL_SPLIT_PERCENT)) // 100
        for i in range(num_objects):
            labels.append(math.floor(tgt[i, 4] / group_factor))
            img = origin_imgs[i:i+1, :, :] * 255
            img = np.concatenate((img, img, img), axis=0)
            imgs.append(img)

        train_imgs = imgs[:train_count]
        train_labels = labels[:train_count]
        val_imgs = imgs[train_count:trainval_count]
        val_labels = labels[train_count:trainval_count]
        test_imgs = imgs[trainval_count:]
        test_labels = labels[trainval_count:]

        if type == 'train1000':
            self.imgs = train_imgs[:800] + val_imgs[:200]
            self.labels = train_labels[:800] + val_labels[:200]
        if type == 'train800':
            self.imgs = train_imgs[:800]
            self.labels = train_labels[:800]
        if type == 'val200':
            self.imgs = val_imgs[:200]
            self.labels = val_labels[:200]
        if type == 'all':
            if split == 'train':
                self.imgs = train_imgs
                self.labels = train_labels
            elif split == 'val':
                self.imgs = val_imgs
                self.labels = val_labels
            elif split == 'test':
                self.imgs = test_imgs
                self.labels = test_labels
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Any:
        img, label = self.imgs[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, label