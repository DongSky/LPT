import json
import pathlib
from typing import Any, Callable, Optional, Tuple, List
from urllib.parse import urlparse
import numpy as np
from PIL import Image
import os

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20

class Resisc45(VisionDataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        type: str = 'all',
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "train1000", "train800", "val", "val200", "test"))
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_folder = pathlib.Path(self.root) / "resisc45"
        self.label_file = os.path.join("vtab", "resisc45_labels.txt")
        self.meta_file = os.path.join("vtab", "resisc45_train.txt")

        with open(self.label_file, "r") as f:
            self.classes = [c.strip() for c in f.readlines()]
            self.class_to_index = {}
            for i in range(len(self.classes)):
                self.class_to_index[self.classes[i]] = i
        
        with open(self.meta_file, "r") as f:
            self._image_files = [c.strip() for c in f.readlines()]
            self._labels = []
            for c in self._image_files:
                self._labels.append(self.class_to_index[c.split("/")[0]])

        train_count = (len(self._image_files) * TRAIN_SPLIT_PERCENT) // 100
        trainval_count = (len(self._image_files) * (TRAIN_SPLIT_PERCENT + VALIDATION_SPLIT_PERCENT)) // 100
        if split == 'train':
            self._image_files = self._image_files[:train_count]
            self._labels = self._labels[:train_count]
        elif split == 'train800':
            self._image_files = self._image_files[:800]
            self._labels = self._labels[:800]
        elif split == 'train1000':
            self._image_files = self._image_files[:800] + self._image_files[train_count:train_count+200]
            self._labels = self._labels[:800] + self._labels[train_count:train_count+200]
        elif split == 'val':
            self._image_files = self._image_files[train_count:trainval_count]
            self._labels = self._labels[train_count:trainval_count]
        elif split == 'val200':
            self._image_files = self._image_files[train_count:train_count+200]
            self._labels = self._labels[train_count:train_count+200]
        elif split == 'test':
            self._image_files = self._image_files[trainval_count:]
            self._labels = self._labels[trainval_count:]
        
    def __len__(self) -> int:
        return len(self._image_files)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_file = self._image_files[index]
        label = self._labels[index]

        image = Image.open(os.path.join(self._data_folder, image_file)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label