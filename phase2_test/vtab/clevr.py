import json
import pathlib
from typing import Any, Callable, Optional, Tuple, List
from urllib.parse import urlparse
import numpy as np
from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

TRAIN_SPLIT_PERCENT = 90

# def _closest_object_preprocess_fn(x):
#   dist = tf.reduce_min(x["objects"]["pixel_coords"][:, 2])
#   # These thresholds are uniformly spaced and result in more or less balanced
#   # distribution of classes, see the resulting histogram:

#   thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
#   label = tf.reduce_max(tf.where((thrs - dist) < 0))
#   return {"image": x["image"],
#           "label": label}

class CLEVRClassification(VisionDataset):
    """`CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`_  classification dataset.
    The number of objects in a scene are used as label.
    Args:
        root (string): Root directory of dataset where directory ``root/clevr`` exists or will be saved to if download is
            set to True.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in them target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If
            dataset is already downloaded, it is not downloaded again.
    """

    _URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    _MD5 = "b11922020e72d0cd9154779b2d3d07d2"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        type: str = 'all',
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "clevr"
        self._data_folder = self._base_folder / pathlib.Path(urlparse(self._URL).path).stem

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._image_files = sorted(self._data_folder.joinpath("images", self._split).glob("*"))

        self._labels: List[Optional[int]]
        if self._split != "test":
            with open(self._data_folder / "scenes" / f"CLEVR_{self._split}_scenes.json") as file:
                content = json.load(file)
            num_objects = {scene["image_filename"]: len(scene["objects"] - 3) for scene in content["scenes"]} # follow task-adaptation design, minus 3
            self._labels = [num_objects[image_file.name] for image_file in self._image_files]
        else:
            self._labels = [None] * len(self._image_files)
        
        trainval_count = len(self.index)
        train_count = (TRAIN_SPLIT_PERCENT * trainval_count) // 100

        if type == 'train1000':
            self._image_files = self._image_files[:800] + self._image_files[train_count:train_count+200]
            self.targets = self.targets[:800] + self.targets[train_count:train_count+200]
        elif type == 'train800':
            self._image_files = self._image_files[:800]
            self.targets = self.targets[:800]
        elif type == 'val200':
            self._image_files = self._image_files[train_count:train_count+200]
            self.targets = self.targets[train_count:train_count+200]
        else:
            pass

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file = self._image_files[idx]
        label = self._labels[idx]

        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_folder.exists() and self._data_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(self._URL, str(self._base_folder), md5=self._MD5)

    def extra_repr(self) -> str:
        return f"split={self._split}"

class CLEVRDistance(VisionDataset):
    """`CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`_  Distance dataset.
    The number of objects in a scene are used as label.
    Args:
        root (string): Root directory of dataset where directory ``root/clevr`` exists or will be saved to if download is
            set to True.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in them target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If
            dataset is already downloaded, it is not downloaded again.
    """

    _URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    _MD5 = "b11922020e72d0cd9154779b2d3d07d2"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        type: str = 'all',
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "clevr"
        self._data_folder = self._base_folder / pathlib.Path(urlparse(self._URL).path).stem

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self._image_files = sorted(self._data_folder.joinpath("images", self._split).glob("*"))

        # thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])

        self._labels: List[Optional[int]]
        if self._split != "test":
            with open(self._data_folder / "scenes" / f"CLEVR_{self._split}_scenes.json") as file:
                content = json.load(file)
            # num_objects = {scene["image_filename"]: len(scene["objects"]) for scene in content["scenes"]}
            closest_distances = {}
            for scene in content['scenes']:
                objects = content['objects']
                min_dist = 1e6
                for object in objects:
                    min_dist = min(min_dist, object['pixel_coords'][2])
                if min_dist >= 0 and min_dist < 8.0:
                    cat = 0
                elif min_dist >= 8.0 and min_dist < 8.5:
                    cat = 1
                elif min_dist >= 8.5 and min_dist < 9.0:
                    cat = 2
                elif min_dist >= 9.0 and min_dist < 9.5:
                    cat = 3
                elif min_dist >= 9.5 and min_dist < 10.0:
                    cat = 4
                elif min_dist >= 10.0 and min_dist < 100.0:
                    cat = 5
                closest_distances[scene["image_filename"]] = cat
            self._labels = [closest_distances[image_file.name] for image_file in self._image_files]
        else:
            self._labels = [None] * len(self._image_files)
        
        trainval_count = len(self.index)
        train_count = (TRAIN_SPLIT_PERCENT * trainval_count) // 100

        if type == 'train1000':
            self._image_files = self._image_files[:800] + self._image_files[train_count:train_count+200]
            self.targets = self.targets[:800] + self.targets[train_count:train_count+200]
        elif type == 'train800':
            self._image_files = self._image_files[:800]
            self.targets = self.targets[:800]
        elif type == 'val200':
            self._image_files = self._image_files[train_count:train_count+200]
            self.targets = self.targets[train_count:train_count+200]
        else:
            pass

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file = self._image_files[idx]
        label = self._labels[idx]

        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_folder.exists() and self._data_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(self._URL, str(self._base_folder), md5=self._MD5)

    def extra_repr(self) -> str:
        return f"split={self._split}"