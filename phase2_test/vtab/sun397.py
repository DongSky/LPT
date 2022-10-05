from pathlib import Path
from typing import Any, Tuple, Callable, Optional
import os

import PIL.Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.
    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.
    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        type: str = 'all'
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUN397"

        if split == 'train':
            meta_file = os.path.join("vtab", 'sun397_tfds_tr.txt')
        elif split == 'val':
            meta_file = os.path.join("vtab", 'sun397_tfds_va.txt')
        elif split == 'test':
            meta_file = os.path.join("vtab", 'sun397_tfds_te.txt')
        
        classes_file = os.path.join("vtab", 'sun397_labels.txt')

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # with open(self._data_dir / "ClassName.txt") as f:
        #     self.classes = [c[3:].strip() for c in f]
        
        with open(classes_file) as f:
            self.classes = [c.strip() for c in f.readlines()]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
         
        with open(meta_file) as f:
            self._image_files = [c.strip() for c in f.readlines()]

        # self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]

        if type == 'train1000':
            meta_file_aux = os.path.join("vtab", 'sun397_tfds_va.txt')
            with open(meta_file) as f:
                image_files_aux = [c.strip() for c in f.readlines()]
            labels_aux = [
                self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in image_files_aux
            ]
        
        if type == 'train1000':
            self._image_files = self._image_files[:800] + image_files_aux[:200]
            self._labels = self._labels[:800] + labels_aux[:200]
        elif type == 'train800':
            self._image_files = self._image_files[:800]
            self._labels = self._labels[:200]
        elif type == 'val200':
            self._image_files = self._image_files[:200]
            self._labels = self._labels[:200]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)