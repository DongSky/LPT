import os
from re import T
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20

class EuroSAT(VisionDataset):
    """RGB version of the `EuroSAT <https://github.com/phelber/eurosat>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``root/eurosat`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")
        super().__init__(self._data_folder, transform=transform, target_transform=target_transform)

        if download:
            self.download()
        self.label_file = os.path.join("vtab", "eurosat_labels.txt")
        self.meta_file = os.path.join("vtab", "eurosat_train.txt")
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
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

        self.root = os.path.expanduser(root)
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

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self._base_folder, exist_ok=True)
        download_and_extract_archive(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
            download_root=self._base_folder,
            md5="c8fa014336c82ac7804f0398fcb19387",
        )