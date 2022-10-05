import torch
from vtab.cifar import CIFAR10, CIFAR100
from vtab.flowers102 import Flowers102
from vtab.clevr import CLEVRClassification, CLEVRDistance
from vtab.dtd import DTD
from vtab.eurosat import EuroSAT
from vtab.caltech import Caltech101
from vtab.svhn import SVHN
from vtab.stanford_cars import StanfordCars
from vtab.cub import Cub2011
from vtab.imbalanced_cifar import IMBALANCECIFAR100
from vtab.LT_dataset import LT_Dataset
from vtab.LT_dataset_twoview import LT_Dataset_twoview


def cifar100_1k_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train1000')
    val_set = CIFAR100(data_path, train=False, transform=val_transforms, download=download, type='test')
    num_classes = 100
    return train_set, val_set, num_classes

def cifar100_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train800')
    val_set = CIFAR100(data_path, train=True, transform=val_transforms, download=download, type='val200')
    num_classes = 100
    return train_set, val_set, num_classes
    # raise NotImplementedError

def cifar100_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CIFAR100(data_path, train=True, transform=train_transforms, download=download, type='train1000')
    val_set = CIFAR100(data_path, train=False, transform=val_transforms, download=download, type='test')
    num_classes = 100
    return train_set, val_set, num_classes
    raise NotImplementedError

def flowers102_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def flowers102_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = Flowers102(data_path, split='val', transform=val_transforms, download=download, type='val200')
    num_classes = 102
    return train_set, val_set, num_classes

def flowers102_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='all')
    num_classes = 102
    return train_set, val_set, num_classes

#TODO: caltech-101 datasets
def caltech101_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    # raise NotImplementedError
    train_set = Caltech101(data_path, split='trainval', transform=train_transforms, download=download, type='train1000')
    val_set = Caltech101(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes

def caltech101_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    # raise NotImplementedError
    train_set = Caltech101(data_path, split='trainval', transform=train_transforms, download=download, type='train800')
    val_set = Caltech101(data_path, split='trainval', transform=val_transforms, download=download, type='val200')
    num_classes = 102
    return train_set, val_set, num_classes

def caltech101_full_datasets(data_path, train_transforms, val_transforms, download=False):
    # raise NotImplementedError
    train_set = Caltech101(data_path, split='trainval', transform=train_transforms, download=download, type='train')
    val_set = Caltech101(data_path, split='test', transform=val_transforms, download=download, type='all')
    num_classes = 102
    return train_set, val_set, num_classes

def clevr_count_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = CLEVRClassification(data_path, split='val', transform=val_transforms, download=download, type='val')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_count_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = CLEVRClassification(data_path, split='train', transform=val_transforms, download=download, type='val200')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_count_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRClassification(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = CLEVRClassification(data_path, split='val', transform=val_transforms, download=download, type='all')
    num_classes = 8
    return train_set, val_set, num_classes

def clevr_distance_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = CLEVRDistance(data_path, split='val', transform=val_transforms, download=download, type='val')
    num_classes = 6
    return train_set, val_set, num_classes

def clevr_distance_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = CLEVRDistance(data_path, split='train', transform=val_transforms, download=download, type='val200')
    num_classes = 6
    return train_set, val_set, num_classes

def clevr_distance_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = CLEVRDistance(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = CLEVRDistance(data_path, split='val', transform=val_transforms, download=download, type='all')
    num_classes = 6
    return train_set, val_set, num_classes

#TODO: Retinopathy datasets
def retinopathy_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    raise NotImplementedError
    train_set = Flowers102(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = Flowers102(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 102
    return train_set, val_set, num_classes


def dtd_1k_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = DTD(data_path, split='train', transform=train_transforms, download=download, type='train1000')
    val_set = DTD(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 47
    return train_set, val_set, num_classes

def dtd_800_200_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = DTD(data_path, split='train', transform=train_transforms, download=download, type='train800')
    val_set = DTD(data_path, split='val', transform=val_transforms, download=download, type='val200')
    num_classes = 47
    return train_set, val_set, num_classes

def dtd_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = DTD(data_path, split='train', transform=train_transforms, download=download, type='all')
    val_set = DTD(data_path, split='test', transform=val_transforms, download=download, type='test')
    num_classes = 47
    return train_set, val_set, num_classes

def eurosat_1k_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = EuroSAT(data_path, 'train1000', train_transforms, download=download)
    val_set = EuroSAT(data_path, 'test', val_transforms, download=False)
    num_classes = 10
    return train_set, val_set, num_classes

def eurosat_800_200_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = EuroSAT(data_path, 'train800', train_transforms, download=download)
    val_set = EuroSAT(data_path, 'val200', val_transforms, download=False)
    num_classes = 10
    return train_set, val_set, num_classes

def eurosat_full_datasets(data_path, train_transforms, val_transforms, download=False):
    train_set = EuroSAT(data_path, 'train', train_transforms, download=download)
    val_set = EuroSAT(data_path, 'test', val_transforms, download=False)
    num_classes = 10
    return train_set, val_set, num_classes

def svhn_1k_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = SVHN(data_path, 'train', train_transforms, download=download, type='train1000')
    val_set = SVHN(data_path, 'test', val_transforms, download=download, type='all')
    num_classes = 10
    return train_set, val_set, num_classes

def svhn_800_200_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = SVHN(data_path, 'train', train_transforms, download=download, type='train800')
    val_set = SVHN(data_path, 'train', val_transforms, download=download, type='val200')
    num_classes = 10
    return train_set, val_set, num_classes

def svhn_full_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = SVHN(data_path, 'train', train_transforms, download=download, type='all')
    val_set = SVHN(data_path, 'test', val_transforms, download=download, type='all')
    num_classes = 10
    return train_set, val_set, num_classes

##############################
#  fine-grained datasets
##############################

def stanfordcars_full_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = StanfordCars(data_path, 'train', train_transforms, download=download, type='all')
    val_set = StanfordCars(data_path, 'test', val_transforms, download=download, type='all')
    num_classes = 196
    return train_set, val_set, num_classes

def cub_full_datasets(data_path, train_transforms, val_transforms, download=True):
    train_set = Cub2011(data_path, True, train_transforms, download=download)
    val_set = Cub2011(data_path, False, val_transforms, download=download)
    num_classes = 200
    return train_set, val_set, num_classes

###############################
#    long-tailed datasets
###############################

def imbalanced_cifar100_full_datasets_100(data_path, train_transforms, val_transforms):
    train_set = IMBALANCECIFAR100(phase='train', imbalance_ratio=0.01, transform=train_transforms, root=data_path)
    val_set = IMBALANCECIFAR100(phase='test', imbalance_ratio=1, transform=val_transforms, root=data_path)
    num_classes = 100
    return train_set, val_set, num_classes

def imbalanced_cifar100_full_datasets_50(data_path, train_transforms, val_transforms):
    train_set = IMBALANCECIFAR100(phase='train', imbalance_ratio=0.02, transform=train_transforms, root=data_path)
    val_set = IMBALANCECIFAR100(phase='test', imbalance_ratio=1, transform=val_transforms, root=data_path)
    num_classes = 100
    return train_set, val_set, num_classes

def imbalanced_cifar100_full_datasets_10(data_path, train_transforms, val_transforms):
    train_set = IMBALANCECIFAR100(phase='train', imbalance_ratio=0.1, transform=train_transforms, root=data_path)
    val_set = IMBALANCECIFAR100(phase='test', imbalance_ratio=1, transform=val_transforms, root=data_path)
    num_classes = 100
    return train_set, val_set, num_classes

def Places365_LT_full_datasets(data_path, train_transforms, val_transforms):
    train_set = LT_Dataset(root=data_path, txt='vtab/Places_LT_train.txt', transform=train_transforms)
    val_set = LT_Dataset(root=data_path, txt='vtab/Places_LT_test.txt', transform=val_transforms)
    num_classes = 365
    return train_set, val_set, num_classes

def ImageNet_LT_full_datasets(data_path, train_transforms, val_transforms):
    train_set = LT_Dataset(root=data_path, txt='vtab/ImageNet_LT_train.txt', transform=train_transforms)
    val_set = LT_Dataset(root=data_path, txt='vtab/ImageNet_LT_test.txt', transform=val_transforms)
    num_classes = 1000
    return train_set, val_set, num_classes

def Places365_LT_twoview_full_datasets(data_path, train_transforms, val_transforms):
    train_set = LT_Dataset_twoview(root=data_path, txt='vtab/Places_LT_train.txt', transform=train_transforms)
    val_set = LT_Dataset(root=data_path, txt='vtab/Places_LT_test.txt', transform=val_transforms)
    num_classes = 365
    return train_set, val_set, num_classes

def create_datasets(data_path, train_transforms, val_transforms, name='cifar100', type='1000'):
    if type == '1000':
        if name == 'cifar100':
            return cifar100_1k_datasets(data_path, train_transforms, val_transforms)
        elif name == 'flowers102':
            return flowers102_1k_datasets(data_path, train_transforms, val_transforms)
        elif name == 'caltech101':
            return caltech101_1k_datasets(data_path, train_transforms, val_transforms)
        elif name == 'svhn':
            return svhn_1k_datasets(data_path, train_transforms, val_transforms)
        elif name == 'eurosat':
            return eurosat_1k_datasets(data_path, train_transforms, val_transforms)
    elif type == '800':
        if name == 'cifar100':
            return cifar100_800_200_datasets(data_path, train_transforms, val_transforms)
        elif name == 'flowers102':
            return flowers102_800_200_datasets(data_path, train_transforms, val_transforms)
        elif name == 'caltech101':
            return caltech101_800_200_datasets(data_path, train_transforms, val_transforms)
        elif name == 'svhn':
            return svhn_800_200_datasets(data_path, train_transforms, val_transforms)
        elif name == 'eurosat':
            return eurosat_800_200_datasets(data_path, train_transforms, val_transforms)
    elif type == 'full':
        if name == 'cifar100':
            return cifar100_full_datasets(data_path, train_transforms, val_transforms)
        elif name == 'flowers102':
            return flowers102_full_datasets(data_path, train_transforms, val_transforms)
        elif name == 'caltech101':
            return caltech101_full_datasets(data_path, train_transforms, val_transforms)
        elif name == 'svhn':
            return svhn_full_datasets(data_path, train_transforms, val_transforms)
        elif name == 'eurosat':
            return eurosat_full_datasets(data_path, train_transforms,val_transforms)
        elif name == 'stanfordcars':
            return stanfordcars_full_datasets(data_path, train_transforms, val_transforms)
        elif name == 'cub2011':
            return cub_full_datasets(data_path, train_transforms, val_transforms)
        elif name == 'imbalancedcifar100_100':
            return imbalanced_cifar100_full_datasets_100(data_path, train_transforms, val_transforms)
        elif name == 'imbalancedcifar100_50':
            return imbalanced_cifar100_full_datasets_50(data_path, train_transforms, val_transforms)
        elif name == 'imbalancedcifar100_10':
            return imbalanced_cifar100_full_datasets_10(data_path, train_transforms, val_transforms)
        elif name == 'places365':
            return Places365_LT_full_datasets('/hdd1/dongbowen/data/places365_standard', train_transforms, val_transforms)
        elif name == 'imagenet_lt':
            return ImageNet_LT_full_datasets('/home/ubuntu/dongbowen/data/imagenet/', train_transforms, val_transforms)
