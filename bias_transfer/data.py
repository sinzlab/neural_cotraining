import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100


def compute_mean_std(train_set):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    mean = np.mean(train_set.dataset.data, axis=(0, 1, 2)) / 255
    std = np.std(train_set.dataset.data, axis=(0, 1, 2)) / 255
    return mean, std


class GaussianNoiseTransform:
    def __init__(self, use_random_seed = False):
        self.use_random_seed = use_random_seed

    def __call__(self, img, index):
        if self.use_random_seed:
            torch.random.manual_seed(index)
            return img * 0 + torch.rand(1)
        else:
            return img * 0 + torch.rand(1)


def get_training_data_loader(mean, std, batch_size=64, num_workers=2, shuffle=True, cifar100=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_list = [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(15),
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)
                      ]
    transform_train = transforms.Compose(transform_list)
    if cifar100:
        cifar_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    else:
        cifar_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=transform_train)
    cifar_training_loader = DataLoader(
        cifar_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_training_loader


def get_test_data_loader(mean, std, batch_size=16, num_workers=2, shuffle=False, cifar100=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize(mean, std)]
    transform_test = transforms.Compose(transform_list)
    if cifar100:
        cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar_test_loader = DataLoader(
        cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar_test_loader
