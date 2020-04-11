import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from bias_transfer.configs.dataset import DatasetConfig
import os

from bias_transfer.dataset.utils import download_dataset, create_ImageFolder_format
from bias_transfer.dataset.npy_dataset import NpyDataset

DATASET_URLS = {
    "TinyImageNet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "CIFAR10-C": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar",
    "CIFAR100-C": "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar",
    "TinyImageNet-C": "https://zenodo.org/record/2536630/files/Tiny-ImageNet-C.tar",
}


def dataset_loader(seed, **config):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    config = DatasetConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform_list_base = [transforms.ToTensor()]
    if config.apply_normalization:
        transform_list_base += [transforms.Normalize(config.train_data_mean, config.train_data_std)]
    if config.apply_augmentation:
        transform_list = [transforms.RandomCrop(config.input_size, padding=4),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomRotation(15),
                          ] + transform_list_base
    else:
        transform_list = transform_list_base
    transform_base = transforms.Compose(transform_list_base)
    transform_train = transforms.Compose(transform_list)

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((config.valid_size >= 0) and (config.valid_size <= 1)), error_msg

    # load the dataset
    if config.dataset_cls in list(torchvision.datasets.__dict__.keys()):
        dataset_cls = eval("torchvision.datasets." + config.dataset_cls)
        train_dataset = dataset_cls(
            root=config.data_dir, train=True,
            download=True, transform=transform_train,
        )

        valid_dataset = dataset_cls(
            root=config.data_dir, train=True,
            download=True, transform=transform_base,
        )

        test_dataset = dataset_cls(
            root=config.data_dir, train=False,
            download=True, transform=transform_base,
        )
    else:
        dataset_dir = download_dataset(DATASET_URLS[config.dataset_cls],
                                       config.data_dir, dataset_cls=config.dataset_cls)
        create_ImageFolder_format(dataset_dir)

        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val', 'images')

        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

        valid_dataset = datasets.ImageFolder(train_dir, transform=transform_base)

        test_dataset = datasets.ImageFolder(val_dir,
                                            transform=transform_base)

    if config.add_corrupted_test:
        dataset_dir = download_dataset(DATASET_URLS[config.dataset_cls + "-C"],
                                       config.data_dir, dataset_cls=config.dataset_cls + "-C")

        c_test_datasets = {}
        for c_category in os.listdir(dataset_dir):
            if c_category == "labels.npy" or not c_category.endswith(".npy"):
                continue
            c_test_datasets[c_category[:-4]] = {}
            if config.dataset_cls in ("CIFAR10", "CIFAR100"):
                for c_level in range(1, 6):
                    start = (c_level - 1) * 10000
                    end = c_level * 10000
                    c_test_datasets[c_category[:-4]][c_level] = NpyDataset(
                        sample_file=c_category, target_file="labels.npy", root=dataset_dir,
                        start=start, end=end, transform=transform_base
                    )
            else:
                for c_level in os.listdir(os.path.join(dataset_dir, c_category)):
                    c_test_datasets[c_category][c_level] = datasets.ImageFolder(
                        os.path.join(dataset_dir, c_category, c_level),
                        transform=transform_base
                    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(config.valid_size * num_train))

    if config.shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=config.pin_memory, shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, sampler=valid_sampler,
        num_workers=config.num_workers, pin_memory=config.pin_memory, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, pin_memory=config.pin_memory, shuffle=False
    )
    data_loaders = {"train": train_loader,
                    "val": valid_loader,
                    "test": test_loader}

    if config.add_corrupted_test:
        c_test_loaders = {}
        for c_category in c_test_datasets.keys():
            c_test_loaders[c_category] = {}
            for c_level, dataset in c_test_datasets[c_category].items():
                c_test_loaders[c_category][c_level] = torch.utils.data.DataLoader(
                    dataset, batch_size=config.batch_size,
                    num_workers=config.num_workers, pin_memory=config.pin_memory, shuffle=False
                )
        data_loaders["c_test"] = c_test_loaders

    return data_loaders
