import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from bias_transfer.configs.dataset import DatasetConfig


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
        transform_list = [transforms.RandomCrop(32, padding=4),
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

    # visualize some images
    if config.show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=config.shuffle,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return {"train": train_loader,
            "val": valid_loader,
            "test": test_loader}
