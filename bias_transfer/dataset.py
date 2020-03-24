import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from bias_transfer.configs.dataset import DatasetConfig
import os
import zipfile
import requests
from io import BytesIO


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


def create_ImageFolder_format(dataset_dir: str):
    '''
    This method is responsible for separating validation images into separate sub folders

    Args:
        dataset_dir (str): "/path_to_your_dataset/dataset_folder"
    '''
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def download_images(url: str, data_dir: str, dataset_folder: str = 'tiny-imagenet-200/') -> str:
    '''
    Downloads the dataset from an online downloadable link and
    sets up the folders according to torch ImageFolder required
    format

    Args:
        url (str): download link of the dataset from the internet
        data_dir (str): the directory where to download the dataset
        dataset_folder (str): name of the dataset's folder
    Returns:
        dataset_dir (str): full path to the dataset incl. dataset folder
    '''
    dataset_dir = data_dir + dataset_folder
    if os.path.isdir(dataset_dir):
        print ('Images already downloaded...')
        return dataset_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    r = requests.get(url, stream=True)
    print ('Downloading ' + url )
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    zip_ref.extractall(data_dir)
    zip_ref.close()
    create_ImageFolder_format(dataset_dir)
    return dataset_dir

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
        dataset_dir = download_images('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                        config.data_dir)

        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val', 'images')

        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

        valid_dataset = datasets.ImageFolder(train_dir, transform=transform_base)

        test_dataset =  datasets.ImageFolder(val_dir,
                                        transform=transform_base)

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

    return {"train": train_loader,
            "val": valid_loader,
            "test": test_loader}
