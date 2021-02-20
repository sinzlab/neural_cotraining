import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from imagecorruptions import corrupt
from imagecorruptions.corruptions import *
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import datasets
from bias_transfer.configs.dataset import ImageDatasetConfig
from bias_transfer.dataset.utils import get_dataset, create_ImageFolder_format, ManyDatasetsInOne
from bias_transfer.dataset.npy_dataset import NpyDataset
from bias_transfer.trainer.main_loop_modules import NoiseAugmentation
from .dataset_filters import *
from functools import partial

DATASET_URLS = {
    "TinyImageNet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "CIFAR10-C": "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar",
    "CIFAR100-C": "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar",
    "TinyImageNet-C": "https://zenodo.org/record/2536630/files/Tiny-ImageNet-C.tar",
    "TinyImageNet-ST": "https://informatikunihamburgde-my.sharepoint.com/:u:/g/personal/shahd_safarani_informatik_uni-hamburg_de/EZhUKKVXTvRHlqi2HXHaIjEBLmAv4tQP8olvdGNRoWrPqA?e=8kSrHI&download=1",
    "ImageNet": None,
    "ImageNet-C": {
        "blur": "https://zenodo.org/record/2235448/files/blur.tar",
        "digital": "https://zenodo.org/record/2235448/files/digital.tar",
        "extra": "https://zenodo.org/record/2235448/files/extra.tar",
        "noise": "https://zenodo.org/record/2235448/files/noise.tar",
        "weather": "https://zenodo.org/record/2235448/files/weather.tar",
    },
}


def img_dataset_loader(seed, **config):
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
    seed = 1000
    config = ImageDatasetConfig.from_dict(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    def apply_noise(x):
        if config.apply_noise.get("noise_std", False):
            std = config.apply_noise.get("noise_std")
            noise_config = {
                "std": {np.random.choice(list(std.keys()), p=list(std.values())): 1.0}
            }
        elif config.apply_noise.get("noise_snr", False):
            snr = config.apply_noise.get("noise_snr")
            noise_config = {
                "snr": {np.random.choice(list(snr.keys()), p=list(snr.values())): 1.0}
            }
        else:
            std = {0.08: 0.1, 0.12: 0.1, 0.18: 0.1, 0.26: 0.1, 0.38: 0.1, -1: 0.5}
            noise_config = {
                "std": {np.random.choice(list(std.keys()), p=list(std.values())): 1.0}
            }
        return NoiseAugmentation.apply_noise(x, device="cpu", **noise_config)[0]

    def apply_only_noise(x):
        std = {0.08: 0.2, 0.12: 0.2, 0.18: 0.2, 0.26: 0.2, 0.38: 0.2}
        noise_config = {
            "std": {np.random.choice(list(std.keys()), p=list(std.values())): 1.0}
        }
        return NoiseAugmentation.apply_noise(x, device="cpu", **noise_config)[0]

    def apply_one_noise(x, std_value=None):
        noise_config = {
            "std": {std_value: 1.0}
        }

        return NoiseAugmentation.apply_noise(x, device="cpu", **noise_config)[0]

    if config.dataset_cls == "ImageNet":
        transform_train = [
            transforms.RandomResizedCrop(config.input_size)
            if config.apply_augmentation
            else None,
            transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
            transforms.ToTensor(),
            transforms.Lambda(apply_noise) if config.apply_noise else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        if config.apply_noise.get("representation_matching", False):
            transform_train_only_noise = [
                transforms.RandomResizedCrop(config.input_size)
                if config.apply_augmentation
                else None,
                transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
                transforms.ToTensor(),
                transforms.Lambda(apply_only_noise) if config.apply_noise else None,
                transforms.Grayscale() if config.apply_grayscale else None,
                transforms.Normalize(config.train_data_mean, config.train_data_std)
                if config.apply_normalization
                else None,
            ]
        transform_val_in_domain = [
            transforms.Resize(config.in_resize),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            transforms.Lambda(apply_noise) if config.apply_noise else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val_out_domain = [
            transforms.Resize(config.in_resize),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            transforms.Lambda(apply_noise) if not config.apply_noise else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_test_c = [
            transforms.Resize(config.in_resize),
            transforms.CenterCrop(config.input_size),
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val_gauss_levels = {}
        for level in [0.0,0.05,0.1,0.2,0.3,0.5,1.0]:
            transform_val_gauss_levels[level] = [
                    transforms.Resize(config.in_resize),
                    transforms.CenterCrop(config.input_size),
                    transforms.ToTensor(),
                    transforms.Lambda(partial(apply_one_noise, std_value=level)),
                    transforms.Grayscale() if config.apply_grayscale else None,
                    transforms.Normalize(config.train_data_mean, config.train_data_std)
                    if config.apply_normalization
                    else None,
                ]
    else:
        transform_train = [
            transforms.RandomCrop(config.input_size, padding=4)
            if config.apply_augmentation
            else None,
            transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
            transforms.RandomRotation(15) if config.apply_augmentation else None,
            transforms.ToTensor(),
            transforms.Lambda(apply_noise) if config.apply_noise else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        if config.apply_noise.get("representation_matching", False):
            transform_train_only_noise = [
                transforms.RandomCrop(config.input_size, padding=4)
                if config.apply_augmentation
                else None,
                transforms.RandomHorizontalFlip() if config.apply_augmentation else None,
                transforms.RandomRotation(15) if config.apply_augmentation else None,
                transforms.ToTensor(),
                transforms.Lambda(apply_only_noise) if config.apply_noise else None,
                transforms.Grayscale() if config.apply_grayscale else None,
                transforms.Normalize(config.train_data_mean, config.train_data_std)
                if config.apply_normalization
                else None,
            ]
        transform_val_in_domain = [
            transforms.ToTensor(),
            transforms.Lambda(apply_noise) if config.apply_noise else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val_out_domain = [
            transforms.ToTensor(),
            transforms.Lambda(apply_noise) if not config.apply_noise else None,
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_test_c = [
            transforms.Grayscale() if config.apply_grayscale else None,
            transforms.ToTensor(),
            transforms.Normalize(config.train_data_mean, config.train_data_std)
            if config.apply_normalization
            else None,
        ]
        transform_val_gauss_levels = {}
        for level in [0.0,0.05,0.1,0.2,0.3,0.5,1.0]:
            transform_val_gauss_levels[level] = [
                    transforms.ToTensor(),
                    transforms.Lambda(partial(apply_one_noise, std_value=level)),
                    transforms.Grayscale() if config.apply_grayscale else None,
                    transforms.Normalize(config.train_data_mean, config.train_data_std)
                    if config.apply_normalization
                    else None,
                ]

    transform_test_c = transforms.Compose(
        list(filter(lambda x: x is not None, transform_test_c))
    )
    transform_val_in_domain = transforms.Compose(
        list(filter(lambda x: x is not None, transform_val_in_domain))
    )
    transform_val_out_domain = transforms.Compose(
        list(filter(lambda x: x is not None, transform_val_out_domain))
    )
    transform_train = transforms.Compose(
        list(filter(lambda x: x is not None, transform_train))
    )
    if config.apply_noise.get("representation_matching", False):
        transform_train_only_noise = transforms.Compose(
            list(filter(lambda x: x is not None, transform_train_only_noise))
        )
    for level in list(transform_val_gauss_levels.keys()):
        transform_val_gauss_levels[level] = transforms.Compose(
                list(filter(lambda x: x is not None, transform_val_gauss_levels[level]))
            )

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (config.valid_size >= 0) and (config.valid_size <= 1), error_msg

    # load the dataset
    if (
        config.dataset_cls in list(torchvision.datasets.__dict__.keys())
        and config.dataset_cls != "ImageNet"
    ):
        dataset_cls = eval("torchvision.datasets." + config.dataset_cls)
        train_dataset = dataset_cls(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=transform_train,
        )

        if config.apply_noise.get("representation_matching", False):
            train_dataset_only_noise = dataset_cls(
                root=config.data_dir,
                train=True,
                download=config.download,
                transform=transform_train_only_noise,
            )
            train_dataset = ManyDatasetsInOne([train_dataset, train_dataset_only_noise])

        valid_dataset_in_domain = dataset_cls(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=transform_val_in_domain,
        )

        test_dataset_in_domain = dataset_cls(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=transform_val_in_domain,
        )

        valid_dataset_out_domain = dataset_cls(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=transform_val_out_domain,
        )

        test_dataset_out_domain = dataset_cls(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=transform_val_out_domain,
        )
        val_gauss_datasets = {}
        for level in list(transform_val_gauss_levels.keys()):
            val_gauss_datasets[level] = dataset_cls(
                    root=config.data_dir,
                    train=True,
                    download=config.download,
                    transform=transform_val_gauss_levels[level],
                )
    else:
        dataset_dir = get_dataset(
            DATASET_URLS[config.dataset_cls],
            config.data_dir,
            dataset_cls=config.dataset_cls,
            download=config.dowload,
        )

        if config.dataset_cls != "ImageNet":
            create_ImageFolder_format(dataset_dir)
            imagenet_val_dir = os.path.join(dataset_dir, "imagenet_val", "images")

        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val", "images")

        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

        if config.apply_noise.get("representation_matching", False):
            train_dataset_only_noise = datasets.ImageFolder(train_dir, transform=transform_train_only_noise)
            train_dataset = ManyDatasetsInOne([train_dataset, train_dataset_only_noise])

        valid_dataset_in_domain = datasets.ImageFolder(train_dir, transform=transform_val_in_domain)

        test_dataset_in_domain = datasets.ImageFolder(val_dir, transform=transform_val_in_domain)

        valid_dataset_out_domain = datasets.ImageFolder(train_dir, transform=transform_val_out_domain)

        test_dataset_out_domain = datasets.ImageFolder(val_dir, transform=transform_val_out_domain)

        val_gauss_datasets = {}
        for level in list(transform_val_gauss_levels.keys()):
            val_gauss_datasets[level] = datasets.ImageFolder(train_dir, transform=transform_val_gauss_levels[level])

        if config.add_fly_corrupted_test:
            fly_test_datasets = {}
            if config.dataset_cls != "ImageNet":
                imagenet_fly_test_datasets = {}
            for fly_noise_type, levels in config.add_fly_corrupted_test.items():
                fly_test_datasets[fly_noise_type] = {}
                if config.dataset_cls != "ImageNet":
                    imagenet_fly_test_datasets[fly_noise_type] = {}
                for level in levels:

                    class Noise(object):
                        def __init__(self, noise_type, severity):
                            self.noise_type = noise_type
                            self.severity = severity

                        def __call__(self, pic):
                            pic = np.asarray(pic)
                            img = corrupt(pic, corruption_name=self.noise_type, severity=self.severity)
                            return img

                    if config.dataset_cls == "ImageNet":
                        transform_fly_test = [
                            transforms.Resize(config.in_resize),
                            transforms.CenterCrop(config.input_size),
                            Noise(fly_noise_type, level),
                            transforms.ToPILImage() if config.apply_grayscale else None,
                            transforms.Grayscale() if config.apply_grayscale else None,
                            transforms.ToTensor(),
                            transforms.Normalize(config.train_data_mean, config.train_data_std)
                            if config.apply_normalization
                            else None,
                        ]
                        transform_fly_test = transforms.Compose(
                            list(filter(lambda x: x is not None, transform_fly_test))
                        )
                        fly_test_datasets[fly_noise_type][level] = datasets.ImageFolder(val_dir,
                                                                                        transform=transform_fly_test)
                    else:
                        transform_fly_test = [
                            Noise(fly_noise_type, level),
                            transforms.ToPILImage() if config.apply_grayscale else None,
                            transforms.Grayscale() if config.apply_grayscale else None,
                            transforms.ToTensor(),
                            transforms.Normalize(config.train_data_mean, config.train_data_std)
                            if config.apply_normalization
                            else None,
                        ]
                        imagenet_transform_fly_test = [
                            transforms.Resize(config.in_resize),
                            transforms.CenterCrop(config.input_size),
                            Noise(fly_noise_type, level),
                            transforms.ToPILImage() if config.apply_grayscale else None,
                            transforms.Grayscale() if config.apply_grayscale else None,
                            transforms.ToTensor(),
                            transforms.Normalize(config.train_data_mean, config.train_data_std)
                            if config.apply_normalization
                            else None,
                        ]
                        transform_fly_test = transforms.Compose(
                            list(filter(lambda x: x is not None, transform_fly_test))
                        )
                        imagenet_transform_fly_test = transforms.Compose(
                            list(filter(lambda x: x is not None, imagenet_transform_fly_test))
                        )
                        fly_test_datasets[fly_noise_type][level] = datasets.ImageFolder(val_dir,
                                                                                        transform=transform_fly_test)
                        imagenet_fly_test_datasets[fly_noise_type][level] = datasets.ImageFolder(imagenet_val_dir,
                                                                                        transform=imagenet_transform_fly_test)

    if config.add_stylized_test:
        st_dataset_dir = get_dataset(
            DATASET_URLS[config.dataset_cls + "-ST"],
            config.data_dir,
            dataset_cls=config.dataset_cls + "-ST",
            download=config.dowload,
        )
        st_test_dataset = datasets.ImageFolder(st_dataset_dir, transform=transform_val_in_domain)


    if config.add_corrupted_test:
        urls = DATASET_URLS[config.dataset_cls + "-C"]
        if not isinstance(urls, dict):
            urls = {"default": urls}
        for key, url in urls.items():
            dataset_dir = get_dataset(
                url,
                config.data_dir,
                dataset_cls=config.dataset_cls + "-C",
                download=config.download,
            )

            c_test_datasets = {}
            for c_category in os.listdir(dataset_dir):
                if config.dataset_cls in ("CIFAR10", "CIFAR100"):
                    if c_category == "labels.npy" or not c_category.endswith(".npy"):
                        continue
                    c_test_datasets[c_category[:-4]] = {}
                    for c_level in range(1, 6):
                        start = (c_level - 1) * 10000
                        end = c_level * 10000
                        c_test_datasets[c_category[:-4]][c_level] = NpyDataset(
                            sample_file=c_category,
                            target_file="labels.npy",
                            root=dataset_dir,
                            start=start,
                            end=end,
                            transform=transform_val_out_domain if config.apply_noise else transform_val_in_domain,
                        )
                else:
                    if not os.path.isdir(os.path.join(dataset_dir, c_category)):
                        continue
                    c_test_datasets[c_category] = {}
                    for c_level in os.listdir(os.path.join(dataset_dir, c_category)):
                        c_test_datasets[c_category][
                            int(c_level)
                        ] = datasets.ImageFolder(
                            os.path.join(dataset_dir, c_category, c_level),
                            transform=transform_test_c,
                        )

    filters = [globals().get(f)(config, train_dataset) for f in config.filters]
    datasets_ = [train_dataset, valid_dataset_in_domain, valid_dataset_out_domain, test_dataset_in_domain, test_dataset_out_domain]
    datasets_ += list(val_gauss_datasets.values())
    if config.add_fly_corrupted_test:
        for fly_ds in fly_test_datasets.values():
            datasets_ += list(fly_ds.values())
        if config.dataset_cls == "TinyImageNet":
            for in_fly_ds in imagenet_fly_test_datasets.values():
                datasets_ += list(in_fly_ds.values())


    if config.add_corrupted_test:
        for c_ds in c_test_datasets.values():
            datasets_ += list(c_ds.values())

    for ds in datasets_:
        for filt in filters:
            filt.apply(ds)

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
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    valid_loader_in_domain = torch.utils.data.DataLoader(
        valid_dataset_in_domain,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    valid_loader_out_domain = torch.utils.data.DataLoader(
        valid_dataset_out_domain,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )

    test_loader_in_domain = torch.utils.data.DataLoader(
        test_dataset_in_domain,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    test_loader_out_domain = torch.utils.data.DataLoader(
        test_dataset_out_domain,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )

    val_gauss_loaders = {}
    for level in val_gauss_datasets.keys():
        val_gauss_loaders[level] = torch.utils.data.DataLoader(
                val_gauss_datasets[level],
                batch_size=config.batch_size,
                sampler=valid_sampler,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                shuffle=False,
            )
    data_loaders = {
        "train": {"img_classification": train_loader},
        "validation": {"img_classification": valid_loader_in_domain},
        "test": {"img_classification": test_loader_in_domain},
        "validation_out_domain": {"img_classification": valid_loader_out_domain},
        "test_out_domain": {"img_classification": test_loader_out_domain},
        "validation_gauss": val_gauss_loaders
    }

    if config.add_stylized_test:
        st_test_loader = torch.utils.data.DataLoader(
            st_test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False,
        )
        data_loaders["st_test"] = st_test_loader

    if config.add_corrupted_test:
        c_test_loaders = {}
        for c_category in c_test_datasets.keys():
            c_test_loaders[c_category] = {}
            for c_level, dataset in c_test_datasets[c_category].items():
                c_test_loaders[c_category][c_level] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                    shuffle=False,
                )
        data_loaders["c_test"] = c_test_loaders

    if config.add_fly_corrupted_test:
        fly_test_loaders = {}
        if config.dataset_cls == "TinyImageNet":
            imagenet_fly_test_loaders = {}
        for fly_noise_type in fly_test_datasets.keys():
            fly_test_loaders[fly_noise_type] = {}
            if config.dataset_cls == "TinyImageNet":
                imagenet_fly_test_loaders[fly_noise_type] = {}
            for level, dataset in fly_test_datasets[fly_noise_type].items():
                fly_test_loaders[fly_noise_type][level] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory,
                    shuffle=False,
                )
                if config.dataset_cls == "TinyImageNet":
                    imagenet_fly_test_loaders[fly_noise_type][level] = torch.utils.data.DataLoader(
                        imagenet_fly_test_datasets[fly_noise_type][level],
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory,
                        shuffle=False,
                    )
        data_loaders["fly_c_test"] = fly_test_loaders
        if config.dataset_cls == "TinyImageNet":
            data_loaders["imagenet_fly_c_test"] = imagenet_fly_test_loaders
    return data_loaders
