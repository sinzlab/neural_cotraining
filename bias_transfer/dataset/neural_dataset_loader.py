import numpy as np
import torch

from nnfabrik import builder
import os
from os import listdir
from os.path import isfile, join


def neural_dataset_loader(seed, **config):
    config.pop("comment", None)
    data_dir = config.pop("data_dir", None)
    neuronal_data_path = os.path.join(data_dir, "neuronal_data/")
    config["neuronal_data_files"] = [
        neuronal_data_path + f
        for f in listdir(neuronal_data_path)
        if isfile(join(neuronal_data_path, f))
    ]
    config["image_cache_path"] = os.path.join(data_dir, "images/individual")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_fn = "nnvision.datasets.monkey_static_loader"
    data_loaders = builder.get_data(dataset_fn, config)
    dataloaders = {
        "train": data_loaders["train"],
        "validation": {"neural": data_loaders["validation"]},
        "test": {"neural": data_loaders["test"]},
    }
    return dataloaders
