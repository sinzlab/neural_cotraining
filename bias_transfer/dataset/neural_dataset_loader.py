import numpy as np
import torch

from nnfabrik import builder
import os
from os import listdir
from os.path import isfile, join


def neural_dataset_loader(seed, **config):
    seed = 1000
    config.pop("comment", None)
    data_dir = config.pop("data_dir", None)
    neuronal_data_path = os.path.join(data_dir, "neuronal_data/")
    config["neuronal_data_files"] = [
        neuronal_data_path + f
        for f in listdir(neuronal_data_path)
        if (isfile(join(neuronal_data_path, f)) and f != "CSRF19_V4_3653663964522.pickle")
    ]
    config["image_cache_path"] = os.path.join(data_dir, "images/individual")
    config["original_image_cache_path"] = os.path.join(data_dir, "images/original/")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_fn = "nnvision.datasets.monkey_static_loader"
    data_loaders = builder.get_data(dataset_fn, config)
    dataloaders = {
        "train": data_loaders["train"],
        "validation": {task: data_loaders["validation"] for task in config['target_types']},
        "test": {task: data_loaders["test"] for task in config['target_types']},
    }
    if "validation_gauss" in data_loaders.keys():
        dataloaders['validation_gauss'] = data_loaders['validation_gauss']
    if "fly_c_test" in data_loaders.keys():
        dataloaders['fly_c_test'] = data_loaders['fly_c_test']
    return dataloaders
