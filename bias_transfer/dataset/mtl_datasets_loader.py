from .img_dataset_loader import img_dataset_loader
from .neural_dataset_loader import neural_dataset_loader
import os
import torch
import numpy as np
from nnfabrik import builder
from os.path import isfile, join

def mtl_datasets_loader(seed, **config):
    neural_dataset_config = config.pop("neural_dataset_config")
    img_dataset_config = config.pop("img_dataset_config")

    neural_dataset_config.pop("seed")

    neural_dataset_loaders = neural_dataset_loader(seed, **neural_dataset_config)
    img_dataset_loaders = img_dataset_loader(seed, **img_dataset_config)

    data_loaders = neural_dataset_loaders
    data_loaders["train"]["img_classification"] = img_dataset_loaders["train"][
        "img_classification"
    ]
    data_loaders["validation"]["img_classification"] = img_dataset_loaders[
        "validation"
    ]["img_classification"]
    data_loaders["test"]["img_classification"] = img_dataset_loaders["test"][
        "img_classification"
    ]
    if "c_test" in img_dataset_loaders:
        data_loaders["c_test"] = img_dataset_loaders["c_test"]
    if "st_test" in img_dataset_loaders:
        data_loaders["st_test"] = img_dataset_loaders["st_test"]
    return data_loaders




def shared_dataset_loader(seed, **config):
    config.pop("comment", None)
    config.pop("img_dataset_dict")
    img_dataset_config = config.pop("img_dataset_config")
    img_dataset_loaders = img_dataset_loader(seed, **img_dataset_config)
    data_dir = config.pop("data_dir", None)
    neuronal_data_path = os.path.join(data_dir, "neuronal_data/")
    config["neuronal_data_files"] = [
        neuronal_data_path + f
        for f in os.listdir(neuronal_data_path)
        if isfile(join(neuronal_data_path, f))
    ]
    config["image_cache_path"] = os.path.join(data_dir, "images/individual")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_fn = "nnvision.datasets.monkey_static_loader"
    data_loaders = builder.get_data(dataset_fn, config)
    if "c_test" in img_dataset_loaders:
        data_loaders["c_test"] = img_dataset_loaders["c_test"]

    return data_loaders